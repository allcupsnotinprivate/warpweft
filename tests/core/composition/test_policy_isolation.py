"""Operation- and scope-local policy settings in the shared link store."""

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

from _support.axes import context_axes
import anyio
import pytest

from warpweft.core.axes import ScopeSpec
from warpweft.core.clock import ManualClock
from warpweft.core.component import AComponent, EmptySettings, Lifetime, invocable
from warpweft.core.component.policy import Policy
from warpweft.core.composition import Container, DictSettingsResolver
from warpweft.core.errors import AttemptTimeout, CircuitOpen, RetryExhausted, TransientError
from warpweft.core.outcome import Outcome

pytestmark = pytest.mark.anyio

Make = Callable[..., AbstractAsyncContextManager[Container]]
ACME = (("tenant", "acme"),)
GLOBEX = (("tenant", "globex"),)


class TimeoutPolicies(AComponent[EmptySettings, None, str]):
    name = "timeout-policies"

    @staticmethod
    async def _wait() -> str:
        await anyio.sleep(0.05)
        return "ok"

    @invocable(policy=Policy(overrides={"timeout": {"seconds": 0.01}}))
    async def short(self) -> str:
        return await self._wait()

    @invocable(policy=Policy(overrides={"timeout": {"seconds": 1.0}}))
    async def long(self) -> str:
        return await self._wait()


class RetryPolicies(AComponent[EmptySettings, None, str]):
    name = "retry-policies"

    @invocable(policy=Policy(overrides={"retry": {"attempts": 1}}))
    async def single(self) -> str:
        raise TransientError("single")

    @invocable(policy=Policy(overrides={"retry": {"attempts": 3}}))
    async def triple(self) -> str:
        raise TransientError("triple")


class CachePolicies(AComponent[EmptySettings, None, str]):
    name = "cache-policies"

    def __init__(self, settings: EmptySettings) -> None:
        super().__init__(settings)
        self.calls = {"short": 0, "long": 0}

    @invocable(policy=Policy(overrides={"cache": {"ttl": 1.0}}))
    async def short(self) -> int:
        self.calls["short"] += 1
        return self.calls["short"]

    @invocable(policy=Policy(overrides={"cache": {"ttl": 100.0}}))
    async def long(self) -> int:
        self.calls["long"] += 1
        return self.calls["long"]


@pytest.mark.parametrize("order", [("short", "long"), ("long", "short")])
async def test_timeout_settings_are_isolated_between_methods(container: Make, order: tuple[str, str]) -> None:
    config = {"timeout-policies": {"policy": {"timeout": {"seconds": 1.0}}}}
    async with container(TimeoutPolicies, config=config) as c:
        for method in order:
            if method == "short":
                with pytest.raises(AttemptTimeout):
                    await c.invoke("timeout-policies", method)
            else:
                assert (await c.invoke("timeout-policies", method)).value == "ok"


@pytest.mark.parametrize("order", [("single", "triple"), ("triple", "single")])
async def test_retry_settings_are_isolated_between_methods(container: Make, order: tuple[str, str]) -> None:
    retry = {"attempts": 3, "base_delay": 0.0, "max_delay": 0.01, "jitter": False}
    async with container(RetryPolicies, config={"retry-policies": {"policy": {"retry": retry}}}) as c:
        for method in order:
            with pytest.raises(RetryExhausted) as caught:
                await c.invoke("retry-policies", method)
            assert caught.value.attempts == (1 if method == "single" else 3)


@pytest.mark.parametrize("order", [("short", "long"), ("long", "short")])
async def test_cache_settings_are_isolated_between_methods(container: Make, order: tuple[str, str]) -> None:
    clock = ManualClock()
    cache = {"ttl": 100.0, "max_entries": 10}
    async with container(CachePolicies, config={"cache-policies": {"policy": {"cache": cache}}}, clock=clock) as c:
        component = await c.get(CachePolicies)
        for method in order:
            assert (await c.invoke("cache-policies", method)).source == "live"
        assert component.calls == {"short": 1, "long": 1}

        clock.advance(2.0)
        for method in order:
            outcome = await c.invoke("cache-policies", method)
            assert outcome.source == ("live" if method == "short" else "cache")
        assert component.calls == {"short": 2, "long": 1}


class ScopedRetry(AComponent[EmptySettings, None, str]):
    name = "scoped-retry"
    lifetime = Lifetime.SCOPED
    scope = ScopeSpec(("tenant",))

    @invocable
    async def fail(self) -> str:
        raise TransientError("down")


async def test_concurrent_first_use_isolates_tenant_retry_overrides(container: Make) -> None:
    axes, handles = context_axes("tenant")
    resolver = DictSettingsResolver(
        {
            ("scoped-retry", ACME): {"policy": {"retry": {"attempts": 1}}},
            ("scoped-retry", GLOBEX): {"policy": {"retry": {"attempts": 3}}},
        }
    )
    retry = {"attempts": 2, "base_delay": 0.0, "max_delay": 0.01, "jitter": False}
    attempts: dict[str, int] = {}

    async def invoke(tenant: str) -> None:
        with handles["tenant"].use(tenant), pytest.raises(RetryExhausted) as caught:
            await c.invoke("scoped-retry", "fail")
        attempts[tenant] = caught.value.attempts

    async with (
        container(
            ScopedRetry, config={"scoped-retry": {"policy": {"retry": retry}}}, axes=axes, resolver=resolver
        ) as c,
        anyio.create_task_group() as tg,
    ):
        tg.start_soon(invoke, "acme")
        tg.start_soon(invoke, "globex")

    assert attempts == {"acme": 1, "globex": 3}


class ScopedCache(AComponent[EmptySettings, None, str]):
    name = "scoped-cache"
    lifetime = Lifetime.SCOPED
    scope = ScopeSpec(("tenant",))

    def __init__(self, settings: EmptySettings) -> None:
        super().__init__(settings)
        self.calls = 0
        self.release = anyio.Event()

    @invocable
    async def fetch(self) -> int:
        self.calls += 1
        await self.release.wait()
        return self.calls


async def let_tasks_run(times: int = 20) -> None:
    for _ in range(times):
        await anyio.lowlevel.checkpoint()


async def test_rebuilt_scoped_chains_preserve_cache_hits_and_single_flight(container: Make) -> None:
    axes, handles = context_axes("tenant")
    cache = {"ttl": 100.0, "max_entries": 10}
    outcomes: list[Outcome[Any]] = []

    async def invoke() -> None:
        outcomes.append(await c.invoke("scoped-cache", "fetch"))

    async with container(ScopedCache, config={"scoped-cache": {"policy": {"cache": cache}}}, axes=axes) as c:
        with handles["tenant"].use("acme"):
            async with anyio.create_task_group() as tg:
                tg.start_soon(invoke)
                tg.start_soon(invoke)
                await let_tasks_run()
                component = await c.get(ScopedCache)
                assert component.calls == 1  # concurrent misses coalesce across rebuilt chains
                component.release.set()

            assert {outcome.source for outcome in outcomes} == {"live", "cache"}
            for _ in range(3):
                assert (await c.invoke("scoped-cache", "fetch")).source == "cache"
            assert component.calls == 1


class EndpointFailure(AComponent[EmptySettings, None, str]):
    calls: dict[str, int] = {}

    def endpoint(self) -> str:
        return "shared"

    @invocable
    async def fail(self) -> str:
        type(self).calls[self.name] = type(self).calls.get(self.name, 0) + 1
        raise TransientError("down")


class EndpointFailureA(EndpointFailure):
    name = "endpoint-failure-a"


class EndpointFailureB(EndpointFailure):
    name = "endpoint-failure-b"


async def test_same_endpoint_breaker_sharing_is_unchanged(container: Make) -> None:
    EndpointFailure.calls = {}
    breaker = {"window": 1, "failure_threshold": 1, "reset_timeout": 100.0}
    config = {
        "endpoint-failure-a": {"policy": {"circuit_breaker": breaker}},
        "endpoint-failure-b": {"policy": {"circuit_breaker": breaker}},
    }
    async with container(EndpointFailureA, EndpointFailureB, config=config) as c:
        with pytest.raises(TransientError):
            await c.invoke("endpoint-failure-a", "fail")
        with pytest.raises(CircuitOpen):
            await c.invoke("endpoint-failure-b", "fail")
    assert EndpointFailure.calls == {"endpoint-failure-a": 1}


class EndpointConcurrency(AComponent[EmptySettings, None, str]):
    active = 0
    peak = 0
    release = anyio.Event()

    def endpoint(self) -> str:
        return "shared"

    @invocable
    async def run(self) -> str:
        cls = EndpointConcurrency
        cls.active += 1
        cls.peak = max(cls.peak, cls.active)
        try:
            await cls.release.wait()
        finally:
            cls.active -= 1
        return "ok"


class EndpointConcurrencyA(EndpointConcurrency):
    name = "endpoint-concurrency-a"


class EndpointConcurrencyB(EndpointConcurrency):
    name = "endpoint-concurrency-b"


async def test_same_endpoint_concurrency_sharing_is_unchanged(container: Make) -> None:
    EndpointConcurrency.active = 0
    EndpointConcurrency.peak = 0
    EndpointConcurrency.release = anyio.Event()
    concurrency = {"inner_limit": 10, "outer_limit": 1}
    config = {
        "endpoint-concurrency-a": {"policy": {"concurrency": concurrency}},
        "endpoint-concurrency-b": {"policy": {"concurrency": concurrency}},
    }

    async def invoke(component: str) -> None:
        await c.invoke(component, "run")

    async with (
        container(EndpointConcurrencyA, EndpointConcurrencyB, config=config) as c,
        anyio.create_task_group() as tg,
    ):
        tg.start_soon(invoke, "endpoint-concurrency-a")
        tg.start_soon(invoke, "endpoint-concurrency-b")
        await let_tasks_run()
        assert EndpointConcurrency.active == 1
        assert EndpointConcurrency.peak == 1
        EndpointConcurrency.release.set()
