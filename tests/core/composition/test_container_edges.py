"""Container edge cases and less-travelled branches."""

from contextvars import ContextVar

import anyio
import pytest

from warpweft.core.axes import Axis, AxisRegistry, ScopeSpec
from warpweft.core.component import AComponent, Criticality, EmptySettings, Health, Lifetime, Policy, invocable
from warpweft.core.composition import Container, Registry
from warpweft.core.composition.endpoint import endpoint_axis
from warpweft.core.errors import ConfigurationError, RetryExhausted, StartupError, TransientError

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

_tenant: ContextVar[str | None] = ContextVar("edges_tenant", default=None)


async def let_tasks_run(times: int = 20) -> None:
    for _ in range(times):
        await anyio.lowlevel.checkpoint()


# --- links builders: timeout / concurrency / cache ---------------------------


class Worker(AComponent[EmptySettings, None, str]):
    name = "worker"

    @invocable
    async def do(self) -> str:
        return "done"


async def test_all_builtin_link_builders_are_exercised() -> None:
    reg = Registry()
    reg.register(Worker)
    config = {
        "worker": {
            "policy": {
                "concurrency": {"inner_limit": 2, "outer_limit": 2},
                "cache": {"ttl": 100.0, "max_entries": 10},
                "circuit_breaker": {"window": 5, "failure_threshold": 5, "reset_timeout": 10.0},
                "retry": {"attempts": 2, "base_delay": 0.0, "max_delay": 1.0},
                "timeout": {"seconds": 5.0},
            }
        }
    }
    container = Container.build(reg, config)
    instances: list[Worker] = []
    for _ in range(2):
        await container.start()
        instances.append(await container.get(Worker))
        assert (await container.invoke("worker", "do")).value == "done"
        await container.stop()
    assert instances[0] is not instances[1]


# --- build/lifecycle branches ------------------------------------------------


async def test_endpoint_axis_supplied_by_caller_is_accepted() -> None:
    reg = Registry()
    reg.register(Worker)
    axes = AxisRegistry()
    axes.register(endpoint_axis())  # container must tolerate a pre-registered endpoint
    container = Container.build(reg, {"worker": {}}, axes=axes)
    await container.start()
    assert (await container.invoke("worker", "do")).value == "done"
    await container.stop()


async def test_start_and_stop_are_idempotent() -> None:
    reg = Registry()
    reg.register(Worker)
    container = Container.build(reg, {"worker": {}})
    await container.stop()  # before start: no-op
    await container.start()
    await container.start()  # second start: no-op
    assert container.started
    await container.stop()


async def test_concurrent_start_and_stop_calls_are_serialized() -> None:
    starts: list[str] = []
    stops: list[str] = []

    class SlowStart(AComponent[EmptySettings, None, str]):
        name = "slow-start"

        async def start(self) -> None:
            starts.append("start")
            await anyio.lowlevel.checkpoint()

        async def stop(self) -> None:
            stops.append("stop")
            await anyio.lowlevel.checkpoint()

        @invocable
        async def go(self) -> str:
            return "ok"

    reg = Registry()
    reg.register(SlowStart)
    container = Container.build(reg, {"slow-start": {}})
    async with anyio.create_task_group() as tg:
        tg.start_soon(container.start)
        tg.start_soon(container.start)
    assert starts == ["start"]
    assert (await container.invoke("slow-start", "go")).value == "ok"

    async with anyio.create_task_group() as tg:
        tg.start_soon(container.stop)
        tg.start_soon(container.stop)
    assert stops == ["stop"]
    assert not container.started


async def test_start_during_drain_waits_and_starts_a_fresh_run() -> None:
    release = anyio.Event()
    starts: list[str] = []

    class Reusable(AComponent[EmptySettings, None, str]):
        async def start(self) -> None:
            starts.append("start")

        @invocable
        async def wait(self) -> str:
            await release.wait()
            return "ok"

    reg = Registry()
    reg.register(Reusable)
    container = Container.build(reg, {"reusable": {}})
    await container.start()

    async def call() -> None:
        await container.invoke("reusable", "wait")

    async with anyio.create_task_group() as tg:
        tg.start_soon(call)
        while container._active_calls == 0:
            await anyio.lowlevel.checkpoint()
        tg.start_soon(container.stop)
        await let_tasks_run()
        tg.start_soon(container.start)
        await let_tasks_run()
        release.set()

    assert container.started
    assert starts == ["start", "start"]
    assert (await container.invoke("reusable", "wait")).value == "ok"
    await container.stop()


async def test_invoke_and_get_are_rejected_immediately_during_drain() -> None:
    release = anyio.Event()

    class Draining(AComponent[EmptySettings, None, str]):
        @invocable
        async def wait(self) -> str:
            await release.wait()
            return "ok"

        @invocable
        async def ping(self) -> str:
            return "pong"

    reg = Registry()
    reg.register(Draining)
    container = Container.build(reg, {"draining": {}})
    await container.start()

    async def call() -> None:
        await container.invoke("draining", "wait")

    async with anyio.create_task_group() as tg:
        tg.start_soon(call)
        while container._active_calls == 0:
            await anyio.lowlevel.checkpoint()
        tg.start_soon(container.stop)
        await let_tasks_run()
        try:
            with pytest.raises(ConfigurationError, match="not started"):
                await container.invoke("draining", "ping")
            with pytest.raises(ConfigurationError, match="not started"):
                await container.get(Draining)
        finally:
            release.set()


async def test_start_failure_can_be_retried_with_fresh_stores() -> None:
    attempts = 0

    class RetryStart(AComponent[EmptySettings, None, str]):
        name = "retry-start"

        async def start(self) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("first attempt fails")

        @invocable
        async def go(self) -> str:
            return "ok"

    reg = Registry()
    reg.register(RetryStart)
    container = Container.build(reg, {"retry-start": {}})
    with pytest.raises(StartupError, match="first attempt fails"):
        await container.start()
    failed_store = container._link_store
    assert failed_store._closed
    assert not container.started

    await container.start()
    assert container._link_store is not failed_store
    assert (await container.invoke("retry-start", "go")).value == "ok"
    await container.stop()


async def test_chain_construction_failure_cleans_up_and_can_be_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    starts: list[str] = []
    stops: list[str] = []

    class ChainWorker(AComponent[EmptySettings, None, str]):
        name = "chain-worker"

        async def start(self) -> None:
            starts.append("start")

        async def stop(self) -> None:
            stops.append("stop")

        @invocable
        async def go(self) -> str:
            return "ok"

    reg = Registry()
    reg.register(ChainWorker)
    container = Container.build(reg, {"chain-worker": {}})
    original_build_chain = container._build_chain
    fail = True

    def fail_once(*args: object, **kwargs: object) -> object:
        nonlocal fail
        if fail:
            fail = False
            raise RuntimeError("chain construction failed")
        return original_build_chain(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(container, "_build_chain", fail_once)
    with pytest.raises(RuntimeError, match="chain construction failed"):
        await container.start()
    failed_store = container._link_store
    assert starts == ["start"]
    assert stops == ["stop"]
    assert failed_store._closed
    assert not container._process
    assert not container.started

    await container.start()
    assert container._link_store is not failed_store
    assert (await container.invoke("chain-worker", "go")).value == "ok"
    await container.stop()
    assert starts == ["start", "start"]
    assert stops == ["stop", "stop"]


async def test_optional_degradation_does_not_leak_into_a_restart() -> None:
    fail = True

    class Recovering(AComponent[EmptySettings, None, str]):
        criticality = Criticality.OPTIONAL

        async def start(self) -> None:
            if fail:
                raise RuntimeError("not yet")

        @invocable
        async def go(self) -> str:
            return "ok"

    reg = Registry()
    reg.register(Recovering)
    container = Container.build(reg, {"recovering": {}})
    await container.start()
    assert container.is_degraded("recovering")
    await container.stop()

    fail = False
    await container.start()
    assert not container.is_degraded("recovering")
    assert (await container.invoke("recovering", "go")).value == "ok"
    await container.stop()


async def test_stop_drains_in_flight_calls() -> None:
    release = anyio.Event()

    class Parker(AComponent[EmptySettings, None, str]):
        name = "parker"

        @invocable
        async def wait(self) -> str:
            await release.wait()
            return "ok"

    reg = Registry()
    reg.register(Parker)
    container = Container.build(reg, {"parker": {}})
    await container.start()

    results: list[str] = []

    async def call() -> None:
        results.append((await container.invoke("parker", "wait")).value)

    async with anyio.create_task_group() as tg:
        tg.start_soon(call)
        while container._active_calls == 0:
            await anyio.lowlevel.checkpoint()
        tg.start_soon(container.stop)
        await let_tasks_run()  # stop enters the drain loop while the call is in flight
        release.set()

    assert results == ["ok"]
    assert not container.started


async def test_forced_drain_does_not_reset_outliving_call_counts() -> None:
    release = anyio.Event()

    class Outliving(AComponent[EmptySettings, None, str]):
        @invocable
        async def wait(self) -> str:
            await release.wait()
            return "ok"

        @invocable
        async def ping(self) -> str:
            return "pong"

    reg = Registry()
    reg.register(Outliving)
    container = Container.build(reg, {"outliving": {}}, drain_timeout=0.001)
    await container.start()

    async def call() -> None:
        await container.invoke("outliving", "wait")

    async with anyio.create_task_group() as tg:
        tg.start_soon(call)
        while container._active_calls == 0:
            await anyio.lowlevel.checkpoint()
        await container.stop()
        assert container._active_calls == 1

        await container.start()
        assert container._active_calls == 1  # the old call still owns its count
        assert (await container.invoke("outliving", "ping")).value == "pong"
        assert container._active_calls == 1
        release.set()

    assert container._active_calls == 0
    await container.stop()


# --- invocation branches -----------------------------------------------------


async def test_invoke_unconfigured_component_is_rejected() -> None:
    reg = Registry()
    reg.register(Worker)
    container = Container.build(reg, {"worker": {}})
    await container.start()
    with pytest.raises(ConfigurationError, match="not configured"):
        await container.invoke("ghost", "do")
    await container.stop()


async def test_component_without_declared_settings() -> None:
    class Bare(AComponent):  # type: ignore[type-arg]  # no settings model
        name = "bare"

        @invocable
        async def go(self) -> bool:
            return True

    reg = Registry()
    reg.register(Bare)
    container = Container.build(reg, {"bare": {}})
    await container.start()
    assert (await container.invoke("bare", "go")).value is True
    await container.stop()


async def test_method_policy_restricts_the_chain() -> None:
    class Restricted(AComponent[EmptySettings, None, str]):
        name = "restricted"

        @invocable(policy=Policy(chain=("timeout",)))
        async def only_timeout(self) -> str:
            return "ok"

    reg = Registry()
    reg.register(Restricted)
    # retry is configured but the method's policy forbids it: it must be skipped.
    config = {
        "restricted": {
            "policy": {"retry": {"attempts": 3, "base_delay": 0.0, "max_delay": 1.0}, "timeout": {"seconds": 5.0}}
        }
    }
    container = Container.build(reg, config)
    await container.start()
    assert (await container.invoke("restricted", "only_timeout")).value == "ok"
    await container.stop()


async def test_unknown_link_name_in_chain_is_skipped() -> None:
    class Future(AComponent[EmptySettings, None, str]):
        name = "future"

        @invocable(policy=Policy(chain=("ghost", "timeout")))
        async def go(self) -> str:
            return "ok"

    reg = Registry()
    reg.register(Future)
    config = {"future": {"policy": {"chain": ["ghost", "timeout"], "timeout": {"seconds": 5.0}}}}
    container = Container.build(reg, config)
    await container.start()
    assert (await container.invoke("future", "go")).value == "ok"  # ghost has no builder, skipped
    await container.stop()


async def test_per_method_override_merges_over_config() -> None:
    class OverrideFlaky(AComponent[EmptySettings, None, str]):
        name = "ov-flaky"

        def __init__(self, settings: EmptySettings) -> None:
            super().__init__(settings)
            self.calls = 0

        @invocable(policy=Policy(overrides={"retry": {"attempts": 2}}))
        async def fetch(self) -> str:
            self.calls += 1
            raise TransientError(f"f{self.calls}")

    reg = Registry()
    reg.register(OverrideFlaky)
    # config allows 5 attempts, the method override caps it at 2.
    config = {"ov-flaky": {"policy": {"retry": {"attempts": 5, "base_delay": 0.0, "max_delay": 1.0}}}}
    container = Container.build(reg, config)
    await container.start()
    with pytest.raises(RetryExhausted) as excinfo:
        await container.invoke("ov-flaky", "fetch")
    assert excinfo.value.attempts == 2  # override won
    await container.stop()


# --- scoped dependency resolution & readiness branches -----------------------


class DepProc(AComponent[EmptySettings, None, str]):
    name = "dep-proc"

    @invocable
    async def ping(self) -> str:
        return "p"


class ScopedA(AComponent[EmptySettings, None, str]):
    name = "scoped-a"
    lifetime = Lifetime.SCOPED
    scope = ScopeSpec(("tenant",))

    @invocable
    async def a(self) -> str:
        return "a"


class ScopedB(AComponent[EmptySettings, None, str]):
    name = "scoped-b"
    lifetime = Lifetime.SCOPED
    scope = ScopeSpec(("tenant",))
    dependencies = ("dep-proc", "scoped-a")

    @invocable
    async def combine(self) -> str:
        proc: DepProc = self.dependency("dep-proc")  # type: ignore[assignment]
        other: ScopedA = self.dependency("scoped-a")  # type: ignore[assignment]
        return f"{await proc.ping()}-{await other.a()}"


def scoped_deps_container() -> Container:
    reg = Registry()
    for cls in (DepProc, ScopedA, ScopedB):
        reg.register(cls)
    axes = AxisRegistry()
    axes.register(Axis(name="tenant", resolver=_tenant.get))
    return Container.build(reg, {"dep-proc": {}, "scoped-a": {}, "scoped-b": {}}, axes=axes)


async def test_scoped_component_resolves_process_and_scoped_dependencies() -> None:
    container = scoped_deps_container()
    await container.start()
    _tenant.set("acme")
    outcome = await container.invoke("scoped-b", "combine")
    assert outcome.value == "p-a"  # reached both a process dep and a scoped dep
    await container.stop()


async def test_readiness_reports_scoped_components_as_ok() -> None:
    container = scoped_deps_container()
    await container.start()
    result = await container.readiness()
    assert result.ready
    assert result.components["scoped-a"].state is Health.OK
    await container.stop()


async def test_readiness_before_start_reports_not_running() -> None:
    reg = Registry()
    reg.register(Worker)
    container = Container.build(reg, {"worker": {}})
    result = await container.readiness()
    assert not result.ready
    assert result.components["worker"].state is Health.UNHEALTHY


async def test_scoped_skips_a_degraded_optional_process_dependency() -> None:
    class OptionalBackend(AComponent[EmptySettings, None, None]):
        name = "opt-backend"
        criticality = Criticality.OPTIONAL

        async def start(self) -> None:
            raise RuntimeError("backend down")

        @invocable
        async def go(self) -> None: ...

    class ScopedFront(AComponent[EmptySettings, None, str]):
        name = "scoped-front"
        lifetime = Lifetime.SCOPED
        scope = ScopeSpec(("tenant",))
        backend: OptionalBackend  # annotated dependency; unset when degraded

        @invocable
        async def hello(self) -> str:
            return "front"  # copes without the degraded backend

    reg = Registry()
    reg.register(OptionalBackend)
    reg.register(ScopedFront)
    axes = AxisRegistry()
    axes.register(Axis(name="tenant", resolver=_tenant.get))
    container = Container.build(reg, {"opt-backend": {}, "scoped-front": {}}, axes=axes)
    await container.start()
    assert container.is_degraded("opt-backend")

    _tenant.set("acme")
    outcome = await container.invoke("scoped-front", "hello")  # resolves deps, skipping the degraded one
    assert outcome.value == "front"
    await container.stop()
