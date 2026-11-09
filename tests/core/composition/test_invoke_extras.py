"""invoke/proxy extras: overall budget and ambient correlation id."""

from typing import Any

from _support.axes import context_axes
import anyio
import pytest

from warpweft.core.axes import ScopeSpec
from warpweft.core.clock import ManualClock
from warpweft.core.component import AComponent, Criticality, EmptySettings, Lifetime, invocable
from warpweft.core.composition import Container, Registry
from warpweft.core.context import InvocationContext, use_correlation_id
from warpweft.core.errors import AttemptTimeout, DeadlineExceeded, TransientError

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


class Flaky(AComponent[EmptySettings, None, str]):
    def __init__(self, settings: EmptySettings) -> None:
        super().__init__(settings)
        self.calls = 0

    @invocable
    async def fetch(self) -> str:
        self.calls += 1
        raise TransientError(f"attempt {self.calls}")


class Echo(AComponent[EmptySettings, None, str]):
    @invocable
    async def whoami(self, ctx: InvocationContext) -> str:
        return ctx.correlation_id


class Slow(AComponent[EmptySettings, None, str]):
    @invocable
    async def fetch(self) -> str:
        await anyio.sleep(0.05)
        return "late"


class ScopedSlow(AComponent[EmptySettings, None, str]):
    name = "scoped-slow"
    lifetime = Lifetime.SCOPED
    scope = ScopeSpec(("tenant",))

    @invocable
    async def fetch(self) -> str:
        await anyio.sleep(0.05)
        return "late"


def container_with(*classes: type[AComponent[Any, Any, Any]], config: Any = None, **options: Any) -> Container:
    reg = Registry()
    for cls in classes:
        reg.register(cls)
    return Container.build(reg, config or {name: {} for name in (c.name for c in classes)}, **options)


# --- budget ------------------------------------------------------------------


async def test_budget_sets_an_overall_deadline() -> None:
    # A tiny budget with a 1s backoff: the first failure cannot afford a retry.
    config = {"flaky": {"policy": {"retry": {"attempts": 5, "base_delay": 1.0, "max_delay": 1.0, "jitter": False}}}}
    container = container_with(Flaky, config=config)
    await container.start()
    flaky = await container.get(Flaky)
    with pytest.raises(DeadlineExceeded):
        await container.invoke("flaky", "fetch", budget=0.05)
    assert flaky.calls == 1  # stopped by the deadline, not by exhausting attempts
    await container.stop()


async def test_no_budget_means_no_deadline() -> None:
    container = container_with(Echo)
    await container.start()
    outcome = await container.invoke("echo", "whoami")  # would raise if a bad deadline slipped in
    assert outcome.value  # a generated correlation id
    await container.stop()


async def test_proxy_budget_applies_to_every_call() -> None:
    config = {"flaky": {"policy": {"retry": {"attempts": 5, "base_delay": 1.0, "max_delay": 1.0, "jitter": False}}}}
    container = container_with(Flaky, config=config)
    await container.start()
    flaky = container.proxy(Flaky, budget=0.05)
    with pytest.raises(DeadlineExceeded):
        await flaky.fetch()
    await container.stop()


@pytest.mark.parametrize("use_proxy", [False, True])
async def test_linkless_process_budget_is_enforced(use_proxy: bool) -> None:
    container = container_with(Slow)
    await container.start()
    try:
        with pytest.raises(DeadlineExceeded):
            if use_proxy:
                await container.proxy(Slow, budget=0.001).fetch()
            else:
                await container.invoke("slow", "fetch", budget=0.001)
    finally:
        await container.stop()


@pytest.mark.parametrize("use_proxy", [False, True])
async def test_linkless_scoped_budget_is_enforced(use_proxy: bool) -> None:
    axes, handles = context_axes("tenant")
    container = container_with(ScopedSlow, axes=axes)
    await container.start()
    try:
        with handles["tenant"].use("acme"), pytest.raises(DeadlineExceeded):
            if use_proxy:
                await container.proxy(ScopedSlow, budget=0.001).fetch()
            else:
                await container.invoke("scoped-slow", "fetch", budget=0.001)
    finally:
        await container.stop()


async def test_scoped_initialization_consumes_the_overall_budget() -> None:
    class ScopedSlowStart(AComponent[EmptySettings, None, str]):
        name = "scoped-slow-start"
        lifetime = Lifetime.SCOPED
        scope = ScopeSpec(("tenant",))

        async def start(self) -> None:
            await anyio.sleep(0.05)

        @invocable
        async def fetch(self) -> str:
            return "late"

    axes, handles = context_axes("tenant")
    container = container_with(ScopedSlowStart, axes=axes, init_timeout=1.0)
    await container.start()
    try:
        with handles["tenant"].use("acme"), anyio.fail_after(0.02), pytest.raises(DeadlineExceeded):
            await container.invoke("scoped-slow-start", "fetch", budget=0.001)
    finally:
        await container.stop()


async def test_zero_budget_rejects_linkless_call_before_it_starts() -> None:
    container = container_with(Echo)
    await container.start()
    try:
        with pytest.raises(DeadlineExceeded, match="before"):
            await container.invoke("echo", "whoami", budget=0)
    finally:
        await container.stop()


async def test_overall_deadline_cancels_a_retry_attempt() -> None:
    class SlowFlaky(AComponent[EmptySettings, None, str]):
        name = "slow-flaky"

        def __init__(self, settings: EmptySettings) -> None:
            super().__init__(settings)
            self.calls = 0

        @invocable
        async def fetch(self) -> str:
            self.calls += 1
            await anyio.sleep(0.05)
            raise TransientError("late failure")

    config = {
        "slow-flaky": {"policy": {"retry": {"attempts": 3, "base_delay": 0.001, "max_delay": 0.001, "jitter": False}}}
    }
    container = container_with(SlowFlaky, config=config)
    await container.start()
    try:
        flaky = await container.get(SlowFlaky)
        with anyio.fail_after(0.02), pytest.raises(DeadlineExceeded):
            await container.invoke("slow-flaky", "fetch", budget=0.005)
        assert flaky.calls == 1
    finally:
        await container.stop()


async def test_overall_deadline_stops_retries_during_backoff() -> None:
    config = {"flaky": {"policy": {"retry": {"attempts": 10, "base_delay": 0.02, "max_delay": 0.02, "jitter": False}}}}
    container = container_with(Flaky, config=config)
    await container.start()
    try:
        flaky = await container.get(Flaky)
        with anyio.fail_after(0.04), pytest.raises(DeadlineExceeded):
            await container.invoke("flaky", "fetch", budget=0.025)
        assert flaky.calls == 2
    finally:
        await container.stop()


async def test_overall_deadline_takes_precedence_over_longer_attempt_timeout() -> None:
    config = {"slow": {"policy": {"timeout": {"seconds": 1.0}}}}
    container = container_with(Slow, config=config)
    await container.start()
    try:
        with pytest.raises(DeadlineExceeded):
            await container.invoke("slow", "fetch", budget=0.005)
    finally:
        await container.stop()


async def test_shorter_attempt_timeout_takes_precedence_over_overall_deadline() -> None:
    config = {"slow": {"policy": {"timeout": {"seconds": 0.005}}}}
    container = container_with(Slow, config=config)
    await container.start()
    try:
        with pytest.raises(AttemptTimeout):
            await container.invoke("slow", "fetch", budget=1.0)
    finally:
        await container.stop()


async def test_external_cancellation_is_not_reported_as_deadline_exceeded() -> None:
    class Waiting(AComponent[EmptySettings, None, str]):
        @invocable
        async def fetch(self) -> str:
            await anyio.Event().wait()
            return "unreachable"

    container = container_with(Waiting)
    await container.start()
    try:
        with anyio.move_on_after(0.01) as scope:
            await container.invoke("waiting", "fetch", budget=30.0)
        assert scope.cancelled_caught
    finally:
        await container.stop()


async def test_no_checkpoint_late_success_is_rejected_with_manual_clock() -> None:
    clock = ManualClock()

    class LateWithoutAwait(AComponent[EmptySettings, None, str]):
        name = "late-without-await"

        @invocable
        async def fetch(self) -> str:
            clock.advance(2)
            return "late"

    container = container_with(LateWithoutAwait, clock=clock)
    await container.start()
    try:
        with pytest.raises(DeadlineExceeded, match="exhausted"):
            await container.invoke("late-without-await", "fetch", budget=1)
    finally:
        await container.stop()


async def test_deadline_still_uses_synchronous_degradation_fallback() -> None:
    class OptionalSlow(Slow):
        name = "optional-slow"
        criticality = Criticality.OPTIONAL

        def stub(self, ctx: InvocationContext) -> str:
            return "fallback"

    config = {"optional-slow": {"policy": {"degradation": {}}}}
    container = container_with(OptionalSlow, config=config)
    await container.start()
    try:
        outcome = await container.invoke("optional-slow", "fetch", budget=0)
        assert outcome.value == "fallback"
        assert outcome.source == "stub"
        assert outcome.degraded
    finally:
        await container.stop()


# --- correlation id ----------------------------------------------------------


async def test_correlation_id_defaults_to_the_ambient_one() -> None:
    container = container_with(Echo)
    await container.start()
    with use_correlation_id("req-42"):
        assert (await container.invoke("echo", "whoami")).value == "req-42"
        assert (await container.proxy(Echo).whoami()) == "req-42"
    await container.stop()


async def test_explicit_correlation_id_wins_over_ambient() -> None:
    container = container_with(Echo)
    await container.start()
    with use_correlation_id("ambient"):
        assert (await container.invoke("echo", "whoami", correlation_id="explicit")).value == "explicit"
    await container.stop()


async def test_correlation_id_is_generated_without_an_ambient_one() -> None:
    container = container_with(Echo)
    await container.start()
    first = (await container.invoke("echo", "whoami")).value
    second = (await container.invoke("echo", "whoami")).value
    assert first and second and first != second  # fresh ids
    await container.stop()
