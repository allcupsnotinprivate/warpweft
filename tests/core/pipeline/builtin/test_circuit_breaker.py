"""Circuit breaker: window-based tripping, half-open probing, transient-only.

No test sleeps for real: ManualClock drives the reset timeout.
"""

from collections.abc import Mapping
import contextlib
from contextlib import AbstractContextManager, nullcontext
import logging
from typing import Any

import anyio
from pydantic import ValidationError
import pytest

from warpweft.core.clock import ManualClock
from warpweft.core.context import InvocationContext
from warpweft.core.errors import CircuitOpen, PermanentError, TransientError
from warpweft.core.observe import (
    ATTR_BREAKER_STATE_FROM,
    ATTR_BREAKER_STATE_TO,
    EVENT_BREAKER_TRANSITION,
    OBSERVER_KEY,
    AttributeValue,
)
from warpweft.core.outcome import Outcome
from warpweft.core.pipeline.builtin.circuit_breaker import (
    ENDPOINT_SCOPE,
    CircuitBreakerFactory,
    CircuitBreakerInterceptor,
    CircuitBreakerSettings,
    CircuitState,
)
from warpweft.core.pipeline.interceptor import Next

pytestmark = [pytest.mark.unit, pytest.mark.anyio]


class Recorder:
    """Minimal observer capturing the events links emit through the seam."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, AttributeValue]]] = []

    def event(self, name: str, attributes: Mapping[str, AttributeValue] | None = None) -> None:
        self.events.append((name, dict(attributes or {})))

    def span(self, name: str, attributes: Mapping[str, AttributeValue] | None = None) -> AbstractContextManager[object]:
        return nullcontext()

    def transitions(self) -> list[tuple[str, str]]:
        return [
            (attrs[ATTR_BREAKER_STATE_FROM], attrs[ATTR_BREAKER_STATE_TO])  # type: ignore[misc]
            for name, attrs in self.events
            if name == EVENT_BREAKER_TRANSITION
        ]


class Call:
    """Base call driven by a script of actions, counting how often it ran."""

    def __init__(self, action: str = "ok") -> None:
        self.action = action
        self.calls = 0

    async def __call__(self, ctx: InvocationContext) -> Outcome[Any]:
        self.calls += 1
        if self.action == "ok":
            return Outcome(value="ok")
        if self.action == "transient":
            raise TransientError("boom")
        if self.action == "permanent":
            raise PermanentError("nope")
        raise AssertionError(self.action)


def breaker(
    clock: Any, *, window: int = 3, failure_threshold: int = 3, reset_timeout: float = 10.0
) -> CircuitBreakerInterceptor:
    settings = CircuitBreakerSettings(window=window, failure_threshold=failure_threshold, reset_timeout=reset_timeout)
    return CircuitBreakerInterceptor(settings, clock)


def ctx(**overrides: Any) -> InvocationContext:
    defaults: dict[str, Any] = {"operation": "op", "correlation_id": "cid"}
    defaults.update(overrides)
    return InvocationContext(**defaults)


async def test_closed_passes_success_through() -> None:
    cb = breaker(ManualClock())
    outcome = await cb.call(Call("ok"), ctx())
    assert outcome.value == "ok"
    assert cb.state is CircuitState.CLOSED


async def test_opens_after_threshold_transient_failures() -> None:
    cb = breaker(ManualClock(), window=3, failure_threshold=3)
    call = Call("transient")
    for _ in range(3):
        with pytest.raises(TransientError):
            await cb.call(call, ctx())
    assert cb.state is CircuitState.OPEN


async def test_open_rejects_without_calling_next() -> None:
    cb = breaker(ManualClock(), window=2, failure_threshold=2)
    fail = Call("transient")
    for _ in range(2):
        with pytest.raises(TransientError):
            await cb.call(fail, ctx())
    assert cb.state is CircuitState.OPEN

    probe = Call("ok")
    with pytest.raises(CircuitOpen):
        await cb.call(probe, ctx())
    assert probe.calls == 0  # the call never reached the base


async def test_open_rejection_reports_time_until_probe() -> None:
    clock = ManualClock()
    cb = breaker(clock, window=2, failure_threshold=2, reset_timeout=10.0)
    fail = Call("transient")
    for _ in range(2):
        with pytest.raises(TransientError):
            await cb.call(fail, ctx())

    clock.advance(4.0)
    with pytest.raises(CircuitOpen) as info:
        await cb.call(Call("ok"), ctx())
    assert info.value.retry_after == pytest.approx(6.0)


async def test_permanent_errors_do_not_open_the_breaker() -> None:
    cb = breaker(ManualClock(), window=3, failure_threshold=2)
    call = Call("permanent")
    for _ in range(5):
        with pytest.raises(PermanentError):
            await cb.call(call, ctx())
    assert cb.state is CircuitState.CLOSED


async def test_sliding_window_forgets_old_failures() -> None:
    cb = breaker(ManualClock(), window=3, failure_threshold=3)
    fail, ok = Call("transient"), Call("ok")

    # F, F, S, F -> the window is [F, S, F], only 2 failures: never trips.
    for action in (fail, fail, ok, fail):
        with contextlib.suppress(TransientError):
            await cb.call(action, ctx())
    assert cb.state is CircuitState.CLOSED


async def test_half_open_after_reset_lets_one_probe_and_closes_on_success() -> None:
    clock = ManualClock()
    cb = breaker(clock, window=2, failure_threshold=2, reset_timeout=10.0)
    for _ in range(2):
        with pytest.raises(TransientError):
            await cb.call(Call("transient"), ctx())
    assert cb.state is CircuitState.OPEN

    clock.advance(10.0)  # reset elapsed -> next call is the probe
    probe = Call("ok")
    outcome = await cb.call(probe, ctx())
    assert outcome.value == "ok"
    assert probe.calls == 1
    assert cb.state is CircuitState.CLOSED


async def test_half_open_probe_failure_reopens() -> None:
    clock = ManualClock()
    cb = breaker(clock, window=1, failure_threshold=1, reset_timeout=5.0)
    with pytest.raises(TransientError):
        await cb.call(Call("transient"), ctx())
    assert cb.state is CircuitState.OPEN

    clock.advance(5.0)
    with pytest.raises(TransientError):
        await cb.call(Call("transient"), ctx())
    assert cb.state is CircuitState.OPEN  # probe failed -> open again


async def test_half_open_permanent_probe_releases_slot_and_stays_half_open() -> None:
    clock = ManualClock()
    cb = breaker(clock, window=1, failure_threshold=1, reset_timeout=5.0)
    with pytest.raises(TransientError):
        await cb.call(Call("transient"), ctx())
    clock.advance(5.0)

    # The probe hits a permanent error: it says nothing about the service being
    # down, so the slot is freed and the breaker stays half-open (not reopened).
    with pytest.raises(PermanentError):
        await cb.call(Call("permanent"), ctx())
    assert cb.state is CircuitState.HALF_OPEN

    # The next call is admitted as a fresh probe and closes the breaker.
    outcome = await cb.call(Call("ok"), ctx())
    assert outcome.value == "ok"
    assert cb.state is CircuitState.CLOSED


async def test_open_before_reset_still_rejects() -> None:
    clock = ManualClock()
    cb = breaker(clock, window=1, failure_threshold=1, reset_timeout=10.0)
    with pytest.raises(TransientError):
        await cb.call(Call("transient"), ctx())

    clock.advance(9.0)  # not yet elapsed
    with pytest.raises(CircuitOpen):
        await cb.call(Call("ok"), ctx())


async def test_half_open_allows_exactly_one_concurrent_probe() -> None:
    clock = ManualClock()
    cb = breaker(clock, window=1, failure_threshold=1, reset_timeout=5.0)
    with pytest.raises(TransientError):
        await cb.call(Call("transient"), ctx())
    clock.advance(5.0)

    released = anyio.Event()
    started = anyio.Event()
    results: list[str] = []

    async def slow_probe(c: InvocationContext) -> Outcome[Any]:
        started.set()
        await released.wait()
        return Outcome(value="probe-ok")

    async def run_probe() -> None:
        outcome = await cb.call(slow_probe, ctx())
        results.append(str(outcome.value))

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_probe)
        await started.wait()  # probe is now in flight and parked
        # A second call while the probe is in flight must be rejected at once.
        with pytest.raises(CircuitOpen):
            await cb.call(Call("ok"), ctx())
        released.set()

    assert results == ["probe-ok"]
    assert cb.state is CircuitState.CLOSED


async def test_circuit_open_is_transient() -> None:
    assert issubclass(CircuitOpen, TransientError)


async def test_settings_validation() -> None:
    with pytest.raises(ValidationError):
        CircuitBreakerSettings(window=0, failure_threshold=1, reset_timeout=1.0)
    with pytest.raises(ValidationError):
        CircuitBreakerSettings(window=3, failure_threshold=1, reset_timeout=0.0)
    with pytest.raises(ValidationError, match="failure_threshold"):
        CircuitBreakerSettings(window=2, failure_threshold=5, reset_timeout=1.0)


async def test_factory_scope_and_creation() -> None:
    clock = ManualClock()
    factory = CircuitBreakerFactory(CircuitBreakerSettings(window=1, failure_threshold=1, reset_timeout=1.0), clock)
    assert factory.state_scope == ENDPOINT_SCOPE
    assert factory.state_scope.axes == ("endpoint",)

    link = factory.create(())
    base: Next = Call("ok").__call__
    outcome = await link.call(base, ctx())
    assert outcome.value == "ok"


# --- manual controls ---------------------------------------------------------


async def test_force_open_rejects_then_probes_after_reset_timeout() -> None:
    clock = ManualClock()
    cb = breaker(clock, reset_timeout=10.0)
    await cb.force_open()
    assert cb.state is CircuitState.OPEN

    probe = Call("ok")
    with pytest.raises(CircuitOpen):
        await cb.call(probe, ctx())
    assert probe.calls == 0

    clock.advance(10.0)  # reset elapsed -> next call is the probe
    outcome = await cb.call(Call("ok"), ctx())
    assert outcome.value == "ok"
    assert cb.state is CircuitState.CLOSED


async def test_force_open_rearms_the_open_timer() -> None:
    clock = ManualClock()
    cb = breaker(clock, window=1, failure_threshold=1, reset_timeout=10.0)
    with pytest.raises(TransientError):
        await cb.call(Call("transient"), ctx())
    assert cb.state is CircuitState.OPEN

    clock.advance(6.0)
    await cb.force_open()  # re-arms: the 10s window restarts from here

    clock.advance(6.0)  # 12s since the original trip, but only 6s since force_open
    with pytest.raises(CircuitOpen):
        await cb.call(Call("ok"), ctx())

    clock.advance(4.0)  # now 10s since force_open
    outcome = await cb.call(Call("ok"), ctx())
    assert outcome.value == "ok"


async def test_reset_closes_and_clears_window() -> None:
    cb = breaker(ManualClock(), window=3, failure_threshold=3)
    fail = Call("transient")
    for _ in range(2):  # 2 of 3 failures - not yet tripped
        with pytest.raises(TransientError):
            await cb.call(fail, ctx())

    await cb.reset()  # forgets the two recorded failures
    assert cb.state is CircuitState.CLOSED

    for _ in range(2):  # two more failures still do not trip (window was cleared)
        with pytest.raises(TransientError):
            await cb.call(fail, ctx())
    assert cb.state is CircuitState.CLOSED

    with pytest.raises(TransientError):  # the third one trips
        await cb.call(fail, ctx())
    assert cb.state is CircuitState.OPEN


async def test_reset_from_half_open_releases_probe_slot() -> None:
    clock = ManualClock()
    cb = breaker(clock, window=1, failure_threshold=1, reset_timeout=5.0)
    with pytest.raises(TransientError):
        await cb.call(Call("transient"), ctx())
    clock.advance(5.0)

    # Park a probe in flight, then reset from half-open.
    started, released = anyio.Event(), anyio.Event()

    async def slow_probe(c: InvocationContext) -> Outcome[Any]:
        started.set()
        await released.wait()
        return Outcome(value="probe-ok")

    async with anyio.create_task_group() as tg:
        tg.start_soon(lambda: cb.call(slow_probe, ctx()))  # type: ignore[arg-type,return-value]
        await started.wait()
        await cb.reset()  # closes and frees the probe slot
        released.set()

    assert cb.state is CircuitState.CLOSED
    # A fresh call is admitted normally (the slot was released, not stuck).
    outcome = await cb.call(Call("ok"), ctx())
    assert outcome.value == "ok"


async def test_reset_is_idempotent_when_closed() -> None:
    cb = breaker(ManualClock())
    await cb.reset()
    await cb.reset()
    assert cb.state is CircuitState.CLOSED


# --- transition metric events ------------------------------------------------


async def test_transition_events_cover_the_full_cycle() -> None:
    clock = ManualClock()
    cb = breaker(clock, window=1, failure_threshold=1, reset_timeout=5.0)
    rec = Recorder()

    with pytest.raises(TransientError):  # closed -> open
        await cb.call(Call("transient"), ctx(bag={OBSERVER_KEY: rec}))
    clock.advance(5.0)
    outcome = await cb.call(Call("ok"), ctx(bag={OBSERVER_KEY: rec}))  # open -> half_open -> closed
    assert outcome.value == "ok"

    assert rec.transitions() == [
        ("closed", "open"),
        ("open", "half_open"),
        ("half_open", "closed"),
    ]


async def test_probe_failure_emits_half_open_to_open() -> None:
    clock = ManualClock()
    cb = breaker(clock, window=1, failure_threshold=1, reset_timeout=5.0)
    rec = Recorder()

    with pytest.raises(TransientError):  # closed -> open
        await cb.call(Call("transient"), ctx(bag={OBSERVER_KEY: rec}))
    clock.advance(5.0)
    with pytest.raises(TransientError):  # open -> half_open, probe fails -> half_open -> open
        await cb.call(Call("transient"), ctx(bag={OBSERVER_KEY: rec}))

    assert rec.transitions() == [
        ("closed", "open"),
        ("open", "half_open"),
        ("half_open", "open"),
    ]


async def test_manual_controls_emit_no_events_but_log(caplog: pytest.LogCaptureFixture) -> None:
    cb = breaker(ManualClock())
    rec = Recorder()
    # An observer is only reachable through a ctx; manual controls take none,
    # so nothing can be emitted - but the state change is still logged.
    with caplog.at_level(logging.INFO, logger="warpweft.core.pipeline.builtin.circuit_breaker"):
        await cb.force_open()
        await cb.reset()

    assert rec.events == []
    messages = [r.getMessage() for r in caplog.records]
    assert any("opened" in m for m in messages)
    assert any("closed" in m for m in messages)
