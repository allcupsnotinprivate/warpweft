"""Observation seam: null default, bag lookup, link emissions."""

from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from datetime import UTC, datetime, timedelta
from typing import Any

import anyio
import pytest

from warpweft.core.clock import ManualClock
from warpweft.core.context import InvocationContext
from warpweft.core.errors import CircuitOpen, TransientError
from warpweft.core.observe import (
    ATTR_ATTEMPT_NUMBER,
    ATTR_BACKOFF_DELAY,
    ATTR_BREAKER_STATE,
    EVENT_BREAKER_REJECTED,
    EVENT_RETRY_BACKOFF,
    NULL_OBSERVER,
    OBSERVER_KEY,
    SPAN_ATTEMPT,
    AttributeValue,
    observer_of,
)
from warpweft.core.outcome import Outcome
from warpweft.core.pipeline.builtin.circuit_breaker import CircuitBreakerInterceptor, CircuitBreakerSettings
from warpweft.core.pipeline.builtin.retry import RetryInterceptor, RetrySettings

pytestmark = [pytest.mark.unit, pytest.mark.anyio]


class RecordingObserver:
    """Observer capturing every event and span for assertions."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, AttributeValue]]] = []
        self.spans: list[tuple[str, dict[str, AttributeValue]]] = []

    def event(self, name: str, attributes: Mapping[str, AttributeValue] | None = None) -> None:
        self.events.append((name, dict(attributes or {})))

    def span(self, name: str, attributes: Mapping[str, AttributeValue] | None = None) -> AbstractContextManager[object]:
        self.spans.append((name, dict(attributes or {})))
        return nullcontext()


class RecordingClock:
    """Clock whose sleep returns instantly, advancing virtual time."""

    def __init__(self) -> None:
        self._time = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self._time

    def now(self) -> datetime:
        return datetime(2020, 1, 1, tzinfo=UTC) + timedelta(seconds=self._time)

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self._time += max(seconds, 0.0)


class Flaky:
    """Base call failing a scripted number of times before succeeding."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    async def __call__(self, ctx: InvocationContext) -> Outcome[Any]:
        self.calls += 1
        if self.calls <= self.failures:
            raise TransientError(f"failure #{self.calls}")
        return Outcome(value="ok")


def ctx(**overrides: Any) -> InvocationContext:
    defaults: dict[str, Any] = {"operation": "op", "correlation_id": "cid"}
    defaults.update(overrides)
    return InvocationContext(**defaults)


def make_retry(clock: RecordingClock, attempts: int = 5) -> RetryInterceptor:
    settings = RetrySettings(attempts=attempts, base_delay=1.0, max_delay=8.0, jitter=False)
    return RetryInterceptor(settings, clock)


async def test_null_observer_is_a_usable_noop() -> None:
    NULL_OBSERVER.event("anything", {"k": 1})
    with NULL_OBSERVER.span("anything", {"k": "v"}):
        pass


async def test_observer_of_defaults_to_null_and_reads_the_bag() -> None:
    assert observer_of(ctx()) is NULL_OBSERVER
    recording = RecordingObserver()
    c = ctx(bag={OBSERVER_KEY: recording})
    assert observer_of(c) is recording


async def test_retry_emits_attempt_spans_and_backoff_events() -> None:
    clock = RecordingClock()
    recording = RecordingObserver()
    flaky = Flaky(failures=2)

    outcome = await make_retry(clock).call(flaky, ctx(bag={OBSERVER_KEY: recording}))

    assert outcome.attempts == 3
    assert recording.spans == [
        (SPAN_ATTEMPT, {ATTR_ATTEMPT_NUMBER: 1}),
        (SPAN_ATTEMPT, {ATTR_ATTEMPT_NUMBER: 2}),
        (SPAN_ATTEMPT, {ATTR_ATTEMPT_NUMBER: 3}),
    ]
    assert recording.events == [
        (EVENT_RETRY_BACKOFF, {ATTR_BACKOFF_DELAY: 1.0, ATTR_ATTEMPT_NUMBER: 2}),
        (EVENT_RETRY_BACKOFF, {ATTR_BACKOFF_DELAY: 2.0, ATTR_ATTEMPT_NUMBER: 3}),
    ]


async def test_retry_without_observer_behaves_identically() -> None:
    clock = RecordingClock()
    flaky = Flaky(failures=2)
    outcome = await make_retry(clock).call(flaky, ctx())
    assert outcome.attempts == 3
    assert clock.sleeps == [1.0, 2.0]


async def test_breaker_emits_rejected_event_when_open() -> None:
    clock = ManualClock()
    settings = CircuitBreakerSettings(window=1, failure_threshold=1, reset_timeout=10.0)
    breaker = CircuitBreakerInterceptor(settings, clock)
    recording = RecordingObserver()

    async def boom(c: InvocationContext) -> Outcome[Any]:
        raise TransientError("down")

    with pytest.raises(TransientError):
        await breaker.call(boom, ctx())
    with pytest.raises(CircuitOpen):
        await breaker.call(boom, ctx(bag={OBSERVER_KEY: recording}))

    assert recording.events == [(EVENT_BREAKER_REJECTED, {ATTR_BREAKER_STATE: "open"})]


async def test_breaker_emits_rejected_event_when_half_open_probe_busy() -> None:
    clock = ManualClock()
    settings = CircuitBreakerSettings(window=1, failure_threshold=1, reset_timeout=5.0)
    breaker = CircuitBreakerInterceptor(settings, clock)
    recording = RecordingObserver()

    async def boom(c: InvocationContext) -> Outcome[Any]:
        raise TransientError("down")

    with pytest.raises(TransientError):
        await breaker.call(boom, ctx())
    clock.advance(5.0)

    started = anyio.Event()
    released = anyio.Event()

    async def slow_probe(c: InvocationContext) -> Outcome[Any]:
        started.set()
        await released.wait()
        return Outcome(value="ok")

    async def run_probe() -> None:
        await breaker.call(slow_probe, ctx())

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_probe)
        await started.wait()
        with pytest.raises(CircuitOpen):
            await breaker.call(boom, ctx(bag={OBSERVER_KEY: recording}))
        released.set()

    assert recording.events == [(EVENT_BREAKER_REJECTED, {ATTR_BREAKER_STATE: "half_open"})]
