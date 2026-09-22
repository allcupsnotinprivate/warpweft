"""Retry link: classification, attempt limits, deadline honesty, backoff and jitter.

No test here sleeps for real: time is either recorded-and-skipped
(RecordingClock) or advanced manually (ManualClock).
"""

from datetime import UTC, datetime, timedelta
import random
from typing import Any

import anyio
from pydantic import ValidationError
import pytest

from warpweft.core.clock import ManualClock
from warpweft.core.context import InvocationContext
from warpweft.core.errors import DeadlineExceeded, ErrorClass, PermanentError, RetryExhausted, TransientError
from warpweft.core.outcome import Outcome
from warpweft.core.pipeline.builtin.retry import RetryFactory, RetryInterceptor, RetrySettings
from warpweft.core.pipeline.interceptor import Next

pytestmark = [pytest.mark.unit, pytest.mark.anyio]


class RecordingClock:
    """Clock whose sleep returns instantly, recording the delay and advancing virtual time."""

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

    def __init__(self, failures: int, exc_factory: Any = TransientError) -> None:
        self.failures = failures
        self.exc_factory = exc_factory
        self.calls = 0
        self.attempts_seen: list[int] = []

    async def __call__(self, ctx: InvocationContext) -> Outcome[Any]:
        self.calls += 1
        self.attempts_seen.append(ctx.attempt)
        if self.calls <= self.failures:
            raise self.exc_factory(f"failure #{self.calls}")
        return Outcome(value="ok")


def make_retry(
    clock: Any,
    attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = False,
    **kwargs: Any,
) -> RetryInterceptor:
    settings = RetrySettings(attempts=attempts, base_delay=base_delay, max_delay=max_delay, jitter=jitter, **kwargs)
    return RetryInterceptor(settings, clock)


def ctx(**overrides: Any) -> InvocationContext:
    defaults: dict[str, Any] = {"operation": "op", "correlation_id": "cid"}
    defaults.update(overrides)
    return InvocationContext(**defaults)


async def test_retries_transient_until_success() -> None:
    clock = RecordingClock()
    flaky = Flaky(failures=2)
    outcome = await make_retry(clock).call(flaky, ctx())

    assert outcome.value == "ok"
    assert outcome.attempts == 3
    assert flaky.calls == 3


async def test_permanent_error_is_raised_immediately() -> None:
    clock = RecordingClock()
    flaky = Flaky(failures=10, exc_factory=PermanentError)
    with pytest.raises(PermanentError):
        await make_retry(clock).call(flaky, ctx())
    assert flaky.calls == 1
    assert clock.sleeps == []


async def test_unknown_exception_is_not_retried() -> None:
    clock = RecordingClock()
    flaky = Flaky(failures=10, exc_factory=ValueError)
    with pytest.raises(ValueError, match="failure"):
        await make_retry(clock).call(flaky, ctx())
    assert flaky.calls == 1


async def test_attempt_limit_is_respected() -> None:
    clock = RecordingClock()
    flaky = Flaky(failures=100)
    with pytest.raises(TransientError, match="#3"):
        await make_retry(clock, attempts=3).call(flaky, ctx())
    assert flaky.calls == 3
    assert len(clock.sleeps) == 2  # no sleep after the final failure


async def test_exhausted_retry_wraps_last_error_in_framework_error() -> None:
    clock = RecordingClock()
    flaky = Flaky(failures=100)
    with pytest.raises(RetryExhausted) as excinfo:
        await make_retry(clock, attempts=3).call(flaky, ctx())

    err = excinfo.value
    assert err.attempts == 3
    assert isinstance(err.last_error, TransientError)
    assert str(err.last_error) == "failure #3"
    assert err.__cause__ is err.last_error  # original preserved for tracebacks
    assert isinstance(err, TransientError)  # an outer retry could still act on it


async def test_each_attempt_gets_child_context_and_parent_is_untouched() -> None:
    clock = RecordingClock()
    flaky = Flaky(failures=2)
    parent = ctx()
    await make_retry(clock).call(flaky, parent)

    assert flaky.attempts_seen == [1, 2, 3]
    assert parent.attempt == 1


async def test_backoff_grows_exponentially_and_is_capped() -> None:
    clock = RecordingClock()
    flaky = Flaky(failures=4)
    await make_retry(clock, attempts=5, base_delay=1.0, max_delay=4.0).call(flaky, ctx())

    assert clock.sleeps == [1.0, 2.0, 4.0, 4.0]


async def test_jitter_stays_within_the_backoff_envelope() -> None:
    settings = RetrySettings(attempts=5, base_delay=1.0, max_delay=4.0, jitter=True)
    link = RetryInterceptor(settings, RecordingClock(), rng=random.Random(42))

    envelopes = [1.0, 2.0, 4.0, 4.0]
    sampled = [[link._delay_before(n + 2) for _ in range(200)] for n in range(4)]

    for cap, samples in zip(envelopes, sampled, strict=True):
        assert all(0.0 <= s <= cap for s in samples)
        assert max(samples) > cap * 0.5  # actually spread, not stuck near zero


async def test_deadline_checked_before_sleeping() -> None:
    """Three 5-second retries must not turn a 6-second budget into 15 seconds of waiting."""
    clock = RecordingClock()
    flaky = Flaky(failures=100)
    retry = make_retry(clock, attempts=4, base_delay=5.0, max_delay=5.0)

    with pytest.raises(DeadlineExceeded, match="backoff"):
        await retry.call(flaky, ctx(deadline=6.0))

    # attempt 1 fails -> sleep 5 (fits in 6) -> attempt 2 fails ->
    # next sleep of 5 does not fit into the remaining 1 -> stop right there.
    assert flaky.calls == 2
    assert clock.sleeps == [5.0]
    assert clock.monotonic() < 6.0  # the caller never waited past its budget


async def test_expired_deadline_prevents_even_the_first_attempt() -> None:
    clock = RecordingClock()
    clock._time = 10.0
    flaky = Flaky(failures=0)
    with pytest.raises(DeadlineExceeded):
        await make_retry(clock).call(flaky, ctx(deadline=5.0))
    assert flaky.calls == 0


async def test_retry_on_can_target_other_classes() -> None:
    clock = RecordingClock()
    flaky = Flaky(failures=1, exc_factory=PermanentError)
    retry = make_retry(clock, retry_on=ErrorClass.PERMANENT)
    outcome = await retry.call(flaky, ctx())
    assert outcome.attempts == 2

    transient_flaky = Flaky(failures=1)
    with pytest.raises(TransientError):
        await retry.call(transient_flaky, ctx())
    assert transient_flaky.calls == 1


async def test_base_exceptions_are_never_swallowed() -> None:
    """Cancellation-like BaseExceptions must escape instantly, not be retried."""
    clock = RecordingClock()
    calls = 0

    async def interrupted(c: InvocationContext) -> Outcome[Any]:
        nonlocal calls
        calls += 1
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        await make_retry(clock).call(interrupted, ctx())
    assert calls == 1


async def test_outcome_carries_attempts_and_elapsed() -> None:
    clock = RecordingClock()
    flaky = Flaky(failures=3)
    outcome = await make_retry(clock, base_delay=1.0, max_delay=8.0).call(flaky, ctx())

    assert outcome.attempts == 4
    assert outcome.elapsed == sum(clock.sleeps)


async def test_works_on_manual_clock_without_real_sleeping() -> None:
    """End-to-end on ManualClock: 10 virtual seconds of backoff, ~0 real time."""
    clock = ManualClock()
    flaky = Flaky(failures=2)
    retry = make_retry(clock, base_delay=5.0, max_delay=5.0)
    done: dict[str, Outcome[Any]] = {}

    async def run() -> None:
        done["outcome"] = await retry.call(flaky, ctx())

    async with anyio.create_task_group() as tg:
        tg.start_soon(run)
        for _ in range(2):
            await clock.wait_for_sleepers(1)
            clock.advance(5.0)

    assert done["outcome"].attempts == 3
    assert clock.monotonic() == 10.0


async def test_settings_validation() -> None:
    with pytest.raises(ValidationError):
        RetrySettings(attempts=0, base_delay=1, max_delay=2)
    with pytest.raises(ValidationError, match="max_delay"):
        RetrySettings(attempts=3, base_delay=10, max_delay=1)


async def test_factory_creates_working_instances() -> None:
    clock = RecordingClock()
    factory = RetryFactory(RetrySettings(attempts=2, base_delay=1, max_delay=2, jitter=False), clock)
    assert not factory.state_scope

    link = factory.create(())
    flaky = Flaky(failures=1)
    base: Next = flaky.__call__
    outcome = await link.call(base, ctx())
    assert outcome.attempts == 2
