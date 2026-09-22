"""Instrumentation wrapper: the five metrics and the cardinality policy.

Fresh in-memory MeterProvider per test (cumulative temporality - asserted
values are absolutes); the OTel globals are never touched.
"""

from typing import Any

from _support.otel import metering, read, sole_point
import pytest

from warpweft.core.clock import ManualClock
from warpweft.core.context import InvocationContext
from warpweft.core.errors import CircuitOpen, TransientError
from warpweft.core.outcome import Outcome
from warpweft.core.pipeline.builtin.circuit_breaker import CircuitBreakerInterceptor, CircuitBreakerSettings
from warpweft.core.pipeline.builtin.retry import RetryInterceptor, RetrySettings
from warpweft.core.pipeline.chain import compose
from warpweft.core.telemetry import conventions as conv
from warpweft.core.telemetry.instrument import TelemetryConfig, instrument

pytestmark = [pytest.mark.unit, pytest.mark.anyio]

TWO_AXES = (("region", "eu"), ("tenant", "acme"))


def ctx(**overrides: Any) -> InvocationContext:
    defaults: dict[str, Any] = {"operation": "op", "correlation_id": "cid"}
    defaults.update(overrides)
    return InvocationContext(**defaults)


async def ok(c: InvocationContext) -> Outcome[Any]:
    return Outcome(value="ok")


async def test_calls_counter_on_success() -> None:
    provider, reader = metering()
    await instrument(ok, meter_provider=provider)(ctx())

    metric = read(reader)[conv.METRIC_CALLS]
    assert metric.unit == conv.UNIT_CALLS
    point = sole_point(metric)
    assert point.value == 1
    assert dict(point.attributes) == {
        conv.ATTR_OPERATION: "op",
        conv.ATTR_STATUS: conv.STATUS_OK,
        conv.ATTR_SOURCE: "live",
        conv.ATTR_DEGRADED: False,
    }


async def test_duration_histogram_measures_with_the_explicit_clock() -> None:
    provider, reader = metering()
    clock = ManualClock()

    async def slow(c: InvocationContext) -> Outcome[Any]:
        clock.advance(1.5)
        return Outcome(value="ok")

    await instrument(slow, meter_provider=provider, clock=clock)(ctx())

    metric = read(reader)[conv.METRIC_DURATION]
    assert metric.unit == conv.UNIT_SECONDS
    point = sole_point(metric)
    assert point.count == 1
    assert point.sum == pytest.approx(1.5)
    assert dict(point.attributes) == {conv.ATTR_OPERATION: "op", conv.ATTR_STATUS: conv.STATUS_OK}


async def test_duration_histogram_falls_back_to_the_context_clock() -> None:
    provider, reader = metering()
    clock = ManualClock()

    async def slow(c: InvocationContext) -> Outcome[Any]:
        clock.advance(2.0)
        return Outcome(value="ok")

    await instrument(slow, meter_provider=provider)(ctx(clock=clock))

    point = sole_point(read(reader)[conv.METRIC_DURATION])
    assert point.sum == pytest.approx(2.0)


async def test_degradations_counter_with_axes() -> None:
    provider, reader = metering()

    async def stubbed(c: InvocationContext) -> Outcome[Any]:
        return Outcome(value=[], source="stub", degraded=True)

    await instrument(stubbed, meter_provider=provider)(ctx(scope_key=TWO_AXES))

    metrics = read(reader)
    degradations = sole_point(metrics[conv.METRIC_DEGRADATIONS])
    assert degradations.value == 1
    assert dict(degradations.attributes) == {
        conv.ATTR_OPERATION: "op",
        f"{conv.AXIS_ATTR_PREFIX}region": "eu",
        f"{conv.AXIS_ATTR_PREFIX}tenant": "acme",
    }
    calls = sole_point(metrics[conv.METRIC_CALLS])
    assert dict(calls.attributes)[conv.ATTR_DEGRADED] is True
    assert dict(calls.attributes)[conv.ATTR_SOURCE] == "stub"


async def test_rejections_counted_even_when_an_outer_retry_recovers() -> None:
    """The decisive case: the rejection never surfaces as an exception."""
    provider, reader = metering()
    clock = ManualClock()

    breaker = CircuitBreakerInterceptor(CircuitBreakerSettings(window=1, failure_threshold=1, reset_timeout=5.0), clock)

    async def boom(c: InvocationContext) -> Outcome[Any]:
        raise TransientError("down")

    with pytest.raises(TransientError):  # trip the breaker: state OPEN at t=0
        await breaker.call(boom, ctx())

    # Retry OUTSIDE the breaker; its 10s backoff sails past the 5s reset.
    retry = RetryInterceptor(RetrySettings(attempts=3, base_delay=10.0, max_delay=10.0, jitter=False), clock)
    chain = compose([retry, breaker], ok)
    wrapped = instrument(chain, meter_provider=provider, clock=clock)

    async def run() -> None:
        outcome = await wrapped(ctx(scope_key=(("endpoint", "api.example"),)))
        assert outcome.value == "ok"
        assert outcome.attempts == 2

    import anyio

    async with anyio.create_task_group() as tg:
        tg.start_soon(run)
        await clock.wait_for_sleepers(1)  # retry parked on its backoff
        clock.advance(10.0)  # past the breaker reset: next attempt is the probe

    metrics = read(reader)
    rejection = sole_point(metrics[conv.METRIC_BREAKER_REJECTIONS])
    assert metrics[conv.METRIC_BREAKER_REJECTIONS].unit == conv.UNIT_REJECTIONS
    assert rejection.value == 1
    assert dict(rejection.attributes) == {
        conv.ATTR_OPERATION: "op",
        f"{conv.AXIS_ATTR_PREFIX}endpoint": "api.example",
        conv.ATTR_BREAKER_STATE: "open",
    }
    # The call as a whole still reports success.
    assert dict(sole_point(metrics[conv.METRIC_CALLS]).attributes)[conv.ATTR_STATUS] == conv.STATUS_OK


async def test_rejection_escaping_to_the_wrapper_counts_once_and_errors() -> None:
    provider, reader = metering()
    clock = ManualClock()
    breaker = CircuitBreakerInterceptor(
        CircuitBreakerSettings(window=1, failure_threshold=1, reset_timeout=100.0), clock
    )

    async def boom(c: InvocationContext) -> Outcome[Any]:
        raise TransientError("down")

    with pytest.raises(TransientError):
        await breaker.call(boom, ctx())

    async def through_breaker(c: InvocationContext) -> Outcome[Any]:
        return await breaker.call(ok, c)

    wrapped = instrument(through_breaker, meter_provider=provider, clock=clock)
    with pytest.raises(CircuitOpen):
        await wrapped(ctx())

    metrics = read(reader)
    assert sole_point(metrics[conv.METRIC_BREAKER_REJECTIONS]).value == 1
    calls = sole_point(metrics[conv.METRIC_CALLS])
    assert dict(calls.attributes)[conv.ATTR_STATUS] == conv.STATUS_ERROR
    assert dict(calls.attributes)[conv.ATTR_ERROR_CLASS] == "transient"


async def test_transitions_counter_records_each_state_change() -> None:
    provider, reader = metering()
    clock = ManualClock()
    breaker = CircuitBreakerInterceptor(CircuitBreakerSettings(window=1, failure_threshold=1, reset_timeout=5.0), clock)

    async def flaky(c: InvocationContext) -> Outcome[Any]:
        raise TransientError("down")

    async def trip(c: InvocationContext) -> Outcome[Any]:
        return await breaker.call(flaky, c)

    async def recover(c: InvocationContext) -> Outcome[Any]:
        return await breaker.call(ok, c)

    key = (("endpoint", "api.example"),)
    wrapped_trip = instrument(trip, meter_provider=provider, clock=clock)
    wrapped_recover = instrument(recover, meter_provider=provider, clock=clock)

    with pytest.raises(TransientError):  # closed -> open
        await wrapped_trip(ctx(scope_key=key))
    clock.advance(5.0)
    await wrapped_recover(ctx(scope_key=key))  # open -> half_open -> closed

    metric = read(reader)[conv.METRIC_BREAKER_TRANSITIONS]
    assert metric.unit == conv.UNIT_TRANSITIONS
    points = {
        (p.attributes[conv.ATTR_BREAKER_STATE_FROM], p.attributes[conv.ATTR_BREAKER_STATE_TO]): p.value
        for p in metric.data.data_points
    }
    assert points == {("closed", "open"): 1, ("open", "half_open"): 1, ("half_open", "closed"): 1}
    for point in metric.data.data_points:  # axes_all + operation attach to every point
        assert point.attributes[conv.ATTR_OPERATION] == "op"
        assert point.attributes[f"{conv.AXIS_ATTR_PREFIX}endpoint"] == "api.example"


async def test_axes_do_not_reach_histogram_or_ok_calls_by_default() -> None:
    provider, reader = metering()
    await instrument(ok, meter_provider=provider)(ctx(scope_key=TWO_AXES))

    metrics = read(reader)
    for name in (conv.METRIC_DURATION, conv.METRIC_CALLS):
        attrs = dict(sole_point(metrics[name]).attributes)
        assert not any(key.startswith(conv.AXIS_ATTR_PREFIX) for key in attrs)


async def test_allowlisted_axis_value_gets_full_breakdown() -> None:
    provider, reader = metering()
    config = TelemetryConfig(axis_allowlist=frozenset({"acme"}))
    await instrument(ok, meter_provider=provider, config=config)(ctx(scope_key=TWO_AXES))

    metrics = read(reader)
    for name in (conv.METRIC_DURATION, conv.METRIC_CALLS):
        attrs = dict(sole_point(metrics[name]).attributes)
        assert attrs[f"{conv.AXIS_ATTR_PREFIX}tenant"] == "acme"  # allowlisted value
        assert f"{conv.AXIS_ATTR_PREFIX}region" not in attrs  # "eu" is not


async def test_error_calls_carry_axes_without_any_allowlist() -> None:
    provider, reader = metering()

    async def bad(c: InvocationContext) -> Outcome[Any]:
        raise TransientError("down")

    with pytest.raises(TransientError):
        await instrument(bad, meter_provider=provider)(ctx(scope_key=TWO_AXES))

    metrics = read(reader)
    calls_attrs = dict(sole_point(metrics[conv.METRIC_CALLS]).attributes)
    assert calls_attrs[f"{conv.AXIS_ATTR_PREFIX}tenant"] == "acme"
    assert calls_attrs[f"{conv.AXIS_ATTR_PREFIX}region"] == "eu"
    # The histogram still follows the allowlist rule even on errors.
    duration_attrs = dict(sole_point(metrics[conv.METRIC_DURATION]).attributes)
    assert not any(key.startswith(conv.AXIS_ATTR_PREFIX) for key in duration_attrs)


async def test_nothing_recorded_means_no_data_points() -> None:
    provider, reader = metering()
    await instrument(ok, meter_provider=provider)(ctx())

    metrics = read(reader)
    assert conv.METRIC_DEGRADATIONS not in metrics
    assert conv.METRIC_BREAKER_REJECTIONS not in metrics
    assert conv.METRIC_BREAKER_TRANSITIONS not in metrics
