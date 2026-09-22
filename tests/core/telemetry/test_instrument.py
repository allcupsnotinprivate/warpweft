"""Instrumentation wrapper: spans, nesting, bag facts, observer lifecycle.

Every test builds its own in-memory TracerProvider; the OTel globals are
never touched (they are set-once per process, and this suite runs twice -
asyncio and trio).
"""

from datetime import UTC, datetime, timedelta
from typing import Any

from _support.otel import by_name, tracing
from opentelemetry.trace import StatusCode
import pytest

from warpweft.core.axes import AxisRegistry
from warpweft.core.context import InvocationContext
from warpweft.core.errors import PermanentError, TransientError
from warpweft.core.observe import OBSERVER_KEY, SPAN_ATTEMPT
from warpweft.core.outcome import Outcome
from warpweft.core.pipeline.builtin.cache import CacheInterceptor, CacheSettings
from warpweft.core.pipeline.builtin.retry import RetryFactory, RetrySettings
from warpweft.core.pipeline.builtin.timeout import TimeoutFactory, TimeoutSettings
from warpweft.core.pipeline.chain import build_chain
from warpweft.core.pipeline.state import InMemoryStateStore
from warpweft.core.telemetry import conventions as conv
from warpweft.core.telemetry.instrument import instrument

pytestmark = [pytest.mark.unit, pytest.mark.anyio]


class RecordingClock:
    """Clock whose sleep returns instantly, advancing virtual time."""

    def __init__(self) -> None:
        self._time = 0.0

    def monotonic(self) -> float:
        return self._time

    def now(self) -> datetime:
        return datetime(2020, 1, 1, tzinfo=UTC) + timedelta(seconds=self._time)

    async def sleep(self, seconds: float) -> None:
        self._time += max(seconds, 0.0)


def ctx(**overrides: Any) -> InvocationContext:
    defaults: dict[str, Any] = {"operation": "op", "correlation_id": "cid"}
    defaults.update(overrides)
    return InvocationContext(**defaults)


async def ok(c: InvocationContext) -> Outcome[Any]:
    return Outcome(value="ok")


async def test_invocation_span_carries_start_and_final_attributes() -> None:
    provider, exporter = tracing()
    wrapped = instrument(ok, tracer_provider=provider)

    await wrapped(ctx(operation="search.query", scope_key=(("tenant", "acme"),)))

    (span,) = exporter.get_finished_spans()
    assert span.name == "search.query"
    attrs = dict(span.attributes or {})
    assert attrs[conv.ATTR_OPERATION] == "search.query"
    assert attrs[conv.ATTR_CORRELATION_ID] == "cid"
    assert attrs[f"{conv.AXIS_ATTR_PREFIX}tenant"] == "acme"
    assert attrs[conv.ATTR_SOURCE] == "live"
    assert attrs[conv.ATTR_DEGRADED] is False
    assert attrs[conv.ATTR_ATTEMPTS] == 1
    assert span.status.status_code is StatusCode.UNSET  # instrumentation never sets OK


async def test_attempt_spans_nest_under_the_invocation_span() -> None:
    provider, exporter = tracing()
    clock = RecordingClock()
    calls = 0

    async def flaky(c: InvocationContext) -> Outcome[Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TransientError("first fails")
        return Outcome(value="ok")

    retry = RetryFactory(RetrySettings(attempts=3, base_delay=0.1, max_delay=1.0, jitter=False), clock)
    timeout = TimeoutFactory(TimeoutSettings(seconds=5.0), clock)
    chain = build_chain([retry, timeout], InMemoryStateStore(), AxisRegistry(), flaky)
    wrapped = instrument(chain, tracer_provider=provider, clock=clock)

    outcome = await wrapped(ctx())
    assert outcome.attempts == 2

    spans = exporter.get_finished_spans()
    (invocation,) = by_name(spans, "op")
    attempts = by_name(spans, SPAN_ATTEMPT)
    assert len(attempts) == 2
    for attempt in attempts:
        assert attempt.parent is not None
        assert attempt.parent.span_id == invocation.context.span_id
        assert attempt.context.trace_id == invocation.context.trace_id

    failed, succeeded = attempts
    assert dict(failed.attributes or {})[conv.ATTR_ATTEMPT_NUMBER] == 1
    assert failed.status.status_code is StatusCode.ERROR
    assert [e.name for e in failed.events] == ["exception"]
    assert succeeded.status.status_code is StatusCode.UNSET

    # The backoff between attempts lands on the invocation span.
    backoffs = [e for e in invocation.events if e.name == conv.EVENT_RETRY_BACKOFF]
    assert len(backoffs) == 1
    assert dict(backoffs[0].attributes or {})[conv.ATTR_BACKOFF_DELAY] == 0.1


async def test_error_path_sets_status_and_error_class() -> None:
    provider, exporter = tracing()

    async def bad(c: InvocationContext) -> Outcome[Any]:
        raise PermanentError("bad request")

    wrapped = instrument(bad, tracer_provider=provider)
    with pytest.raises(PermanentError):
        await wrapped(ctx())

    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert [e.name for e in span.events] == ["exception"]
    assert dict(span.attributes or {})[conv.ATTR_ERROR_CLASS] == "permanent"


async def test_cache_fact_becomes_a_span_attribute() -> None:
    provider, exporter = tracing()
    cache = CacheInterceptor(CacheSettings(ttl=100.0, max_entries=10), RecordingClock())

    async def through_cache(c: InvocationContext) -> Outcome[Any]:
        return await cache.call(ok, c)

    wrapped = instrument(through_cache, tracer_provider=provider)
    await wrapped(ctx())  # prime: miss
    await wrapped(ctx())  # hit

    first, second = exporter.get_finished_spans()
    assert dict(first.attributes or {})[conv.ATTR_CACHE] == "miss"
    assert dict(second.attributes or {})[conv.ATTR_CACHE] == "hit"
    assert dict(second.attributes or {})[conv.ATTR_SOURCE] == "cache"


async def test_without_sdk_the_wrapper_is_a_transparent_noop() -> None:
    # Default providers resolve to the OTel globals, which no tests configure,
    # so this genuinely exercises the proxy/no-op path.
    wrapped = instrument(ok)
    outcome = await wrapped(ctx())
    assert outcome.value == "ok"
    assert outcome.source == "live"


async def test_previous_bag_observer_is_restored() -> None:
    provider, _ = tracing()
    sentinel = object()
    seen: list[object] = []

    async def base(c: InvocationContext) -> Outcome[Any]:
        seen.append(c.bag[OBSERVER_KEY])  # the OTel observer during the call
        return Outcome(value="ok")

    c = ctx(bag={OBSERVER_KEY: sentinel})
    await instrument(base, tracer_provider=provider)(c)
    assert c.bag[OBSERVER_KEY] is sentinel
    assert seen[0] is not sentinel


async def test_observer_key_is_removed_when_there_was_none() -> None:
    provider, _ = tracing()
    c = ctx()
    await instrument(ok, tracer_provider=provider)(c)
    assert OBSERVER_KEY not in c.bag


async def test_observer_is_restored_on_error_too() -> None:
    provider, _ = tracing()

    async def bad(c: InvocationContext) -> Outcome[Any]:
        raise TransientError("down")

    c = ctx()
    with pytest.raises(TransientError):
        await instrument(bad, tracer_provider=provider)(c)
    assert OBSERVER_KEY not in c.bag


async def test_span_enricher_runs_on_success_with_outcome() -> None:
    provider, exporter = tracing()
    seen: list[tuple[InvocationContext, Outcome[Any] | None, BaseException | None]] = []

    def enrich(span: Any, c: InvocationContext, outcome: Outcome[Any] | None, exc: BaseException | None) -> None:
        seen.append((c, outcome, exc))
        span.set_attribute("app.input", dict(c.arguments or {})["q"])
        assert outcome is not None
        span.set_attribute("app.output", outcome.value)

    wrapped = instrument(ok, tracer_provider=provider, span_enricher=enrich)
    await wrapped(ctx(arguments={"q": "hello"}))

    (c, outcome, exc) = seen[0]
    assert dict(c.arguments or {}) == {"q": "hello"}
    assert outcome is not None and outcome.value == "ok"
    assert exc is None

    (span,) = exporter.get_finished_spans()
    attrs = dict(span.attributes or {})
    assert attrs["app.input"] == "hello"
    assert attrs["app.output"] == "ok"


async def test_span_enricher_runs_on_error_with_exception() -> None:
    provider, exporter = tracing()
    seen: list[tuple[Outcome[Any] | None, BaseException | None]] = []
    boom = PermanentError("bad request")

    async def bad(c: InvocationContext) -> Outcome[Any]:
        raise boom

    def enrich(span: Any, c: InvocationContext, outcome: Outcome[Any] | None, exc: BaseException | None) -> None:
        seen.append((outcome, exc))
        span.set_attribute("app.failed", True)

    wrapped = instrument(bad, tracer_provider=provider, span_enricher=enrich)
    with pytest.raises(PermanentError):
        await wrapped(ctx())

    (outcome, exc) = seen[0]
    assert outcome is None
    assert exc is boom

    (span,) = exporter.get_finished_spans()
    assert dict(span.attributes or {})["app.failed"] is True


async def test_span_enricher_failure_does_not_break_success() -> None:
    provider, _ = tracing()

    def enrich(span: Any, c: InvocationContext, outcome: Outcome[Any] | None, exc: BaseException | None) -> None:
        raise RuntimeError("enricher boom")

    wrapped = instrument(ok, tracer_provider=provider, span_enricher=enrich)
    outcome = await wrapped(ctx())  # the enricher's error is swallowed
    assert outcome.value == "ok"


async def test_span_enricher_failure_preserves_original_error() -> None:
    provider, _ = tracing()

    async def bad(c: InvocationContext) -> Outcome[Any]:
        raise TransientError("down")

    def enrich(span: Any, c: InvocationContext, outcome: Outcome[Any] | None, exc: BaseException | None) -> None:
        raise RuntimeError("enricher boom")

    wrapped = instrument(bad, tracer_provider=provider, span_enricher=enrich)
    # The original error propagates, not the enricher's.
    with pytest.raises(TransientError):
        await wrapped(ctx())


async def test_no_span_enricher_is_the_default_behaviour() -> None:
    provider, exporter = tracing()
    wrapped = instrument(ok, tracer_provider=provider)  # span_enricher defaults to None
    await wrapped(ctx(operation="search.query"))

    (span,) = exporter.get_finished_spans()
    attrs = dict(span.attributes or {})
    assert span.name == "search.query"
    assert not any(k.startswith("app.") for k in attrs)
