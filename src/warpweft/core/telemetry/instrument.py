"""The instrumentation wrapper: one span per invocation, five metrics.

``instrument`` wraps an assembled chain (any ``Next``) and is meant to be
applied unconditionally, outside the outermost link. It installs an
OTel-backed observer into the context ``bag`` so links emit attempt spans and
events through the ``warpweft.core.observe`` seam without importing OTel.

Cardinality policy (see ``docs/telemetry.md``): axis values always attach to
the degradation, breaker-rejection and breaker-transition counters and to
error-status call counts; they attach to the duration histogram and ok-status
call counts only for values in the allowlist.
"""

from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Final

from opentelemetry import metrics, trace

from warpweft.core import __version__
from warpweft.core.axes import ScopeKey
from warpweft.core.clock import Clock, SystemClock
from warpweft.core.context import InvocationContext
from warpweft.core.errors import DefaultErrorClassifier, ErrorClassifier
from warpweft.core.observe import (
    EVENT_BREAKER_REJECTED,
    EVENT_BREAKER_TRANSITION,
    FACT_CACHE,
    OBSERVER_KEY,
    AttributeValue,
)
from warpweft.core.outcome import Outcome
from warpweft.core.pipeline.interceptor import Next

from . import conventions as conv


@dataclass(frozen=True)
class TelemetryConfig:
    """Knobs of the wrapper. Collection itself is not one of them.

    ``axis_allowlist`` holds axis *values* granted full metric breakdown:
    matching pairs are attached to the duration histogram and ok-status call
    counts, which never carry axes otherwise.
    """

    axis_allowlist: frozenset[str] = frozenset()


DEFAULT_CONFIG: Final = TelemetryConfig()

#: Fallback for contexts carrying no clock when none is passed explicitly.
_SYSTEM_CLOCK: Final = SystemClock()

#: Sentinel distinguishing "no previous observer" from a stored None.
_MISSING: Final = object()


def _axis_attrs(scope_key: ScopeKey, allowlist: frozenset[str] | None = None) -> dict[str, AttributeValue]:
    """Axis pairs as attributes; ``None`` allowlist means all pairs."""
    return {
        f"{conv.AXIS_ATTR_PREFIX}{name}": value for name, value in scope_key if allowlist is None or value in allowlist
    }


class _OtelObserver:
    """Observer forwarding link emissions to the current OTel span.

    Events land on whatever span is current: a backoff emitted between
    attempts lands on the invocation span, one emitted inside an attempt on
    the attempt span. Breaker rejections additionally drive their counter -
    an exception at the wrapper is not a reliable signal, since an outer
    retry or a degradation stub may swallow ``CircuitOpen`` entirely.
    """

    def __init__(
        self,
        tracer: trace.Tracer,
        rejections: metrics.Counter,
        transitions: metrics.Counter,
        counter_attrs: Mapping[str, AttributeValue],
    ) -> None:
        self._tracer = tracer
        self._rejections = rejections
        self._transitions = transitions
        self._counter_attrs = counter_attrs

    def event(self, name: str, attributes: Mapping[str, AttributeValue] | None = None) -> None:
        trace.get_current_span().add_event(name, attributes)
        if name == EVENT_BREAKER_REJECTED:
            self._rejections.add(1, {**self._counter_attrs, **(attributes or {})})
        elif name == EVENT_BREAKER_TRANSITION:
            self._transitions.add(1, {**self._counter_attrs, **(attributes or {})})

    def span(self, name: str, attributes: Mapping[str, AttributeValue] | None = None) -> AbstractContextManager[object]:
        return self._tracer.start_as_current_span(name, attributes=attributes)


def instrument(
    next_: Next,
    *,
    clock: Clock | None = None,
    tracer_provider: trace.TracerProvider | None = None,
    meter_provider: metrics.MeterProvider | None = None,
    classifier: ErrorClassifier | None = None,
    config: TelemetryConfig = DEFAULT_CONFIG,
) -> Next:
    """Wrap ``next_`` with the invocation span and the metrics contract.

    Providers default to the OTel globals, which are no-ops until the host
    application configures an SDK - so wrapping is always safe and nearly
    free. The clock resolves per call: this parameter, else ``ctx.clock``,
    else a process-wide system clock.
    """
    tracer = (tracer_provider or trace.get_tracer_provider()).get_tracer(conv.INSTRUMENTATION_NAME, __version__)
    meter = (meter_provider or metrics.get_meter_provider()).get_meter(conv.INSTRUMENTATION_NAME, __version__)
    calls = meter.create_counter(conv.METRIC_CALLS, unit=conv.UNIT_CALLS, description="Invocations")
    duration = meter.create_histogram(
        conv.METRIC_DURATION, unit=conv.UNIT_SECONDS, description="Invocation duration, retries included"
    )
    degradations = meter.create_counter(
        conv.METRIC_DEGRADATIONS, unit=conv.UNIT_CALLS, description="Calls answered by a stub"
    )
    rejections = meter.create_counter(
        conv.METRIC_BREAKER_REJECTIONS, unit=conv.UNIT_REJECTIONS, description="Calls rejected by an open breaker"
    )
    transitions = meter.create_counter(
        conv.METRIC_BREAKER_TRANSITIONS, unit=conv.UNIT_TRANSITIONS, description="Circuit breaker state transitions"
    )
    error_classifier = classifier or DefaultErrorClassifier()

    async def call(ctx: InvocationContext) -> Outcome[Any]:
        resolved_clock = clock or ctx.clock or _SYSTEM_CLOCK
        axes_all = _axis_attrs(ctx.scope_key)
        axes_allowed = _axis_attrs(ctx.scope_key, config.axis_allowlist)
        operation_attr: dict[str, AttributeValue] = {conv.ATTR_OPERATION: ctx.operation}
        observer = _OtelObserver(tracer, rejections, transitions, {**operation_attr, **axes_all})

        start_attrs: dict[str, AttributeValue] = {
            **operation_attr,
            conv.ATTR_CORRELATION_ID: ctx.correlation_id,
            **axes_all,
        }
        started = resolved_clock.monotonic()
        with tracer.start_as_current_span(ctx.operation, attributes=start_attrs) as span:
            previous = ctx.bag.get(OBSERVER_KEY, _MISSING)
            ctx.bag[OBSERVER_KEY] = observer
            try:
                outcome = await next_(ctx)
            except Exception as exc:
                # Span status and exception recording are handled by the span
                # context manager itself; only metrics and the class are ours.
                # BaseException (cancellation) passes through without metrics.
                elapsed = resolved_clock.monotonic() - started
                error_class = error_classifier.classify(exc).value
                span.set_attribute(conv.ATTR_ERROR_CLASS, error_class)
                calls.add(
                    1,
                    {
                        **operation_attr,
                        conv.ATTR_STATUS: conv.STATUS_ERROR,
                        conv.ATTR_ERROR_CLASS: error_class,
                        **axes_all,
                    },
                )
                duration.record(elapsed, {**operation_attr, conv.ATTR_STATUS: conv.STATUS_ERROR, **axes_allowed})
                raise
            else:
                elapsed = resolved_clock.monotonic() - started
                span.set_attribute(conv.ATTR_SOURCE, outcome.source)
                span.set_attribute(conv.ATTR_DEGRADED, outcome.degraded)
                span.set_attribute(conv.ATTR_ATTEMPTS, outcome.attempts)
                cache_fact = ctx.bag.get(FACT_CACHE)
                if isinstance(cache_fact, str):
                    span.set_attribute(conv.ATTR_CACHE, cache_fact)
                calls.add(
                    1,
                    {
                        **operation_attr,
                        conv.ATTR_STATUS: conv.STATUS_OK,
                        conv.ATTR_SOURCE: outcome.source,
                        conv.ATTR_DEGRADED: outcome.degraded,
                        **axes_allowed,
                    },
                )
                duration.record(elapsed, {**operation_attr, conv.ATTR_STATUS: conv.STATUS_OK, **axes_allowed})
                if outcome.degraded:
                    degradations.add(1, {**operation_attr, **axes_all})
                return outcome
            finally:
                if previous is _MISSING:
                    ctx.bag.pop(OBSERVER_KEY, None)
                else:
                    ctx.bag[OBSERVER_KEY] = previous

    return call
