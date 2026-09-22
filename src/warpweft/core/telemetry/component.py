"""Domain metrics for components: the object behind ``self.telemetry``.

A component records its own business metrics (tokens consumed, rows synced)
through ``self.telemetry`` - no meter plumbing, no boilerplate. The container
binds a `ComponentTelemetry` to every instance it creates; a component
constructed directly (unit tests) falls back to a process-wide no-op, so the
same code runs anywhere.

Every recorded point automatically carries ``warpweft.component`` (the
component's name) and the instance's axis pairs as ``warpweft.axis.<name>`` -
the latter filtered by the same ``axis_allowlist`` cardinality policy the
framework metrics follow. User attributes merge underneath: they cannot
overwrite the automatic ones.

Metric names are entirely the component author's (e.g. ``o2.llm.tokens``); no
prefix is imposed. The meter uses its own instrumentation scope,
``warpweft.component``, keeping host-defined metrics apart from the
``warpweft`` scope whose names are the framework's stability contract.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Final

from opentelemetry import metrics
from opentelemetry.metrics import NoOpMeter

from warpweft.core import __version__
from warpweft.core.axes import GLOBAL_SCOPE, ScopeKey
from warpweft.core.observe import AttributeValue

from . import conventions as conv
from .instrument import _axis_attrs


class ComponentCounter:
    """Counter with the component's identity attributes mixed in."""

    def __init__(self, counter: metrics.Counter, auto: Mapping[str, AttributeValue]) -> None:
        self._counter = counter
        self._auto = auto

    def add(self, value: int | float, attributes: Mapping[str, AttributeValue] | None = None) -> None:
        """Record an increment; automatic attributes win over ``attributes``."""
        self._counter.add(value, {**(attributes or {}), **self._auto})


class ComponentHistogram:
    """Histogram with the component's identity attributes mixed in."""

    def __init__(self, histogram: metrics.Histogram, auto: Mapping[str, AttributeValue]) -> None:
        self._histogram = histogram
        self._auto = auto

    def record(self, value: int | float, attributes: Mapping[str, AttributeValue] | None = None) -> None:
        """Record a measurement; automatic attributes win over ``attributes``."""
        self._histogram.record(value, {**(attributes or {}), **self._auto})


class ComponentTelemetry:
    """A component's channel for its own metrics.

    Instruments are created on first use and cached by name, so
    ``self.telemetry.counter("o2.llm.tokens").add(n)`` in a hot path costs a
    dict lookup. A scoped component's instance gets its slice's axis pairs
    baked in, so per-tenant breakdown needs no code.
    """

    def __init__(
        self,
        component: str,
        scope_key: ScopeKey,
        meter: metrics.Meter,
        axis_allowlist: frozenset[str] = frozenset(),
    ) -> None:
        self._meter = meter
        self._auto: dict[str, AttributeValue] = {
            conv.ATTR_COMPONENT: component,
            **_axis_attrs(scope_key, axis_allowlist),
        }
        self._counters: dict[str, ComponentCounter] = {}
        self._histograms: dict[str, ComponentHistogram] = {}

    def counter(self, name: str, *, unit: str = "", description: str = "") -> ComponentCounter:
        """A named counter (cached); ``unit``/``description`` apply on first creation."""
        cached = self._counters.get(name)
        if cached is None:
            cached = ComponentCounter(self._meter.create_counter(name, unit=unit, description=description), self._auto)
            self._counters[name] = cached
        return cached

    def histogram(self, name: str, *, unit: str = "", description: str = "") -> ComponentHistogram:
        """A named histogram (cached); ``unit``/``description`` apply on first creation."""
        cached = self._histograms.get(name)
        if cached is None:
            cached = ComponentHistogram(
                self._meter.create_histogram(name, unit=unit, description=description), self._auto
            )
            self._histograms[name] = cached
        return cached


def component_meter(meter_provider: Any = None) -> metrics.Meter:
    """The meter component metrics record through (scope ``warpweft.component``)."""
    provider = meter_provider or metrics.get_meter_provider()
    return provider.get_meter(conv.INSTRUMENTATION_COMPONENT_NAME, __version__)


#: Fallback for components created outside a container: records nothing, never fails.
NOOP_TELEMETRY: Final = ComponentTelemetry("", GLOBAL_SCOPE, NoOpMeter(conv.INSTRUMENTATION_COMPONENT_NAME))


#: The telemetry channel the container binds *around* a component's construction,
#: so ``self.telemetry`` already works inside ``__init__`` (an instrument cached
#: there then points at the live channel, not the no-op). ``None`` outside a build.
_CONSTRUCTING: ContextVar[ComponentTelemetry | None] = ContextVar("warpweft_component_telemetry", default=None)


@contextmanager
def binding_telemetry(telemetry: ComponentTelemetry) -> Iterator[None]:
    """Expose ``telemetry`` to ``AComponent.telemetry`` for the duration of construction."""
    token = _CONSTRUCTING.set(telemetry)
    try:
        yield
    finally:
        _CONSTRUCTING.reset(token)


def constructing_telemetry() -> ComponentTelemetry | None:
    """The telemetry channel bound around the component currently being constructed, if any."""
    return _CONSTRUCTING.get()
