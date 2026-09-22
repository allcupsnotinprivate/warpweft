"""In-memory OpenTelemetry helpers shared across the telemetry suites.

Every helper builds isolated in-memory providers/readers so tests never touch
process-global OTel state. Extracted from the per-file copies that previously
lived in ``test_telemetry``, ``test_telemetry_metrics``, ``test_component_telemetry``
and ``test_dependency_telemetry``.
"""

from typing import Any

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader, Metric
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from warpweft.core.telemetry import conventions as conv


def tracing() -> tuple[TracerProvider, InMemorySpanExporter]:
    """A tracer provider wired to an in-memory span exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def metering() -> tuple[MeterProvider, InMemoryMetricReader]:
    """A meter provider wired to an in-memory metric reader."""
    reader = InMemoryMetricReader()
    return MeterProvider(metric_readers=[reader]), reader


def scoped_metrics(reader: InMemoryMetricReader, scope_name: str | None = None) -> dict[str, Metric]:
    """Collected metrics by name; ``scope_name`` limits to one instrumentation scope."""
    flat: dict[str, Metric] = {}
    data = reader.get_metrics_data()
    for resource_metrics in data.resource_metrics if data else ():
        for scope_metrics in resource_metrics.scope_metrics:
            if scope_name is not None and scope_metrics.scope.name != scope_name:
                continue
            for metric in scope_metrics.metrics:
                flat[metric.name] = metric
    return flat


def read(reader: InMemoryMetricReader) -> dict[str, Metric]:
    """Flatten collected metrics by name across all scopes; absent name = nothing recorded."""
    return scoped_metrics(reader)


def sole_point(metric: Metric) -> Any:
    """The single data point of ``metric`` (raises if there is not exactly one)."""
    (point,) = metric.data.data_points
    return point


def points_by_operation(metric: Metric) -> dict[tuple[Any, Any], Any]:
    """Data points keyed by ``(operation, status)``."""
    points: dict[tuple[Any, Any], Any] = {}
    for p in metric.data.data_points:
        attrs = dict(p.attributes or {})
        points[(attrs[conv.ATTR_OPERATION], attrs.get(conv.ATTR_STATUS, ""))] = p
    return points


def points_by_attrs(metric: Metric, *attr_names: str) -> dict[tuple[Any, ...], Any]:
    """Data points keyed by the values of ``attr_names`` (missing → ``None``)."""
    points: dict[tuple[Any, ...], Any] = {}
    for p in metric.data.data_points:
        attrs = dict(p.attributes or {})
        points[tuple(attrs.get(name) for name in attr_names)] = p
    return points


def by_name(spans: tuple[ReadableSpan, ...], name: str) -> list[ReadableSpan]:
    """The finished spans whose name is exactly ``name``."""
    return [s for s in spans if s.name == name]


def axis_attrs_of(carrier: Any) -> dict[str, Any]:
    """The ``warpweft.axis.<name>`` attributes of a span or metric point, prefix stripped.

    ``carrier`` is anything with an ``attributes`` mapping (a ReadableSpan or a
    metric data point). Returns ``{axis_name: value}``.
    """
    attrs = dict(getattr(carrier, "attributes", None) or {})
    prefix = conv.AXIS_ATTR_PREFIX
    return {key[len(prefix) :]: value for key, value in attrs.items() if key.startswith(prefix)}
