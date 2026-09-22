"""Dependency-call telemetry: injected dependencies get spans and metrics.

The container wraps every injected dependency in a telemetry proxy: each
``@invocable`` call gets the standard span/metrics contract while running no
policy links. In-memory OTel providers per test; globals untouched.
"""

from typing import Any

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader, Metric
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
import pytest

from warpweft.core.component import AComponent, EmptySettings, invocable
from warpweft.core.composition import Container, Registry
from warpweft.core.context import InvocationContext
from warpweft.core.errors import PermanentError, TransientError
from warpweft.core.outcome import Outcome
from warpweft.core.telemetry import conventions as conv

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


# --- test components ---------------------------------------------------------


class Dep(AComponent[EmptySettings, None, str]):
    name = "dep"

    def __init__(self, settings: EmptySettings) -> None:
        super().__init__(settings)
        self.calls = 0
        self.fail_times = 0

    @invocable
    async def fetch(self, tag: str = "default") -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise TransientError("down")
        return f"data:{tag}"


class BadDep(AComponent[EmptySettings, None, str]):
    name = "bad-dep"

    @invocable
    async def boom(self) -> str:
        raise PermanentError("bad request")


class Upper(AComponent[EmptySettings, None, str]):
    name = "upper"
    dep: Dep  # annotation-declared dependency

    @invocable
    async def run(self, tag: str = "default") -> str:
        return await self.dep.fetch(tag)


class UpperBad(AComponent[EmptySettings, None, str]):
    name = "upper-bad"
    dependencies = ("bad-dep",)  # name-declared dependency

    @invocable
    async def run(self) -> str:
        bad: BadDep = self.dependency("bad-dep")  # type: ignore[assignment]
        return await bad.boom()


def fresh_registry() -> Registry:
    reg = Registry()
    for cls in (Dep, BadDep, Upper, UpperBad):
        reg.register(cls)
    return reg


# --- helpers -----------------------------------------------------------------


def tracing() -> tuple[TracerProvider, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def metering() -> tuple[MeterProvider, InMemoryMetricReader]:
    reader = InMemoryMetricReader()
    return MeterProvider(metric_readers=[reader]), reader


def read(reader: InMemoryMetricReader) -> dict[str, Metric]:
    flat: dict[str, Metric] = {}
    data = reader.get_metrics_data()
    for resource_metrics in data.resource_metrics if data else ():
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                flat[metric.name] = metric
    return flat


def points_by_operation(metric: Metric) -> dict[tuple[str, str], Any]:
    """Data points keyed by ``(operation, status)``."""
    return {
        (p.attributes[conv.ATTR_OPERATION], p.attributes.get(conv.ATTR_STATUS, "")): p for p in metric.data.data_points
    }


def by_name(spans: tuple[ReadableSpan, ...], name: str) -> list[ReadableSpan]:
    return [s for s in spans if s.name == name]


# --- spans and metrics -------------------------------------------------------


async def test_dependency_call_emits_a_child_span_and_the_metrics() -> None:
    tracer_provider, exporter = tracing()
    meter_provider, reader = metering()
    container = Container.build(
        fresh_registry(),
        {"dep": {}, "upper": {}},
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )
    await container.start()
    outcome = await container.invoke("upper", "run", correlation_id="corr-1", tag="x")
    assert outcome.value == "data:x"
    await container.stop()

    spans = exporter.get_finished_spans()
    (dep_span,) = by_name(spans, "dep.fetch")
    (upper_span,) = by_name(spans, "upper.run")
    assert dep_span.parent is not None
    assert dep_span.parent.span_id == upper_span.context.span_id  # child of the invocation
    assert dep_span.context.trace_id == upper_span.context.trace_id

    attrs = dict(dep_span.attributes or {})
    assert attrs[conv.ATTR_OPERATION] == "dep.fetch"
    assert attrs[conv.ATTR_CORRELATION_ID] == "corr-1"  # inherited from the caller
    assert attrs[conv.ATTR_SOURCE] == "live"
    assert attrs[conv.ATTR_ATTEMPTS] == 1

    metrics = read(reader)
    calls = points_by_operation(metrics[conv.METRIC_CALLS])
    assert calls[("dep.fetch", conv.STATUS_OK)].value == 1
    assert calls[("upper.run", conv.STATUS_OK)].value == 1
    duration = points_by_operation(metrics[conv.METRIC_DURATION])
    assert duration[("dep.fetch", conv.STATUS_OK)].count == 1


async def test_dependency_policies_do_not_run_on_raw_calls() -> None:
    # dep configures retry, but the injected proxy runs no policy links: the
    # first failure flies up to upper's own retry, which re-invokes the whole
    # method - dep is called exactly twice, once per upper attempt.
    config = {
        "dep": {"policy": {"retry": {"attempts": 3, "base_delay": 0.0, "max_delay": 1.0}}},
        "upper": {"policy": {"retry": {"attempts": 2, "base_delay": 0.0, "max_delay": 1.0}}},
    }
    container = Container.build(fresh_registry(), config)
    await container.start()
    dep = await container.get(Dep)
    dep.fail_times = 1

    outcome = await container.invoke("upper", "run")
    assert outcome.value == "data:default"
    assert outcome.attempts == 2  # upper's retry recovered
    assert dep.calls == 2  # dep's own retry never ran (would have been 2 calls in 1 attempt)
    await container.stop()


async def test_dependency_error_sets_status_and_error_class_and_propagates() -> None:
    tracer_provider, exporter = tracing()
    meter_provider, reader = metering()
    container = Container.build(
        fresh_registry(),
        {"bad-dep": {}, "upper-bad": {}},
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )
    await container.start()
    with pytest.raises(PermanentError):
        await container.invoke("upper-bad", "run")
    await container.stop()

    (dep_span,) = by_name(exporter.get_finished_spans(), "bad-dep.boom")
    assert dep_span.status.status_code is StatusCode.ERROR
    assert dict(dep_span.attributes or {})[conv.ATTR_ERROR_CLASS] == "permanent"

    calls = points_by_operation(read(reader)[conv.METRIC_CALLS])
    point = calls[("bad-dep.boom", conv.STATUS_ERROR)]
    assert point.value == 1
    assert point.attributes[conv.ATTR_ERROR_CLASS] == "permanent"


async def test_span_enricher_runs_for_dependency_calls_too() -> None:
    tracer_provider, exporter = tracing()
    seen: list[tuple[str, Any, Any]] = []

    def enrich(span: Any, ctx: InvocationContext, outcome: Outcome[Any] | None, exc: BaseException | None) -> None:
        seen.append((ctx.operation, dict(ctx.arguments or {}), None if outcome is None else outcome.value))
        span.set_attribute("app.op", ctx.operation)

    container = Container.build(
        fresh_registry(),
        {"dep": {}, "upper": {}},
        tracer_provider=tracer_provider,
        span_enricher=enrich,
    )
    await container.start()
    await container.invoke("upper", "run", tag="q")
    await container.stop()

    operations = [op for op, _, _ in seen]
    assert operations == ["dep.fetch", "upper.run"]  # dependency finished first
    op, arguments, value = seen[0]
    assert arguments == {"tag": "q"}  # bound via the method's signature
    assert value == "data:q"
    (dep_span,) = by_name(exporter.get_finished_spans(), "dep.fetch")
    assert dict(dep_span.attributes or {})["app.op"] == "dep.fetch"


# --- proxy transparency ------------------------------------------------------


async def test_injected_dependency_passes_isinstance_and_attribute_access() -> None:
    container = Container.build(fresh_registry(), {"dep": {}, "upper": {}})
    await container.start()
    upper = await container.get(Upper)
    dep = await container.get(Dep)

    assert isinstance(upper.dep, Dep)  # the proxy reports the dependency's class
    assert type(upper.dep) is not Dep  # ...but it is a proxy, not the instance
    assert upper.dep.settings is dep.settings  # plain attributes pass through
    assert upper.dep.calls == 0
    await container.stop()


async def test_guarded_invocation_of_the_dependency_is_not_doubled() -> None:
    # Invoking dep directly goes through its own chain built on the bare
    # instance - exactly one dep.fetch span, not one per wrapping layer.
    tracer_provider, exporter = tracing()
    container = Container.build(fresh_registry(), {"dep": {}, "upper": {}}, tracer_provider=tracer_provider)
    await container.start()
    await container.invoke("dep", "fetch")
    await container.stop()

    assert len(by_name(exporter.get_finished_spans(), "dep.fetch")) == 1
