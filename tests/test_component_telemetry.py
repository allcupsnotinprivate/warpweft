"""Component domain metrics: ``self.telemetry`` inside and outside a container.

Inside a container each instance records through the container's meter
provider with ``warpweft.component`` and its slice's allowlisted axis pairs
attached; outside, the property degrades to a process-wide no-op.
"""

from contextvars import ContextVar
from typing import Any

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader, Metric
import pytest

from warpweft.core.axes import Axis, AxisRegistry, ScopeSpec
from warpweft.core.component import AComponent, EmptySettings, Lifetime, invocable
from warpweft.core.composition import Container, Registry
from warpweft.core.telemetry import conventions as conv
from warpweft.core.telemetry.component import NOOP_TELEMETRY
from warpweft.core.telemetry.instrument import TelemetryConfig

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

_tenant: ContextVar[str | None] = ContextVar("test_tenant", default=None)


# --- test components ---------------------------------------------------------


class Meterful(AComponent[EmptySettings, None, str]):
    name = "meterful"

    @invocable
    async def work(self) -> str:
        self.telemetry.counter("app.things", unit="{thing}", description="Things processed").add(2, {"kind": "x"})
        self.telemetry.histogram("app.size", unit="By").record(1.5)
        return "ok"


class TenantMeterful(AComponent[EmptySettings, None, str]):
    name = "tenant-meterful"
    lifetime = Lifetime.SCOPED
    scope = ScopeSpec(("tenant",))

    @invocable
    async def work(self) -> str:
        self.telemetry.counter("app.tenant.things").add(1)
        return "ok"


class Spoofer(AComponent[EmptySettings, None, str]):
    name = "spoofer"

    @invocable
    async def work(self) -> str:
        # User attributes must not overwrite the automatic ones.
        self.telemetry.counter("app.spoofed").add(1, {conv.ATTR_COMPONENT: "not-me"})
        return "ok"


class EagerMeterful(AComponent[EmptySettings, None, str]):
    name = "eager-meterful"

    def __init__(self, settings: EmptySettings) -> None:
        super().__init__(settings)
        # Cache the instrument at construction: it must bind to the live channel,
        # not the no-op, so runtime .add() calls are actually recorded.
        self._counter = self.telemetry.counter("app.eager")

    @invocable
    async def work(self) -> str:
        self._counter.add(3)
        return "ok"


# --- helpers -----------------------------------------------------------------


def _registry_of(*classes: type[AComponent[Any, Any, Any]]) -> Registry:
    registry = Registry()
    for cls in classes:
        registry.register(cls)
    return registry


def metering() -> tuple[MeterProvider, InMemoryMetricReader]:
    reader = InMemoryMetricReader()
    return MeterProvider(metric_readers=[reader]), reader


def scoped_metrics(reader: InMemoryMetricReader, scope_name: str) -> dict[str, Metric]:
    """Metrics recorded under one instrumentation scope, by name."""
    flat: dict[str, Metric] = {}
    data = reader.get_metrics_data()
    for resource_metrics in data.resource_metrics if data else ():
        for scope_metrics in resource_metrics.scope_metrics:
            if scope_metrics.scope.name != scope_name:
                continue
            for metric in scope_metrics.metrics:
                flat[metric.name] = metric
    return flat


def sole_point(metric: Metric) -> Any:
    (point,) = metric.data.data_points
    return point


# --- inside a container ------------------------------------------------------


async def test_component_metric_carries_the_component_attribute() -> None:
    provider, reader = metering()
    container = Container.build(_registry_of(Meterful), {"meterful": {}}, meter_provider=provider)
    await container.start()
    await container.invoke("meterful", "work")
    await container.stop()

    metrics = scoped_metrics(reader, conv.INSTRUMENTATION_COMPONENT_NAME)
    counter = metrics["app.things"]
    assert counter.unit == "{thing}"
    point = sole_point(counter)
    assert point.value == 2
    assert dict(point.attributes) == {conv.ATTR_COMPONENT: "meterful", "kind": "x"}

    histogram = sole_point(metrics["app.size"])
    assert histogram.count == 1
    assert histogram.sum == pytest.approx(1.5)
    assert dict(histogram.attributes) == {conv.ATTR_COMPONENT: "meterful"}


async def test_telemetry_cached_in_init_records_to_the_live_channel() -> None:
    provider, reader = metering()
    container = Container.build(_registry_of(EagerMeterful), {"eager-meterful": {}}, meter_provider=provider)
    await container.start()
    await container.invoke("eager-meterful", "work")
    await container.stop()

    metric = scoped_metrics(reader, conv.INSTRUMENTATION_COMPONENT_NAME)["app.eager"]
    point = sole_point(metric)
    assert point.value == 3  # not silently dropped by an __init__-time no-op
    assert dict(point.attributes) == {conv.ATTR_COMPONENT: "eager-meterful"}


async def test_scoped_component_axes_follow_the_allowlist() -> None:
    provider, reader = metering()
    axes = AxisRegistry()
    axes.register(Axis(name="tenant", resolver=_tenant.get))
    container = Container.build(
        _registry_of(TenantMeterful),
        {"tenant-meterful": {}},
        axes=axes,
        meter_provider=provider,
        telemetry=TelemetryConfig(axis_allowlist=frozenset({"acme"})),
    )
    await container.start()
    _tenant.set("acme")
    await container.invoke("tenant-meterful", "work")
    _tenant.set("globex")
    await container.invoke("tenant-meterful", "work")
    await container.stop()

    metric = scoped_metrics(reader, conv.INSTRUMENTATION_COMPONENT_NAME)["app.tenant.things"]
    attr_sets = [dict(p.attributes) for p in metric.data.data_points]
    # acme is allowlisted: its point carries the axis pair; globex's does not.
    assert {conv.ATTR_COMPONENT: "tenant-meterful", f"{conv.AXIS_ATTR_PREFIX}tenant": "acme"} in attr_sets
    assert {conv.ATTR_COMPONENT: "tenant-meterful"} in attr_sets
    assert len(attr_sets) == 2


async def test_user_attributes_cannot_overwrite_automatic_ones() -> None:
    provider, reader = metering()
    container = Container.build(_registry_of(Spoofer), {"spoofer": {}}, meter_provider=provider)
    await container.start()
    await container.invoke("spoofer", "work")
    await container.stop()

    point = sole_point(scoped_metrics(reader, conv.INSTRUMENTATION_COMPONENT_NAME)["app.spoofed"])
    assert dict(point.attributes)[conv.ATTR_COMPONENT] == "spoofer"


async def test_instruments_are_cached_by_name() -> None:
    provider, _ = metering()
    container = Container.build(_registry_of(Meterful), {"meterful": {}}, meter_provider=provider)
    await container.start()
    instance = await container.get(Meterful)
    assert instance.telemetry.counter("app.things") is instance.telemetry.counter("app.things")
    assert instance.telemetry.histogram("app.size") is instance.telemetry.histogram("app.size")
    await container.stop()


# --- outside a container -----------------------------------------------------


def test_component_created_directly_gets_a_working_noop() -> None:
    component = Meterful(EmptySettings())
    assert component.telemetry is NOOP_TELEMETRY
    component.telemetry.counter("app.things").add(1, {"kind": "x"})  # records nothing, raises nothing
    component.telemetry.histogram("app.size").record(2.0)
