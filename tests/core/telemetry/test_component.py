"""Component domain metrics: ``self.telemetry`` inside and outside a container.

Inside a container each instance records through the container's meter
provider with ``warpweft.component`` and its slice's allowlisted axis pairs
attached; outside, the property degrades to a process-wide no-op.
"""

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from contextvars import ContextVar

from _support.otel import metering, scoped_metrics, sole_point
import pytest

from warpweft.core.axes import Axis, AxisRegistry, ScopeSpec
from warpweft.core.component import AComponent, EmptySettings, Lifetime, invocable
from warpweft.core.composition import Container
from warpweft.core.telemetry import conventions as conv
from warpweft.core.telemetry.component import NOOP_TELEMETRY
from warpweft.core.telemetry.instrument import TelemetryConfig

#: A container factory; requesting it tags each test ``integration`` via the auto-marker.
Make = Callable[..., AbstractAsyncContextManager[Container]]

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


# --- inside a container ------------------------------------------------------


async def test_component_metric_carries_the_component_attribute(container: Make) -> None:
    provider, reader = metering()
    async with container(Meterful, config={"meterful": {}}, meter_provider=provider) as c:
        await c.invoke("meterful", "work")

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


async def test_telemetry_cached_in_init_records_to_the_live_channel(container: Make) -> None:
    provider, reader = metering()
    async with container(EagerMeterful, config={"eager-meterful": {}}, meter_provider=provider) as c:
        await c.invoke("eager-meterful", "work")

    metric = scoped_metrics(reader, conv.INSTRUMENTATION_COMPONENT_NAME)["app.eager"]
    point = sole_point(metric)
    assert point.value == 3  # not silently dropped by an __init__-time no-op
    assert dict(point.attributes) == {conv.ATTR_COMPONENT: "eager-meterful"}


async def test_scoped_component_axes_follow_the_allowlist(container: Make) -> None:
    provider, reader = metering()
    axes = AxisRegistry()
    axes.register(Axis(name="tenant", resolver=_tenant.get))
    async with container(
        TenantMeterful,
        config={"tenant-meterful": {}},
        axes=axes,
        meter_provider=provider,
        telemetry=TelemetryConfig(axis_allowlist=frozenset({"acme"})),
    ) as c:
        _tenant.set("acme")
        await c.invoke("tenant-meterful", "work")
        _tenant.set("globex")
        await c.invoke("tenant-meterful", "work")

    metric = scoped_metrics(reader, conv.INSTRUMENTATION_COMPONENT_NAME)["app.tenant.things"]
    attr_sets = [dict(p.attributes) for p in metric.data.data_points]
    # acme is allowlisted: its point carries the axis pair; globex's does not.
    assert {conv.ATTR_COMPONENT: "tenant-meterful", f"{conv.AXIS_ATTR_PREFIX}tenant": "acme"} in attr_sets
    assert {conv.ATTR_COMPONENT: "tenant-meterful"} in attr_sets
    assert len(attr_sets) == 2


async def test_user_attributes_cannot_overwrite_automatic_ones(container: Make) -> None:
    provider, reader = metering()
    async with container(Spoofer, config={"spoofer": {}}, meter_provider=provider) as c:
        await c.invoke("spoofer", "work")

    point = sole_point(scoped_metrics(reader, conv.INSTRUMENTATION_COMPONENT_NAME)["app.spoofed"])
    assert dict(point.attributes)[conv.ATTR_COMPONENT] == "spoofer"


async def test_instruments_are_cached_by_name(container: Make) -> None:
    provider, _ = metering()
    async with container(Meterful, config={"meterful": {}}, meter_provider=provider) as c:
        instance = await c.get(Meterful)
        assert instance.telemetry.counter("app.things") is instance.telemetry.counter("app.things")
        assert instance.telemetry.histogram("app.size") is instance.telemetry.histogram("app.size")


# --- outside a container -----------------------------------------------------


def test_component_created_directly_gets_a_working_noop() -> None:
    component = Meterful(EmptySettings())
    assert component.telemetry is NOOP_TELEMETRY
    component.telemetry.counter("app.things").add(1, {"kind": "x"})  # records nothing, raises nothing
    component.telemetry.histogram("app.size").record(2.0)
