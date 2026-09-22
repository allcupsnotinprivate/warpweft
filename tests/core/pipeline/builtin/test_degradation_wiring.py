"""Degradation wired end-to-end: config + criticality + stub() through a real Container."""

from contextvars import ContextVar
from typing import Any

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader, Metric
import pytest

from warpweft.core.axes import Axis, AxisRegistry, ScopeSpec
from warpweft.core.component import AComponent, Criticality, EmptySettings, Lifetime, invocable
from warpweft.core.composition import Container, DictSettingsResolver, Registry
from warpweft.core.context import InvocationContext
from warpweft.core.errors import ConfigurationError, PermanentError, TransientError
from warpweft.core.telemetry import conventions as conv

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

_tenant: ContextVar[str | None] = ContextVar("degradation_tenant", default=None)


# --- test components ---------------------------------------------------------


class Fallback(AComponent[EmptySettings, None, Any]):
    name = "fallback"
    criticality = Criticality.OPTIONAL

    def __init__(self, settings: EmptySettings) -> None:
        super().__init__(settings)
        self.healthy = False
        self.calls = 0

    def stub(self, ctx: InvocationContext) -> Any:
        return {"stubbed": ctx.operation}

    @invocable
    async def fetch(self) -> Any:
        self.calls += 1
        if not self.healthy:
            raise TransientError("down")
        return {"live": True}


class OptionalNoStub(AComponent[EmptySettings, None, Any]):
    name = "optional-no-stub"
    criticality = Criticality.OPTIONAL

    @invocable
    async def fetch(self) -> Any:
        raise TransientError("down")


class RequiredWithStub(AComponent[EmptySettings, None, Any]):
    name = "required-with-stub"

    def stub(self, ctx: InvocationContext) -> Any:
        return {"stubbed": True}

    @invocable
    async def fetch(self) -> Any:
        raise TransientError("down")


class Picky(AComponent[EmptySettings, None, Any]):
    """Degrades permanent errors, re-raises transient ones (degrade_on override)."""

    name = "picky"
    criticality = Criticality.OPTIONAL

    def stub(self, ctx: InvocationContext) -> Any:
        return {"stubbed": True}

    @invocable
    async def fetch(self, *, permanent: bool) -> Any:
        if permanent:
            raise PermanentError("nope")
        raise TransientError("down")


class ScopedFallback(AComponent[EmptySettings, None, Any]):
    name = "scoped-fallback"
    criticality = Criticality.OPTIONAL
    lifetime = Lifetime.SCOPED
    scope = ScopeSpec(("tenant",))

    def stub(self, ctx: InvocationContext) -> Any:
        return {"stubbed": ctx.operation}

    @invocable
    async def fetch(self) -> Any:
        raise TransientError("down")


DEGRADE = {"policy": {"degradation": {}}}


def _tenant_axes() -> AxisRegistry:
    axes = AxisRegistry()
    axes.register(Axis(name="tenant", resolver=_tenant.get))
    return axes


def _metering() -> tuple[MeterProvider, InMemoryMetricReader]:
    reader = InMemoryMetricReader()
    return MeterProvider(metric_readers=[reader]), reader


def _read(reader: InMemoryMetricReader) -> dict[str, Metric]:
    flat: dict[str, Metric] = {}
    data = reader.get_metrics_data()
    for resource_metrics in data.resource_metrics if data else ():
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                flat[metric.name] = metric
    return flat


# --- build-time validation ---------------------------------------------------


async def test_build_rejects_degradation_on_required_component() -> None:
    reg = Registry()
    reg.register(RequiredWithStub)
    with pytest.raises(ConfigurationError, match="only an optional component"):
        Container.build(reg, {"required-with-stub": DEGRADE})


async def test_build_rejects_degradation_without_stub() -> None:
    reg = Registry()
    reg.register(OptionalNoStub)
    with pytest.raises(ConfigurationError, match="defines no stub"):
        Container.build(reg, {"optional-no-stub": DEGRADE})


# --- runtime behaviour -------------------------------------------------------


async def test_outage_serves_stub_then_recovers() -> None:
    reg = Registry()
    reg.register(Fallback)
    container = Container.build(reg, {"fallback": DEGRADE})
    await container.start()
    try:
        outcome = await container.invoke("fallback", "fetch")
        assert outcome.value == {"stubbed": "fallback.fetch"}
        assert outcome.source == "stub"
        assert outcome.degraded is True

        instance = await container.get(Fallback)
        instance.healthy = True
        recovered = await container.invoke("fallback", "fetch")
        assert recovered.value == {"live": True}
        assert recovered.source == "live"
        assert recovered.degraded is False
    finally:
        await container.stop()


async def test_no_degradation_without_config_even_with_stub() -> None:
    reg = Registry()
    reg.register(Fallback)
    container = Container.build(reg, {"fallback": {}})
    await container.start()
    try:
        with pytest.raises(TransientError):
            await container.invoke("fallback", "fetch")
    finally:
        await container.stop()


async def test_degrade_on_override() -> None:
    reg = Registry()
    reg.register(Picky)
    container = Container.build(reg, {"picky": {"policy": {"degradation": {"degrade_on": "permanent"}}}})
    await container.start()
    try:
        # A permanent error now degrades...
        degraded = await container.invoke("picky", "fetch", permanent=True)
        assert degraded.source == "stub"
        # ...while a transient one surfaces.
        with pytest.raises(TransientError):
            await container.invoke("picky", "fetch", permanent=False)
    finally:
        await container.stop()


async def test_degradations_metric_increments_through_a_real_container() -> None:
    provider, reader = _metering()
    reg = Registry()
    reg.register(Fallback)
    container = Container.build(reg, {"fallback": DEGRADE}, meter_provider=provider)
    await container.start()
    try:
        await container.invoke("fallback", "fetch")
    finally:
        await container.stop()

    metric = _read(reader)[conv.METRIC_DEGRADATIONS]
    (point,) = metric.data.data_points
    assert point.value == 1
    assert dict(point.attributes)[conv.ATTR_OPERATION] == "fallback.fetch"


async def test_breaker_records_failures_while_stubs_are_served() -> None:
    reg = Registry()
    reg.register(Fallback)
    config = {
        "fallback": {
            "policy": {
                "degradation": {},
                "circuit_breaker": {"window": 4, "failure_threshold": 2, "reset_timeout": 60.0},
            }
        }
    }
    container = Container.build(reg, config)
    await container.start()
    try:
        for _ in range(3):
            outcome = await container.invoke("fallback", "fetch")
            assert outcome.source == "stub"  # every call is served a stub

        breakers = container.snapshot().breakers
        assert any(b.state == "open" for b in breakers)

        instance = await container.get(Fallback)
        calls_when_open = instance.calls
        await container.invoke("fallback", "fetch")  # rejected by the open breaker
        assert instance.calls == calls_when_open  # never reached the method
    finally:
        await container.stop()


async def test_stub_results_are_not_cached() -> None:
    reg = Registry()
    reg.register(Fallback)
    config = {"fallback": {"policy": {"degradation": {}, "cache": {"ttl": 60.0, "max_entries": 10}}}}
    container = Container.build(reg, config)
    await container.start()
    try:
        first = await container.invoke("fallback", "fetch")
        second = await container.invoke("fallback", "fetch")
        assert first.source == "stub"
        assert second.source == "stub"  # a cached stub would read "cache"

        instance = await container.get(Fallback)
        instance.healthy = True
        live = await container.invoke("fallback", "fetch")
        assert live.source == "live"
        cached = await container.invoke("fallback", "fetch")
        assert cached.source == "cache"
    finally:
        await container.stop()


async def test_retry_exhausts_then_stub_substitutes() -> None:
    reg = Registry()
    reg.register(Fallback)
    config = {
        "fallback": {"policy": {"degradation": {}, "retry": {"attempts": 3, "base_delay": 0.0, "max_delay": 0.1}}}
    }
    container = Container.build(reg, config)
    await container.start()
    try:
        outcome = await container.invoke("fallback", "fetch")
        assert outcome.source == "stub"
        instance = await container.get(Fallback)
        assert instance.calls == 3  # retried to exhaustion, then degraded
    finally:
        await container.stop()


async def test_scoped_component_degrades() -> None:
    reg = Registry()
    reg.register(ScopedFallback)
    container = Container.build(reg, {"scoped-fallback": DEGRADE}, axes=_tenant_axes())
    await container.start()
    _tenant.set("acme")
    try:
        outcome = await container.invoke("scoped-fallback", "fetch")
        assert outcome.source == "stub"
        assert outcome.value == {"stubbed": "scoped-fallback.fetch"}
    finally:
        await container.stop()


async def test_slice_override_contradiction_rejected_at_first_use() -> None:
    reg = Registry()

    class ScopedNoStub(AComponent[EmptySettings, None, Any]):
        name = "scoped-no-stub"
        criticality = Criticality.OPTIONAL
        lifetime = Lifetime.SCOPED
        scope = ScopeSpec(("tenant",))

        @invocable
        async def fetch(self) -> Any:
            raise TransientError("down")

    reg.register(ScopedNoStub)
    resolver = DictSettingsResolver({("scoped-no-stub", (("tenant", "acme"),)): DEGRADE})
    container = Container.build(reg, {"scoped-no-stub": {}}, axes=_tenant_axes(), resolver=resolver)
    await container.start()
    _tenant.set("acme")
    try:
        with pytest.raises(ConfigurationError, match="defines no stub"):
            await container.invoke("scoped-no-stub", "fetch")
    finally:
        await container.stop()


async def test_degradation_in_policy_chain_is_ignored() -> None:
    reg = Registry()
    reg.register(Fallback)
    config = {"fallback": {"policy": {"chain": ["retry", "timeout", "degradation"], "degradation": {}}}}
    container = Container.build(reg, config)
    await container.start()
    try:
        outcome = await container.invoke("fallback", "fetch")
        assert outcome.source == "stub"  # still degrades, wrapped outside the chain
        assert "degradation" not in container.explain("fallback", "fetch").chain
    finally:
        await container.stop()
