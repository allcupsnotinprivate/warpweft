"""Axis capture on spans and metrics, driven through a real container.

Spans carry every axis pair of the invocation's slice (the cardinality allowlist
governs metrics only, never spans). Under concurrency each invocation's span and
metric points carry that invocation's own slice, with no cross-tenant bleed.
"""

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

from _support.axes import context_axes
from _support.components import scoped_recorder
from _support.otel import axis_attrs_of, by_name, metering, points_by_attrs, read, tracing
import anyio
import pytest

from warpweft.core.axes import ScopeSpec
from warpweft.core.component import AComponent, EmptySettings, Lifetime, invocable
from warpweft.core.composition import Container
from warpweft.core.telemetry import conventions as conv
from warpweft.core.telemetry.instrument import TelemetryConfig

pytestmark = pytest.mark.anyio

Make = Callable[..., AbstractAsyncContextManager[Container]]


async def test_invocation_span_carries_all_multi_axis_pairs(container: Make) -> None:
    tracer_provider, exporter = tracing()
    axes, handles = context_axes("region", "tenant")
    recorder = scoped_recorder(
        "rec", scope=("region", "tenant"), events=[], value_of=lambda: handles["tenant"].current()
    )
    async with container(recorder, config={"rec": {}}, axes=axes, tracer_provider=tracer_provider) as c:
        with handles["region"].use("eu"), handles["tenant"].use("acme"):
            await c.invoke("rec", "whoami")

    (span,) = by_name(exporter.get_finished_spans(), "rec.whoami")
    assert axis_attrs_of(span) == {"region": "eu", "tenant": "acme"}


async def test_concurrent_tenant_spans_and_metrics_do_not_cross(container: Make) -> None:
    tracer_provider, exporter = tracing()
    meter_provider, reader = metering()
    axes, handles = context_axes("tenant")
    gate = anyio.Event()

    class Svc(AComponent[EmptySettings, None, str]):
        name = "svc"
        lifetime = Lifetime.SCOPED
        scope = ScopeSpec(("tenant",))

        @invocable
        async def go(self) -> str:
            tenant = handles["tenant"].current() or "?"
            await gate.wait()
            return tenant

    async with container(
        Svc,
        config={"svc": {}},
        axes=axes,
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        telemetry=TelemetryConfig(axis_allowlist=frozenset({"acme", "globex"})),  # ok-calls carry the tenant pair
    ) as c:

        async def call(tenant: str) -> None:
            with handles["tenant"].use(tenant):
                await c.invoke("svc", "go")

        async with anyio.create_task_group() as tg:
            tg.start_soon(call, "acme")
            tg.start_soon(call, "globex")
            await anyio.sleep(0.05)
            gate.set()

    # Each finished span carries exactly its own tenant.
    tenants = sorted(axis_attrs_of(s)["tenant"] for s in by_name(exporter.get_finished_spans(), "svc.go"))
    assert tenants == ["acme", "globex"]
    # calls counter: one ok point per tenant.
    calls = points_by_attrs(read(reader)[conv.METRIC_CALLS], f"{conv.AXIS_ATTR_PREFIX}tenant", conv.ATTR_STATUS)
    assert calls[("acme", conv.STATUS_OK)].value == 1
    assert calls[("globex", conv.STATUS_OK)].value == 1
