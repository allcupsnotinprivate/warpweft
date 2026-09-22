"""Axis capture on spans and metrics, driven through a real container.

Spans carry every axis pair of the invocation's slice (the cardinality allowlist
governs metrics only, never spans). Under concurrency each invocation's span and
metric points carry that invocation's own slice, with no cross-tenant bleed.
"""

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

from _support.axes import context_axes
from _support.components import domain_metric_emitter, scoped_recorder
from _support.otel import axis_attrs_of, by_name, metering, points_by_attrs, read, scoped_metrics, sole_point, tracing
import anyio
import pytest

from warpweft.core.axes import ScopeSpec
from warpweft.core.component import AComponent, EmptySettings, Lifetime, invocable
from warpweft.core.composition import Container
from warpweft.core.errors import TransientError
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


@pytest.mark.characterization
async def test_error_duration_point_has_no_error_class(container: Make) -> None:
    # CHARACTERIZATION: the error-path calls counter carries warpweft.error.class,
    # but the duration histogram's error point does not - you can count errors by
    # class yet cannot get error latency by class.
    meter_provider, reader = metering()

    class Boom(AComponent[EmptySettings, None, str]):
        name = "boom"

        @invocable
        async def go(self) -> str:
            raise TransientError("down")

    async with container(Boom, config={"boom": {}}, meter_provider=meter_provider) as c:
        with pytest.raises(TransientError):
            await c.invoke("boom", "go")

    metrics = read(reader)
    duration = sole_point(metrics[conv.METRIC_DURATION])
    calls = sole_point(metrics[conv.METRIC_CALLS])
    assert conv.ATTR_ERROR_CLASS not in dict(duration.attributes)  # missing on the histogram
    assert conv.ATTR_ERROR_CLASS in dict(calls.attributes)  # present on the counter


@pytest.mark.characterization
async def test_process_domain_metrics_carry_no_axis_pairs(container: Make) -> None:
    # CHARACTERIZATION: a process component is instantiated under GLOBAL_SCOPE, so
    # its self.telemetry metrics never carry per-request axis pairs, even when the
    # value is allowlisted.
    meter_provider, reader = metering()
    axes, handles = context_axes("tenant", defaults={"tenant": "public"})
    emitter = domain_metric_emitter("emit", lifetime=Lifetime.PROCESS)
    async with container(
        emitter,
        config={"emit": {}},
        axes=axes,
        meter_provider=meter_provider,
        telemetry=TelemetryConfig(axis_allowlist=frozenset({"acme"})),
    ) as c:
        with handles["tenant"].use("acme"):
            await c.invoke("emit", "work")

    point = sole_point(scoped_metrics(reader, conv.INSTRUMENTATION_COMPONENT_NAME)["test.hits"])
    assert dict(point.attributes) == {conv.ATTR_COMPONENT: "emit"}  # no axis breakdown, ever


async def test_scoped_domain_metrics_split_per_tenant(container: Make) -> None:
    meter_provider, reader = metering()
    axes, handles = context_axes("tenant")
    emitter = domain_metric_emitter("emit", lifetime=Lifetime.SCOPED, scope=("tenant",))
    async with container(
        emitter,
        config={"emit": {}},
        axes=axes,
        meter_provider=meter_provider,
        telemetry=TelemetryConfig(axis_allowlist=frozenset({"acme", "globex"})),
    ) as c:
        for tenant in ("acme", "globex"):
            with handles["tenant"].use(tenant):
                await c.invoke("emit", "work")

    points = points_by_attrs(
        scoped_metrics(reader, conv.INSTRUMENTATION_COMPONENT_NAME)["test.hits"], f"{conv.AXIS_ATTR_PREFIX}tenant"
    )
    assert points[("acme",)].value == 1
    assert points[("globex",)].value == 1


@pytest.mark.characterization
async def test_endpoint_never_reaches_axis_attributes(container: Make) -> None:
    # CHARACTERIZATION: use_endpoint drives [endpoint]-sliced link state, but the
    # endpoint is not part of the telemetry scope_key unless declared in ScopeSpec.
    # A component with a real endpoint() emits spans without warpweft.axis.endpoint.
    tracer_provider, exporter = tracing()

    class Svc(AComponent[EmptySettings, None, str]):
        name = "svc"

        def endpoint(self) -> str | None:
            return "host-x"

        @invocable
        async def go(self) -> str:
            return "ok"

    async with container(Svc, config={"svc": {}}, tracer_provider=tracer_provider) as c:
        await c.invoke("svc", "go")

    (span,) = by_name(exporter.get_finished_spans(), "svc.go")
    assert axis_attrs_of(span) == {"component": "svc"}  # endpoint 'host-x' is absent from the axes
