"""Dependency-call telemetry: injected dependencies get spans and metrics.

The container wraps every injected dependency in a telemetry proxy: each
``@invocable`` call gets the standard span/metrics contract while running no
policy links. In-memory OTel providers per test; globals untouched.
"""

from collections.abc import Mapping
from typing import Any

from _support.axes import context_axes
from _support.containers import registry_of
from _support.otel import axis_attrs_of, by_name, metering, points_by_operation, read, tracing
import anyio
from opentelemetry.trace import StatusCode
from pydantic import BaseModel
import pytest

from warpweft.core.axes import GLOBAL_SCOPE, ScopeSpec
from warpweft.core.component import AComponent, EmptySettings, Lifetime, invocable
from warpweft.core.component.invocable import InputBinding, set_input_binding
from warpweft.core.composition import Container, Registry
from warpweft.core.context import InvocationContext
from warpweft.core.errors import PermanentError, TransientError
from warpweft.core.outcome import Outcome
from warpweft.core.telemetry import conventions as conv
from warpweft.core.telemetry.dependency import DependencyTelemetryProxy

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
    return registry_of(Dep, BadDep, Upper, UpperBad)


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


async def test_dependency_outcome_is_reported_not_rewrapped() -> None:
    # #44: a method that returns its own Outcome (a degraded, cached result) must
    # have the span reflect that source/degraded/attempts, not dataclass defaults.
    tracer_provider, exporter = tracing()
    meter_provider, reader = metering()

    class Cache:
        async def load(self, key: str) -> Outcome[str]:
            return Outcome(value=f"v:{key}", source="cache", degraded=True, attempts=2)

    proxy = DependencyTelemetryProxy(
        Cache(),
        component="cache",
        methods={"load": None},
        scope_key=GLOBAL_SCOPE,
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )
    assert await proxy.load("k") == "v:k"  # caller still receives the bare value

    (span,) = by_name(exporter.get_finished_spans(), "cache.load")
    attrs = dict(span.attributes or {})
    assert attrs[conv.ATTR_SOURCE] == "cache"
    assert attrs[conv.ATTR_DEGRADED] is True
    assert attrs[conv.ATTR_ATTEMPTS] == 2

    degradations = points_by_operation(read(reader)[conv.METRIC_DEGRADATIONS])
    assert degradations[("cache.load", "")].value == 1  # a degradation point is emitted


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


# --- proxy transparency ------------------------------------------------------


class _Widget:
    """A value-like, container-like, context-manager-like dependency stand-in."""

    def __init__(self, items: list[int]) -> None:
        self.items = items
        self.entered = False

    def __len__(self) -> int:
        return len(self.items)

    def __bool__(self) -> bool:
        return bool(self.items)

    def __contains__(self, x: int) -> bool:
        return x in self.items

    def __iter__(self) -> Any:
        return iter(self.items)

    def __getitem__(self, i: int) -> int:
        return self.items[i]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Widget) and other.items == self.items

    def __hash__(self) -> int:
        return hash(tuple(self.items))

    def __repr__(self) -> str:
        return f"_Widget({self.items!r})"

    async def __aenter__(self) -> str:
        self.entered = True
        return "conn"

    async def __aexit__(self, *exc: Any) -> bool:
        self.entered = False
        return False


def _proxy_for(instance: object, methods: Mapping[str, Any] | None = None) -> Any:
    return DependencyTelemetryProxy(instance, component="w", methods=methods or {}, scope_key=GLOBAL_SCOPE)


async def test_proxy_forwards_object_protocols() -> None:
    inst = _Widget([1, 2, 3])
    proxy = _proxy_for(inst)

    assert isinstance(proxy, _Widget)  # __class__ spoof
    assert len(proxy) == 3
    assert bool(proxy) is True
    assert bool(_proxy_for(_Widget([]))) is False
    assert 2 in proxy
    assert list(proxy) == [1, 2, 3]
    assert proxy[0] == 1
    assert repr(proxy) == "_Widget([1, 2, 3])"  # not the wrapper's repr

    async with proxy as conn:  # (async) context-manager protocol forwards
        assert conn == "conn"
        assert inst.entered is True
    assert inst.entered is False


async def test_proxy_equals_and_hashes_like_the_wrapped_instance() -> None:
    inst = _Widget([1, 2, 3])
    proxy = _proxy_for(inst)

    assert proxy == inst
    assert inst == proxy
    assert proxy == _Widget([1, 2, 3])
    assert hash(proxy) == hash(inst)
    assert {proxy, inst} == {inst}  # interchangeable as set members
    assert proxy is not inst  # identity still cannot be forwarded


async def test_callable_dependency_methods_are_not_proxy_instrumented() -> None:
    # A callable dependency (an Action) guards+instruments itself via its own
    # chain; the proxy must not add a second, unguarded span for its invocables.
    provider, exporter = tracing()

    class Act:
        def __call__(self) -> str:
            return "called"

        async def run(self) -> str:
            return "raw"

    proxy = DependencyTelemetryProxy(
        Act(), component="act", methods={"run": None}, scope_key=GLOBAL_SCOPE, tracer_provider=provider
    )
    assert proxy() == "called"  # __call__ forwards to the instance's own chain
    assert await proxy.run() == "raw"  # invocable is left raw, not wrapped
    assert exporter.get_finished_spans() == ()  # no proxy-emitted span


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


# --- dependency-call scope attribution ---------------------------------------


async def test_dependency_span_carries_the_deps_slice_not_the_callers() -> None:
    tracer_provider, exporter = tracing()
    axes, handles = context_axes("tenant", "region")

    class RegionDep(AComponent[EmptySettings, None, str]):
        name = "region-dep"
        lifetime = Lifetime.SCOPED
        scope = ScopeSpec(("region",))

        @invocable
        async def where(self) -> str:
            return "x"

    class TenantCaller(AComponent[EmptySettings, None, str]):
        name = "tenant-caller"
        lifetime = Lifetime.SCOPED
        scope = ScopeSpec(("tenant",))
        dependencies = ("region-dep",)

        @invocable
        async def ask(self) -> str:
            return await self.dependency("region-dep").where()

    container = Container.build(
        registry_of(RegionDep, TenantCaller),
        {"region-dep": {}, "tenant-caller": {}},
        axes=axes,
        tracer_provider=tracer_provider,
    )
    await container.start()
    with handles["tenant"].use("acme"), handles["region"].use("eu"):
        await container.invoke("tenant-caller", "ask")
    await container.stop()

    (parent,) = by_name(exporter.get_finished_spans(), "tenant-caller.ask")
    (child,) = by_name(exporter.get_finished_spans(), "region-dep.where")
    assert axis_attrs_of(parent) == {"tenant": "acme"}  # caller's own slice
    assert axis_attrs_of(child) == {"region": "eu"}  # the dependency's own slice, not the caller's


async def test_process_dep_span_carries_component_scope_key() -> None:
    tracer_provider, exporter = tracing()
    container = Container.build(fresh_registry(), {"dep": {}, "upper": {}}, tracer_provider=tracer_provider)
    await container.start()
    await container.invoke("upper", "run", tag="x")
    await container.stop()

    (dep_span,) = by_name(exporter.get_finished_spans(), "dep.fetch")
    assert axis_attrs_of(dep_span) == {"component": "dep"}  # process dep is sliced by its component key


# --- dependency-call argument shape, deadline and correlation ----------------


async def test_positional_dependency_call_captures_named_arguments() -> None:
    seen: dict[str, dict[str, Any]] = {}

    def enrich(span: Any, ctx: InvocationContext, outcome: Any, exc: Any) -> None:
        seen[ctx.operation] = dict(ctx.arguments or {})

    class Dep(AComponent[EmptySettings, None, str]):
        name = "dep"

        @invocable
        async def fetch(self, x: int, y: int = 5) -> str:
            return f"{x},{y}"

    class Up(AComponent[EmptySettings, None, str]):
        name = "up"
        dependencies = ("dep",)

        @invocable
        async def run(self) -> str:
            return await self.dependency("dep").fetch(1)  # positional

    container = Container.build(registry_of(Dep, Up), {"dep": {}, "up": {}}, span_enricher=enrich)
    await container.start()
    await container.invoke("up", "run")
    await container.stop()
    assert seen["dep.fetch"] == {"x": 1}  # positional mapped to the parameter name via signature.bind


async def test_kwargs_dependency_call_reports_flat_arguments() -> None:
    # #45: a **kwargs method's ctx.arguments must be flat on the dependency path,
    # matching the guarded path - not nested under the VAR_KEYWORD parameter name.
    seen: dict[str, dict[str, Any]] = {}

    def enrich(span: Any, ctx: InvocationContext, outcome: Any, exc: Any) -> None:
        seen[ctx.operation] = dict(ctx.arguments or {})

    class Dep(AComponent[EmptySettings, None, str]):
        name = "dep"

        @invocable
        async def fetch(self, **filters: Any) -> str:
            return ",".join(f"{k}={v}" for k, v in sorted(filters.items()))

    class Up(AComponent[EmptySettings, None, str]):
        name = "up"
        dependencies = ("dep",)

        @invocable
        async def run(self) -> str:
            return await self.dependency("dep").fetch(a=1, b=2)

    container = Container.build(registry_of(Dep, Up), {"dep": {}, "up": {}}, span_enricher=enrich)
    await container.start()
    await container.invoke("up", "run")
    await container.stop()
    assert seen["dep.fetch"] == {"a": 1, "b": 2}  # flattened, not {"filters": {...}}


async def test_boxed_binding_dependency_call_reports_caller_facing_fields() -> None:
    # #45 repro: an invocable with a custom InputBinding (flat fields -> one model
    # param) must report the flat caller-facing fields on the dependency path, so a
    # span_enricher reads the same keys as on the guarded path.
    seen: dict[str, dict[str, Any]] = {}

    def enrich(span: Any, ctx: InvocationContext, outcome: Any, exc: Any) -> None:
        seen[ctx.operation] = dict(ctx.arguments or {})

    class Query(BaseModel):
        field_a: int
        field_b: str

    class Dep(AComponent[EmptySettings, None, str]):
        name = "dep"

        @invocable
        async def search(self, query: Query) -> str:
            return f"{query.field_a}:{query.field_b}"

    set_input_binding(
        Dep.search,
        InputBinding(
            model=Query,
            bind=lambda args: {"query": Query.model_validate(dict(args))},
            caller_view=lambda kwargs: dict(kwargs["query"].model_dump()),
        ),
    )

    class Up(AComponent[EmptySettings, None, str]):
        name = "up"
        dependencies = ("dep",)

        @invocable
        async def run(self) -> str:
            return await self.dependency("dep").search(Query(field_a=1, field_b="x"))

    container = Container.build(registry_of(Dep, Up), {"dep": {}, "up": {}}, span_enricher=enrich)
    await container.start()
    await container.invoke("up", "run")
    await container.stop()
    assert seen["dep.search"] == {"field_a": 1, "field_b": "x"}  # flat, not {"query": <model>}


@pytest.mark.characterization
async def test_bad_arity_falls_back_to_kwargs_and_raises_the_real_error() -> None:
    # CHARACTERIZATION: a wrong-arity dependency call cannot be bound, so the proxy
    # records kwargs only (positionals dropped) and lets the method's own TypeError
    # propagate.
    seen: dict[str, dict[str, Any]] = {}

    def enrich(span: Any, ctx: InvocationContext, outcome: Any, exc: Any) -> None:
        seen[ctx.operation] = dict(ctx.arguments or {})

    class Dep(AComponent[EmptySettings, None, str]):
        name = "dep"

        @invocable
        async def fetch(self, x: int) -> str:
            return str(x)

    class Up(AComponent[EmptySettings, None, str]):
        name = "up"
        dependencies = ("dep",)

        @invocable
        async def run(self) -> str:
            return await self.dependency("dep").fetch(1, 2, 3)  # too many positionals

    container = Container.build(registry_of(Dep, Up), {"dep": {}, "up": {}}, span_enricher=enrich)
    await container.start()
    with pytest.raises(TypeError):
        await container.invoke("up", "run")
    await container.stop()
    assert seen["dep.fetch"] == {}  # signature.bind failed -> kwargs-only (empty), positionals dropped


@pytest.mark.characterization
async def test_expired_deadline_is_inherited_but_not_enforced() -> None:
    # CHARACTERIZATION: a dependency call inherits the caller's deadline into its
    # ctx, but the raw (link-less) dependency path enforces nothing - the call runs
    # to completion even when the budget is already spent.
    seen: dict[str, float | None] = {}

    def enrich(span: Any, ctx: InvocationContext, outcome: Any, exc: Any) -> None:
        seen[ctx.operation] = ctx.deadline

    class Slow(AComponent[EmptySettings, None, str]):
        name = "slow"

        @invocable
        async def go(self) -> str:
            await anyio.sleep(0.05)  # outlives the tiny budget below
            return "done"

    class Caller(AComponent[EmptySettings, None, str]):
        name = "caller"
        dependencies = ("slow",)

        @invocable
        async def run(self) -> str:
            return await self.dependency("slow").go()

    container = Container.build(registry_of(Slow, Caller), {"slow": {}, "caller": {}}, span_enricher=enrich)
    await container.start()
    outcome = await container.invoke("caller", "run", budget=0.001)
    await container.stop()
    assert seen["slow.go"] is not None  # deadline inherited
    assert outcome.value == "done"  # yet the dependency call was not cut off


async def test_correlation_id_propagates_invoke_to_nested_deps() -> None:
    tracer_provider, exporter = tracing()

    class Leaf(AComponent[EmptySettings, None, str]):
        name = "leaf"

        @invocable
        async def tip(self) -> str:
            return "leaf"

    class Mid(AComponent[EmptySettings, None, str]):
        name = "mid"
        dependencies = ("leaf",)

        @invocable
        async def hop(self) -> str:
            return await self.dependency("leaf").tip()

    class Root(AComponent[EmptySettings, None, str]):
        name = "root"
        dependencies = ("mid",)

        @invocable
        async def run(self) -> str:
            return await self.dependency("mid").hop()

    container = Container.build(
        registry_of(Leaf, Mid, Root), {"leaf": {}, "mid": {}, "root": {}}, tracer_provider=tracer_provider
    )
    await container.start()
    await container.invoke("root", "run", correlation_id="corr-9")
    await container.stop()

    spans = exporter.get_finished_spans()
    ids = {s.name: dict(s.attributes or {})[conv.ATTR_CORRELATION_ID] for s in spans}
    assert ids["root.run"] == ids["mid.hop"] == ids["leaf.tip"] == "corr-9"  # one id through the whole tree
