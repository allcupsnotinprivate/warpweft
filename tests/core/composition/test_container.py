"""Container: build validation, lifecycle, invocation, endpoint sharing, scope."""

from contextvars import ContextVar
from typing import Any

from _support.containers import registry_of
import anyio
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel
import pytest

from warpweft.core.axes import Axis, AxisRegistry, ScopeSpec
from warpweft.core.component import AComponent, Criticality, EmptySettings, Lifetime, invocable
from warpweft.core.composition import Container, Registry
from warpweft.core.context import InvocationContext
from warpweft.core.errors import CircuitOpen, ComponentUnavailable, ConfigurationError, StartupError, TransientError

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

EVENTS: list[str] = []
CALLS: dict[str, int] = {}
SCOPED: list[str] = []
_tenant: ContextVar[str | None] = ContextVar("test_tenant", default=None)


# --- test components ---------------------------------------------------------


class DB(AComponent[EmptySettings, None, None]):
    name = "db"

    async def start(self) -> None:
        EVENTS.append("db.start")

    async def stop(self) -> None:
        EVENTS.append("db.stop")

    @invocable
    async def ping(self) -> str:
        return "pong"


class API(AComponent[EmptySettings, None, None]):
    name = "api"
    dependencies = ("db",)

    async def start(self) -> None:
        EVENTS.append("api.start")

    async def stop(self) -> None:
        EVENTS.append("api.stop")

    @invocable
    async def call_db(self) -> str:
        db: DB = self.dependency("db")  # type: ignore[assignment]
        return await db.ping()


class EchoSettings(BaseModel):
    prefix: str


class Echo(AComponent[EchoSettings, str, str]):
    name = "echo"

    @invocable
    async def echo(self, text: str) -> str:
        return f"{self.settings.prefix}{text}"


class Flaky(AComponent[EmptySettings, None, str]):
    name = "flaky"

    def __init__(self, settings: EmptySettings) -> None:
        super().__init__(settings)
        self.calls = 0

    @invocable
    async def fetch(self) -> str:
        self.calls += 1
        if self.calls < 3:
            raise TransientError(f"transient #{self.calls}")
        return "ok"


class CtxWanter(AComponent[EmptySettings, None, str]):
    name = "ctxw"

    @invocable
    async def op(self, ctx: InvocationContext) -> str:
        return ctx.operation


class BoomBase(AComponent[EmptySettings, None, None]):
    host = "?"

    def endpoint(self) -> str | None:
        return self.host

    @invocable
    async def go(self) -> bool:
        CALLS[self.name] = CALLS.get(self.name, 0) + 1
        raise TransientError("down")


class BoomA(BoomBase):
    name = "a"
    host = "host1"


class BoomB(BoomBase):
    name = "b"
    host = "host1"


class BoomC(BoomBase):
    name = "c"
    host = "host2"


class TenantThing(AComponent[EmptySettings, None, str]):
    name = "tenant-thing"
    lifetime = Lifetime.SCOPED
    scope = ScopeSpec(("tenant",))

    def __init__(self, settings: EmptySettings) -> None:
        super().__init__(settings)
        self.tenant = _tenant.get()

    async def start(self) -> None:
        SCOPED.append(f"start:{self.tenant}")

    async def stop(self) -> None:
        SCOPED.append(f"stop:{self.tenant}")

    @invocable
    async def whoami(self) -> str:
        return self.tenant or "?"


def fresh_registry() -> Registry:
    return registry_of(DB, API, Echo, Flaky, CtxWanter, BoomA, BoomB, BoomC, TenantThing)


def setup_function() -> None:
    EVENTS.clear()
    CALLS.clear()
    SCOPED.clear()


# --- build validation --------------------------------------------------------


async def test_build_rejects_unknown_component() -> None:
    with pytest.raises(ConfigurationError, match="not registered"):
        Container.build(fresh_registry(), {"ghost": {}})


async def test_build_validates_config_against_the_model() -> None:
    with pytest.raises(ConfigurationError, match="component 'echo' at 'prefix'"):
        Container.build(fresh_registry(), {"echo": {}})  # missing required 'prefix'


async def test_build_validates_the_dependency_graph() -> None:
    with pytest.raises(ConfigurationError, match="depends on 'db'"):
        Container.build(fresh_registry(), {"api": {}})  # db not configured


# --- lifecycle ---------------------------------------------------------------


async def test_start_orders_dependencies_and_injects_them() -> None:
    container = Container.build(fresh_registry(), {"db": {}, "api": {}})
    await container.start()
    assert EVENTS == ["db.start", "api.start"]
    outcome = await container.invoke("api", "call_db")
    assert outcome.value == "pong"  # api reached its injected db dependency
    await container.stop()
    assert EVENTS[-2:] == ["api.stop", "db.stop"]  # reverse order


async def test_required_failure_aborts_start() -> None:
    class BadRequired(AComponent[EmptySettings, None, None]):
        name = "bad-required"

        async def start(self) -> None:
            raise RuntimeError("boom")

        @invocable
        async def go(self) -> bool:
            return True

    reg = Registry()
    reg.register(BadRequired)
    container = Container.build(reg, {"bad-required": {}})
    with pytest.raises(StartupError, match="required component 'bad-required'"):
        await container.start()
    assert not container.started


async def test_optional_failure_degrades_but_starts() -> None:
    class BadOptional(AComponent[EmptySettings, None, None]):
        name = "bad-optional"
        criticality = Criticality.OPTIONAL

        async def start(self) -> None:
            raise RuntimeError("nope")

        @invocable
        async def go(self) -> bool:
            return True

    reg = Registry()
    reg.register(BadOptional)
    container = Container.build(reg, {"bad-optional": {}})
    await container.start()
    assert container.started
    assert container.is_degraded("bad-optional")
    with pytest.raises(ComponentUnavailable, match="degraded"):
        await container.invoke("bad-optional", "go")
    await container.stop()


async def test_init_timeout_fails_a_hanging_start() -> None:
    class Hanger(AComponent[EmptySettings, None, None]):
        name = "hanger"

        async def start(self) -> None:
            await anyio.Event().wait()  # never completes

        @invocable
        async def go(self) -> bool:
            return True

    reg = Registry()
    reg.register(Hanger)
    container = Container.build(reg, {"hanger": {}}, init_timeout=0.02)
    with pytest.raises(StartupError):
        await container.start()


# --- invocation --------------------------------------------------------------


async def test_invoke_runs_the_method_and_wraps_the_result() -> None:
    container = Container.build(fresh_registry(), {"echo": {"prefix": ">>"}})
    await container.start()
    outcome = await container.invoke("echo", "echo", text="hi")
    assert outcome.value == ">>hi"
    assert outcome.source == "live"
    await container.stop()


async def test_span_enricher_is_forwarded_through_build() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    seen: list[tuple[Any, Any]] = []

    def enrich(span: Any, ctx: InvocationContext, outcome: Any, exc: Any) -> None:
        seen.append((dict(ctx.arguments or {}), None if outcome is None else outcome.value))
        span.set_attribute("app.echoed", outcome.value)

    container = Container.build(
        fresh_registry(),
        {"echo": {"prefix": ">>"}},
        tracer_provider=provider,
        span_enricher=enrich,
    )
    await container.start()
    await container.invoke("echo", "echo", text="hi")
    await container.stop()

    assert seen == [({"text": "hi"}, ">>hi")]
    (span,) = exporter.get_finished_spans()
    assert dict(span.attributes or {})["app.echoed"] == ">>hi"


async def test_invoke_applies_retry_from_config() -> None:
    config = {"flaky": {"policy": {"retry": {"attempts": 3, "base_delay": 0.0, "max_delay": 1.0}}}}
    container = Container.build(fresh_registry(), config)
    await container.start()
    outcome = await container.invoke("flaky", "fetch")
    assert outcome.value == "ok"
    assert outcome.attempts == 3
    await container.stop()


async def test_invoke_injects_context_when_the_method_asks() -> None:
    container = Container.build(fresh_registry(), {"ctxw": {}})
    await container.start()
    outcome = await container.invoke("ctxw", "op")
    assert outcome.value == "ctxw.op"
    await container.stop()


async def test_invoke_unknown_method_and_before_start() -> None:
    container = Container.build(fresh_registry(), {"echo": {"prefix": ""}})
    with pytest.raises(ConfigurationError, match="not started"):
        await container.invoke("echo", "echo", text="x")
    await container.start()
    with pytest.raises(ConfigurationError, match="no invocable"):
        await container.invoke("echo", "missing")
    await container.stop()


# --- endpoint slicing (axes in action) --------------------------------------


async def test_endpoint_state_is_shared_by_endpoint_across_components() -> None:
    breaker = {"window": 1, "failure_threshold": 1, "reset_timeout": 100.0}
    config = {
        "a": {"policy": {"circuit_breaker": breaker}},
        "b": {"policy": {"circuit_breaker": breaker}},
        "c": {"policy": {"circuit_breaker": breaker}},
    }
    container = Container.build(fresh_registry(), config)
    await container.start()

    # 'a' trips the breaker for host1.
    with pytest.raises(TransientError):
        await container.invoke("a", "go")
    # 'b' shares host1: rejected without ever running its method.
    with pytest.raises(CircuitOpen):
        await container.invoke("b", "go")
    # 'c' is on host2: its own breaker is still closed.
    with pytest.raises(TransientError):
        await container.invoke("c", "go")

    assert CALLS == {"a": 1, "c": 1}  # 'b' never executed
    await container.stop()


# --- scoped components (lazy per-key, LRU) -----------------------------------


def scoped_container(**opts: Any) -> Container:
    reg = Registry()
    reg.register(TenantThing)
    axes = AxisRegistry()
    axes.register(Axis(name="tenant", resolver=_tenant.get))
    return Container.build(reg, {"tenant-thing": {}}, axes=axes, **opts)


async def test_scoped_instances_are_created_lazily_per_key() -> None:
    container = scoped_container()
    await container.start()
    assert SCOPED == []  # nothing created at start

    _tenant.set("acme")
    assert (await container.invoke("tenant-thing", "whoami")).value == "acme"
    _tenant.set("globex")
    assert (await container.invoke("tenant-thing", "whoami")).value == "globex"
    _tenant.set("acme")
    assert (await container.invoke("tenant-thing", "whoami")).value == "acme"  # reused

    assert SCOPED == ["start:acme", "start:globex"]  # acme created once
    await container.stop()


async def test_scoped_lru_evicts_and_stops_the_oldest() -> None:
    container = scoped_container(scoped_max_entries=1)
    await container.start()

    _tenant.set("acme")
    await container.invoke("tenant-thing", "whoami")
    _tenant.set("globex")
    await container.invoke("tenant-thing", "whoami")  # evicts acme

    assert "stop:acme" in SCOPED
    await container.stop()


# --- isolation & telemetry ---------------------------------------------------


async def test_two_containers_are_independent() -> None:
    reg = fresh_registry()
    one = Container.build(reg, {"echo": {"prefix": "1:"}})
    two = Container.build(reg, {"echo": {"prefix": "2:"}})
    await one.start()
    await two.start()
    assert (await one.invoke("echo", "echo", text="x")).value == "1:x"
    assert (await two.invoke("echo", "echo", text="x")).value == "2:x"
    await one.stop()
    assert two.started  # stopping one leaves the other running
    assert (await two.invoke("echo", "echo", text="y")).value == "2:y"
    await two.stop()


async def test_invocation_is_instrumented() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    container = Container.build(fresh_registry(), {"echo": {"prefix": ""}}, tracer_provider=provider)
    await container.start()
    await container.invoke("echo", "echo", text="hi")
    await container.stop()

    names = [s.name for s in exporter.get_finished_spans()]
    assert "echo.echo" in names
