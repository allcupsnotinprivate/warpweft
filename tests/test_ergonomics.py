"""Ergonomics: annotation-declared dependencies, container.get and proxy."""

from typing import ClassVar

import pytest

from warpweft.core.component import (
    AComponent,
    EmptySettings,
    component_dependencies,
    dependency_annotations,
    describe,
    invocable,
)
from warpweft.core.composition import Container, Registry
from warpweft.core.errors import ComponentUnavailable, ConfigurationError, TransientError

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


class Embedder(AComponent[EmptySettings, str, list[float]]):
    @invocable
    async def embed(self, text: str) -> list[float]:
        return [float(len(text))]


class Search(AComponent[EmptySettings, str, str]):
    embedder: Embedder  # annotation-declared dependency
    label: ClassVar[str] = "x"  # ClassVar: not a dependency
    note: str = "y"  # plain non-component annotation: not a dependency

    @invocable
    async def query(self, text: str) -> str:
        vector = await self.embedder.embed(text)  # typed access
        return f"{text}:{vector[0]}"


def fresh_registry() -> Registry:
    reg = Registry()
    reg.register(Embedder)
    reg.register(Search)
    return reg


# --- annotation-declared dependencies ----------------------------------------


def test_annotation_dependencies_are_detected() -> None:
    assert dependency_annotations(Search) == {"embedder": Embedder}
    assert component_dependencies(Search) == ("embedder",)
    assert describe(Search).dependencies == ("embedder",)


def test_explicit_and_annotated_dependencies_merge() -> None:
    class Combo(AComponent[EmptySettings, None, None]):
        dependencies = ("embedder",)  # explicit, same as the annotation below
        embedder: Embedder
        search: Search  # annotated only

        @invocable
        async def go(self) -> None: ...

    assert component_dependencies(Combo) == ("embedder", "search")


async def test_container_binds_annotated_dependency_as_attribute() -> None:
    container = Container.build(fresh_registry(), {"embedder": {}, "search": {}})
    await container.start()
    outcome = await container.invoke("search", "query", text="hi")
    assert outcome.value == "hi:2.0"  # reached the injected embedder through self.embedder
    await container.stop()


async def test_graph_validation_sees_annotated_dependencies() -> None:
    reg = Registry()
    reg.register(Search)  # embedder NOT configured
    with pytest.raises(ConfigurationError, match="depends on 'embedder'"):
        Container.build(reg, {"search": {}})


# --- container.get -----------------------------------------------------------


async def test_get_returns_the_running_instance() -> None:
    container = Container.build(fresh_registry(), {"embedder": {}, "search": {}})
    await container.start()
    search = await container.get(Search)
    assert isinstance(search, Search)
    # The injected dependency is the live instance behind a telemetry proxy:
    # isinstance still holds, and the proxy wraps the very same instance.
    embedder = await container.get(Embedder)
    assert isinstance(search.embedder, Embedder)
    assert search.embedder._ww_instance is embedder  # type: ignore[attr-defined]
    assert await container.get("search") is search  # by-name form
    await container.stop()


async def test_get_before_start_and_unknown_are_rejected() -> None:
    container = Container.build(fresh_registry(), {"embedder": {}, "search": {}})
    with pytest.raises(ConfigurationError, match="not started"):
        await container.get(Search)
    await container.start()
    with pytest.raises(ConfigurationError, match="not configured"):
        await container.get("ghost")
    await container.stop()


async def test_get_degraded_optional_raises_unavailable() -> None:
    from warpweft.core.component import Criticality

    class Fragile(AComponent[EmptySettings, None, None]):
        criticality = Criticality.OPTIONAL

        async def start(self) -> None:
            raise RuntimeError("nope")

        @invocable
        async def go(self) -> None: ...

    reg = Registry()
    reg.register(Fragile)
    container = Container.build(reg, {"fragile": {}})
    await container.start()
    with pytest.raises(ComponentUnavailable):
        await container.get(Fragile)
    await container.stop()


# --- container.proxy ---------------------------------------------------------


class FlakyOnce(AComponent[EmptySettings, None, str]):
    def __init__(self, settings: EmptySettings) -> None:
        super().__init__(settings)
        self.calls = 0

    @invocable
    async def fetch(self) -> str:
        self.calls += 1
        if self.calls < 3:
            raise TransientError("warming up")
        return "payload"


async def test_proxy_routes_through_the_chain() -> None:
    reg = Registry()
    reg.register(FlakyOnce)
    config = {"flaky_once": {"policy": {"retry": {"attempts": 3, "base_delay": 0.0, "max_delay": 1.0}}}}
    container = Container.build(reg, config)
    await container.start()

    flaky = container.proxy(FlakyOnce)
    value = await flaky.fetch()  # typed call; retry recovered underneath
    assert value == "payload"
    assert (await container.get(FlakyOnce)).calls == 3  # the chain really retried
    await container.stop()


async def test_get_resolves_a_scoped_instance_for_the_current_axis() -> None:
    from contextvars import ContextVar

    from warpweft.core.axes import Axis, AxisRegistry, ScopeSpec
    from warpweft.core.component import Lifetime

    tenant: ContextVar[str | None] = ContextVar("ergo_tenant", default=None)

    class PerTenant(AComponent[EmptySettings, None, str]):
        lifetime = Lifetime.SCOPED
        scope = ScopeSpec(("tenant",))

        @invocable
        async def go(self) -> str:
            return "ok"

    reg = Registry()
    reg.register(PerTenant)
    axes = AxisRegistry()
    axes.register(Axis(name="tenant", resolver=tenant.get))
    container = Container.build(reg, {"per_tenant": {}}, axes=axes)
    await container.start()

    tenant.set("acme")
    acme = await container.get(PerTenant)
    tenant.set("globex")
    globex = await container.get(PerTenant)
    assert acme is not globex  # one live instance per tenant slice
    tenant.set("acme")
    assert await container.get(PerTenant) is acme  # reused
    await container.stop()


async def test_proxy_rejects_non_invocables_and_unknown_components() -> None:
    container = Container.build(fresh_registry(), {"embedder": {}, "search": {}})
    await container.start()
    proxy = container.proxy(Search)
    assert "search" in repr(proxy)
    with pytest.raises(AttributeError, match="no invocable"):
        _ = proxy.start  # lifecycle methods are not exposed through the proxy
    with pytest.raises(ConfigurationError, match="not configured"):
        container.proxy("ghost")
    await container.stop()
