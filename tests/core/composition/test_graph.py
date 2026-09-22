"""Dependency graph: validation and lifecycle ordering."""

import pytest

from warpweft.core.component import Lifetime
from warpweft.core.composition.graph import DependencyGraph, GraphNode
from warpweft.core.errors import ConfigurationError

pytestmark = pytest.mark.unit


def node(name: str, *deps: str, lifetime: Lifetime = Lifetime.PROCESS) -> GraphNode:
    return GraphNode(name=name, dependencies=deps, lifetime=lifetime)


def graph(*nodes: GraphNode) -> DependencyGraph:
    return DependencyGraph({n.name: n for n in nodes})


def test_missing_dependency_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="depends on 'db', which is not configured"):
        graph(node("api", "db"))


def test_cycle_is_reported_with_the_path() -> None:
    with pytest.raises(ConfigurationError, match="dependency cycle:"):
        graph(node("a", "b"), node("b", "c"), node("c", "a"))


def test_self_dependency_is_a_cycle() -> None:
    with pytest.raises(ConfigurationError, match="dependency cycle:"):
        graph(node("a", "a"))


def test_process_cannot_depend_on_scoped() -> None:
    with pytest.raises(ConfigurationError, match="cannot depend on scoped component 'tenant'"):
        graph(
            node("api", "tenant", lifetime=Lifetime.PROCESS),
            node("tenant", lifetime=Lifetime.SCOPED),
        )


def test_scoped_may_depend_on_process() -> None:
    g = graph(
        node("tenant", "db", lifetime=Lifetime.SCOPED),
        node("db", lifetime=Lifetime.PROCESS),
    )
    assert g.startup_order() == ("db", "tenant")


def test_startup_layers_group_independent_components() -> None:
    g = graph(node("a"), node("b"), node("c", "a", "b"))
    layers = g.startup_layers()
    assert layers[0] == ("a", "b")  # independent, sorted
    assert layers[1] == ("c",)


def test_startup_order_places_dependencies_first() -> None:
    g = graph(node("api", "cache", "db"), node("cache", "db"), node("db"))
    order = g.startup_order()
    assert order.index("db") < order.index("cache") < order.index("api")


def test_shutdown_is_the_reverse_of_startup() -> None:
    g = graph(node("api", "db"), node("db"))
    assert g.shutdown_order() == tuple(reversed(g.startup_order()))
