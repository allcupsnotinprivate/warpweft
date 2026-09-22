"""Contextvar-backed axes for tenancy tests.

``context_axes`` builds an ``AxisRegistry`` with one contextvar-backed axis per
name and returns ``AxisHandle``s so a test binds a slice with a context manager
(``handles["tenant"].use("acme")``) instead of raw ``ContextVar.set`` calls that
leak across tests. Mirrors what ``App.axis`` does, without needing a full App.
"""

from collections.abc import Mapping
from contextvars import ContextVar

from warpweft.core.axes import Axis, AxisRegistry
from warpweft.runtime.tenancy import AxisHandle


def context_axes(
    *names: str,
    defaults: Mapping[str, str] | None = None,
    max_cardinality: int = 1000,
) -> tuple[AxisRegistry, dict[str, AxisHandle]]:
    """An ``AxisRegistry`` plus a handle per axis.

    An axis listed in ``defaults`` is optional (``on_missing="default"``); any
    other is required. ``max_cardinality`` applies to every axis built here.
    """
    defaults = defaults or {}
    registry = AxisRegistry()
    handles: dict[str, AxisHandle] = {}
    for name in names:
        default = defaults.get(name)
        var: ContextVar[str | None] = ContextVar(f"test_axis_{name}", default=default)
        axis = (
            Axis(name=name, resolver=var.get, on_missing="default", default=default, max_cardinality=max_cardinality)
            if default is not None
            else Axis(name=name, resolver=var.get, on_missing="required", max_cardinality=max_cardinality)
        )
        registry.register(axis)
        handles[name] = AxisHandle(name, var)
    return registry, handles
