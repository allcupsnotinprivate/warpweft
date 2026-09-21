"""``@tool``: expose an invocable to LLMs as an MCP tool.

Stacks on top of ``@invocable`` and carries only MCP-facing metadata, so the
core never learns about tools. A method is exposed only when it is marked -
everything else (health checks, internal helpers) stays private by default.

    class Weather(AComponent[WeatherSettings, str, Forecast]):
        @tool(description="Get the forecast for a city.", read_only=True)
        @invocable
        async def forecast(self, city: str) -> Forecast: ...
"""

from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any, TypeVar, overload

F = TypeVar("F", bound=Callable[..., Any])

#: Attribute the decorator stamps on the function object.
_MARK = "__warpweft_tool__"


@dataclass(frozen=True)
class ToolMeta:
    """MCP-facing metadata attached to a tool method.

    ``name`` overrides the derived tool name; ``description`` overrides the
    method docstring. The hint flags map to the MCP tool annotations that help
    a model reason about a call's effects. ``tags`` are free-form labels used
    to select which tools a server exposes (see ``collect_tools``); they are
    never sent to the client. ``background=True`` exposes the tool as a
    non-blocking *submit* (it returns a ``task_id`` at once and runs in the
    background); the caller then polls the shared ``task_status`` / ``task_result``
    tools and may ``task_cancel`` it.
    """

    name: str | None = None
    title: str | None = None
    description: str | None = None
    read_only: bool | None = None
    destructive: bool | None = None
    idempotent: bool | None = None
    open_world: bool | None = None
    tags: frozenset[str] = frozenset()
    background: bool = False


@overload
def tool(fn: F) -> F: ...
@overload
def tool(
    *,
    name: str | None = ...,
    title: str | None = ...,
    description: str | None = ...,
    read_only: bool | None = ...,
    destructive: bool | None = ...,
    idempotent: bool | None = ...,
    open_world: bool | None = ...,
    tags: Collection[str] | None = ...,
    background: bool = ...,
) -> Callable[[F], F]: ...
def tool(
    fn: F | None = None,
    *,
    name: str | None = None,
    title: str | None = None,
    description: str | None = None,
    read_only: bool | None = None,
    destructive: bool | None = None,
    idempotent: bool | None = None,
    open_world: bool | None = None,
    tags: Collection[str] | None = None,
    background: bool = False,
) -> F | Callable[[F], F]:
    """Mark an invocable method as an MCP tool. Usable bare or with arguments."""
    meta = ToolMeta(
        name=name,
        title=title,
        description=description,
        read_only=read_only,
        destructive=destructive,
        idempotent=idempotent,
        open_world=open_world,
        tags=frozenset(tags or ()),
        background=background,
    )

    def stamp(func: F) -> F:
        setattr(func, _MARK, meta)
        return func

    return stamp if fn is None else stamp(fn)


def is_tool(obj: object) -> bool:
    return hasattr(obj, _MARK)


def tool_meta(obj: object) -> ToolMeta:
    """Return the ToolMeta stamped on ``obj`` (raises if it is not a tool)."""
    meta = getattr(obj, _MARK)
    assert isinstance(meta, ToolMeta)  # noqa: S101 - invariant guarded by is_tool
    return meta
