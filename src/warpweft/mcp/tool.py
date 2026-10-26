"""``@tool``: expose an invocable to LLMs as an MCP tool.

Stacks on top of ``@invocable`` and carries only MCP-facing metadata, so the
core never learns about tools. A method is exposed only when it is marked -
everything else (health checks, internal helpers) stays private by default.

    class Weather(AComponent[WeatherSettings, str, Forecast]):
        @tool(description="Get the forecast for a city.", read_only=True)
        @invocable
        async def forecast(self, city: str) -> Forecast: ...
"""

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, overload

F = TypeVar("F", bound=Callable[..., Any])

#: Attribute the decorator stamps on the function object.
_MARK = "__warpweft_tool__"

#: URI schemes an icon ``src`` may use. The MCP spec has clients reject every
#: other scheme (``http:`` included), so an icon is checked when it is defined.
_ICON_SCHEMES = frozenset({"https", "data"})
_ICON_THEMES = frozenset({"light", "dark"})


@dataclass(frozen=True)
class Icon:
    """An icon for a server or a tool - MCP's ``icons`` metadata.

    ``src`` is an ``https:`` URL or a ``data:`` URI carrying the image inline;
    ``mime_type`` names the image type when the source does not; ``sizes``
    lists the sizes the icon renders well at (``"48x48"``, or ``"any"`` for a
    scalable image) and ``theme`` the background it was drawn for. Clients pick
    the icon that fits their UI. A ``data:`` icon travels with every message
    that carries it, so prefer URLs for server icons (see ``ServerOptions``).
    """

    src: str
    mime_type: str | None = None
    sizes: Sequence[str] = ()
    theme: Literal["light", "dark"] | None = None

    def __post_init__(self) -> None:
        scheme, sep, _ = self.src.partition(":")
        if not sep or scheme.lower() not in _ICON_SCHEMES:
            raise ValueError(f"icon src must be an https: URL or a data: URI, got {self.src!r}")
        if self.theme is not None and self.theme not in _ICON_THEMES:
            raise ValueError(f"icon theme must be 'light' or 'dark', got {self.theme!r}")
        object.__setattr__(self, "sizes", tuple(self.sizes))  # hashable, like the rest of the metadata


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
    tools and may ``task_cancel`` it. ``icons`` are shown by clients next to
    the tool.
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
    icons: tuple[Icon, ...] = ()


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
    icons: Sequence[Icon] | None = ...,
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
    icons: Sequence[Icon] | None = None,
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
        icons=tuple(icons or ()),
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
