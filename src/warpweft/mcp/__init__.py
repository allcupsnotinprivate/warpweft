"""Warpweft MCP: expose component invocables as Model Context Protocol tools.

Importing this package (or ``warpweft.mcp.tool``) does **not** require the
``mcp`` extra: the ``@tool`` marker and its metadata are pure Python. Only the
server, task and HTTP symbols (`build_server`, `run_stdio`, `asgi_app`, ...)
pull in the mcp SDK, and they are imported lazily on first access - so an action
can carry tool metadata whether or not the SDK is installed.
"""

from typing import TYPE_CHECKING, Any

from .tool import ToolMeta, is_tool, tool

if TYPE_CHECKING:
    from .http import MCPHttpApp, asgi_app, header_binder, run_http
    from .server import AxisBinder, ServerOptions, ToolBinding, build_server, collect_tools, open_server, run_stdio
    from .tasks import InMemoryTaskStore, TaskRecord, TaskRunner, TaskStore, task_runner

__all__ = [
    "AxisBinder",
    "InMemoryTaskStore",
    "MCPHttpApp",
    "ServerOptions",
    "TaskRecord",
    "TaskRunner",
    "TaskStore",
    "ToolBinding",
    "ToolMeta",
    "asgi_app",
    "build_server",
    "collect_tools",
    "header_binder",
    "is_tool",
    "open_server",
    "run_http",
    "run_stdio",
    "task_runner",
    "tool",
]

#: Names served lazily from ``.server`` (which imports the mcp SDK).
_LAZY_SERVER = frozenset(
    {"AxisBinder", "ServerOptions", "ToolBinding", "build_server", "collect_tools", "open_server", "run_stdio"}
)
#: Names served lazily from ``.http`` (which imports the mcp SDK and starlette).
_LAZY_HTTP = frozenset({"MCPHttpApp", "asgi_app", "header_binder", "run_http"})
#: Names served lazily from ``.tasks`` (which imports the mcp SDK).
_LAZY_TASKS = frozenset({"InMemoryTaskStore", "TaskRecord", "TaskRunner", "TaskStore", "task_runner"})


def __getattr__(name: str) -> Any:
    if name in _LAZY_SERVER:
        from . import server

        return getattr(server, name)
    if name in _LAZY_HTTP:
        from . import http

        return getattr(http, name)
    if name in _LAZY_TASKS:
        from . import tasks

        return getattr(tasks, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
