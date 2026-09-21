"""Warpweft MCP: expose component invocables as Model Context Protocol tools.

Importing this package (or ``warpweft.mcp.tool``) does **not** require the
``mcp`` extra: the ``@tool`` marker and its metadata are pure Python. Only the
server symbols (`build_server`, `run_stdio`, `collect_tools`, `ToolBinding`)
pull in the mcp SDK, and they are imported lazily on first access - so an action
can carry tool metadata whether or not the SDK is installed.
"""

from typing import TYPE_CHECKING, Any

from .tool import ToolMeta, is_tool, tool

if TYPE_CHECKING:
    from .server import ToolBinding, build_server, collect_tools, run_stdio
    from .tasks import InMemoryTaskStore, TaskRecord, TaskRunner, TaskStore, task_runner

__all__ = [
    "InMemoryTaskStore",
    "TaskRecord",
    "TaskRunner",
    "TaskStore",
    "ToolBinding",
    "ToolMeta",
    "build_server",
    "collect_tools",
    "is_tool",
    "run_stdio",
    "task_runner",
    "tool",
]

#: Names served lazily from ``.server`` (which imports the mcp SDK).
_LAZY_SERVER = frozenset({"ToolBinding", "build_server", "collect_tools", "run_stdio"})
#: Names served lazily from ``.tasks`` (which imports the mcp SDK).
_LAZY_TASKS = frozenset({"InMemoryTaskStore", "TaskRecord", "TaskRunner", "TaskStore", "task_runner"})


def __getattr__(name: str) -> Any:
    if name in _LAZY_SERVER:
        from . import server

        return getattr(server, name)
    if name in _LAZY_TASKS:
        from . import tasks

        return getattr(tasks, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
