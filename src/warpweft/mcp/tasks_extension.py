"""The MCP Tasks extension (``io.modelcontextprotocol/tasks``) over warpweft's task runner.

On the 2026-07-28 protocol a server may answer ``tools/call`` with a task handle
instead of a result; the client then polls ``tasks/get`` until the task settles
and reads the result from it. This module projects warpweft's background tasks
(`TaskRunner` / `TaskStore`) onto that wire:

- `supports_native_tasks` - whether this request's client declared the
  extension, so a ``@tool(background=True)`` call may return a task handle.
- `create_task_result` - the ``resultType: "task"`` handle answering such a call.
- `detailed_task` - a stored task as ``tasks/get`` reports it.
- `install` - advertise the extension and serve ``tasks/get``,
  ``tasks/update`` and ``tasks/cancel``.

A client without the extension keeps the plain submit/poll tools
(``task_status`` / ``task_result`` / ``task_cancel``) - same runner, same store.

Status mapping. The runner marks a job whose result is a tool error (``isError``)
as ``failed``, which the plain tools report as such. The extension reserves
``failed`` for JSON-RPC errors, so such a task is reported ``completed`` with the
tool error as its result; ``failed`` (with an ``error`` object) is left for a job
that crashed without producing a result.

Not implemented (all optional in the spec): ``input_required`` tasks - a warpweft
job never asks for input mid-flight, so ``tasks/update`` accepts and ignores any
responses - and ``notifications/tasks`` push updates; clients poll.

Spec: https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks
"""

from typing import Any, Protocol

from mcp.server.lowlevel.server import Server
from mcp.shared.exceptions import MCPError
import mcp.types as mt
from mcp_types import INTERNAL_ERROR, INVALID_PARAMS, MISSING_REQUIRED_CLIENT_CAPABILITY
from mcp_types.version import MODERN_PROTOCOL_VERSIONS

from warpweft.core.errors import ConfigurationError

from .errors import error_metadata
from .tasks import TERMINAL, TaskRecord, TaskRunner, TaskScope

#: The extension identifier, advertised under ``capabilities.extensions``.
TASKS_EXTENSION = "io.modelcontextprotocol/tasks"

#: The polling interval suggested to clients in every task handle.
POLL_INTERVAL_MS = 1000

_TASKS_CAPABILITY = mt.ClientCapabilities(extensions={TASKS_EXTENSION: {}})


class TaskIdParams(mt.RequestParams):
    """Params of ``tasks/get`` and ``tasks/cancel``."""

    task_id: str


class UpdateTaskParams(TaskIdParams):
    """Params of ``tasks/update``: answers to a task's ``inputRequests``."""

    input_responses: dict[str, Any] | None = None


class TaskScopeProvider(Protocol):
    def scope(self, ctx: Any, params: mt.CallToolRequestParams) -> TaskScope: ...


def supports_native_tasks(ctx: Any) -> bool:
    """Whether the client of this request may be answered with a task handle.

    The extension lives on the 2026-07-28 protocol, where the client declares
    its capabilities per request; the spec forbids returning a task to a client
    that did not declare it.
    """
    return ctx.protocol_version in MODERN_PROTOCOL_VERSIONS and bool(
        ctx.session.check_client_capability(_TASKS_CAPABILITY)
    )


def create_task_result(record: TaskRecord) -> dict[str, Any]:
    """The ``resultType: "task"`` handle that answers a task-augmented ``tools/call``."""
    return {"resultType": "task", **_task_fields(record)}


def task_backend_error(exc: Exception, operation: str) -> MCPError:
    """Translate a store failure into a deliberate protocol-level error."""
    return MCPError(
        code=INTERNAL_ERROR,
        message=f"Task store {operation} failed: {exc}",
        data=error_metadata(exc),
    )


def install(server: Server[Any], runner: TaskRunner, *, scope_provider: TaskScopeProvider) -> None:
    """Advertise the extension on ``server`` and serve its methods from ``runner``."""

    async def get_task(ctx: Any, params: TaskIdParams) -> dict[str, Any]:
        return detailed_task(await _record(ctx, runner, params.task_id, "tasks/get", params, scope_provider))

    async def update_task(ctx: Any, params: UpdateTaskParams) -> dict[str, Any]:
        # A warpweft task never enters input_required, so every response is for
        # an unknown key - which the spec says to ignore.
        await _record(ctx, runner, params.task_id, "tasks/update", params, scope_provider)
        return {}

    async def cancel_task(ctx: Any, params: TaskIdParams) -> dict[str, Any]:
        await _record(ctx, runner, params.task_id, "tasks/cancel", params, scope_provider)
        await runner.cancel(params.task_id)  # cooperative; settles to cancelled as the job unwinds
        return {}

    server.extensions[TASKS_EXTENSION] = {}
    server.add_request_handler("tasks/get", TaskIdParams, get_task)
    server.add_request_handler("tasks/update", UpdateTaskParams, update_task)
    server.add_request_handler("tasks/cancel", TaskIdParams, cancel_task)


async def _record(
    ctx: Any,
    runner: TaskRunner,
    task_id: str,
    method: str,
    params: TaskIdParams,
    scope_provider: TaskScopeProvider,
) -> TaskRecord:
    """Load an owned task for a ``tasks/*`` request, or raise its protocol error."""
    if not supports_native_tasks(ctx):
        raise MCPError(
            code=MISSING_REQUIRED_CLIENT_CAPABILITY,
            message="Missing required client capability",
            data={"requiredCapabilities": {"extensions": {TASKS_EXTENSION: {}}}},
        )
    synthetic = mt.CallToolRequestParams(
        name=method,
        arguments=params.model_dump(mode="python", by_alias=False, exclude_none=True),
    )
    try:
        scope = scope_provider.scope(ctx, synthetic)
    except ConfigurationError as exc:
        raise _unknown_task(task_id) from exc
    except Exception as exc:
        raise task_backend_error(exc, "authorize") from exc
    try:
        record = await runner.store.get(task_id)
    except Exception as exc:
        raise task_backend_error(exc, "get") from exc
    if record is None or record.scope != scope:
        raise _unknown_task(task_id)
    return record


def _unknown_task(task_id: str) -> MCPError:
    return MCPError(code=INVALID_PARAMS, message=f"Failed to retrieve task: task '{task_id}' not found")


def _task_fields(record: TaskRecord) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "taskId": record.task_id,
        "status": record.status,
        "createdAt": record.created_at.isoformat(),
        "lastUpdatedAt": record.last_updated_at.isoformat(),
        "ttlMs": _ttl_from_creation(record),
        "pollIntervalMs": POLL_INTERVAL_MS,
    }
    if record.status_message is not None:
        fields["statusMessage"] = record.status_message
    return fields


def _ttl_from_creation(record: TaskRecord) -> int | None:
    """The spec's ``ttlMs``: required, counted from creation, ``null`` for unlimited.

    warpweft never expires a running task, so while it runs the honest answer
    is ``null``. Once settled it is retained ``ttl_ms`` past its final update,
    which from creation is the run time plus that retention.
    """
    if record.ttl_ms is None or record.status not in TERMINAL:
        return None
    ran_ms = int((record.last_updated_at - record.created_at).total_seconds() * 1000)
    return ran_ms + record.ttl_ms


def detailed_task(record: TaskRecord) -> dict[str, Any]:
    """A ``tasks/get`` answer: the task plus its status-specific field."""
    task = _task_fields(record)
    if record.result is not None and record.status in ("completed", "failed"):
        # A tool error is still a completed call: the error lives in the result.
        task["status"] = "completed"
        task["result"] = record.result.model_dump(by_alias=True, mode="json", exclude_none=True)
    elif record.status == "failed":
        task["error"] = {"code": INTERNAL_ERROR, "message": record.status_message or "task failed"}
    return task
