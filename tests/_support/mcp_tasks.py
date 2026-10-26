"""A minimal client for the MCP Tasks extension, for tests.

The mcp SDK ships no client for ``io.modelcontextprotocol/tasks``. `TasksClient`
is one built on the SDK's client-extension hooks: it declares the extension,
claims the ``resultType: "task"`` answer to ``tools/call`` and resolves it by
polling ``tasks/get`` - what a real host does.
"""

from typing import Any, ClassVar, Literal

import anyio
from mcp.client.extension import ClaimContext, ClientExtension, ResultClaim
import mcp.types as mt

from warpweft.mcp.tasks_extension import TASKS_EXTENSION, TaskIdParams, UpdateTaskParams

MODERN = "2026-07-28"


class TaskFields(mt.Result):
    """The fields every task shape carries."""

    task_id: str
    status: str
    created_at: str
    last_updated_at: str
    ttl_ms: int | None
    poll_interval_ms: int | None = None
    status_message: str | None = None


class TaskHandle(TaskFields):
    """The ``resultType: "task"`` answer to a task-augmented ``tools/call``."""

    result_type: Literal["task"] = "task"


class DetailedTask(TaskFields):
    """A ``tasks/get`` answer."""

    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class GetTask(mt.Request[TaskIdParams, Literal["tasks/get"]]):
    method: Literal["tasks/get"] = "tasks/get"
    params: TaskIdParams
    name_param: ClassVar[str | None] = "taskId"  # the spec's Mcp-Name routing header


class CancelTask(mt.Request[TaskIdParams, Literal["tasks/cancel"]]):
    method: Literal["tasks/cancel"] = "tasks/cancel"
    params: TaskIdParams
    name_param: ClassVar[str | None] = "taskId"


class TaskFailed(Exception):
    def __init__(self, task: DetailedTask) -> None:
        super().__init__(f"task {task.task_id} ended {task.status}")
        self.task = task


class TasksClient(ClientExtension):
    """Declares the extension and resolves task handles by polling ``tasks/get``."""

    identifier = TASKS_EXTENSION

    def __init__(self) -> None:
        self.handles: list[TaskHandle] = []
        self.started = anyio.Event()

    def claims(self) -> list[ResultClaim[Any]]:
        return [
            ResultClaim(
                result_type="task", model=TaskHandle, resolve=self._resolve, protocol_versions=frozenset({MODERN})
            )
        ]

    async def _resolve(self, handle: TaskHandle, ctx: ClaimContext) -> mt.CallToolResult:
        self.handles.append(handle)
        self.started.set()
        with anyio.fail_after(5):
            while True:
                task = await get_task(ctx.session, handle.task_id)
                if task.status == "completed":
                    return mt.CallToolResult.model_validate(task.result)
                if task.status in ("failed", "cancelled"):
                    raise TaskFailed(task)
                await anyio.sleep(0.01)


async def get_task(session: Any, task_id: str) -> DetailedTask:
    return await session.send_request(GetTask(params=TaskIdParams(task_id=task_id)), DetailedTask)


class UpdateTask(mt.Request[UpdateTaskParams, Literal["tasks/update"]]):
    method: Literal["tasks/update"] = "tasks/update"
    params: UpdateTaskParams
    name_param: ClassVar[str | None] = "taskId"
