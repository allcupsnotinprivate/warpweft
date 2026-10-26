"""warpweft.mcp.tasks_extension: background tools as native MCP tasks (2026-07-28).

The client side is `_support.mcp_tasks.TasksClient`, a minimal tasks-extension
client (the mcp SDK ships none).
"""

from collections.abc import AsyncIterator
import contextlib
from datetime import UTC, datetime, timedelta
from typing import Any

from _support.mcp_tasks import MODERN, CancelTask, DetailedTask, TaskFailed, TasksClient, UpdateTask, get_task
import anyio
from mcp import Client
from mcp.client.extension import ClientExtension
from mcp.shared.exceptions import MCPError
import mcp.types as mt
from mcp_types import INTERNAL_ERROR, INVALID_PARAMS, MISSING_REQUIRED_CLIENT_CAPABILITY
from pydantic import BaseModel
import pytest

from warpweft.core.component import AComponent, EmptySettings, invocable
from warpweft.core.composition import Registry
from warpweft.core.errors import PermanentError
from warpweft.mcp import build_server, task_runner, tool
from warpweft.mcp.tasks import TaskRecord
from warpweft.mcp.tasks_extension import TASKS_EXTENSION, TaskIdParams, UpdateTaskParams, detailed_task
from warpweft.runtime import App

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


# --- app under test ------------------------------------------------------------------


def jobs_app(gate: anyio.Event | None = None) -> App:
    class Jobs(AComponent[EmptySettings, None, int]):
        name = "jobs"

        @tool(background=True, description="Double a number, slowly.")
        @invocable
        async def double(self, n: int) -> int:
            if gate is not None:
                await gate.wait()
            return n * 2

        @tool(background=True)
        @invocable
        async def broken(self) -> int:
            raise PermanentError("upstream rejected the job")

        @tool
        @invocable
        async def now(self) -> int:
            return 7

    reg = Registry()
    reg.register(Jobs)
    return App(registry=reg)


@contextlib.asynccontextmanager
async def connected(app: App, *, extensions: list[ClientExtension], mode: str = MODERN) -> AsyncIterator[Client]:
    async with app.run(), task_runner() as runner:
        server = build_server(app, runner=runner)
        async with Client(server, mode=mode, extensions=extensions) as client:
            yield client


# --- tools/call answered with a task -------------------------------------------------


async def test_background_tool_runs_as_a_native_task() -> None:
    tasks = TasksClient()
    async with connected(jobs_app(), extensions=[tasks]) as client:
        result = await client.call_tool("jobs__double", {"n": 21})
    assert not result.is_error
    assert result.structured_content == {"result": 42}  # the tool's own result, as if inline
    [handle] = tasks.handles
    assert handle.status == "working"
    assert handle.ttl_ms is None  # a running task never expires
    assert handle.poll_interval_ms is not None


async def test_a_tool_error_completes_the_task_with_an_error_result() -> None:
    # The extension reserves "failed" for JSON-RPC errors; a tool error is a
    # completed call whose result says isError.
    async with connected(jobs_app(), extensions=[TasksClient()]) as client:
        result = await client.call_tool("jobs__broken", {})
    assert result.is_error
    assert result.meta is not None
    assert result.meta["warpweft.error"] == "permanent"


async def test_inline_tools_stay_inline() -> None:
    tasks = TasksClient()
    async with connected(jobs_app(), extensions=[tasks]) as client:
        result = await client.call_tool("jobs__now", {})
    assert result.structured_content == {"result": 7}
    assert tasks.handles == []


@pytest.mark.parametrize("mode", [pytest.param(MODERN, id="per-request"), pytest.param("legacy", id="handshake")])
async def test_clients_without_the_extension_get_the_plain_submit(mode: str) -> None:
    extensions: list[ClientExtension] = [TasksClient()] if mode == "legacy" else []  # legacy: never offered
    async with connected(jobs_app(), extensions=extensions, mode=mode) as client:
        submitted = await client.call_tool("jobs__double", {"n": 1})
    assert submitted.structured_content is not None
    assert set(submitted.structured_content) == {"task_id"}


# --- discovery -------------------------------------------------------------------------


async def test_native_clients_see_background_tools_as_ordinary_tools() -> None:
    async with connected(jobs_app(), extensions=[TasksClient()]) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {"jobs__double", "jobs__broken", "jobs__now"}  # no task_status / task_result / task_cancel
    assert tools["jobs__double"].output_schema == tools["jobs__now"].output_schema  # the real {"result": int}
    assert "task_id" not in (tools["jobs__double"].description or "")


async def test_other_clients_see_the_submit_and_poll_tools() -> None:
    async with connected(jobs_app(), extensions=[]) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert {"task_status", "task_result", "task_cancel"} <= set(tools)
    assert tools["jobs__double"].output_schema is not None
    assert set(tools["jobs__double"].output_schema["properties"]) == {"task_id"}


async def test_the_extension_is_advertised_only_with_background_tools() -> None:
    # mode="auto" runs a real server/discover (a version pin synthesizes one).
    async with connected(jobs_app(), extensions=[TasksClient()], mode="auto") as client:
        assert TASKS_EXTENSION in (client.server_capabilities.extensions or {})

    class Plain(AComponent[EmptySettings, None, int]):
        name = "plain"

        @tool
        @invocable
        async def one(self) -> int:
            return 1

    reg = Registry()
    reg.register(Plain)
    async with connected(App(registry=reg), extensions=[TasksClient()], mode="auto") as client:
        assert TASKS_EXTENSION not in (client.server_capabilities.extensions or {})


# --- tasks/get, tasks/cancel -------------------------------------------------------------


async def test_cancel_settles_the_task_as_cancelled() -> None:
    tasks = TasksClient()
    gate = anyio.Event()  # never set: the job blocks until cancelled
    failure: list[TaskFailed] = []

    async def call(client: Client) -> None:
        with pytest.raises(TaskFailed) as raised:
            await client.call_tool("jobs__double", {"n": 1})
        failure.append(raised.value)

    async with connected(jobs_app(gate), extensions=[tasks]) as client, anyio.create_task_group() as tg:
        tg.start_soon(call, client)
        await tasks.started.wait()
        task_id = tasks.handles[0].task_id
        await client.session.send_request(CancelTask(params=TaskIdParams(task_id=task_id)), mt.EmptyResult)
    assert failure[0].task.status == "cancelled"


async def test_unknown_task_is_invalid_params() -> None:
    async with connected(jobs_app(), extensions=[TasksClient()]) as client:
        with pytest.raises(MCPError) as raised:
            await get_task(client.session, "no-such-task")
    assert raised.value.error.code == INVALID_PARAMS


async def test_tasks_methods_require_the_client_capability() -> None:
    async with connected(jobs_app(), extensions=[]) as client:
        submitted = await client.call_tool("jobs__double", {"n": 1})  # a plain submit: the task exists
        assert submitted.structured_content is not None
        with pytest.raises(MCPError) as raised:
            await get_task(client.session, submitted.structured_content["task_id"])
    assert raised.value.error.code == MISSING_REQUIRED_CLIENT_CAPABILITY


async def test_update_accepts_and_ignores_responses() -> None:
    # A warpweft task never asks for input, so every response is for an unknown
    # key - which the spec says to ignore.
    tasks = TasksClient()
    gate = anyio.Event()
    async with connected(jobs_app(gate), extensions=[tasks]) as client, anyio.create_task_group() as tg:
        tg.start_soon(client.call_tool, "jobs__double", {"n": 2})
        await tasks.started.wait()
        task_id = tasks.handles[0].task_id
        params = UpdateTaskParams(task_id=task_id, input_responses={"name": {"action": "accept", "content": {}}})
        await client.session.send_request(UpdateTask(params=params), mt.EmptyResult)
        assert (await get_task(client.session, task_id)).status == "working"
        gate.set()


def test_a_job_that_crashed_without_a_result_is_failed() -> None:
    now = datetime.now(UTC)
    record = TaskRecord(
        task_id="t",
        tool="jobs__double",
        status="failed",
        created_at=now,
        last_updated_at=now,
        ttl_ms=None,
        status_message="store went away",
    )
    task = detailed_task(record)
    assert task["status"] == "failed"
    assert task["error"] == {"code": INTERNAL_ERROR, "message": "store went away"}
    assert task["ttlMs"] is None  # required on the wire, null meaning unlimited


def test_a_settled_task_reports_its_ttl_from_creation() -> None:
    created = datetime.now(UTC)
    record = TaskRecord(
        task_id="t",
        tool="jobs__double",
        status="completed",
        created_at=created,
        last_updated_at=created + timedelta(seconds=10),  # ran for 10 s
        ttl_ms=1000,
        result=mt.CallToolResult(content=[]),
    )
    assert detailed_task(record)["ttlMs"] == 11_000  # run time + retention


# --- one store, two polling surfaces ---------------------------------------------------


class Report(BaseModel):
    month: str
    total: int


def reports_app() -> App:
    class Reports(AComponent[EmptySettings, None, Report]):
        name = "reports"

        @tool(background=True)
        @invocable
        async def monthly(self, month: str) -> Report:
            return Report(month=month, total=3)

    reg = Registry()
    reg.register(Reports)
    return App(registry=reg)


@contextlib.asynccontextmanager
async def two_clients(app: App) -> AsyncIterator[tuple[Client, TasksClient, Client]]:
    """A native-tasks client (and its extension) plus a plain client on one server: one runner, one store."""
    tasks = TasksClient()
    async with app.run(), task_runner() as runner:
        server = build_server(app, runner=runner)
        async with (
            Client(server, mode=MODERN, extensions=[tasks]) as native,
            Client(server, mode=MODERN) as plain,
        ):
            yield native, tasks, plain


async def settled(session: Any, task_id: str) -> DetailedTask:
    with anyio.fail_after(5):
        while True:
            task = await get_task(session, task_id)
            if task.status != "working":
                return task
            await anyio.sleep(0.01)


async def test_a_submitted_task_reads_inline_shaped_through_tasks_get() -> None:
    async with two_clients(reports_app()) as (native, _, plain):
        submitted = await plain.call_tool("reports__monthly", {"month": "may"})
        assert submitted.structured_content is not None
        task = await settled(native.session, submitted.structured_content["task_id"])
    assert task.result is not None
    assert task.result["structuredContent"] == {"month": "may", "total": 3}  # the tool's own schema


async def test_a_native_task_reads_wrapped_through_task_result() -> None:
    async with two_clients(reports_app()) as (native, tasks, plain):
        await native.call_tool("reports__monthly", {"month": "june"})  # resolves once the task completed
        result = await plain.call_tool("task_result", {"task_id": tasks.handles[0].task_id})
    assert result.structured_content == {"result": {"month": "june", "total": 3}}  # task_result's own schema
