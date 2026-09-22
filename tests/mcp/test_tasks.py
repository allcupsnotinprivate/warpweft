"""warpweft.mcp background tools: the substrate (store + runner) and the
plain-``tools/call`` submit / poll surface built on it.

The wire tests drive an ordinary in-memory client: submit, task_status,
task_result and task_cancel are all normal tools returning ``CallToolResult``,
so they work over the stdio handshake with no special transport.
"""

import json
from typing import Any

import anyio
import mcp.types as mt
from pydantic import BaseModel
import pytest

from warpweft.core.clock import ManualClock
from warpweft.core.component import AComponent, EmptySettings, invocable
from warpweft.core.composition import Registry
from warpweft.core.context import report_progress
from warpweft.core.errors import FrameworkError, PermanentError
from warpweft.mcp import build_server, task_runner, tool
from warpweft.mcp.tasks import DEFAULT_TTL_MS, InMemoryTaskStore, TaskRecord
from warpweft.runtime import App

pytestmark = pytest.mark.anyio


def app_with(*classes: type[AComponent[Any, Any, Any]]) -> App:
    reg = Registry()
    for cls in classes:
        reg.register(cls)
    return App(registry=reg)


# --- substrate: InMemoryTaskStore -------------------------------------------


def ok(text: str = "ok") -> mt.CallToolResult:
    return mt.CallToolResult(content=[mt.TextContent(type="text", text=text)])


def err(text: str = "boom") -> mt.CallToolResult:
    return mt.CallToolResult(content=[mt.TextContent(type="text", text=text)], is_error=True)


def record(clock: ManualClock, task_id: str = "t1", *, ttl_ms: int | None = DEFAULT_TTL_MS) -> TaskRecord:
    now = clock.now()
    return TaskRecord(
        task_id=task_id,
        tool="demo",
        status="working",
        created_at=now,
        last_updated_at=now,
        ttl_ms=ttl_ms,
    )


async def test_create_and_get_roundtrip() -> None:
    clock = ManualClock()
    store = InMemoryTaskStore(clock)
    rec = record(clock)
    await store.create(rec)
    assert await store.get("t1") == rec


async def test_get_unknown_is_none() -> None:
    assert await InMemoryTaskStore(ManualClock()).get("missing") is None


async def test_update_advances_status_and_stamps_time() -> None:
    clock = ManualClock()
    store = InMemoryTaskStore(clock)
    await store.create(record(clock))
    clock.advance(1.0)
    result = ok("done")
    await store.update("t1", status="completed", status_message="finished", result=result)

    got = await store.get("t1")
    assert got is not None
    assert got.status == "completed"
    assert got.status_message == "finished"
    assert got.result is result
    assert got.last_updated_at == clock.now()


async def test_update_unknown_is_a_noop() -> None:
    store = InMemoryTaskStore(ManualClock())
    await store.update("missing", status="completed")
    assert await store.get("missing") is None


async def test_expired_record_is_dropped_on_access() -> None:
    clock = ManualClock()
    store = InMemoryTaskStore(clock)
    await store.create(record(clock, ttl_ms=1000))
    clock.advance(2.0)
    assert await store.get("t1") is None


async def test_unlimited_ttl_never_expires() -> None:
    clock = ManualClock()
    store = InMemoryTaskStore(clock)
    await store.create(record(clock, ttl_ms=None))
    clock.advance(10_000)
    assert await store.get("t1") is not None


# --- substrate: TaskRunner ---------------------------------------------------


def const_job(result: mt.CallToolResult) -> Any:
    async def job(task_id: str) -> mt.CallToolResult:
        return result

    return job


async def poll_store(store: InMemoryTaskStore, task_id: str, status: str) -> TaskRecord:
    with anyio.fail_after(2):
        while True:
            rec = await store.get(task_id)
            if rec is not None and rec.status == status:
                return rec
            await anyio.sleep(0.01)


async def test_runner_mints_ids_and_runs_to_completion() -> None:
    counter = iter(range(1, 10))
    factory = lambda: f"task-{next(counter)}"  # noqa: E731 - test-local
    async with task_runner(clock=ManualClock(), id_factory=factory) as runner:
        result = ok("value")
        rec = await runner.start("demo", const_job(result), ttl_ms=DEFAULT_TTL_MS)
        assert rec.task_id == "task-1"  # from the injected factory
        assert rec.status == "working"  # durably created before returning

        done = await poll_store(runner.store, rec.task_id, "completed")  # type: ignore[arg-type]
        assert done.result is result


async def test_default_ids_are_unique_across_runners() -> None:
    # The default uuid factory keeps ids unique even for runners sharing a store,
    # so records don't clobber each other (the old per-runner counter collided).
    async with task_runner(clock=ManualClock()) as a, task_runner(clock=ManualClock()) as b:
        rec_a = await a.start("demo", const_job(ok()), ttl_ms=None)
        rec_b = await b.start("demo", const_job(ok()), ttl_ms=None)
        assert rec_a.task_id != rec_b.task_id


async def test_runner_marks_error_result_as_failed() -> None:
    async with task_runner(clock=ManualClock()) as runner:
        rec = await runner.start("demo", const_job(err()), ttl_ms=None)
        failed = await poll_store(runner.store, rec.task_id, "failed")  # type: ignore[arg-type]
        assert failed.result is not None
        assert failed.result.is_error is True


async def test_runner_guards_against_a_raising_job() -> None:
    async def boom(task_id: str) -> mt.CallToolResult:
        raise RuntimeError("kaboom")

    async with task_runner(clock=ManualClock()) as runner:
        rec = await runner.start("demo", boom, ttl_ms=None)
        failed = await poll_store(runner.store, rec.task_id, "failed")  # type: ignore[arg-type]
        assert failed.status_message == "kaboom"
        # The nursery survives: a second job still runs.
        rec2 = await runner.start("demo", const_job(ok()), ttl_ms=None)
        await poll_store(runner.store, rec2.task_id, "completed")  # type: ignore[arg-type]


async def test_runner_cancel_stops_a_job() -> None:
    started = anyio.Event()
    cancelled = anyio.Event()

    async def waiter(task_id: str) -> mt.CallToolResult:
        started.set()
        try:
            await anyio.sleep(60)
        except anyio.get_cancelled_exc_class():
            cancelled.set()
            raise
        return ok("never")

    async with task_runner(clock=ManualClock()) as runner:
        rec = await runner.start("demo", waiter, ttl_ms=None)
        with anyio.fail_after(2):
            await started.wait()
        await runner.cancel(rec.task_id)
        with anyio.fail_after(2):
            await cancelled.wait()
        settled = await poll_store(runner.store, rec.task_id, "cancelled")  # type: ignore[arg-type]
        assert settled.status == "cancelled"


async def test_custom_id_factory() -> None:
    ids = iter(["a", "b"])
    async with task_runner(clock=ManualClock(), id_factory=lambda: next(ids)) as runner:
        rec = await runner.start("demo", const_job(ok()), ttl_ms=None)
        assert rec.task_id == "a"


async def test_cancel_before_the_job_starts_is_honored() -> None:
    # The scope is registered in start() before it returns, so a cancel issued
    # immediately after start() - before the job has begun running - is not lost
    # (regression: it used to no-op because the scope wasn't registered yet).
    proceed = anyio.Event()  # never set: the job hangs unless cancelled

    async def job(task_id: str) -> mt.CallToolResult:
        await proceed.wait()
        return ok("never")

    async with task_runner(clock=ManualClock()) as runner:
        rec = await runner.start("demo", job, ttl_ms=None)
        await runner.cancel(rec.task_id)
        settled = await poll_store(runner.store, rec.task_id, "cancelled")  # type: ignore[arg-type]
        assert settled.status == "cancelled"
        assert settled.status_message == "cancelled"  # stale progress overwritten


async def test_update_does_not_resurrect_an_expired_record() -> None:
    clock = ManualClock()
    store = InMemoryTaskStore(clock)
    await store.create(record(clock, ttl_ms=1000))
    clock.advance(2.0)  # past TTL
    # A late progress update must not un-expire the record.
    await store.update("t1", status="working", status_message="late")
    assert await store.get("t1") is None


# --- wire: background tools over a plain client ------------------------------


class Report(BaseModel):
    month: str
    total: int


class Reports(AComponent[EmptySettings, str, Report]):
    name = "reports"

    @tool(background=True, description="Generate a monthly report.")
    @invocable
    async def report(self, month: str) -> Report:
        await report_progress(0.5, message="crunching")
        return Report(month=month, total=42)


class Quick(AComponent[EmptySettings, None, str]):
    name = "quick"

    @tool(background=True)
    @invocable
    async def run(self) -> str:
        return "done"


class Boom(AComponent[EmptySettings, None, str]):
    name = "boom"

    @tool(background=True)
    @invocable
    async def run(self) -> str:
        raise PermanentError("nope")


async def submit(client: Any, name: str, args: dict[str, Any] | None = None) -> str:
    result = await client.call_tool(name, args or {})
    assert result.is_error is False, result.content
    assert result.structured_content is not None
    return str(result.structured_content["task_id"])


async def poll_client(client: Any, task_id: str, status: str) -> mt.CallToolResult:
    with anyio.fail_after(2):
        while True:
            res = await client.call_tool("task_status", {"task_id": task_id})
            if res.structured_content is not None and res.structured_content.get("status") == status:
                return res
            await anyio.sleep(0.01)


async def test_submit_advertises_task_id_output(connect) -> None:
    async with task_runner() as runner, connect(app_with(Quick), runner=runner) as client:
        listed = await client.list_tools()
    by_name = {t.name: t for t in listed.tools}
    submit_tool = by_name["quick__run"]
    assert submit_tool.output_schema == {
        "type": "object",
        "properties": {"task_id": {"type": "string"}},
        "required": ["task_id"],
    }
    # The shared poll tools are present.
    assert {"task_status", "task_result", "task_cancel"} <= set(by_name)


async def test_task_result_schema_is_a_union_of_op_results(connect) -> None:
    async with task_runner() as runner, connect(app_with(Quick, Reports), runner=runner) as client:
        listed = await client.list_tools()
    result_tool = next(t for t in listed.tools if t.name == "task_result")
    schema = result_tool.output_schema
    assert schema is not None
    union = schema["properties"]["result"]["oneOf"]
    assert {"type": "string"} in union  # quick__run returns str
    # reports__report returns a model -> its object schema is a union member.
    assert any(m.get("title") == "Report" and m.get("type") == "object" for m in union)


async def test_task_result_schema_hoists_nested_defs(connect) -> None:
    class Line(BaseModel):
        sku: str

    class Invoice(BaseModel):
        lines: list[Line]

    class Billing(AComponent[EmptySettings, str, Invoice]):
        name = "billing"

        @tool(background=True)
        @invocable
        async def invoice(self, customer: str) -> Invoice:
            return Invoice(lines=[])

    async with task_runner() as runner, connect(app_with(Billing), runner=runner) as client:
        listed = await client.list_tools()
    result_tool = next(t for t in listed.tools if t.name == "task_result")
    schema = result_tool.output_schema
    assert schema is not None
    # A nested model forces $defs, which must be hoisted to the wrapper root.
    assert "Line" in schema.get("$defs", {})


def _alpha_component() -> type[AComponent[Any, Any, Any]]:
    class Row(BaseModel):
        a: int

    class Out(BaseModel):
        rows: list[Row]

    class Alpha(AComponent[EmptySettings, None, Out]):
        name = "alpha"

        @tool(background=True)
        @invocable
        async def go(self) -> Out:
            return Out(rows=[])

    return Alpha


def _beta_component() -> type[AComponent[Any, Any, Any]]:
    class Row(BaseModel):  # same class name as Alpha's, different fields
        b: str

    class Out(BaseModel):
        rows: list[Row]

    class Beta(AComponent[EmptySettings, None, Out]):
        name = "beta"

        @tool(background=True)
        @invocable
        async def go(self) -> Out:
            return Out(rows=[])

    return Beta


async def test_task_result_union_namespaces_colliding_defs(connect) -> None:
    # Two ops each nest a model named "Row" with different fields. Hoisting both
    # into one $defs would clobber one; namespacing by tool name keeps both.
    app = app_with(_alpha_component(), _beta_component())
    async with task_runner() as runner, connect(app, runner=runner) as client:
        listed = await client.list_tools()
    schema = next(t for t in listed.tools if t.name == "task_result").output_schema
    assert schema is not None
    defs = schema["$defs"]
    assert "a" in defs["alpha__go.Row"]["properties"]  # Alpha's Row survives
    assert "b" in defs["beta__go.Row"]["properties"]  # Beta's Row survives, un-clobbered
    dumped = json.dumps(schema)  # each member's $ref points at its own namespaced def
    assert "#/$defs/alpha__go.Row" in dumped
    assert "#/$defs/beta__go.Row" in dumped


async def test_submit_then_poll_then_result(connect) -> None:
    async with task_runner() as runner, connect(app_with(Reports), runner=runner) as client:
        task_id = await submit(client, "reports__report", {"month": "june"})

        done = await poll_client(client, task_id, "completed")
        assert done.structured_content is not None

        result = await client.call_tool("task_result", {"task_id": task_id})
        assert result.is_error is False
        assert result.structured_content == {"result": {"month": "june", "total": 42}}


async def test_progress_is_reflected_in_status(connect) -> None:
    release = anyio.Event()

    class Slow(AComponent[EmptySettings, None, str]):
        name = "slow"

        @tool(background=True)
        @invocable
        async def run(self) -> str:
            await report_progress(0.5, message="halfway")
            await release.wait()
            return "done"

    async with task_runner() as runner, connect(app_with(Slow), runner=runner) as client:
        task_id = await submit(client, "slow__run")
        with anyio.fail_after(2):
            while True:
                res = await client.call_tool("task_status", {"task_id": task_id})
                if res.structured_content and res.structured_content.get("message") == "halfway":
                    break
                await anyio.sleep(0.01)
        release.set()
        await poll_client(client, task_id, "completed")


async def test_result_before_finished_is_an_error(connect) -> None:
    release = anyio.Event()

    class Slow(AComponent[EmptySettings, None, str]):
        name = "slow"

        @tool(background=True)
        @invocable
        async def run(self) -> str:
            await release.wait()
            return "done"

    async with task_runner() as runner, connect(app_with(Slow), runner=runner) as client:
        task_id = await submit(client, "slow__run")
        early = await client.call_tool("task_result", {"task_id": task_id})
        assert early.is_error is True
        assert early.meta is not None
        assert early.meta["warpweft.error"] == "not_ready"
        # Still running -> the result will appear, so the poll is retryable.
        assert early.meta["warpweft.retryable"] is True
        release.set()
        await poll_client(client, task_id, "completed")


async def test_failed_task_result_carries_the_error(connect) -> None:
    async with task_runner() as runner, connect(app_with(Boom), runner=runner) as client:
        task_id = await submit(client, "boom__run")
        await poll_client(client, task_id, "failed")
        result = await client.call_tool("task_result", {"task_id": task_id})
        assert result.is_error is True
        assert result.meta is not None
        assert result.meta["warpweft.error"] == "permanent"


async def test_cancel_stops_a_running_task(connect) -> None:
    started = anyio.Event()
    cancelled = anyio.Event()

    class Waiter(AComponent[EmptySettings, None, str]):
        name = "waiter"

        @tool(background=True)
        @invocable
        async def run(self) -> str:
            started.set()
            try:
                await anyio.sleep(60)
            except anyio.get_cancelled_exc_class():
                cancelled.set()
                raise
            return "never"

    async with task_runner() as runner, connect(app_with(Waiter), runner=runner) as client:
        task_id = await submit(client, "waiter__run")
        with anyio.fail_after(2):
            await started.wait()
        ack = await client.call_tool("task_cancel", {"task_id": task_id})
        assert ack.is_error is False
        with anyio.fail_after(2):
            await cancelled.wait()
        settled = await poll_client(client, task_id, "cancelled")
        # Terminal status carries a clear message, not stale in-flight progress.
        assert settled.structured_content is not None
        assert settled.structured_content["message"] == "cancelled"

        # task_result on a terminal-but-result-less task is a distinct,
        # non-retryable terminal error, never the retry-implying not_ready.
        after = await client.call_tool("task_result", {"task_id": task_id})
        assert after.is_error is True
        assert after.meta is not None
        assert after.meta["warpweft.error"] == "cancelled"
        assert after.meta["warpweft.retryable"] is False


async def test_unknown_task_is_a_tool_error(connect) -> None:
    async with task_runner() as runner, connect(app_with(Quick), runner=runner) as client:
        res = await client.call_tool("task_status", {"task_id": "nope"})
    assert res.is_error is True
    assert res.meta is not None
    assert res.meta["warpweft.error"] == "unknown_task"


# --- wire: build-time guards -------------------------------------------------


def test_background_tool_without_runner_fails_loudly() -> None:
    with pytest.raises(FrameworkError, match="background=True.*no task runner"):
        build_server(app_with(Quick))


async def test_reserved_tool_name_is_rejected() -> None:
    class Clash(AComponent[EmptySettings, None, str]):
        name = "svc"

        @tool(name="task_status")
        @invocable
        async def go(self) -> str:
            return "x"

    async with task_runner() as runner:
        with pytest.raises(FrameworkError, match="reserved for background task tools"):
            build_server(app_with(Clash), runner=runner)


async def test_confirm_destructive_gates_a_background_submit(connect) -> None:
    class Danger(AComponent[EmptySettings, None, str]):
        name = "danger"

        @tool(background=True, destructive=True)
        @invocable
        async def wipe(self) -> str:
            return "wiped"

    async def decline(context: Any, params: Any) -> Any:
        return mt.ElicitResult(action="decline")

    async with (
        task_runner() as runner,
        connect(app_with(Danger), runner=runner, elicitation_callback=decline, confirm_destructive=True) as client,
    ):
        result = await client.call_tool("danger__wipe", {})
    assert result.is_error is True
    assert result.meta is not None
    assert result.meta["warpweft.error"] == "declined"
