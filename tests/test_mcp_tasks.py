"""warpweft.mcp.tasks: the background-execution substrate (store + runner).

These exercise the transport-agnostic groundwork only. The MCP task *wire*
surface is not wired yet (see the module docstring), so nothing here drives a
client session; jobs are plain callables returning a ``CallToolResult``.
"""

import anyio
import mcp.types as mt
import pytest

from warpweft.core.clock import ManualClock
from warpweft.mcp.tasks import DEFAULT_TTL_MS, InMemoryTaskStore, TaskRecord, task_runner

pytestmark = pytest.mark.anyio


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
        poll_interval_ms=500,
    )


# --- InMemoryTaskStore -------------------------------------------------------


async def test_create_and_get_roundtrip() -> None:
    clock = ManualClock()
    store = InMemoryTaskStore(clock)
    rec = record(clock)
    await store.create(rec)
    assert await store.get("t1") == rec


async def test_get_unknown_is_none() -> None:
    store = InMemoryTaskStore(ManualClock())
    assert await store.get("missing") is None


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
    await store.update("missing", status="completed")  # must not raise
    assert await store.get("missing") is None


async def test_expired_record_is_dropped_on_access() -> None:
    clock = ManualClock()
    store = InMemoryTaskStore(clock)
    await store.create(record(clock, ttl_ms=1000))
    clock.advance(2.0)  # 2s > 1000ms
    assert await store.get("t1") is None


async def test_unlimited_ttl_never_expires() -> None:
    clock = ManualClock()
    store = InMemoryTaskStore(clock)
    await store.create(record(clock, ttl_ms=None))
    clock.advance(10_000)
    assert await store.get("t1") is not None


# --- TaskRunner --------------------------------------------------------------


async def poll_until(store: InMemoryTaskStore, task_id: str, status: str) -> TaskRecord:
    with anyio.fail_after(2):
        while True:
            rec = await store.get(task_id)
            if rec is not None and rec.status == status:
                return rec
            await anyio.sleep(0.01)


async def test_runner_runs_a_job_to_completion() -> None:
    async with task_runner(clock=ManualClock()) as runner:
        store = runner.store
        result = ok("value")
        rec = await runner.start("t1", "demo", lambda: _immediate(result), ttl_ms=DEFAULT_TTL_MS)
        assert rec.status == "working"  # durably created before returning

        done = await poll_until(store, "t1", "completed")  # type: ignore[arg-type]
        assert done.result is result


async def test_runner_marks_error_result_as_failed() -> None:
    async with task_runner(clock=ManualClock()) as runner:
        await runner.start("t1", "demo", lambda: _immediate(err()), ttl_ms=None)
        failed = await poll_until(runner.store, "t1", "failed")  # type: ignore[arg-type]
        assert failed.result is not None
        assert failed.result.is_error is True


async def test_runner_guards_against_a_raising_job() -> None:
    async def boom() -> mt.CallToolResult:
        raise RuntimeError("kaboom")

    async with task_runner(clock=ManualClock()) as runner:
        # A job that raises must be caught, not tear down the shared nursery.
        await runner.start("t1", "demo", boom, ttl_ms=None)
        failed = await poll_until(runner.store, "t1", "failed")  # type: ignore[arg-type]
        assert failed.status_message == "kaboom"
        # The nursery survives: a second job still runs.
        await runner.start("t2", "demo", lambda: _immediate(ok()), ttl_ms=None)
        await poll_until(runner.store, "t2", "completed")  # type: ignore[arg-type]


async def test_cancel_stops_a_running_job() -> None:
    started = anyio.Event()
    cancelled = anyio.Event()

    async def waiter() -> mt.CallToolResult:
        started.set()
        try:
            await anyio.sleep(60)
        except anyio.get_cancelled_exc_class():
            cancelled.set()
            raise
        return ok("never")

    async with task_runner(clock=ManualClock()) as runner:
        await runner.start("t1", "demo", waiter, ttl_ms=None)
        with anyio.fail_after(2):
            await started.wait()

        await runner.cancel("t1")
        with anyio.fail_after(2):
            await cancelled.wait()
        settled = await poll_until(runner.store, "t1", "cancelled")  # type: ignore[arg-type]
        assert settled.status == "cancelled"


async def test_cancel_of_unknown_task_is_a_noop() -> None:
    async with task_runner(clock=ManualClock()) as runner:
        await runner.cancel("missing")  # must not raise


async def _immediate(result: mt.CallToolResult) -> mt.CallToolResult:
    return result
