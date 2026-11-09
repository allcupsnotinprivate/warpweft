"""Reusable conformance tests for custom ``TaskStore`` implementations."""

from dataclasses import replace

import anyio
import mcp.types as mt
from mcp_types import TaskStatus
import pytest

from warpweft.core.clock import ManualClock
from warpweft.mcp.tasks import (
    InMemoryTaskStore,
    InvalidTaskTransitionError,
    TaskAlreadyExistsError,
    TaskRecord,
    TaskStore,
)


def result(text: str = "ok") -> mt.CallToolResult:
    return mt.CallToolResult(content=[mt.TextContent(type="text", text=text)])


def make_record(
    clock: ManualClock,
    task_id: str = "t1",
    *,
    tool: str = "demo",
    status: TaskStatus = "working",
    ttl_ms: int | None = 1000,
) -> TaskRecord:
    now = clock.now()
    return TaskRecord(
        task_id=task_id,
        tool=tool,
        status=status,
        created_at=now,
        last_updated_at=now,
        ttl_ms=ttl_ms,
    )


class TaskStoreContract:
    """Subclass with a ``store`` fixture to check a custom store's contract."""

    @pytest.fixture
    def clock(self) -> ManualClock:
        return ManualClock()

    @pytest.fixture
    def store(self, clock: ManualClock) -> TaskStore:
        raise NotImplementedError

    async def test_live_duplicate_create_is_rejected_without_overwriting(
        self, store: TaskStore, clock: ManualClock
    ) -> None:
        original = make_record(clock, tool="original")
        await store.create(original)
        with pytest.raises(TaskAlreadyExistsError, match="t1"):
            await store.create(make_record(clock, tool="replacement"))
        assert await store.get("t1") == original

    async def test_concurrent_duplicate_create_has_exactly_one_winner(
        self, store: TaskStore, clock: ManualClock
    ) -> None:
        gate = anyio.Event()
        errors: list[Exception] = []

        async def create(tool: str) -> None:
            await gate.wait()
            try:
                await store.create(make_record(clock, tool=tool))
            except Exception as exc:
                errors.append(exc)

        async with anyio.create_task_group() as tg:
            tg.start_soon(create, "a")
            tg.start_soon(create, "b")
            gate.set()
        assert len(errors) == 1
        assert isinstance(errors[0], TaskAlreadyExistsError)
        stored = await store.get("t1")
        assert stored is not None
        assert stored.tool in {"a", "b"}

    async def test_expired_record_can_be_replaced(self, store: TaskStore, clock: ManualClock) -> None:
        await store.create(make_record(clock, status="completed", ttl_ms=100))
        clock.advance(0.2)
        replacement = make_record(clock, tool="replacement")
        await store.create(replacement)
        assert await store.get("t1") == replacement

    @pytest.mark.parametrize("status", ["working", "completed", "failed", "cancelled"])
    async def test_working_transitions_are_allowed(
        self, store: TaskStore, clock: ManualClock, status: TaskStatus
    ) -> None:
        await store.create(make_record(clock))
        clock.advance(1)
        await store.update("t1", status=status, status_message="new", result=result())
        stored = await store.get("t1")
        assert stored is not None
        assert stored.status == status
        assert stored.status_message == "new"
        assert stored.last_updated_at == clock.now()

    @pytest.mark.parametrize("target", ["working", "input_required", "completed", "failed", "cancelled"])
    async def test_input_required_cannot_transition(
        self, store: TaskStore, clock: ManualClock, target: TaskStatus
    ) -> None:
        original = make_record(clock, status="input_required")
        await store.create(original)
        with pytest.raises(InvalidTaskTransitionError):
            await store.update("t1", status=target)
        assert await store.get("t1") == original

    async def test_working_to_input_required_is_a_typed_error(self, store: TaskStore, clock: ManualClock) -> None:
        original = make_record(clock)
        await store.create(original)
        with pytest.raises(InvalidTaskTransitionError, match="working.*input_required"):
            await store.update("t1", status="input_required")
        assert await store.get("t1") == original

    @pytest.mark.parametrize("initial", ["completed", "failed", "cancelled"])
    @pytest.mark.parametrize("target", ["working", "input_required", "completed", "failed", "cancelled"])
    async def test_terminal_records_are_immutable(
        self, store: TaskStore, clock: ManualClock, initial: TaskStatus, target: TaskStatus
    ) -> None:
        original = make_record(clock, status=initial, ttl_ms=None)
        original = replace(original, result=result("original"))
        await store.create(original)
        clock.advance(1)
        await store.update("t1", status=target, status_message="ignored", result=result("ignored"))
        assert await store.get("t1") == original

    async def test_first_terminal_write_wins_a_race(self, store: TaskStore, clock: ManualClock) -> None:
        await store.create(make_record(clock, ttl_ms=None))
        gate = anyio.Event()

        async def settle(status: TaskStatus) -> None:
            await gate.wait()
            await store.update("t1", status=status, status_message=status, result=result(status))

        async with anyio.create_task_group() as tg:
            tg.start_soon(settle, "completed")
            tg.start_soon(settle, "cancelled")
            gate.set()
        stored = await store.get("t1")
        assert stored is not None
        assert stored.status in {"completed", "cancelled"}
        assert stored.status_message == stored.status
        assert stored.result is not None
        assert stored.result.content[0].text == stored.status  # type: ignore[union-attr]

    async def test_terminal_write_is_atomic_with_cancellation(self, store: TaskStore, clock: ManualClock) -> None:
        await store.create(make_record(clock, ttl_ms=None))
        await store.update("t1", status="completed", status_message="done", result=result("done"))
        settled = await store.get("t1")
        await store.update("t1", status="cancelled", status_message="late")
        assert await store.get("t1") == settled

    async def test_ignored_terminal_update_does_not_extend_ttl(self, store: TaskStore, clock: ManualClock) -> None:
        await store.create(make_record(clock, ttl_ms=1000))
        clock.advance(1)
        await store.update("t1", status="completed", result=result())
        settled = await store.get("t1")
        assert settled is not None
        clock.advance(0.6)
        await store.update("t1", status="failed", status_message="ignored")
        assert await store.get("t1") == settled
        clock.advance(0.6)
        assert await store.get("t1") is None

    async def test_unknown_and_expired_updates_are_noops(self, store: TaskStore, clock: ManualClock) -> None:
        await store.update("missing", status="completed")
        assert await store.get("missing") is None
        await store.create(make_record(clock, status="completed", ttl_ms=100))
        clock.advance(0.2)
        await store.update("t1", status="working")
        assert await store.get("t1") is None


class TestInMemoryTaskStore(TaskStoreContract):
    @pytest.fixture
    def store(self, clock: ManualClock) -> TaskStore:
        return InMemoryTaskStore(clock)
