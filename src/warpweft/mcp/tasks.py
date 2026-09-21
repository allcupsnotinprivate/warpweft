"""Background-execution substrate for long-running tools (MCP Tasks groundwork).

A long-running tool need not block its request for minutes: it can run in the
background while the caller polls for status and collects the result later. This
module owns the parts of that story that are warpweft's to own and that are
transport-agnostic - a `TaskStore` protocol for durability (state + result), an
in-memory default for dev and tests, and a `TaskRunner` that spawns each
background job into a nursery tied to the server's lifetime. A job still flows
through ``container.invoke`` - the full policy chain and telemetry - so a task
is just a call whose result is retained instead of returned inline. Persistence
beyond a single process (surviving a restart) is the store's concern.

The MCP *wire* surface for tasks (a client augmenting ``tools/call``, a
``CreateTaskResult`` handle, ``tasks/get`` / ``tasks/result`` / ``tasks/cancel``)
is **not** wired here yet: in the current SDK that feature only exists on the
modern ``2026-07-28`` streamable-HTTP transport, which is unreachable over the
stdio initialize handshake warpweft serves today. This substrate is the
foundation that transport will build on when it lands.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Protocol

import anyio
import anyio.abc
import mcp.types as mt
from mcp_types import TaskStatus

from warpweft.core.clock import Clock, SystemClock

#: Default retention and suggested poll cadence, in milliseconds. A client may
#: request a shorter TTL via the augmentation; it can never extend past this.
DEFAULT_TTL_MS = 5 * 60 * 1000
DEFAULT_POLL_INTERVAL_MS = 500

#: A task in a terminal state never changes again.
TERMINAL: frozenset[TaskStatus] = frozenset({"completed", "failed", "cancelled"})


@dataclass(frozen=True)
class TaskRecord:
    """The durable state of one task. ``result`` is set only once terminal."""

    task_id: str
    tool: str
    status: TaskStatus
    created_at: datetime
    last_updated_at: datetime
    ttl_ms: int | None
    poll_interval_ms: int | None
    status_message: str | None = None
    #: The final ``tools/call`` payload, retained for ``tasks/result``.
    result: mt.CallToolResult | None = None


class TaskStore(Protocol):
    """Durable home for task state and results.

    warpweft ships `InMemoryTaskStore`; a deployment that must survive a
    process restart supplies its own (Redis, Postgres, ...). All methods are
    async so a backing store can be remote. ``get`` returns ``None`` for an
    unknown or expired id.
    """

    async def create(self, record: TaskRecord) -> None: ...

    async def get(self, task_id: str) -> TaskRecord | None: ...

    async def update(
        self,
        task_id: str,
        *,
        status: TaskStatus,
        status_message: str | None = None,
        result: mt.CallToolResult | None = None,
    ) -> None:
        """Advance a task's state. A no-op if the id is unknown."""
        ...


@dataclass
class InMemoryTaskStore:
    """Process-local `TaskStore` for development and tests.

    Records live in a dict guarded by a lock; nothing survives a restart, and
    expired records are dropped lazily on access. Timestamps come from the
    injected `Clock`, so a `ManualClock` makes retention deterministic.
    """

    clock: Clock = field(default_factory=SystemClock)
    _records: dict[str, TaskRecord] = field(default_factory=dict)
    _lock: anyio.Lock = field(default_factory=anyio.Lock)

    async def create(self, record: TaskRecord) -> None:
        async with self._lock:
            self._records[record.task_id] = record

    async def get(self, task_id: str) -> TaskRecord | None:
        async with self._lock:
            record = self._records.get(task_id)
            if record is None:
                return None
            if self._expired(record):
                del self._records[task_id]
                return None
            return record

    async def update(
        self,
        task_id: str,
        *,
        status: TaskStatus,
        status_message: str | None = None,
        result: mt.CallToolResult | None = None,
    ) -> None:
        async with self._lock:
            record = self._records.get(task_id)
            if record is None:
                return
            self._records[task_id] = replace(
                record,
                status=status,
                status_message=status_message if status_message is not None else record.status_message,
                result=result if result is not None else record.result,
                last_updated_at=self.clock.now(),
            )

    def _expired(self, record: TaskRecord) -> bool:
        if record.ttl_ms is None:
            return False
        age_ms = (self.clock.now() - record.created_at).total_seconds() * 1000
        return age_ms > record.ttl_ms


#: A background job: given its task id, run to completion and return the
#: final tool payload. The runner handles storing status and catching cancel.
TaskJob = Callable[[], Awaitable[mt.CallToolResult]]


class TaskRunner:
    """Spawns and tracks background task executions.

    The nursery is injected (see `task_runner`) so it outlives individual
    requests but is bounded by the server's lifetime. Each running task keeps a
    `CancelScope` so ``tasks/cancel`` can stop exactly one job; on shutdown the
    nursery is cancelled and every outstanding job unwinds through its policy
    chain.
    """

    def __init__(self, task_group: anyio.abc.TaskGroup, store: TaskStore, clock: Clock) -> None:
        self._tg = task_group
        self._store = store
        self._clock = clock
        self._scopes: dict[str, anyio.CancelScope] = {}

    @property
    def store(self) -> TaskStore:
        return self._store

    async def start(self, task_id: str, tool: str, job: TaskJob, *, ttl_ms: int | None) -> TaskRecord:
        """Durably create the task, then spawn it. Returns the initial record.

        The record is persisted before we return so a poll that races the
        spawn always finds the task (the spec requires durable creation before
        the response is sent).
        """
        now = self._clock.now()
        record = TaskRecord(
            task_id=task_id,
            tool=tool,
            status="working",
            created_at=now,
            last_updated_at=now,
            ttl_ms=ttl_ms,
            poll_interval_ms=DEFAULT_POLL_INTERVAL_MS,
        )
        await self._store.create(record)
        self._tg.start_soon(self._run, task_id, job)
        return record

    async def cancel(self, task_id: str) -> None:
        """Request cancellation of a running task; a no-op if it already ended."""
        scope = self._scopes.get(task_id)
        if scope is not None:
            scope.cancel()

    async def _run(self, task_id: str, job: TaskJob) -> None:
        scope = anyio.CancelScope()
        self._scopes[task_id] = scope
        try:
            with scope:
                result = await job()
                status: TaskStatus = "failed" if result.is_error else "completed"
                await self._store.update(task_id, status=status, result=result)
                return
            # Falls through here only when the scope caught its own
            # cancellation (a tasks/cancel). Shield the bookkeeping so the
            # cancellation cannot also interrupt the status write.
            with anyio.CancelScope(shield=True):
                await self._store.update(task_id, status="cancelled")
        except Exception as exc:
            # ``job`` is expected to turn every failure into an error result, so
            # this is a last-resort guard: a bug here must not tear down the
            # shared nursery and its sibling tasks.
            with anyio.CancelScope(shield=True):
                await self._store.update(task_id, status="failed", status_message=str(exc))
        finally:
            self._scopes.pop(task_id, None)


@asynccontextmanager
async def task_runner(store: TaskStore | None = None, *, clock: Clock | None = None) -> AsyncIterator[TaskRunner]:
    """Open a `TaskRunner` whose nursery lives for the duration of the block.

    Defaults to an `InMemoryTaskStore`. On exit the nursery is cancelled, so no
    background task outlives the server; a store that must retain results past
    shutdown persists them as it goes.
    """
    clock = clock or SystemClock()
    store = store or InMemoryTaskStore(clock)
    async with anyio.create_task_group() as tg:
        try:
            yield TaskRunner(tg, store, clock)
        finally:
            tg.cancel_scope.cancel()
