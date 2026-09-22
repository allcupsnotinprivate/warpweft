"""Concurrency link: two-level limits, fair share, deadline-aware waiting.

Concurrency is observed with a base call that parks on a gate while tracking
how many calls are active at once; virtual time is not involved except in the
deadline tests.
"""

from typing import Any

import anyio
from pydantic import ValidationError
import pytest

from warpweft.core.axes import ScopeKey
from warpweft.core.clock import ManualClock, SystemClock
from warpweft.core.context import InvocationContext
from warpweft.core.errors import DeadlineExceeded, TransientError
from warpweft.core.outcome import Outcome
from warpweft.core.pipeline.builtin.concurrency import (
    ENDPOINT_SCOPE,
    ConcurrencyFactory,
    ConcurrencyInterceptor,
    ConcurrencySettings,
)
from warpweft.core.pipeline.interceptor import Next

pytestmark = [pytest.mark.unit, pytest.mark.anyio]

KEY_A: ScopeKey = (("i", "a"),)
KEY_B: ScopeKey = (("i", "b"),)


class Gate:
    """Base call that parks on an event, tracking active/peak concurrency."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.total = 0
        self.open = anyio.Event()

    async def __call__(self, ctx: InvocationContext) -> Outcome[Any]:
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.total += 1
        try:
            await self.open.wait()
        finally:
            self.active -= 1
        return Outcome(value="ok")


def link(clock: Any, *, inner_limit: int, outer_limit: int) -> ConcurrencyInterceptor:
    return ConcurrencyInterceptor(ConcurrencySettings(inner_limit=inner_limit, outer_limit=outer_limit), clock)


def ctx(**overrides: Any) -> InvocationContext:
    defaults: dict[str, Any] = {"operation": "op", "correlation_id": "cid"}
    defaults.update(overrides)
    return InvocationContext(**defaults)


async def let_tasks_run(times: int = 20) -> None:
    """Yield to the scheduler enough for spawned calls to reach their block."""
    for _ in range(times):
        await anyio.lowlevel.checkpoint()


async def test_success_passes_through_and_releases_slots() -> None:
    cb = link(ManualClock(), inner_limit=1, outer_limit=1)

    async def base(c: InvocationContext) -> Outcome[Any]:
        return Outcome(value="ok")

    # Two sequential calls succeed: if slots were not released, the second
    # would block forever.
    assert (await cb.call(base, ctx())).value == "ok"
    assert (await cb.call(base, ctx())).value == "ok"


async def test_outer_limit_caps_total_across_instances() -> None:
    cb = link(ManualClock(), inner_limit=10, outer_limit=1)
    gate = Gate()

    async def run(key: ScopeKey) -> None:
        await cb.call(gate, ctx(scope_key=key))

    async with anyio.create_task_group() as tg:
        tg.start_soon(run, KEY_A)
        tg.start_soon(run, KEY_B)  # different instance, same endpoint
        await let_tasks_run()
        assert gate.active == 1  # outer_limit=1 admits only one, despite inner room
        assert gate.peak == 1
        gate.open.set()
    assert gate.total == 2  # both eventually completed


async def test_inner_limit_serializes_same_instance() -> None:
    cb = link(ManualClock(), inner_limit=1, outer_limit=10)
    gate = Gate()

    async def run() -> None:
        await cb.call(gate, ctx(scope_key=KEY_A))

    async with anyio.create_task_group() as tg:
        tg.start_soon(run)
        tg.start_soon(run)  # same instance slice
        await let_tasks_run()
        assert gate.active == 1  # inner_limit=1 serializes one instance's calls
        gate.open.set()
    assert gate.total == 2


async def test_different_instances_run_concurrently() -> None:
    cb = link(ManualClock(), inner_limit=1, outer_limit=10)
    gate = Gate()

    async def run(key: ScopeKey) -> None:
        await cb.call(gate, ctx(scope_key=key))

    async with anyio.create_task_group() as tg:
        tg.start_soon(run, KEY_A)
        tg.start_soon(run, KEY_B)  # separate inner semaphores
        await let_tasks_run()
        assert gate.active == 2  # each instance has its own fair share
        assert gate.peak == 2
        gate.open.set()
    assert gate.total == 2


async def test_slot_is_released_on_error() -> None:
    cb = link(ManualClock(), inner_limit=1, outer_limit=1)

    async def boom(c: InvocationContext) -> Outcome[Any]:
        raise TransientError("boom")

    async def ok(c: InvocationContext) -> Outcome[Any]:
        return Outcome(value="ok")

    with pytest.raises(TransientError):
        await cb.call(boom, ctx())
    # If the failed call had leaked its slots, this would block forever.
    assert (await cb.call(ok, ctx())).value == "ok"


async def test_expired_deadline_skips_acquisition() -> None:
    clock = ManualClock()
    cb = link(clock, inner_limit=1, outer_limit=1)
    calls = 0

    async def base(c: InvocationContext) -> Outcome[Any]:
        nonlocal calls
        calls += 1
        return Outcome(value="ok")

    c = ctx(deadline=5.0, clock=clock)
    clock.advance(6)
    with pytest.raises(DeadlineExceeded):
        await cb.call(base, c)
    assert calls == 0


async def test_wait_is_bounded_by_the_deadline() -> None:
    clock = SystemClock()
    cb = link(clock, inner_limit=10, outer_limit=1)
    gate = Gate()
    called = 0

    async def hold(c: InvocationContext) -> None:
        await cb.call(gate, c)

    async def other(c: InvocationContext) -> Outcome[Any]:
        nonlocal called
        called += 1
        return Outcome(value="late")

    async with anyio.create_task_group() as tg:
        tg.start_soon(hold, ctx(scope_key=KEY_A))
        await let_tasks_run()
        assert gate.active == 1  # the only outer slot is taken

        # A second call cannot get the outer slot within its small budget.
        waiter = InvocationContext.start("op", "cid", clock=clock, budget=0.02, scope_key=KEY_B)
        with pytest.raises(DeadlineExceeded):
            await cb.call(other, waiter)
        assert called == 0
        gate.open.set()


async def test_many_calls_complete_without_deadlock() -> None:
    cb = link(ManualClock(), inner_limit=2, outer_limit=3)
    gate = Gate()
    gate.open.set()  # no parking: every call runs to completion

    async def run(key: ScopeKey) -> None:
        await cb.call(gate, ctx(scope_key=key))

    async with anyio.create_task_group() as tg:
        for i in range(20):
            tg.start_soon(run, ((("i", str(i % 4))),))
    assert gate.total == 20
    assert gate.peak <= 3  # never exceeds the outer cap


async def test_settings_validation() -> None:
    with pytest.raises(ValidationError):
        ConcurrencySettings(inner_limit=0, outer_limit=1)
    with pytest.raises(ValidationError):
        ConcurrencySettings(inner_limit=1, outer_limit=0)


async def test_factory_scope_and_creation() -> None:
    factory = ConcurrencyFactory(ConcurrencySettings(inner_limit=1, outer_limit=1), ManualClock())
    assert factory.state_scope == ENDPOINT_SCOPE
    assert factory.state_scope.axes == ("endpoint",)

    inst = factory.create(())
    base: Next = _ok
    assert (await inst.call(base, ctx())).value == "ok"


async def _ok(c: InvocationContext) -> Outcome[Any]:
    return Outcome(value="ok")
