"""SystemClock smoke and ManualClock virtual-time behavior."""

from datetime import UTC

import anyio
import pytest

from warpweft.core.clock import ManualClock, SystemClock

pytestmark = [pytest.mark.unit, pytest.mark.anyio]


async def test_system_clock_smoke() -> None:
    clock = SystemClock()
    before = clock.monotonic()
    await clock.sleep(0)
    assert clock.monotonic() >= before
    assert clock.now().tzinfo is not None
    # Negative values must not raise (and must still be a checkpoint).
    await clock.sleep(-1)


async def test_manual_clock_does_not_really_sleep() -> None:
    clock = ManualClock()
    woke_at: list[float] = []

    async def sleeper() -> None:
        await clock.sleep(3600)  # an hour of virtual time
        woke_at.append(clock.monotonic())

    async with anyio.create_task_group() as tg:
        tg.start_soon(sleeper)
        await clock.wait_for_sleepers(1)
        clock.advance(3600)

    assert woke_at == [3600.0]


async def test_manual_clock_wakes_in_order_of_due_time() -> None:
    clock = ManualClock()
    woke: list[str] = []

    async def sleeper(name: str, seconds: float) -> None:
        await clock.sleep(seconds)
        woke.append(name)

    async with anyio.create_task_group() as tg:
        tg.start_soon(sleeper, "late", 10.0)
        tg.start_soon(sleeper, "early", 1.0)
        await clock.wait_for_sleepers(2)
        clock.advance(1.0)  # wakes only "early"
        await anyio.lowlevel.checkpoint()
        assert clock.pending == 1
        clock.advance(9.0)  # wakes "late"

    assert woke == ["early", "late"]
    assert clock.monotonic() == 10.0


async def test_manual_clock_zero_sleep_is_checkpoint_not_park() -> None:
    clock = ManualClock()
    await clock.sleep(0)  # must return without advance()
    assert clock.pending == 0


async def test_manual_clock_now_follows_virtual_time() -> None:
    clock = ManualClock()
    start = clock.now()
    assert start.tzinfo is UTC
    clock.advance(90)
    assert (clock.now() - start).total_seconds() == 90


async def test_manual_clock_rejects_backward_advance() -> None:
    clock = ManualClock()
    with pytest.raises(ValueError, match="backwards"):
        clock.advance(-1)
