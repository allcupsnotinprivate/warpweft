"""Timeout link: budget enforcement, deadline trimming, no call after expiry."""

from typing import Any

import anyio
from pydantic import ValidationError
import pytest

from warpweft.core.axes import AxisRegistry
from warpweft.core.clock import ManualClock, SystemClock
from warpweft.core.context import InvocationContext
from warpweft.core.errors import AttemptTimeout, DeadlineExceeded
from warpweft.core.outcome import Outcome
from warpweft.core.pipeline.builtin.timeout import TimeoutFactory, TimeoutInterceptor, TimeoutSettings
from warpweft.core.pipeline.chain import build_chain
from warpweft.core.pipeline.interceptor import Next
from warpweft.core.pipeline.state import InMemoryStateStore

pytestmark = [pytest.mark.unit, pytest.mark.anyio]


def ctx(**overrides: Any) -> InvocationContext:
    defaults: dict[str, Any] = {"operation": "op", "correlation_id": "cid"}
    defaults.update(overrides)
    return InvocationContext(**defaults)


async def ok(c: InvocationContext) -> Outcome[Any]:
    return Outcome(value="ok")


async def hang_forever(c: InvocationContext) -> Outcome[Any]:
    await anyio.Event().wait()
    raise AssertionError("unreachable")


async def test_success_passes_through() -> None:
    link = TimeoutInterceptor(TimeoutSettings(seconds=1), SystemClock())
    outcome = await link.call(ok, ctx())
    assert outcome.value == "ok"


async def test_fires_on_slow_call() -> None:
    link = TimeoutInterceptor(TimeoutSettings(seconds=0.01), SystemClock())
    with pytest.raises(AttemptTimeout, match="op"):
        await link.call(hang_forever, ctx())


async def test_limit_is_trimmed_by_remaining_deadline() -> None:
    """seconds=10, but only 0.01 virtual seconds of deadline remain: the deadline wins."""
    clock = ManualClock()
    link = TimeoutInterceptor(TimeoutSettings(seconds=10), clock)
    with pytest.raises(AttemptTimeout):
        # If the limit were taken from settings, this would hang for 10 real seconds.
        await link.call(hang_forever, ctx(deadline=0.01))


async def test_expired_deadline_means_no_call_at_all() -> None:
    clock = ManualClock()
    clock.advance(10)
    called = False

    async def base(c: InvocationContext) -> Outcome[Any]:
        nonlocal called
        called = True
        return Outcome(value="ok")

    link = TimeoutInterceptor(TimeoutSettings(seconds=1), clock)
    with pytest.raises(DeadlineExceeded):
        await link.call(base, ctx(deadline=5.0))
    assert not called


async def test_generous_deadline_does_not_trim() -> None:
    clock = ManualClock()
    link = TimeoutInterceptor(TimeoutSettings(seconds=0.01), clock)
    with pytest.raises(AttemptTimeout):
        await link.call(hang_forever, ctx(deadline=1000.0))


async def test_settings_validation() -> None:
    with pytest.raises(ValidationError):
        TimeoutSettings(seconds=0)
    with pytest.raises(ValidationError):
        TimeoutSettings(seconds=-1)


async def test_factory_wires_into_chain() -> None:
    factory = TimeoutFactory(TimeoutSettings(seconds=1), SystemClock())
    assert not factory.state_scope
    chain: Next = build_chain([factory], InMemoryStateStore(), AxisRegistry(), ok)
    assert (await chain(ctx())).value == "ok"
