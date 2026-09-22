"""Cache link: key composition, TTL, LRU, single-flight, negative caching."""

from typing import Any

import anyio
from pydantic import ValidationError
import pytest

from warpweft.core.axes import EMPTY_SCOPE, ScopeSpec
from warpweft.core.clock import ManualClock
from warpweft.core.context import InvocationContext
from warpweft.core.errors import TransientError
from warpweft.core.outcome import Outcome
from warpweft.core.pipeline.builtin.cache import (
    CacheFactory,
    CacheInterceptor,
    CacheSettings,
)

pytestmark = [pytest.mark.unit, pytest.mark.anyio]


class Counter:
    """Base call returning an incrementing value; reveals real invocations."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, ctx: InvocationContext) -> Outcome[Any]:
        self.calls += 1
        return Outcome(value=self.calls)


class Boom:
    """Base call that always fails transiently."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, ctx: InvocationContext) -> Outcome[Any]:
        self.calls += 1
        raise TransientError("boom")


def cache(clock: Any, *, ttl: float = 100.0, max_entries: int = 100, cache_errors: bool = False) -> CacheInterceptor:
    return CacheInterceptor(CacheSettings(ttl=ttl, max_entries=max_entries, cache_errors=cache_errors), clock)


def ctx(**overrides: Any) -> InvocationContext:
    defaults: dict[str, Any] = {"operation": "op", "correlation_id": "cid"}
    defaults.update(overrides)
    return InvocationContext(**defaults)


async def let_tasks_run(times: int = 20) -> None:
    for _ in range(times):
        await anyio.lowlevel.checkpoint()


async def test_miss_then_hit() -> None:
    c = cache(ManualClock())
    counter = Counter()

    first = await c.call(counter, ctx(arguments={"q": "x"}))
    assert first.source == "live"
    assert first.value == 1

    second_ctx = ctx(arguments={"q": "x"})
    second = await c.call(counter, second_ctx)
    assert second.source == "cache"
    assert second.value == 1
    assert counter.calls == 1  # the base ran only once
    assert second_ctx.bag["cache"] == "hit"


async def test_key_depends_on_arguments() -> None:
    c = cache(ManualClock())
    counter = Counter()
    await c.call(counter, ctx(arguments={"q": "x"}))
    await c.call(counter, ctx(arguments={"q": "y"}))
    assert counter.calls == 2  # different arguments -> different entries


async def test_key_is_order_independent() -> None:
    c = cache(ManualClock())
    counter = Counter()
    await c.call(counter, ctx(arguments={"a": 1, "b": 2}))
    hit = await c.call(counter, ctx(arguments={"b": 2, "a": 1}))
    assert hit.source == "cache"
    assert counter.calls == 1


async def test_key_depends_on_scope() -> None:
    c = cache(ManualClock())
    counter = Counter()
    await c.call(counter, ctx(arguments={"q": "x"}, scope_key=(("i", "a"),)))
    await c.call(counter, ctx(arguments={"q": "x"}, scope_key=(("i", "b"),)))
    assert counter.calls == 2  # same args, different slice -> no leak


async def test_key_depends_on_operation() -> None:
    c = cache(ManualClock())
    counter = Counter()
    await c.call(counter, ctx(operation="one", arguments={"q": "x"}))
    await c.call(counter, ctx(operation="two", arguments={"q": "x"}))
    assert counter.calls == 2


async def test_ttl_expiry_triggers_a_fresh_call() -> None:
    clock = ManualClock()
    c = cache(clock, ttl=10.0)
    counter = Counter()

    assert (await c.call(counter, ctx())).value == 1
    assert (await c.call(counter, ctx())).source == "cache"  # still fresh

    clock.advance(10.0)  # entry expired
    fresh = await c.call(counter, ctx())
    assert fresh.source == "live"
    assert fresh.value == 2


async def test_lru_evicts_the_oldest() -> None:
    c = cache(ManualClock(), max_entries=2)
    counter = Counter()

    def k(x: str) -> InvocationContext:
        return ctx(arguments={"k": x})

    await c.call(counter, k("A"))  # {A}
    await c.call(counter, k("B"))  # {A, B}
    await c.call(counter, k("C"))  # inserts C, evicts A -> {B, C}

    assert (await c.call(counter, k("C"))).source == "cache"  # C still there
    evicted = await c.call(counter, k("A"))  # A was evicted -> real call
    assert evicted.source == "live"
    assert counter.calls == 4


async def test_single_flight_coalesces_concurrent_misses() -> None:
    c = cache(ManualClock())

    class Parking:
        def __init__(self) -> None:
            self.calls = 0
            self.open = anyio.Event()

        async def __call__(self, ct: InvocationContext) -> Outcome[Any]:
            self.calls += 1
            await self.open.wait()
            return Outcome(value="v")

    parking = Parking()
    leader_ctx = ctx(arguments={"q": "x"})
    waiter_ctx = ctx(arguments={"q": "x"})
    sources: list[str] = []

    async def run(c_ctx: InvocationContext) -> None:
        sources.append((await c.call(parking, c_ctx)).source)

    async with anyio.create_task_group() as tg:
        tg.start_soon(run, leader_ctx)
        await let_tasks_run()  # ensure the leader is in flight first
        tg.start_soon(run, waiter_ctx)
        await let_tasks_run()
        assert parking.calls == 1  # only the leader hit the base
        parking.open.set()

    assert sorted(sources) == ["cache", "live"]
    assert leader_ctx.bag["cache"] == "miss"
    assert waiter_ctx.bag["cache"] == "coalesced"


async def test_single_flight_propagates_error_without_caching() -> None:
    c = cache(ManualClock(), cache_errors=False)

    class FailParking:
        def __init__(self) -> None:
            self.calls = 0
            self.open = anyio.Event()

        async def __call__(self, ct: InvocationContext) -> Outcome[Any]:
            self.calls += 1
            await self.open.wait()
            raise TransientError("boom")

    fail = FailParking()

    async def run() -> None:
        with pytest.raises(TransientError):
            await c.call(fail, ctx(arguments={"q": "x"}))

    async with anyio.create_task_group() as tg:
        tg.start_soon(run)
        await let_tasks_run()
        tg.start_soon(run)
        await let_tasks_run()
        fail.open.set()

    assert fail.calls == 1  # coalesced: one real attempt, both see the error

    # Nothing was cached, so a later call runs the base again.
    counter = Counter()
    assert (await c.call(counter, ctx(arguments={"q": "x"}))).source == "live"


async def test_negative_caching_off_by_default() -> None:
    c = cache(ManualClock())
    boom = Boom()
    for _ in range(3):
        with pytest.raises(TransientError):
            await c.call(boom, ctx())
    assert boom.calls == 3  # every call retried the base


async def test_negative_caching_when_enabled() -> None:
    clock = ManualClock()
    c = cache(clock, ttl=10.0, cache_errors=True)
    boom = Boom()

    with pytest.raises(TransientError):
        await c.call(boom, ctx())
    hit_ctx = ctx()
    with pytest.raises(TransientError):
        await c.call(boom, hit_ctx)
    assert boom.calls == 1  # the failure was cached
    assert hit_ctx.bag["cache"] == "hit"

    clock.advance(10.0)  # cached error expires
    with pytest.raises(TransientError):
        await c.call(boom, ctx())
    assert boom.calls == 2


async def test_settings_validation() -> None:
    with pytest.raises(ValidationError):
        CacheSettings(ttl=0.0, max_entries=1)
    with pytest.raises(ValidationError):
        CacheSettings(ttl=1.0, max_entries=0)


async def test_factory_default_and_override_scope() -> None:
    clock = ManualClock()
    default = CacheFactory(CacheSettings(ttl=1.0, max_entries=1), clock)
    assert default.state_scope == EMPTY_SCOPE

    scoped = CacheFactory(CacheSettings(ttl=1.0, max_entries=1), clock, state_scope=ScopeSpec(("endpoint",)))
    assert scoped.state_scope.axes == ("endpoint",)

    inst = default.create(())
    counter = Counter()
    assert (await inst.call(counter, ctx())).source == "live"
    assert (await inst.call(counter, ctx())).source == "cache"
