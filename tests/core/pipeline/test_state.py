"""InMemoryStateStore: single-creation under concurrency, LRU eviction, lifecycle."""

import anyio
import pytest

from warpweft.core.axes import ScopeKey
from warpweft.core.pipeline.state import InMemoryStateStore

pytestmark = [pytest.mark.unit, pytest.mark.anyio]


class Resource:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True


def key(value: str) -> ScopeKey:
    return (("axis", value),)


async def test_concurrent_get_or_create_creates_exactly_one() -> None:
    store = InMemoryStateStore()
    created: list[Resource] = []

    def factory() -> Resource:
        resource = Resource()
        created.append(resource)
        return resource

    results: list[Resource] = []

    async def worker() -> None:
        results.append(await store.get_or_create(key("k"), factory))

    async with anyio.create_task_group() as tg:
        for _ in range(20):
            tg.start_soon(worker)

    assert len(created) == 1
    assert all(r is created[0] for r in results)


async def test_startable_is_started_before_publication() -> None:
    store = InMemoryStateStore()
    resource = await store.get_or_create(key("k"), Resource)
    assert resource.started


async def test_plain_objects_need_no_lifecycle() -> None:
    store = InMemoryStateStore()
    obj = await store.get_or_create(key("k"), object)
    assert await store.get_or_create(key("k"), object) is obj
    await store.close()  # must not blow up on non-Stoppable entries


async def test_lru_evicts_oldest_and_stops_it() -> None:
    store = InMemoryStateStore(max_entries=2)
    first = await store.get_or_create(key("1"), Resource)
    second = await store.get_or_create(key("2"), Resource)
    third = await store.get_or_create(key("3"), Resource)

    assert first.stopped  # oldest is out
    assert not second.stopped
    assert not third.stopped
    assert len(store) == 2


async def test_lru_touch_on_access_changes_eviction_order() -> None:
    store = InMemoryStateStore(max_entries=2)
    first = await store.get_or_create(key("1"), Resource)
    second = await store.get_or_create(key("2"), Resource)

    # Touch "1": now "2" is the least recently used.
    assert await store.get_or_create(key("1"), Resource) is first
    await store.get_or_create(key("3"), Resource)

    assert second.stopped
    assert not first.stopped


async def test_explicit_evict_stops_instance() -> None:
    store = InMemoryStateStore()
    resource = await store.get_or_create(key("k"), Resource)
    await store.evict(key("k"))
    assert resource.stopped
    assert len(store) == 0
    await store.evict(key("k"))  # idempotent


async def test_close_stops_everything_and_forbids_use() -> None:
    store = InMemoryStateStore()
    a = await store.get_or_create(key("a"), Resource)
    b = await store.get_or_create(key("b"), Resource)

    await store.close()
    assert a.stopped and b.stopped
    await store.close()  # idempotent

    with pytest.raises(RuntimeError, match="closed"):
        await store.get_or_create(key("c"), Resource)


async def test_max_entries_validated() -> None:
    with pytest.raises(ValueError, match="positive"):
        InMemoryStateStore(max_entries=0)
