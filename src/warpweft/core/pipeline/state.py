"""StateStore protocol and in-memory LRU implementation.

Link instances live here, one per scope key. The store owns their
lifecycle tail: whatever it evicts, it stops.
"""

from collections.abc import Awaitable, Callable
import inspect
from typing import Protocol, TypeVar, cast

import anyio

from warpweft.core.axes import ScopeKey
from warpweft.core.unit import Startable, Stoppable

T = TypeVar("T")

#: A factory may be synchronous or return an awaitable. An async factory lets a
#: caller defer expensive work (e.g. resolving a component's dependencies) until
#: a cache miss actually requires construction, all under the store's lock.
Factory = Callable[[], T | Awaitable[T]]


class StateStore(Protocol):
    """Keyed storage of link instances with bounded cardinality."""

    async def get_or_create(self, key: ScopeKey, factory: Factory[T]) -> T:
        """Return the instance for ``key``, creating it exactly once."""
        ...

    async def evict(self, key: ScopeKey) -> None:
        """Drop the instance for ``key`` (stopping it if it is Stoppable)."""
        ...

    async def close(self) -> None:
        """Stop and drop everything; the store is unusable afterwards."""
        ...


class InMemoryStateStore:
    """Default store: in-process LRU bounded by ``max_entries``.

    ``max_entries`` is expected to come from the ``max_cardinality`` of the
    axes the state is sliced along (the most conservative of them).

    Concurrency: creation is serialized by a single lock, so two concurrent
    ``get_or_create`` calls with the same key produce exactly one object.
    If the created object is Startable it is started before being published.
    On eviction (LRU overflow, explicit evict, close) Stoppable objects are
    stopped.
    """

    def __init__(self, max_entries: int = 1000) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._entries: dict[ScopeKey, object] = {}
        self._lock = anyio.Lock()
        self._closed = False

    def __len__(self) -> int:
        return len(self._entries)

    def items(self) -> list[tuple[ScopeKey, object]]:
        """Snapshot of the live ``(key, instance)`` pairs (for introspection)."""
        return list(self._entries.items())

    def keys(self) -> list[ScopeKey]:
        """Snapshot of the live keys, most-recently-used last."""
        return list(self._entries)

    async def get_or_create(self, key: ScopeKey, factory: Factory[T]) -> T:
        async with self._lock:
            if self._closed:
                raise RuntimeError("state store is closed")
            if key in self._entries:
                # LRU touch: re-insert to mark as most recently used.
                self._entries[key] = self._entries.pop(key)
                return cast(T, self._entries[key])

            # Only ever runs on a cache miss, under the lock, so a sync-or-async
            # factory's work (dependency resolution) is never wasted on a hit.
            created = factory()
            instance = await created if inspect.isawaitable(created) else created
            if isinstance(instance, Startable):
                await instance.start()
            self._entries[key] = instance

            while len(self._entries) > self._max_entries:
                oldest_key = next(iter(self._entries))
                await self._stop(self._entries.pop(oldest_key))
            return instance

    async def evict(self, key: ScopeKey) -> None:
        async with self._lock:
            instance = self._entries.pop(key, None)
            if instance is not None:
                await self._stop(instance)

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            instances = list(self._entries.values())
            self._entries.clear()
            for instance in instances:
                await self._stop(instance)

    @staticmethod
    async def _stop(instance: object) -> None:
        if isinstance(instance, Stoppable):
            await instance.stop()
