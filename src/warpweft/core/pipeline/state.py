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

    def acquire(self, instance: object) -> None:
        """Pin ``instance`` so eviction defers its ``stop()`` until released."""
        ...

    async def release(self, instance: object) -> None:
        """Drop one pin; stop the instance if it was evicted while pinned."""
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

    In-flight protection: a caller that is actively using an instance can pin it
    with ``acquire``/``release``. An eviction (LRU overflow or explicit evict)
    of a pinned instance drops it from the live set immediately but defers its
    ``stop()`` until the last reference is released, so a call parked on a slice
    never runs on a stopped object. ``close`` stops everything regardless (the
    caller is expected to have drained in-flight work first).
    """

    def __init__(self, max_entries: int = 1000) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._entries: dict[ScopeKey, object] = {}
        #: Live pin counts, keyed by ``id(instance)``: how many callers are
        #: currently using each instance (see ``acquire``/``release``).
        self._refs: dict[int, int] = {}
        #: Instances evicted while still pinned, keyed by ``id(instance)``. They
        #: are already out of ``_entries``; ``stop()`` runs on final release.
        self._pending_stop: dict[int, object] = {}
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
                await self._evict_instance(self._entries.pop(oldest_key))
            return instance

    def acquire(self, instance: object) -> None:
        """Pin ``instance`` against eviction-stop for the length of a call.

        Synchronous on purpose: the caller pins the instance in the same step it
        obtained it from ``get_or_create``, with no intervening checkpoint, so a
        concurrent eviction can never stop it out from under an in-flight call.
        """
        self._refs[id(instance)] = self._refs.get(id(instance), 0) + 1

    async def release(self, instance: object) -> None:
        """Drop one pin; if it was the last and the instance was evicted, stop it."""
        ident = id(instance)
        count = self._refs.get(ident, 0) - 1
        if count > 0:
            self._refs[ident] = count
            return
        self._refs.pop(ident, None)
        deferred = self._pending_stop.pop(ident, None)
        if deferred is not None:
            await self._stop(deferred)

    async def evict(self, key: ScopeKey) -> None:
        async with self._lock:
            instance = self._entries.pop(key, None)
        if instance is not None:
            await self._evict_instance(instance)

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            instances = [*self._entries.values(), *self._pending_stop.values()]
            self._entries.clear()
            self._pending_stop.clear()
            self._refs.clear()
            for instance in instances:
                await self._stop(instance)

    async def _evict_instance(self, instance: object) -> None:
        """Stop an evicted instance now, or defer it if a call still pins it."""
        if self._refs.get(id(instance), 0) > 0:
            self._pending_stop[id(instance)] = instance  # busy: stop on final release
            return
        await self._stop(instance)

    @staticmethod
    async def _stop(instance: object) -> None:
        if isinstance(instance, Stoppable):
            await instance.stop()
