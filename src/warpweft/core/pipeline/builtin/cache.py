"""Cache link: in-memory LRU + TTL with single-flight.

Sliced per operation and instance by default so isolated consumers never share
entries or policy settings; a shared cache is an explicit choice. The cache key is
``operation + normalized arguments + ScopeKey`` - the ScopeKey in the key means
even a shared store cannot leak between slices, and argument normalization goes
through pydantic serialization so it is deterministic regardless of field order.

Requirements handled here:

- **Single-flight**: concurrent misses on one key make a single real call; the
  rest await its result, so an expired popular key does not cause an avalanche.
- A cache hit sets ``Outcome.source = "cache"`` and records the fact in
  ``ctx.bag`` for telemetry.
- Negative caching (caching failures) is **off** by default and opt-in.

Only in-memory LRU + TTL lives in the core; a Redis backend is a separate
package.
"""

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Literal

import anyio
from pydantic import BaseModel, Field
from pydantic_core import to_json

from warpweft.core.axes import EMPTY_SCOPE, ScopeKey, ScopeSpec
from warpweft.core.clock import Clock
from warpweft.core.context import InvocationContext
from warpweft.core.observe import FACT_CACHE
from warpweft.core.outcome import Outcome
from warpweft.core.pipeline.interceptor import Interceptor, Next
from warpweft.core.unit import Identity

#: Cache key: (operation, normalized arguments, scope key).
CacheKey = tuple[str, str, ScopeKey]
_EntryKind = Literal["value", "error"]


class CacheSettings(BaseModel):
    """Settings of the cache link."""

    ttl: float = Field(gt=0, description="Time-to-live of an entry, seconds")
    max_entries: int = Field(ge=1, description="LRU capacity; the oldest entry is evicted past this")
    cache_errors: bool = Field(default=False, description="Also cache failures (negative caching)")


class _Entry:
    """A stored value or error with its expiry."""

    __slots__ = ("expires_at", "kind", "payload")

    def __init__(self, kind: _EntryKind, payload: Any, expires_at: float) -> None:
        self.kind = kind
        self.payload = payload
        self.expires_at = expires_at


class _InFlight:
    """Single-flight coordination for one in-progress key."""

    __slots__ = ("done", "error", "ok", "value")

    def __init__(self) -> None:
        self.done = anyio.Event()
        self.value: Any = None
        self.error: BaseException | None = None
        self.ok = False


class CacheInterceptor:
    """One in-memory LRU+TTL cache with single-flight coalescing."""

    settings_model: type[BaseModel] | None = CacheSettings

    def __init__(self, settings: CacheSettings, clock: Clock, identity: Identity | None = None) -> None:
        self.identity = identity or Identity.of("cache")
        self._settings = settings
        self._clock = clock
        self._store: OrderedDict[CacheKey, _Entry] = OrderedDict()
        self._inflight: dict[CacheKey, _InFlight] = {}

    @staticmethod
    def _key(ctx: InvocationContext) -> CacheKey:
        args: Mapping[str, Any] = ctx.arguments or {}
        ordered = {name: args[name] for name in sorted(args)}
        return (ctx.operation, to_json(ordered).decode(), ctx.scope_key)

    def _get(self, key: CacheKey, now: float) -> _Entry | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.expires_at <= now:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return entry

    def _put(self, key: CacheKey, kind: _EntryKind, payload: Any, now: float) -> None:
        self._store[key] = _Entry(kind, payload, now + self._settings.ttl)
        self._store.move_to_end(key)
        while len(self._store) > self._settings.max_entries:
            self._store.popitem(last=False)

    async def call(self, next: Next, ctx: InvocationContext) -> Outcome[object]:
        key = self._key(ctx)
        while True:
            entry = self._get(key, self._clock.monotonic())
            if entry is not None:
                ctx.bag[FACT_CACHE] = "hit"
                if entry.kind == "error":
                    raise entry.payload
                return Outcome(value=entry.payload, source="cache")

            inflight = self._inflight.get(key)
            if inflight is None:
                break  # no await since the miss check: we become the leader

            ctx.bag[FACT_CACHE] = "coalesced"
            await inflight.done.wait()
            if inflight.error is not None:
                raise inflight.error
            if inflight.ok:
                return Outcome(value=inflight.value, source="cache")
            # The leader ended without a usable result (e.g. cancelled): retry.

        return await self._lead(next, ctx, key)

    async def _lead(self, next: Next, ctx: InvocationContext, key: CacheKey) -> Outcome[object]:
        inflight = _InFlight()
        self._inflight[key] = inflight
        ctx.bag[FACT_CACHE] = "miss"
        try:
            outcome = await next(ctx)
        except Exception as exc:
            inflight.error = exc
            if self._settings.cache_errors:
                self._put(key, "error", exc, self._clock.monotonic())
            raise
        else:
            self._put(key, "value", outcome.value, self._clock.monotonic())
            inflight.value = outcome.value
            inflight.ok = True
            return outcome
        finally:
            self._inflight.pop(key, None)
            inflight.done.set()


class CacheFactory:
    """Factory of the cache link.

    ``state_scope`` defaults to the empty scope: the chain isolates each
    operation and invocation scope, and the ScopeKey inside every cache key
    prevents entry leaks. Pass an endpoint/instance scope to share the cache
    across operations and invocation slices, or keep the default for isolation.
    """

    settings_model: type[BaseModel] | None = CacheSettings

    def __init__(self, settings: CacheSettings, clock: Clock, state_scope: ScopeSpec = EMPTY_SCOPE) -> None:
        self.identity = Identity.of("cache")
        self.state_scope = state_scope
        self._settings = settings
        self._clock = clock

    def create(self, key: ScopeKey) -> Interceptor:
        return CacheInterceptor(self._settings, self._clock, identity=self.identity)
