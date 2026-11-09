"""Chain composition: pure fold of interceptors around a base call.

Ordering convention: **the first link in the list is the outermost one** -
it sees the call first and the result last. ``chain([a, b], base)`` runs
``a`` around ``b`` around ``base``.

The chain itself is stateless and built once. What *is* resolved on every
call is the scope: each link's state key is resolved through the axis
registry, and the instance for that key is fetched from the state store.
"""

from collections.abc import Sequence
from functools import reduce
from typing import Any

from warpweft.core.axes import AxisRegistry, ScopeKey
from warpweft.core.context import InvocationContext
from warpweft.core.outcome import Outcome

from .interceptor import Interceptor, InterceptorFactory, Next
from .state import StateStore

#: Default order of the full built-in set, outermost first. Links that do
#: not exist yet are listed deliberately: the order is a contract, fixed
#: before the implementations land. Telemetry is not part of the list -
#: it is always outside and always on.
DEFAULT_ORDER: tuple[str, ...] = ("concurrency", "cache", "circuit_breaker", "retry", "timeout")


def compose(interceptors: Sequence[Interceptor], base: Next) -> Next:
    """Fold pre-resolved link instances around ``base``. First = outermost."""

    def wrap(inner: Next, interceptor: Interceptor) -> Next:
        async def call(ctx: InvocationContext) -> Outcome[Any]:
            return await interceptor.call(inner, ctx)

        return call

    return reduce(wrap, reversed(interceptors), base)


def build_chain(
    factories: Sequence[InterceptorFactory],
    store: StateStore,
    registry: AxisRegistry,
    base: Next,
) -> Next:
    """Build a reusable chain over factories. First factory = outermost link.

    The returned callable resolves each link's scope key and fetches (or
    lazily creates) the instance from the store on every invocation, so one
    chain object serves all slices.

    Store keys are namespaced with the factory uid: different links with
    identical scopes must not collide in a shared store. Empty-scope links
    also include the operation and invocation scope so independently
    configured methods and slices never reuse each other's settings.
    Endpoint-scoped links deliberately omit both to preserve endpoint sharing.
    """

    def bind(inner: Next, factory: InterceptorFactory) -> Next:
        async def call(ctx: InvocationContext) -> Outcome[Any]:
            scope: ScopeKey = registry.resolve(factory.state_scope)
            unit = ("__unit__", factory.identity.uid)
            if factory.state_scope:
                store_key: ScopeKey = (unit, *scope)
            else:
                invocation_scope: ScopeKey = tuple(sorted(ctx.scope_key))
                store_key = (unit, ("__operation__", ctx.operation), *invocation_scope)
            instance = await store.get_or_create(store_key, lambda: factory.create(scope))
            return await instance.call(inner, ctx)

        return call

    return reduce(bind, reversed(factories), base)
