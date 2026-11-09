"""Interceptor protocol and factory.

An interceptor wraps an arbitrary awaitable call. This layer knows nothing
about what is being called - only the context travels through.
"""

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeAlias

from warpweft.core.axes import ScopeKey, ScopeSpec
from warpweft.core.context import InvocationContext
from warpweft.core.outcome import Outcome
from warpweft.core.unit import Identity

#: Continuation of the pipeline: the next link (or the base call itself).
Next: TypeAlias = Callable[[InvocationContext], Awaitable[Outcome[Any]]]


class Interceptor(Protocol):
    """One link of the pipeline.

    Links never reference each other; the only ways to influence the rest of
    the pipeline are calling ``next`` and writing to ``ctx.bag``.
    """

    identity: Identity

    async def call(self, next: Next, ctx: InvocationContext) -> Outcome[Any]:
        """Run the link around ``next``."""
        ...


class InterceptorFactory(Protocol):
    """Creates link instances per scope key.

    The registry stores factories, not instances: one link may have many
    instances, one per slice of its state. Settings are bound to the factory
    at construction time; ``create`` only needs the key.
    """

    identity: Identity

    #: Axes this link's state is sliced along. Empty means one instance per
    #: operation and invocation scope, keeping independently configured calls
    #: isolated in the shared store.
    state_scope: ScopeSpec

    def create(self, key: ScopeKey) -> Interceptor:
        """Build an instance for the given slice."""
        ...
