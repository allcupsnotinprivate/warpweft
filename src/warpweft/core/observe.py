"""Observation seam between pipeline links and telemetry.

Links never import a telemetry backend. Instead, an `Observer` may be
installed into the context ``bag`` (the telemetry wrapper does this), and links
emit spans and events through it via `observer_of`. When no observer is
installed, everything degrades to a no-op, so links behave identically with
telemetry absent.

Contract for observer implementations: methods must be synchronous,
non-blocking and must never raise - links emit from latency-sensitive spots,
including under locks.
"""

from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias, cast

if TYPE_CHECKING:
    from .context import InvocationContext

#: Values an observer attribute may carry (matches OTel's scalar attributes).
AttributeValue: TypeAlias = str | bool | int | float

#: Reserved ``bag`` key the active observer is installed under.
OBSERVER_KEY: Final = "__observer__"

#: Name of the child span wrapping one retry attempt.
SPAN_ATTEMPT: Final = "warpweft.attempt"
#: Attempt number attribute (on attempt spans and backoff events).
ATTR_ATTEMPT_NUMBER: Final = "warpweft.attempt.number"
#: Event emitted right before a retry backoff sleep.
EVENT_RETRY_BACKOFF: Final = "warpweft.retry.backoff"
#: Backoff delay attribute, seconds.
ATTR_BACKOFF_DELAY: Final = "warpweft.backoff.delay"
#: Event emitted when the circuit breaker rejects a call.
EVENT_BREAKER_REJECTED: Final = "warpweft.circuit_breaker.rejected"
#: Breaker state attribute on rejection events ("open" | "half_open").
ATTR_BREAKER_STATE: Final = "warpweft.circuit_breaker.state"
#: Event emitted when the circuit breaker changes state during a call.
EVENT_BREAKER_TRANSITION: Final = "warpweft.circuit_breaker.transition"
#: State a transition moved from / to ("closed" | "open" | "half_open").
ATTR_BREAKER_STATE_FROM: Final = "warpweft.circuit_breaker.state.from"
ATTR_BREAKER_STATE_TO: Final = "warpweft.circuit_breaker.state.to"

#: Bag fact written by the cache link: "hit" | "miss" | "coalesced".
FACT_CACHE: Final = "cache"
#: Bag fact written by the degradation link when a stub was substituted.
FACT_DEGRADED: Final = "degraded"


class Observer(Protocol):
    """Sink for spans and events emitted by pipeline links."""

    def event(self, name: str, attributes: Mapping[str, AttributeValue] | None = None) -> None:
        """Record a point-in-time event."""
        ...

    def span(self, name: str, attributes: Mapping[str, AttributeValue] | None = None) -> AbstractContextManager[object]:
        """Open a child span for the duration of the ``with`` block."""
        ...


class NullObserver:
    """Observer that records nothing; the default when telemetry is absent."""

    def event(self, name: str, attributes: Mapping[str, AttributeValue] | None = None) -> None:
        return None

    def span(self, name: str, attributes: Mapping[str, AttributeValue] | None = None) -> AbstractContextManager[object]:
        return nullcontext()


NULL_OBSERVER: Final = NullObserver()


def observer_of(ctx: "InvocationContext") -> Observer:
    """Return the observer installed in ``ctx.bag``, or the no-op one."""
    return cast("Observer", ctx.bag.get(OBSERVER_KEY, NULL_OBSERVER))
