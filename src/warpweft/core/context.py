"""InvocationContext: the single channel interceptors communicate through.

Interceptors never hold references to each other; everything they need to
agree on (deadline, attempt number, shared facts) travels in the context.

!!! warning
    The current-context contextvar does not survive ``run_in_executor`` or
    manually spawned threads - contextvars are copied at task creation, not
    shared. Pass the context explicitly when crossing thread boundaries.
"""

from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any
import uuid

from .axes import GLOBAL_SCOPE, ScopeKey
from .clock import Clock


@dataclass(frozen=True, slots=True)
class InvocationContext:
    """Immutable description of one invocation.

    Interceptors must not mutate the context; the only mutable part is
    ``bag``, a scratch space for exchanging facts between links. A new
    attempt is a new object created via `child`.
    """

    operation: str
    correlation_id: str
    deadline: float | None = None
    attempt: int = 1
    scope_key: ScopeKey = GLOBAL_SCOPE
    #: Bound call arguments (parameter name -> value), the input a link such as
    #: cache keys on. ``None`` until a caller populates it; the mapping itself
    #: is read-only to links.
    arguments: Mapping[str, Any] | None = None
    #: Clock that interprets ``deadline``. Optional so a plain context still
    #: works, but when set it lets the base call and links read the remaining
    #: budget without threading a clock through by hand.
    clock: Clock | None = None
    bag: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def begin(
        cls,
        operation: str,
        *,
        parent: "InvocationContext | None" = None,
        correlation_id: str | None = None,
        clock: Clock | None = None,
        budget: float | None = None,
        **fields: Any,
    ) -> "InvocationContext":
        """Open a context for a new invocation, resolving the shared entry policy.

        The single place every invocation builder goes through, so the id,
        clock and deadline rules cannot drift between the call paths:

        - **correlation id**: explicit → parent's → ambient (`use_correlation_id`)
          → a fresh uuid.
        - **clock**: explicit → parent's.
        - **deadline**: derived from ``budget`` (relative seconds) via the
          resolved clock; unset when there is no budget. The clock is retained so
          `remaining` and `expired` can be called without one.
        """
        resolved_clock = clock or (parent.clock if parent is not None else None)
        resolved_id = (
            correlation_id
            or (parent.correlation_id if parent is not None else None)
            or current_correlation_id()
            or uuid.uuid4().hex
        )
        deadline = None
        if budget is not None:
            if resolved_clock is None:
                raise ValueError("a budget needs a clock to derive the deadline")
            deadline = resolved_clock.monotonic() + budget
        return cls(
            operation=operation,
            correlation_id=resolved_id,
            deadline=deadline,
            clock=resolved_clock,
            **fields,
        )

    @classmethod
    def start(
        cls,
        operation: str,
        correlation_id: str,
        *,
        clock: Clock,
        budget: float | None = None,
        **fields: Any,
    ) -> "InvocationContext":
        """Build a fresh context from an explicit id and clock (see `begin`).

        A thin wrapper over `begin` for callers that already hold both: the
        resolution chain is a no-op here, but routing through one factory keeps
        the deadline-from-``budget`` rule in a single place.
        """
        return cls.begin(operation, correlation_id=correlation_id, clock=clock, budget=budget, **fields)

    def _clock(self, clock: Clock | None) -> Clock:
        chosen = clock if clock is not None else self.clock
        if chosen is None:
            raise ValueError("no clock available: pass one or build the context with a clock")
        return chosen

    def remaining(self, clock: Clock | None = None) -> float | None:
        """Seconds left until the deadline; ``None`` when there is no deadline.

        Uses the context's own clock when ``clock`` is omitted. Never negative:
        an expired deadline yields ``0.0``.
        """
        if self.deadline is None:
            return None
        return max(self.deadline - self._clock(clock).monotonic(), 0.0)

    def expired(self, clock: Clock | None = None) -> bool:
        """Whether the overall deadline has already passed.

        Uses the context's own clock when ``clock`` is omitted.
        """
        return self.deadline is not None and self._clock(clock).monotonic() >= self.deadline

    def child(self, **overrides: Any) -> "InvocationContext":
        """Derive a context for a new attempt (or any other override).

        ``bag`` is shared with the parent by default - it is the exchange
        channel, not per-attempt state. Override it explicitly if isolation
        is needed.
        """
        return replace(self, **overrides)


_current: ContextVar[InvocationContext | None] = ContextVar("warpweft_current_context", default=None)


def current_context() -> InvocationContext | None:
    """Return the context of the invocation the caller is running inside, if any."""
    return _current.get()


@contextmanager
def use_context(ctx: InvocationContext) -> Iterator[InvocationContext]:
    """Install ``ctx`` as the current context for the duration of the block."""
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)


_correlation_id: ContextVar[str | None] = ContextVar("warpweft_correlation_id", default=None)


def current_correlation_id() -> str | None:
    """Return the ambient correlation id, if one is bound (see `use_correlation_id`)."""
    return _correlation_id.get()


@contextmanager
def use_correlation_id(correlation_id: str) -> Iterator[str]:
    """Bind a correlation id so calls in this block inherit it without passing it.

    A caller (a web request, a worker task) sets it once; every invocation made
    inside the block uses it unless one is passed explicitly.
    """
    token = _correlation_id.set(correlation_id)
    try:
        yield correlation_id
    finally:
        _correlation_id.reset(token)


#: Where progress reports go: an async ``(progress, total, message)`` callable.
ProgressSink = Callable[[float, "float | None", "str | None"], Awaitable[None]]

_progress_sink: ContextVar[ProgressSink | None] = ContextVar("warpweft_progress_sink", default=None)


@contextmanager
def use_progress_sink(sink: ProgressSink) -> Iterator[ProgressSink]:
    """Install a progress sink for the duration of the block.

    A *transport* (an MCP server, an HTTP handler) sets this around an
    invocation so `report_progress` calls made inside reach the caller.
    Component code never installs a sink - it only reports.
    """
    token = _progress_sink.set(sink)
    try:
        yield sink
    finally:
        _progress_sink.reset(token)


async def report_progress(progress: float, total: float | None = None, message: str | None = None) -> None:
    """Report progress of the current invocation to whoever is listening.

    Call this from a long-running invocable; it is a no-op unless the
    transport driving the call installed a sink (see `use_progress_sink`),
    so components stay transport-agnostic.
    """
    sink = _progress_sink.get()
    if sink is not None:
        await sink(progress, total, message)
