"""Circuit breaker link: protect a failing external system from more load.

Sliced by ``[endpoint]`` - the breaker guards the *called* system, not the
caller, so all callers of one endpoint share a breaker. A count-based sliding
window (simpler and more predictable than a time window) decides when to trip.

Rules that matter:

- Only transient failures count. A revoked key giving 401 to one caller is
  permanent and must not open the breaker for everyone else.
- In half-open exactly one probe is let through; the rest are rejected at once,
  so a recovering service is not hit by the whole backlog.
- Rejection in the open state is a ``CircuitOpen`` (a TransientError), so an
  outer retry, if present, handles it like any other transient failure.

The breaker sits outside retry in the default order, so it counts one logical
call (after retries), not one attempt.
"""

from collections import deque
from enum import StrEnum
import logging

import anyio
from pydantic import BaseModel, Field, model_validator

from warpweft.core.axes import ScopeKey, ScopeSpec
from warpweft.core.clock import Clock
from warpweft.core.context import InvocationContext
from warpweft.core.errors import CircuitOpen, DefaultErrorClassifier, ErrorClass, ErrorClassifier
from warpweft.core.observe import (
    ATTR_BREAKER_STATE,
    ATTR_BREAKER_STATE_FROM,
    ATTR_BREAKER_STATE_TO,
    EVENT_BREAKER_REJECTED,
    EVENT_BREAKER_TRANSITION,
    observer_of,
)
from warpweft.core.outcome import Outcome
from warpweft.core.pipeline.interceptor import Interceptor, Next
from warpweft.core.unit import Identity

logger = logging.getLogger(__name__)

#: The breaker's state is sliced per endpoint. The axis itself is registered
#: where the pipeline is wired to components; the slice is declared here.
ENDPOINT_SCOPE = ScopeSpec(("endpoint",))


class CircuitState(StrEnum):
    """The three breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerSettings(BaseModel):
    """Settings of the circuit breaker link."""

    window: int = Field(ge=1, description="Number of most recent calls the failure count looks at")
    failure_threshold: int = Field(ge=1, description="Transient failures within the window that trip the breaker")
    reset_timeout: float = Field(gt=0, description="Seconds the breaker stays open before allowing a probe")

    @model_validator(mode="after")
    def _threshold_fits_window(self) -> "CircuitBreakerSettings":
        if self.failure_threshold > self.window:
            raise ValueError("failure_threshold must not exceed window")
        return self


class CircuitBreakerInterceptor:
    """One breaker instance, shared by every concurrent call of its slice."""

    settings_model: type[BaseModel] | None = CircuitBreakerSettings

    def __init__(
        self,
        settings: CircuitBreakerSettings,
        clock: Clock,
        classifier: ErrorClassifier | None = None,
        identity: Identity | None = None,
    ) -> None:
        self.identity = identity or Identity.of("circuit_breaker")
        self._settings = settings
        self._clock = clock
        self._classifier = classifier or DefaultErrorClassifier()
        self._lock = anyio.Lock()
        self._state = CircuitState.CLOSED
        #: Recent outcomes, True = transient failure. Bounded to the window.
        self._window: deque[bool] = deque(maxlen=settings.window)
        self._opened_at = 0.0
        self._probe_in_flight = False

    @property
    def state(self) -> CircuitState:
        """Current state (for tests and future introspection)."""
        return self._state

    async def call(self, next: Next, ctx: InvocationContext) -> Outcome[object]:
        await self._admit(ctx)
        try:
            outcome = await next(ctx)
        except BaseException as exc:
            await self._on_error(exc, ctx)
            raise
        await self._on_success(ctx)
        return outcome

    async def force_open(self) -> None:
        """Manually trip the breaker: OPEN now, a probe after ``reset_timeout``.

        Unconditional - re-arms the reset timer when already open - and clears
        the window and any probe slot. Logged, but emits no transition metric:
        there is no invocation to attribute the change to.
        """
        async with self._lock:
            self._open(None)

    async def reset(self) -> None:
        """Manually close the breaker, forgetting recorded failures.

        Idempotent: an already-closed breaker just clears its window. Logged,
        but emits no transition metric (no invocation to attribute it to).
        """
        async with self._lock:
            if self._state is CircuitState.CLOSED:
                self._window.clear()
                self._probe_in_flight = False
                return
            self._close(None)

    async def _admit(self, ctx: InvocationContext) -> None:
        """Decide whether this call may proceed; may flip open -> half-open."""
        async with self._lock:
            if self._state is CircuitState.OPEN:
                elapsed = self._clock.monotonic() - self._opened_at
                if elapsed >= self._settings.reset_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._probe_in_flight = True  # this call is the probe
                    self._emit_transition(ctx, CircuitState.OPEN, CircuitState.HALF_OPEN)
                    return
                observer_of(ctx).event(EVENT_BREAKER_REJECTED, {ATTR_BREAKER_STATE: "open"})
                raise CircuitOpen(
                    f"circuit for '{ctx.operation}' is open",
                    retry_after=self._settings.reset_timeout - elapsed,
                )
            if self._state is CircuitState.HALF_OPEN:
                if self._probe_in_flight:
                    observer_of(ctx).event(EVENT_BREAKER_REJECTED, {ATTR_BREAKER_STATE: "half_open"})
                    raise CircuitOpen(f"circuit for '{ctx.operation}' is half-open; a probe is already in flight")
                self._probe_in_flight = True
                return
            # CLOSED: proceed normally.

    async def _on_success(self, ctx: InvocationContext) -> None:
        async with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                self._close(ctx)
            elif self._state is CircuitState.CLOSED:
                self._window.append(False)

    async def _on_error(self, exc: BaseException, ctx: InvocationContext) -> None:
        transient = self._classifier.classify(exc) == ErrorClass.TRANSIENT
        async with self._lock:
            if not transient:
                # Permanent failures (and cancellation) do not reflect on the
                # external system; release the probe slot but change nothing.
                if self._state is CircuitState.HALF_OPEN:
                    self._probe_in_flight = False
                return
            if self._state is CircuitState.HALF_OPEN:
                self._open(ctx)
            elif self._state is CircuitState.CLOSED:
                self._window.append(True)
                if sum(self._window) >= self._settings.failure_threshold:
                    self._open(ctx)

    def _open(self, ctx: InvocationContext | None) -> None:
        previous = self._state
        self._state = CircuitState.OPEN
        self._opened_at = self._clock.monotonic()
        self._window.clear()
        self._probe_in_flight = False
        logger.warning("circuit breaker %r opened", self.identity.uid)
        self._emit_transition(ctx, previous, CircuitState.OPEN)

    def _close(self, ctx: InvocationContext | None) -> None:
        previous = self._state
        self._state = CircuitState.CLOSED
        self._window.clear()
        self._probe_in_flight = False
        logger.info("circuit breaker %r closed", self.identity.uid)
        self._emit_transition(ctx, previous, CircuitState.CLOSED)

    def _emit_transition(self, ctx: InvocationContext | None, from_state: CircuitState, to_state: CircuitState) -> None:
        """Emit a transition event when the change happens inside a call.

        Manual transitions (``force_open``/``reset``) pass ``ctx=None``: there
        is no invocation to attribute the event to, so nothing is emitted and
        the logger calls in ``_open``/``_close`` remain the only manual signal.
        """
        if ctx is None:
            return
        observer_of(ctx).event(
            EVENT_BREAKER_TRANSITION,
            {ATTR_BREAKER_STATE_FROM: from_state.value, ATTR_BREAKER_STATE_TO: to_state.value},
        )


class CircuitBreakerFactory:
    """Factory of the breaker link. One instance per endpoint slice."""

    settings_model: type[BaseModel] | None = CircuitBreakerSettings
    state_scope: ScopeSpec = ENDPOINT_SCOPE

    def __init__(
        self,
        settings: CircuitBreakerSettings,
        clock: Clock,
        classifier: ErrorClassifier | None = None,
    ) -> None:
        self.identity = Identity.of("circuit_breaker")
        self._settings = settings
        self._clock = clock
        self._classifier = classifier

    def create(self, key: ScopeKey) -> Interceptor:
        return CircuitBreakerInterceptor(self._settings, self._clock, self._classifier, identity=self.identity)
