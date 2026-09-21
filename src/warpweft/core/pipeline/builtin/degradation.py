"""Degradation link: substitute a stub when the component is unavailable.

Activated for a component whose ``criticality`` is ``optional`` **and** that
defines a ``stub()`` method, when its ``policy.degradation`` config block is
present. Any other combination is a build-time error, not a silent no-op: a
required component must fail loudly, and degradation without a stub is a dead
config (see ``warpweft.core.composition.wiring.validate_degradation``).

The link is not part of the ordered policy chain. The container and the test
harness wrap it at a fixed position - outside the whole chain (a stub must
never be cached or retried) but inside ``instrument()`` (so the substitution
is counted). Like telemetry, its position is fixed, not listed: it never
appears in ``policy.chain`` or ``explain()``.

Degradation triggers only on *unavailability* - transient failures (the service
is down, retries exhausted, breaker open, deadline gone). A permanent error
(a bad request, a validation failure) is a real error and is re-raised: masking
it with a stub would hide a bug. If the stub itself raises, that exception
propagates (with the original failure as ``__context__``) - a broken stub must
be loud, never masked by a second fallback.

The result is never substituted silently: a stubbed outcome carries
``source = "stub"`` and ``degraded = True``, the substitution is counted for
metrics (not only reflected in health), marked in ``ctx.bag``, and logged.
"""

from collections.abc import Callable
import logging
from typing import Any

from pydantic import BaseModel, Field

from warpweft.core.context import InvocationContext
from warpweft.core.errors import DefaultErrorClassifier, ErrorClass, ErrorClassifier
from warpweft.core.observe import FACT_DEGRADED
from warpweft.core.outcome import Outcome
from warpweft.core.pipeline.interceptor import Next
from warpweft.core.unit import Identity

logger = logging.getLogger(__name__)

#: Produces the fallback value for a degraded call. Receives the context so the
#: stub can depend on the operation or arguments.
StubProvider = Callable[[InvocationContext], Any]


class DegradationSettings(BaseModel):
    """Settings of the degradation link."""

    degrade_on: ErrorClass = Field(
        default=ErrorClass.TRANSIENT,
        description="Error class treated as unavailability and replaced by the stub",
    )


class DegradationInterceptor:
    """Falls back to a stub on unavailability, counting each substitution."""

    settings_model: type[BaseModel] | None = DegradationSettings

    def __init__(
        self,
        settings: DegradationSettings,
        stub: StubProvider,
        classifier: ErrorClassifier | None = None,
        identity: Identity | None = None,
    ) -> None:
        self.identity = identity or Identity.of("degradation")
        self._settings = settings
        self._stub = stub
        self._classifier = classifier or DefaultErrorClassifier()
        self._degraded_count = 0

    @property
    def degraded_count(self) -> int:
        """How many calls this instance has degraded (for metrics)."""
        return self._degraded_count

    async def call(self, next: Next, ctx: InvocationContext) -> Outcome[object]:
        try:
            return await next(ctx)
        except Exception as exc:
            if self._classifier.classify(exc) != self._settings.degrade_on:
                raise  # not unavailability: a real error must surface
            self._degraded_count += 1
            if self._degraded_count == 1:
                logger.warning("operation %r degraded to stub: %s", ctx.operation, exc)
            else:
                logger.debug("operation %r degraded to stub", ctx.operation)
            ctx.bag[FACT_DEGRADED] = True
            # A raising stub propagates with `exc` as its __context__ (inside
            # `except`): a broken stub must be loud, not masked.
            return Outcome(value=self._stub(ctx), source="stub", degraded=True)
