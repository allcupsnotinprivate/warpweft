"""Semantic conventions: the fixed telemetry names contract.

Every span, event, attribute and metric name lives here (link-emitted names
are re-exported from ``warpweft.core.observe``). These names are a public
contract - dashboards and alerts are built on them, so renaming any of them
is a breaking change. See ``docs/telemetry.md``.

This module is deliberately OTel-import-free: pure constants.
"""

from typing import Final

from warpweft.core.observe import (
    ATTR_ATTEMPT_NUMBER,
    ATTR_BACKOFF_DELAY,
    ATTR_BREAKER_STATE,
    ATTR_BREAKER_STATE_FROM,
    ATTR_BREAKER_STATE_TO,
    EVENT_BREAKER_REJECTED,
    EVENT_BREAKER_TRANSITION,
    EVENT_RETRY_BACKOFF,
    FACT_CACHE,
    FACT_DEGRADED,
    SPAN_ATTEMPT,
)

__all__ = [
    "ATTR_ATTEMPTS",
    "ATTR_ATTEMPT_NUMBER",
    "ATTR_BACKOFF_DELAY",
    "ATTR_BREAKER_STATE",
    "ATTR_BREAKER_STATE_FROM",
    "ATTR_BREAKER_STATE_TO",
    "ATTR_CACHE",
    "ATTR_CORRELATION_ID",
    "ATTR_DEGRADED",
    "ATTR_ERROR_CLASS",
    "ATTR_OPERATION",
    "ATTR_SOURCE",
    "ATTR_STATUS",
    "AXIS_ATTR_PREFIX",
    "EVENT_BREAKER_REJECTED",
    "EVENT_BREAKER_TRANSITION",
    "EVENT_RETRY_BACKOFF",
    "FACT_CACHE",
    "FACT_DEGRADED",
    "INSTRUMENTATION_NAME",
    "METRIC_BREAKER_REJECTIONS",
    "METRIC_BREAKER_TRANSITIONS",
    "METRIC_CALLS",
    "METRIC_DEGRADATIONS",
    "METRIC_DURATION",
    "SPAN_ATTEMPT",
    "STATUS_ERROR",
    "STATUS_OK",
    "UNIT_CALLS",
    "UNIT_REJECTIONS",
    "UNIT_SECONDS",
    "UNIT_TRANSITIONS",
]

#: Tracer and meter instrumentation name.
INSTRUMENTATION_NAME: Final = "warpweft"

# --- span / metric attribute keys -------------------------------------------
ATTR_OPERATION: Final = "warpweft.operation"
ATTR_CORRELATION_ID: Final = "warpweft.correlation_id"
ATTR_SOURCE: Final = "warpweft.source"
ATTR_DEGRADED: Final = "warpweft.degraded"
ATTR_ATTEMPTS: Final = "warpweft.attempts"
ATTR_STATUS: Final = "warpweft.status"
ATTR_ERROR_CLASS: Final = "warpweft.error.class"
ATTR_CACHE: Final = "warpweft.cache"
#: Axis attributes are ``warpweft.axis.<axis name>`` = axis value.
AXIS_ATTR_PREFIX: Final = "warpweft.axis."

#: Values of `ATTR_STATUS`.
STATUS_OK: Final = "ok"
STATUS_ERROR: Final = "error"

# --- metric names and units --------------------------------------------------
METRIC_CALLS: Final = "warpweft.calls"
METRIC_DURATION: Final = "warpweft.call.duration"
METRIC_DEGRADATIONS: Final = "warpweft.degradations"
METRIC_BREAKER_REJECTIONS: Final = "warpweft.circuit_breaker.rejections"
METRIC_BREAKER_TRANSITIONS: Final = "warpweft.circuit_breaker.transitions"

UNIT_CALLS: Final = "{call}"
UNIT_SECONDS: Final = "s"
UNIT_REJECTIONS: Final = "{rejection}"
UNIT_TRANSITIONS: Final = "{transition}"
