"""LLM-facing tool errors: tell the model whether a retry can help.

A bare exception string answers "what happened" but not "what should the
caller do next". Every tool error therefore carries machine-readable meta -
``warpweft.error`` (a stable code) and ``warpweft.retryable`` - plus a
one-sentence hint appended to the message, so a model can choose between
retrying, backing off and giving up without guessing from prose.

The codes mirror the framework error taxonomy; anything unclassified is
reported as non-retryable ``error``, the same stance `DefaultErrorClassifier`
takes (retrying the unknown means retrying bugs).
"""

from typing import Any

import mcp.types as mt
from pydantic import ValidationError

from warpweft.core.errors import (
    AttemptTimeout,
    CircuitOpen,
    ComponentUnavailable,
    DeadlineExceeded,
    PermanentError,
    RetryExhausted,
    TransientError,
)


def _classify(exc: BaseException) -> tuple[str, bool, str]:
    """Return (code, retryable, hint) for an exception; hint may be empty."""
    if isinstance(exc, ValidationError):
        return "invalid_arguments", True, "Fix the arguments and call the tool again."
    if isinstance(exc, CircuitOpen):
        if exc.retry_after is not None:
            return "circuit_open", True, f"Do not retry for ~{exc.retry_after:.1f}s; the circuit breaker is open."
        return "circuit_open", True, "Do not retry immediately; wait for the circuit breaker to allow a probe."
    if isinstance(exc, (DeadlineExceeded, AttemptTimeout)):
        return "timeout", True, "The call timed out; a retry may succeed."
    if isinstance(exc, RetryExhausted):
        return "retry_exhausted", True, f"{exc.attempts} attempts already failed; wait before retrying."
    if isinstance(exc, ComponentUnavailable):
        return "unavailable", True, "The component is not available right now; retry later."
    if isinstance(exc, TransientError):
        return "transient", True, "This failure is temporary; it is safe to retry."
    if isinstance(exc, PermanentError):
        return "permanent", False, "Do not retry with the same arguments."
    return "error", False, ""


def error_metadata(exc: BaseException) -> dict[str, Any]:
    """Classify ``exc`` into the metadata shared by tool and protocol errors."""
    code, retryable, _ = _classify(exc)
    meta: dict[str, Any] = {"warpweft.error": code, "warpweft.retryable": retryable}
    if isinstance(exc, CircuitOpen) and exc.retry_after is not None:
        meta["warpweft.retry_after_s"] = round(exc.retry_after, 3)
    if isinstance(exc, RetryExhausted):
        meta["warpweft.attempts"] = exc.attempts
    return meta


def error_result(exc: BaseException) -> mt.CallToolResult:
    """Map an exception to a tool error result with retry guidance in meta."""
    code, _, hint = _classify(exc)
    meta = error_metadata(exc)
    text = f"{exc}\n{hint}" if hint else str(exc)
    return mt.CallToolResult(content=[mt.TextContent(type="text", text=text)], is_error=True, meta=meta)
