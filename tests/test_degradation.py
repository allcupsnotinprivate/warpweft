"""Degradation link: stub on unavailability, loud on real errors, counted."""

import logging
from typing import Any

from pydantic import ValidationError
import pytest

from warpweft.core.context import InvocationContext
from warpweft.core.errors import (
    CircuitOpen,
    DeadlineExceeded,
    ErrorClass,
    PermanentError,
    RetryExhausted,
    TransientError,
)
from warpweft.core.outcome import Outcome
from warpweft.core.pipeline.builtin.degradation import (
    DegradationInterceptor,
    DegradationSettings,
)
from warpweft.core.pipeline.interceptor import Next

pytestmark = [pytest.mark.unit, pytest.mark.anyio]


def link(stub: Any = None, *, degrade_on: ErrorClass = ErrorClass.TRANSIENT) -> DegradationInterceptor:
    provider = stub if callable(stub) else (lambda ctx: stub)
    return DegradationInterceptor(DegradationSettings(degrade_on=degrade_on), provider)


def ctx(**overrides: Any) -> InvocationContext:
    defaults: dict[str, Any] = {"operation": "op", "correlation_id": "cid"}
    defaults.update(overrides)
    return InvocationContext(**defaults)


def raiser(exc: BaseException) -> Next:
    async def base(c: InvocationContext) -> Outcome[Any]:
        raise exc

    return base


async def test_success_passes_through_untouched() -> None:
    cb = link(stub="fallback")

    async def ok(c: InvocationContext) -> Outcome[Any]:
        return Outcome(value="real")

    outcome = await cb.call(ok, ctx())
    assert outcome.value == "real"
    assert outcome.source == "live"
    assert not outcome.degraded
    assert cb.degraded_count == 0


async def test_transient_failure_is_replaced_by_the_stub() -> None:
    cb = link(stub=[])
    c = ctx()
    outcome = await cb.call(raiser(TransientError("down")), c)
    assert outcome.value == []
    assert outcome.source == "stub"
    assert outcome.degraded is True
    assert cb.degraded_count == 1
    assert c.bag["degraded"] is True


async def test_permanent_failure_is_re_raised_not_stubbed() -> None:
    cb = link(stub="fallback")
    with pytest.raises(PermanentError):
        await cb.call(raiser(PermanentError("bad request")), ctx())
    assert cb.degraded_count == 0


@pytest.mark.parametrize(
    "exc",
    [
        TransientError("x"),
        CircuitOpen("open"),
        DeadlineExceeded("late"),
        RetryExhausted("gave up", attempts=3, last_error=TransientError("y")),
    ],
)
async def test_all_unavailability_signals_degrade(exc: BaseException) -> None:
    cb = link(stub="stub")
    outcome = await cb.call(raiser(exc), ctx())
    assert outcome.source == "stub"
    assert outcome.degraded is True


async def test_stub_provider_receives_context() -> None:
    cb = link(stub=lambda c: f"stub-for-{c.operation}")
    outcome = await cb.call(raiser(TransientError("down")), ctx(operation="search"))
    assert outcome.value == "stub-for-search"


async def test_counter_accumulates_across_calls() -> None:
    cb = link(stub="s")
    for _ in range(3):
        await cb.call(raiser(TransientError("down")), ctx())
    assert cb.degraded_count == 3


async def test_degrade_on_can_target_permanent() -> None:
    cb = link(stub="s", degrade_on=ErrorClass.PERMANENT)
    # A permanent error now degrades...
    assert (await cb.call(raiser(PermanentError("no")), ctx())).source == "stub"
    # ...while a transient one is re-raised.
    with pytest.raises(TransientError):
        await cb.call(raiser(TransientError("down")), ctx())


async def test_base_exceptions_are_never_degraded() -> None:
    cb = link(stub="s")
    with pytest.raises(KeyboardInterrupt):
        await cb.call(raiser(KeyboardInterrupt()), ctx())
    assert cb.degraded_count == 0


async def test_settings_default_degrade_on_is_transient() -> None:
    assert DegradationSettings().degrade_on is ErrorClass.TRANSIENT
    with pytest.raises(ValidationError):
        DegradationSettings(degrade_on="nonsense")  # type: ignore[arg-type]


async def test_stub_exception_propagates_with_original_as_context() -> None:
    def boom(c: InvocationContext) -> Any:
        raise RuntimeError("stub is broken")

    cb = link(stub=boom)
    original = TransientError("down")
    with pytest.raises(RuntimeError) as excinfo:
        await cb.call(raiser(original), ctx())
    assert excinfo.value.__context__ is original
    # The call was counted before the stub ran.
    assert cb.degraded_count == 1


async def test_first_substitution_warns_then_debug(caplog: pytest.LogCaptureFixture) -> None:
    cb = link(stub="s")
    logger_name = "warpweft.core.pipeline.builtin.degradation"
    with caplog.at_level(logging.DEBUG, logger=logger_name):
        await cb.call(raiser(TransientError("down")), ctx())
        await cb.call(raiser(TransientError("down")), ctx())
    records = [r for r in caplog.records if r.name == logger_name]
    assert [r.levelno for r in records] == [logging.WARNING, logging.DEBUG]
