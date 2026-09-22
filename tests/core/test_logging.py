"""Library logging: NullHandler default, lifecycle log lines, correlation filter."""

import logging

import pytest

from warpweft.core.clock import ManualClock
from warpweft.core.component import AComponent, Criticality, EmptySettings, invocable
from warpweft.core.composition import Container, Registry
from warpweft.core.context import InvocationContext, use_correlation_id
from warpweft.core.errors import TransientError
from warpweft.core.logging import CorrelationIdFilter
from warpweft.core.outcome import Outcome
from warpweft.core.pipeline.builtin.circuit_breaker import CircuitBreakerInterceptor, CircuitBreakerSettings

pytestmark = [pytest.mark.unit, pytest.mark.anyio]


def test_warpweft_logger_has_a_null_handler() -> None:
    handlers = logging.getLogger("warpweft").handlers
    assert any(isinstance(h, logging.NullHandler) for h in handlers)


def test_module_loggers_are_under_the_warpweft_hierarchy() -> None:
    from warpweft.core.composition import container

    assert container.logger.name == "warpweft.core.composition.container"


class Worker(AComponent[EmptySettings, None, str]):
    @invocable
    async def go(self) -> str:
        return "ok"


async def test_container_logs_start_and_stop(caplog: pytest.LogCaptureFixture) -> None:
    reg = Registry()
    reg.register(Worker)
    container = Container.build(reg, {"worker": {}})
    with caplog.at_level(logging.INFO, logger="warpweft"):
        await container.start()
        await container.stop()
    messages = [r.getMessage() for r in caplog.records]
    assert any("container started" in m for m in messages)
    assert any("container stopping" in m for m in messages)


async def test_degraded_optional_component_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    class Fragile(AComponent[EmptySettings, None, None]):
        criticality = Criticality.OPTIONAL

        async def start(self) -> None:
            raise RuntimeError("boom")

        @invocable
        async def go(self) -> None: ...

    reg = Registry()
    reg.register(Fragile)
    container = Container.build(reg, {"fragile": {}})
    with caplog.at_level(logging.WARNING, logger="warpweft"):
        await container.start()
    assert any("degraded" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
    await container.stop()


async def test_breaker_open_and_close_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    clock = ManualClock()
    breaker = CircuitBreakerInterceptor(CircuitBreakerSettings(window=1, failure_threshold=1, reset_timeout=5.0), clock)

    async def boom(c: InvocationContext) -> Outcome[object]:
        raise TransientError("down")

    async def ok(c: InvocationContext) -> Outcome[object]:
        return Outcome(value="ok")

    ctx = InvocationContext(operation="op", correlation_id="cid")
    with caplog.at_level(logging.INFO, logger="warpweft"):
        with pytest.raises(TransientError):
            await breaker.call(boom, ctx)  # opens
        clock.advance(5.0)
        await breaker.call(ok, ctx)  # probe succeeds -> closes
    messages = [r.getMessage() for r in caplog.records]
    assert any("opened" in m for m in messages)
    assert any("closed" in m for m in messages)


def test_correlation_id_filter_stamps_records() -> None:
    filt = CorrelationIdFilter()
    record = logging.LogRecord("warpweft", logging.INFO, __file__, 1, "msg", (), None)

    assert filt.filter(record) is True
    assert record.correlation_id == ""  # unset -> empty

    with use_correlation_id("req-7"):
        assert filt.filter(record) is True
        assert record.correlation_id == "req-7"
