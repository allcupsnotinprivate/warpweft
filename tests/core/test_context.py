"""InvocationContext: deadlines on ManualClock, child immutability, contextvar."""

import pytest

from warpweft.core.clock import ManualClock
from warpweft.core.context import (
    InvocationContext,
    current_context,
    current_correlation_id,
    report_progress,
    use_context,
    use_correlation_id,
    use_progress_sink,
)

pytestmark = pytest.mark.unit


def ctx(**overrides: object) -> InvocationContext:
    defaults: dict[str, object] = {"operation": "op", "correlation_id": "cid"}
    defaults.update(overrides)
    return InvocationContext(**defaults)  # type: ignore[arg-type]


def test_remaining_without_deadline_is_none() -> None:
    clock = ManualClock()
    assert ctx().remaining(clock) is None
    assert not ctx().expired(clock)


def test_remaining_counts_down_on_manual_clock() -> None:
    clock = ManualClock()
    c = ctx(deadline=10.0)
    assert c.remaining(clock) == 10.0
    clock.advance(4)
    assert c.remaining(clock) == 6.0
    assert not c.expired(clock)


def test_expired_deadline_detected_and_remaining_never_negative() -> None:
    clock = ManualClock()
    c = ctx(deadline=5.0)
    clock.advance(7)
    assert c.expired(clock)
    assert c.remaining(clock) == 0.0


def test_deadline_boundary_is_expired() -> None:
    clock = ManualClock()
    c = ctx(deadline=5.0)
    clock.advance(5)
    assert c.expired(clock)


def test_child_does_not_mutate_parent() -> None:
    parent = ctx(attempt=1, deadline=5.0)
    child = parent.child(attempt=2)
    assert parent.attempt == 1
    assert child.attempt == 2
    assert child.deadline == parent.deadline
    assert child.operation == parent.operation


def test_context_is_immutable() -> None:
    with pytest.raises(AttributeError):
        ctx().attempt = 5  # type: ignore[misc]


def test_bag_is_shared_between_parent_and_child() -> None:
    """The bag is the exchange channel between links, not per-attempt state."""
    parent = ctx()
    child = parent.child(attempt=2)
    child.bag["seen"] = True
    assert parent.bag["seen"] is True


def test_bag_can_be_isolated_explicitly() -> None:
    parent = ctx()
    parent.bag["k"] = "v"
    child = parent.child(bag={})
    child.bag["k"] = "other"
    assert parent.bag["k"] == "v"


def test_remaining_uses_the_contexts_own_clock() -> None:
    clock = ManualClock()
    c = ctx(deadline=10.0, clock=clock)
    assert c.remaining() == 10.0
    clock.advance(4)
    assert c.remaining() == 6.0
    assert not c.expired()


def test_remaining_without_any_clock_is_an_error() -> None:
    c = ctx(deadline=10.0)
    with pytest.raises(ValueError, match="no clock"):
        c.remaining()
    with pytest.raises(ValueError, match="no clock"):
        c.expired()


def test_explicit_clock_overrides_the_context_clock() -> None:
    context_clock = ManualClock(start=0.0)
    other = ManualClock(start=8.0)
    c = ctx(deadline=10.0, clock=context_clock)
    assert c.remaining(other) == 2.0  # measured against the passed clock


def test_start_derives_absolute_deadline_from_budget() -> None:
    clock = ManualClock(start=100.0)
    c = InvocationContext.start("op", "cid", clock=clock, budget=5.0)
    assert c.deadline == 105.0
    assert c.clock is clock
    assert c.remaining() == 5.0


def test_start_without_budget_has_no_deadline() -> None:
    clock = ManualClock()
    c = InvocationContext.start("op", "cid", clock=clock)
    assert c.deadline is None
    assert c.remaining() is None


def test_arguments_channel_is_carried_and_survives_child() -> None:
    c = ctx(arguments={"query": "sre", "limit": 10})
    assert c.arguments == {"query": "sre", "limit": 10}
    child = c.child(attempt=2)
    assert child.arguments == {"query": "sre", "limit": 10}


def test_current_context_helpers() -> None:
    assert current_context() is None
    c = ctx()
    with use_context(c) as installed:
        assert installed is c
        assert current_context() is c
        inner = c.child(attempt=2)
        with use_context(inner):
            assert current_context() is inner
        assert current_context() is c
    assert current_context() is None


def test_correlation_id_contextvar() -> None:
    assert current_correlation_id() is None
    with use_correlation_id("req-1") as bound:
        assert bound == "req-1"
        assert current_correlation_id() == "req-1"
        with use_correlation_id("req-2"):
            assert current_correlation_id() == "req-2"
        assert current_correlation_id() == "req-1"
    assert current_correlation_id() is None


@pytest.mark.anyio
async def test_report_progress_is_a_noop_without_a_sink() -> None:
    await report_progress(0.5, total=1.0, message="halfway")  # must not raise


@pytest.mark.anyio
async def test_report_progress_reaches_the_installed_sink() -> None:
    seen: list[tuple[float, float | None, str | None]] = []

    async def sink(progress: float, total: float | None, message: str | None) -> None:
        seen.append((progress, total, message))

    with use_progress_sink(sink):
        await report_progress(0.25)
        await report_progress(0.5, total=1.0, message="halfway")
    await report_progress(1.0)  # the sink is uninstalled again
    assert seen == [(0.25, None, None), (0.5, 1.0, "halfway")]
