"""Outcome defaults and Unit/Identity protocols."""

from pydantic import BaseModel
import pytest

from warpweft.core.outcome import Outcome
from warpweft.core.unit import Identity, Startable, Stoppable, Unit

pytestmark = pytest.mark.unit


def test_outcome_defaults_are_live() -> None:
    outcome = Outcome(value=42)
    assert outcome.value == 42
    assert outcome.source == "live"
    assert not outcome.degraded
    assert outcome.attempts == 1
    assert outcome.elapsed == 0.0


def test_outcome_is_immutable() -> None:
    with pytest.raises(AttributeError):
        Outcome(value=1).value = 2  # type: ignore[misc]


def test_identity_of_builds_stable_uid() -> None:
    identity = Identity.of("retry", "1")
    assert identity == Identity(name="retry", version="1", uid="retry@1")
    assert Identity.of("retry", "1") == identity  # stable across calls


class _Settings(BaseModel):
    limit: int = 1


class _FullUnit:
    """Structurally conforms to Unit + Startable + Stoppable without subclassing."""

    identity = Identity.of("full")
    settings_model: type[BaseModel] | None = _Settings
    started = False
    stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True


class _BareUnit:
    """Minimal unit: no lifecycle, no settings."""

    identity = Identity.of("bare")
    settings_model: type[BaseModel] | None = None


def test_units_conform_structurally() -> None:
    full: Unit = _FullUnit()
    bare: Unit = _BareUnit()
    assert full.identity.uid == "full@0"
    assert bare.settings_model is None


def test_lifecycle_protocols_are_runtime_checkable() -> None:
    full = _FullUnit()
    bare = _BareUnit()
    assert isinstance(full, Startable)
    assert isinstance(full, Stoppable)
    assert not isinstance(bare, Startable)
    assert not isinstance(bare, Stoppable)


@pytest.mark.anyio
async def test_lifecycle_runs() -> None:
    unit = _FullUnit()
    await unit.start()
    await unit.stop()
    assert unit.started and unit.stopped
