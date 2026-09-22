"""System health: readiness aggregation and container liveness/readiness."""

from _support.axes import context_axes
from _support.components import health_reporter
from _support.containers import registry_of
import anyio
import pytest

from warpweft.core.component import AComponent, Criticality, EmptySettings, Health, HealthStatus, invocable
from warpweft.core.composition import Container, Registry
from warpweft.core.composition.health import aggregate_readiness

pytestmark = [pytest.mark.unit, pytest.mark.anyio]


# --- pure aggregation --------------------------------------------------------


def test_all_healthy_is_ready() -> None:
    result = aggregate_readiness(
        {"a": (), "b": ()},
        {"a": Criticality.REQUIRED, "b": Criticality.REQUIRED},
        {"a": HealthStatus.ok(), "b": HealthStatus.ok()},
    )
    assert result.ready
    assert all(s.is_ok for s in result.components.values())


def test_unhealthy_required_breaks_readiness() -> None:
    result = aggregate_readiness(
        {"a": ()},
        {"a": Criticality.REQUIRED},
        {"a": HealthStatus.unhealthy("down")},
    )
    assert not result.ready


def test_degraded_optional_does_not_break_readiness() -> None:
    result = aggregate_readiness(
        {"req": (), "opt": ()},
        {"req": Criticality.REQUIRED, "opt": Criticality.OPTIONAL},
        {"req": HealthStatus.ok(), "opt": HealthStatus.degraded("partial")},
    )
    assert result.ready
    assert result.components["opt"].state is Health.DEGRADED


def test_unhealthy_required_dependency_downgrades_the_dependent() -> None:
    result = aggregate_readiness(
        {"api": ("db",), "db": ()},
        {"api": Criticality.REQUIRED, "db": Criticality.REQUIRED},
        {"api": HealthStatus.ok(), "db": HealthStatus.unhealthy("down")},
    )
    assert not result.ready
    assert not result.components["api"].is_ok  # downgraded because db is down


def test_unhealthy_optional_dependency_leaves_the_dependent_alone() -> None:
    result = aggregate_readiness(
        {"api": ("cache",), "cache": ()},
        {"api": Criticality.REQUIRED, "cache": Criticality.OPTIONAL},
        {"api": HealthStatus.ok(), "cache": HealthStatus.unhealthy("down")},
    )
    assert result.ready  # api copes without the optional cache
    assert result.components["api"].is_ok


# --- container integration ---------------------------------------------------


class Healthy(AComponent[EmptySettings, None, None]):
    name = "healthy"

    @invocable
    async def go(self) -> bool:
        return True


class Sick(AComponent[EmptySettings, None, None]):
    name = "sick"

    async def health(self) -> HealthStatus:
        return HealthStatus.unhealthy("bad")

    @invocable
    async def go(self) -> bool:
        return True


class Slow(AComponent[EmptySettings, None, None]):
    name = "slow"

    async def health(self) -> HealthStatus:
        await anyio.Event().wait()  # never returns
        return HealthStatus.ok()  # pragma: no cover

    @invocable
    async def go(self) -> bool:
        return True


async def test_liveness_tracks_started_state() -> None:
    reg = Registry()
    reg.register(Healthy)
    container = Container.build(reg, {"healthy": {}})
    assert not (await container.liveness()).is_ok
    await container.start()
    assert (await container.liveness()).is_ok
    await container.stop()


async def test_readiness_ok_when_all_required_healthy() -> None:
    reg = Registry()
    reg.register(Healthy)
    container = Container.build(reg, {"healthy": {}})
    await container.start()
    result = await container.readiness()
    assert result.ready
    await container.stop()


async def test_readiness_fails_on_unhealthy_required_component() -> None:
    reg = Registry()
    reg.register(Sick)
    container = Container.build(reg, {"sick": {}})
    await container.start()
    result = await container.readiness()
    assert not result.ready
    assert result.components["sick"].state is Health.UNHEALTHY
    await container.stop()


async def test_readiness_survives_a_degraded_optional_component() -> None:
    class OptSick(AComponent[EmptySettings, None, None]):
        name = "opt-sick"
        criticality = Criticality.OPTIONAL

        async def start(self) -> None:
            raise RuntimeError("cannot start")

        @invocable
        async def go(self) -> bool:
            return True

    reg = Registry()
    reg.register(Healthy)
    reg.register(OptSick)
    container = Container.build(reg, {"healthy": {}, "opt-sick": {}})
    await container.start()
    result = await container.readiness()
    assert result.ready  # the degraded optional does not break readiness
    assert result.components["opt-sick"].state is Health.DEGRADED
    await container.stop()


async def test_health_check_timeout_is_unhealthy() -> None:
    reg = Registry()
    reg.register(Slow)
    container = Container.build(reg, {"slow": {}}, health_timeout=0.02)
    await container.start()
    result = await container.readiness()
    assert not result.ready
    assert result.components["slow"].state is Health.UNHEALTHY
    await container.stop()


@pytest.mark.characterization
async def test_readiness_ignores_an_unhealthy_live_scoped_instance() -> None:
    # CHARACTERIZATION: a scoped component always reports HealthStatus.ok() in
    # readiness ("created on demand; nothing running to poll") - even when a live
    # slice's health() would return unhealthy, and that health() is never called.
    axes, handles = context_axes("tenant")
    polls: list[int] = []
    reporter = health_reporter("rep", healthy=lambda: False, on_check=lambda: polls.append(1))
    container = Container.build(registry_of(reporter), {"rep": {}}, axes=axes)
    await container.start()
    with handles["tenant"].use("acme"):
        await container.invoke("rep", "go")  # a live, unhealthy scoped slice now exists

    result = await container.readiness()
    assert result.ready  # readiness does not see the unhealthy scoped slice
    assert polls == []  # the scoped instance's health() is never polled
    await container.stop()
