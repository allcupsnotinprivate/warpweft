"""AComponent lifecycle defaults, identity, generic settings, and HealthStatus."""

from pydantic import BaseModel
import pytest

from warpweft.core.axes import EMPTY_SCOPE
from warpweft.core.component import (
    AComponent,
    Criticality,
    EmptySettings,
    Health,
    HealthStatus,
    Lifetime,
    invocable,
    settings_model_of,
)

pytestmark = [pytest.mark.unit, pytest.mark.anyio]


class Settings(BaseModel):
    url: str


class Sample(AComponent[Settings, str, int]):
    name = "sample"
    version = "2"

    @invocable
    async def do(self, x: int) -> int:
        return x


class Plain(AComponent[EmptySettings, None, None]):
    name = "plain"

    @invocable
    async def ping(self) -> bool:
        return True


async def test_identity_from_name_and_version() -> None:
    c = Sample(Settings(url="http://h"))
    assert c.identity.name == "sample"
    assert c.identity.version == "2"
    assert c.identity.uid == "sample@2"


async def test_lifecycle_defaults_are_noops_and_healthy() -> None:
    c = Sample(Settings(url="http://h"))
    await c.start()
    await c.stop()
    assert (await c.health()).is_ok


async def test_base_stub_raises_until_overridden() -> None:
    from warpweft.core.context import InvocationContext

    c = Sample(Settings(url="http://h"))
    with pytest.raises(NotImplementedError, match="no stub"):
        c.stub(InvocationContext(operation="op", correlation_id="cid"))


async def test_settings_is_a_typed_instance_field() -> None:
    s = Settings(url="http://h")
    c = Sample(s)
    assert c.settings is s
    assert c.settings.url == "http://h"  # precisely typed, no BaseModel | None


async def test_settings_model_is_recovered_from_the_generic_argument() -> None:
    assert settings_model_of(Sample) is Settings
    assert Sample(Settings(url="x")).settings_model is Settings


async def test_empty_settings_component() -> None:
    c = Plain(EmptySettings())
    assert settings_model_of(Plain) is EmptySettings
    assert isinstance(c.settings, EmptySettings)


async def test_unparameterised_component_has_no_settings_model() -> None:
    class Bare(AComponent):  # type: ignore[type-arg]
        name = "bare"

        @invocable
        async def go(self) -> None: ...

    assert settings_model_of(Bare) is None


async def test_defaults_of_optional_class_attributes() -> None:
    assert Sample.version == "2"
    assert Sample.policy is None
    assert Sample.dependencies == ()
    assert Sample.criticality is Criticality.REQUIRED
    assert Sample.lifetime is Lifetime.PROCESS
    assert Sample.scope == EMPTY_SCOPE
    assert Sample(Settings(url="x")).endpoint() is None


def test_health_status_helpers() -> None:
    assert HealthStatus.ok().state is Health.OK
    assert HealthStatus.ok().is_ok
    assert HealthStatus.degraded("half").state is Health.DEGRADED
    assert HealthStatus.degraded("half").detail == "half"
    assert not HealthStatus.degraded().is_ok
    assert HealthStatus.unhealthy("down").state is Health.UNHEALTHY
    assert not HealthStatus.unhealthy().is_ok


async def test_subclass_can_override_lifecycle_and_health() -> None:
    events: list[str] = []

    class Live(AComponent[EmptySettings, None, None]):
        name = "live"

        @invocable
        async def ping(self) -> bool:
            return True

        async def start(self) -> None:
            events.append("start")

        async def stop(self) -> None:
            events.append("stop")

        async def health(self) -> HealthStatus:
            return HealthStatus.degraded("warming up")

    c = Live(EmptySettings())
    await c.start()
    await c.stop()
    assert events == ["start", "stop"]
    assert (await c.health()).state is Health.DEGRADED
