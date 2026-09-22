"""Type registry: explicit registration, lookup, entry-point discovery."""

import pytest

from warpweft.core.component import AComponent, EmptySettings, invocable
from warpweft.core.composition.registry import Registry
from warpweft.core.errors import ConfigurationError

pytestmark = pytest.mark.unit


class Widget(AComponent[EmptySettings, None, None]):
    name = "widget"

    @invocable
    async def go(self) -> bool:
        return True


class Gadget(AComponent[EmptySettings, None, None]):
    name = "gadget"

    @invocable
    async def go(self) -> bool:
        return True


class EpWidget(AComponent[EmptySettings, None, None]):
    name = "ep-widget"

    @invocable
    async def go(self) -> bool:
        return True


def test_register_get_and_metadata() -> None:
    reg = Registry()
    reg.register(Widget)
    assert reg.get("widget") is Widget
    assert reg.descriptor("widget").identity.name == "widget"
    assert reg.names() == frozenset({"widget"})
    assert "widget" in reg


def test_register_is_usable_as_a_decorator() -> None:
    reg = Registry()

    @reg.register
    class Local(AComponent[EmptySettings, None, None]):
        name = "local"

        @invocable
        async def go(self) -> None: ...

    assert reg.get("local") is Local


def test_registering_the_same_class_twice_is_idempotent() -> None:
    reg = Registry()
    reg.register(Widget)
    reg.register(Widget)  # no error
    assert reg.get("widget") is Widget


def test_name_collision_with_a_different_class_is_rejected() -> None:
    reg = Registry()

    class OtherWidget(AComponent[EmptySettings, None, None]):
        name = "widget"

        @invocable
        async def go(self) -> None: ...

    reg.register(Widget)
    with pytest.raises(ConfigurationError, match="already registered"):
        reg.register(OtherWidget)


def test_unknown_component_is_a_configuration_error() -> None:
    reg = Registry()
    with pytest.raises(ConfigurationError, match="not registered"):
        reg.get("nope")


def test_registering_an_invalid_component_fails_fast() -> None:
    reg = Registry()

    class NoEntryPoints(AComponent[EmptySettings, None, None]):
        pass  # no @invocable methods

    with pytest.raises(ValueError, match="no @invocable"):
        reg.register(NoEntryPoints)


def test_auto_named_component_registers_under_the_derived_name() -> None:
    reg = Registry()

    class AutoNamed(AComponent[EmptySettings, None, None]):
        @invocable
        async def go(self) -> None: ...

    reg.register(AutoNamed)
    assert reg.get("auto_named") is AutoNamed


def test_two_registries_are_independent() -> None:
    a, b = Registry(), Registry()
    a.register(Widget)
    b.register(Gadget)
    assert "widget" in a and "widget" not in b
    assert "gadget" in b and "gadget" not in a


class _FakeEntryPoint:
    """Minimal stand-in: load_entry_points only calls .load()."""

    def __init__(self, cls: type[AComponent]) -> None:
        self._cls = cls

    def load(self) -> type[AComponent]:
        return self._cls


def test_entry_point_discovery() -> None:
    reg = Registry()
    loaded = reg.load_entry_points(source=[_FakeEntryPoint(EpWidget)])
    assert loaded == {"ep-widget": EpWidget}
    assert reg.get("ep-widget") is EpWidget
