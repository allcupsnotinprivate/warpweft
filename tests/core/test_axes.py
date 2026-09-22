"""Axes: canonicalization, required/default semantics, registry validation."""

import logging

import pytest

from warpweft.core.axes import GLOBAL_SCOPE, Axis, AxisRegistry, ScopeSpec
from warpweft.core.errors import ConfigurationError

pytestmark = pytest.mark.unit


def make_registry(**values: str | None) -> AxisRegistry:
    registry = AxisRegistry()
    for name, value in values.items():
        registry.register(Axis(name=name, resolver=lambda v=value: v))
    return registry


def test_canonicalization_order_independent() -> None:
    """ScopeSpec([a, b]) and ScopeSpec([b, a]) must yield an identical key."""
    registry = make_registry(a="1", b="2")
    key_ab = registry.resolve(ScopeSpec(["a", "b"]))
    key_ba = registry.resolve(ScopeSpec(["b", "a"]))
    assert key_ab == key_ba == (("a", "1"), ("b", "2"))


def test_empty_spec_resolves_to_global_scope() -> None:
    assert AxisRegistry().resolve(ScopeSpec()) == GLOBAL_SCOPE


def test_spec_truthiness() -> None:
    assert not ScopeSpec()
    assert ScopeSpec(["a"])


def test_required_missing_names_the_axis() -> None:
    registry = make_registry(tenant=None)
    with pytest.raises(ConfigurationError, match="tenant"):
        registry.resolve(ScopeSpec(["tenant"]))


def test_default_is_substituted() -> None:
    registry = AxisRegistry()
    registry.register(Axis(name="region", resolver=lambda: None, on_missing="default", default="eu"))
    assert registry.resolve(ScopeSpec(["region"])) == (("region", "eu"),)


def test_resolved_value_wins_over_default() -> None:
    registry = AxisRegistry()
    registry.register(Axis(name="region", resolver=lambda: "us", on_missing="default", default="eu"))
    assert registry.resolve(ScopeSpec(["region"])) == (("region", "us"),)


def test_default_mode_requires_default_value() -> None:
    with pytest.raises(ConfigurationError, match="default"):
        Axis(name="x", resolver=lambda: None, on_missing="default")


def test_axis_validation() -> None:
    with pytest.raises(ConfigurationError):
        Axis(name="", resolver=lambda: None)
    with pytest.raises(ConfigurationError, match="max_cardinality"):
        Axis(name="x", resolver=lambda: None, max_cardinality=0)


def test_duplicate_axis_names_in_spec_rejected() -> None:
    with pytest.raises(ConfigurationError, match="duplicate"):
        ScopeSpec(["a", "a"])


def test_double_registration_rejected() -> None:
    registry = make_registry(a="1")
    with pytest.raises(ConfigurationError, match="already registered"):
        registry.register(Axis(name="a", resolver=lambda: "2"))


def test_unregistered_axis_in_spec_rejected() -> None:
    with pytest.raises(ConfigurationError, match="ghost"):
        AxisRegistry().resolve(ScopeSpec(["ghost"]))


def test_scope_key_is_hashable_dict_key() -> None:
    registry = make_registry(a="1", b="2")
    key = registry.resolve(ScopeSpec(["a", "b"]))
    state = {key: object()}
    assert state[registry.resolve(ScopeSpec(["b", "a"]))] is state[key]


def test_cardinality_warning_fires_once(caplog: pytest.LogCaptureFixture) -> None:
    """The soft cap warns exactly once per axis, and never blocks resolution."""
    current = ["a"]
    registry = AxisRegistry()
    registry.register(Axis(name="user", resolver=lambda: current[0], max_cardinality=2))

    with caplog.at_level(logging.WARNING, logger="warpweft.core.axes"):
        for value in ("a", "b", "c", "d", "a"):
            current[0] = value
            # Resolution stays soft: every value still yields a valid key.
            assert registry.resolve(ScopeSpec(["user"])) == (("user", value),)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "user" in warnings[0].getMessage()
    assert "max_cardinality" in warnings[0].getMessage()


def test_repeated_values_do_not_count_toward_cap(caplog: pytest.LogCaptureFixture) -> None:
    registry = AxisRegistry()
    registry.register(Axis(name="region", resolver=lambda: "eu", max_cardinality=1))

    with caplog.at_level(logging.WARNING, logger="warpweft.core.axes"):
        for _ in range(5):
            registry.resolve(ScopeSpec(["region"]))

    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_no_warning_within_cap(caplog: pytest.LogCaptureFixture) -> None:
    current = ["a"]
    registry = AxisRegistry()
    registry.register(Axis(name="user", resolver=lambda: current[0], max_cardinality=2))

    with caplog.at_level(logging.WARNING, logger="warpweft.core.axes"):
        for value in ("a", "b"):
            current[0] = value
            registry.resolve(ScopeSpec(["user"]))

    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
