"""Policy resolution: method over component over framework default."""

import pytest

from warpweft.core.component.policy import DEFAULT_POLICY, Policy, resolve_policy
from warpweft.core.pipeline.chain import DEFAULT_ORDER

pytestmark = pytest.mark.unit


def test_default_when_nothing_is_declared() -> None:
    eff = resolve_policy(None, None)
    assert eff.chain == DEFAULT_ORDER
    assert eff.overrides == {}


def test_default_policy_pins_the_full_order() -> None:
    assert DEFAULT_POLICY.chain == DEFAULT_ORDER


def test_component_chain_overrides_default() -> None:
    eff = resolve_policy(None, Policy(chain=("retry", "timeout")))
    assert eff.chain == ("retry", "timeout")


def test_method_chain_overrides_component() -> None:
    eff = resolve_policy(Policy(chain=("timeout",)), Policy(chain=("retry", "timeout")))
    assert eff.chain == ("timeout",)


def test_component_inherits_default_when_method_pins_none() -> None:
    eff = resolve_policy(Policy(overrides={"retry": {"attempts": 2}}), Policy(chain=("retry",)))
    assert eff.chain == ("retry",)


def test_overrides_layer_field_by_field() -> None:
    component = Policy(overrides={"retry": {"attempts": 5, "base_delay": 1.0}})
    method = Policy(overrides={"retry": {"attempts": 2}, "timeout": {"seconds": 3}})
    eff = resolve_policy(method, component)
    # method wins on attempts, component's base_delay survives, timeout added
    assert eff.overrides["retry"] == {"attempts": 2, "base_delay": 1.0}
    assert eff.overrides["timeout"] == {"seconds": 3}


def test_resolution_does_not_mutate_inputs() -> None:
    component = Policy(overrides={"retry": {"attempts": 5}})
    resolve_policy(Policy(overrides={"retry": {"attempts": 1}}), component)
    assert component.overrides == {"retry": {"attempts": 5}}
