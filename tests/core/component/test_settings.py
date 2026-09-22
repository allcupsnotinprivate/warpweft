"""Dynamic config-model assembly and the reserved-name guard."""

from pydantic import BaseModel, SecretStr
import pytest

from warpweft.core.component.settings import build_config_model, build_policy_model
from warpweft.core.errors import ConfigurationError
from warpweft.core.pipeline.builtin.retry import RetrySettings
from warpweft.core.pipeline.builtin.timeout import TimeoutSettings

pytestmark = pytest.mark.unit

CHAIN = ("concurrency", "cache", "circuit_breaker", "retry", "timeout")
LINKS = {"retry": RetrySettings, "timeout": TimeoutSettings}


def test_policy_model_has_chain_default_and_optional_link_fields() -> None:
    model = build_policy_model("c", CHAIN, LINKS)
    inst = model()
    assert inst.chain == list(CHAIN)
    assert inst.retry is None
    assert inst.timeout is None


def test_config_model_merges_own_fields_and_policy() -> None:
    class Own(BaseModel):
        base_url: str
        api_key: SecretStr

    config = build_config_model("comp", Own, CHAIN, LINKS)
    assert set(config.model_fields) == {"base_url", "api_key", "policy"}

    inst = config.model_validate(
        {
            "base_url": "http://h",
            "api_key": "secret",
            "policy": {"retry": {"attempts": 3, "base_delay": 0.1, "max_delay": 1.0}},
        }
    )
    assert inst.base_url == "http://h"
    assert inst.policy.retry.attempts == 3
    assert inst.policy.timeout is None
    assert inst.policy.chain == list(CHAIN)


def test_config_model_without_own_settings() -> None:
    config = build_config_model("comp", None, CHAIN, LINKS)
    assert set(config.model_fields) == {"policy"}
    assert config().policy.chain == list(CHAIN)


def test_own_field_named_policy_is_rejected() -> None:
    class Bad(BaseModel):
        policy: str

    with pytest.raises(ConfigurationError, match="reserved"):
        build_config_model("bad", Bad, CHAIN, LINKS)


def test_links_without_settings_model_only_appear_in_chain() -> None:
    config = build_config_model("comp", None, CHAIN, {"retry": RetrySettings})
    policy_fields = config.model_fields["policy"].annotation.model_fields  # type: ignore[union-attr]
    assert "retry" in policy_fields
    assert "cache" not in policy_fields  # no settings model -> no sub-field
    assert config().policy.chain == list(CHAIN)  # but still in the chain contract
