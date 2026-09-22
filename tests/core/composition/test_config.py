"""Layered config: per-field merge, provenance, resolver, error quality."""

from pydantic import BaseModel, Field
import pytest

from warpweft.core.composition.config import (
    SOURCE_COMPONENT,
    SOURCE_DEPLOYMENT,
    SOURCE_FRAMEWORK,
    SOURCE_SLICE,
    DictSettingsResolver,
    assemble_config,
    deep_merge,
)
from warpweft.core.errors import ConfigurationError

pytestmark = pytest.mark.unit


def test_merge_is_per_field_not_whole_object() -> None:
    merged, _ = deep_merge(
        [
            (SOURCE_DEPLOYMENT, {"a": 1, "nested": {"x": 1, "y": 2}}),
            (SOURCE_SLICE, {"nested": {"y": 3}}),  # overrides only y
        ]
    )
    assert merged == {"a": 1, "nested": {"x": 1, "y": 3}}


def test_later_layer_wins_scalar() -> None:
    merged, _ = deep_merge([(SOURCE_FRAMEWORK, {"n": 1}), (SOURCE_DEPLOYMENT, {"n": 2})])
    assert merged["n"] == 2


def test_provenance_tracks_the_winning_source_per_leaf() -> None:
    merged, provenance = deep_merge(
        [
            (SOURCE_FRAMEWORK, {"a": 0, "b": 0}),
            (SOURCE_COMPONENT, {"b": 1}),
            (SOURCE_DEPLOYMENT, {"policy": {"retry": {"attempts": 3}}}),
        ]
    )
    assert provenance["a"] == SOURCE_FRAMEWORK
    assert provenance["b"] == SOURCE_COMPONENT  # last writer wins
    assert provenance["policy.retry.attempts"] == SOURCE_DEPLOYMENT


def test_scalar_replaces_an_earlier_mapping() -> None:
    merged, provenance = deep_merge([(SOURCE_FRAMEWORK, {"x": {"deep": 1}}), (SOURCE_DEPLOYMENT, {"x": 5})])
    assert merged["x"] == 5
    assert provenance["x"] == SOURCE_DEPLOYMENT


def test_dict_resolver_returns_override_or_empty() -> None:
    resolver = DictSettingsResolver({("svc", (("tenant", "acme"),)): {"rate": 10}})
    assert resolver.resolve("svc", (("tenant", "acme"),)) == {"rate": 10}
    assert resolver.resolve("svc", (("tenant", "other"),)) == {}


class Cfg(BaseModel):
    url: str
    retries: int = Field(default=3, ge=0)


def test_assemble_validates_the_merged_result() -> None:
    config, _ = assemble_config(
        "svc",
        Cfg,
        [(SOURCE_DEPLOYMENT, {"url": "http://h"}), (SOURCE_SLICE, {"retries": 5})],
    )
    assert isinstance(config, Cfg)
    assert config.url == "http://h"
    assert config.retries == 5


def test_error_names_component_path_and_source() -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        assemble_config("svc", Cfg, [(SOURCE_DEPLOYMENT, {"url": "http://h", "retries": -1})])
    message = str(excinfo.value)
    assert "component 'svc'" in message
    assert "retries" in message
    assert SOURCE_DEPLOYMENT in message


def test_error_source_for_a_missing_required_field() -> None:
    with pytest.raises(ConfigurationError, match="url"):
        assemble_config("svc", Cfg, [(SOURCE_DEPLOYMENT, {"retries": 1})])
