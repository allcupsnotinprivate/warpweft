"""Descriptor: a component type's contract, built once and cached.

``describe`` scans a component class for invocables, derives their IO schemas
and effective policies, and assembles the dynamic config model. The result is
cached on the class, so building is idempotent and per-type (never per-instance).

The known links whose settings feed the config model are passed in: the full
type registry and third-party discovery live elsewhere. The built-in links
(retry, timeout) are the default set.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import inspect
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel

from warpweft.core.axes import ScopeSpec
from warpweft.core.pipeline.builtin.cache import CacheSettings
from warpweft.core.pipeline.builtin.circuit_breaker import CircuitBreakerSettings
from warpweft.core.pipeline.builtin.concurrency import ConcurrencySettings
from warpweft.core.pipeline.builtin.degradation import DegradationSettings
from warpweft.core.pipeline.builtin.retry import RetrySettings
from warpweft.core.pipeline.builtin.timeout import TimeoutSettings
from warpweft.core.pipeline.chain import DEFAULT_ORDER
from warpweft.core.unit import Identity

from .component import AComponent, Lifetime, component_dependencies, settings_model_of
from .invocable import (
    InvocableSpec,
    build_input_model,
    build_output_adapter,
    input_binding_of,
    is_invocable,
    policy_override,
)
from .policy import Criticality, resolve_policy
from .settings import build_config_model

#: Attribute the built descriptor is cached under on the component class.
_CACHE = "__warpweft_descriptor__"

#: Settings models of the links the config model knows about by default.
BUILTIN_LINK_MODELS: Mapping[str, type[BaseModel]] = MappingProxyType(
    {
        "concurrency": ConcurrencySettings,
        "cache": CacheSettings,
        "circuit_breaker": CircuitBreakerSettings,
        "retry": RetrySettings,
        "timeout": TimeoutSettings,
        "degradation": DegradationSettings,
    }
)


@dataclass(frozen=True)
class Descriptor:
    """Everything known about a component type without instantiating it."""

    identity: Identity
    settings_model: type[BaseModel] | None
    config_model: type[BaseModel]
    invocables: Mapping[str, InvocableSpec]
    dependencies: tuple[str, ...]
    criticality: Criticality
    lifetime: Lifetime
    scope: ScopeSpec

    def config_json_schema(self) -> dict[str, Any]:
        """JSON Schema of the full config (own fields + policy)."""
        return self.config_model.model_json_schema()


def describe(
    cls: type[AComponent[Any, Any, Any]],
    *,
    link_models: Mapping[str, type[BaseModel]] | None = None,
    rebuild: bool = False,
) -> Descriptor:
    """Build (or return the cached) descriptor for a component class."""
    cached = cls.__dict__.get(_CACHE)
    if isinstance(cached, Descriptor) and not rebuild:
        return cached

    if not getattr(cls, "name", None):
        raise ValueError(f"component {cls.__qualname__} must set a 'name' class attribute")

    links = BUILTIN_LINK_MODELS if link_models is None else link_models
    owner = cls.name
    own_settings = settings_model_of(cls)

    invocables: dict[str, InvocableSpec] = {}
    for method_name, member in inspect.getmembers(cls, predicate=is_invocable):
        effective = resolve_policy(policy_override(member), cls.policy)
        binding = input_binding_of(member)
        if binding is not None:
            invocables[method_name] = InvocableSpec(
                method_name=method_name,
                input_model=binding.model,
                output_adapter=build_output_adapter(member),
                policy=effective,
                arg_binder=binding.bind,
            )
        else:
            invocables[method_name] = InvocableSpec(
                method_name=method_name,
                input_model=build_input_model(owner, method_name, member),
                output_adapter=build_output_adapter(member),
                policy=effective,
            )

    # A component that is not an external entry point may be pure infrastructure,
    # reached only through raw dependency access, so it need not declare invocables.
    if not invocables and cls.entrypoint:
        raise ValueError(f"component '{owner}' declares no @invocable methods")

    if cls.lifetime is Lifetime.SCOPED and not cls.scope:
        raise ValueError(f"scoped component '{owner}' must declare a non-empty 'scope'")
    if cls.lifetime is Lifetime.PROCESS and cls.scope:
        raise ValueError(f"process component '{owner}' must not declare a 'scope'")

    config_model = build_config_model(
        owner,
        own_settings=own_settings,
        default_chain=DEFAULT_ORDER,
        link_models=links,
    )

    descriptor = Descriptor(
        identity=Identity.of(cls.name, cls.version),
        settings_model=own_settings,
        config_model=config_model,
        invocables=MappingProxyType(invocables),
        dependencies=component_dependencies(cls),
        criticality=cls.criticality,
        lifetime=cls.lifetime,
        scope=cls.scope,
    )
    setattr(cls, _CACHE, descriptor)
    return descriptor
