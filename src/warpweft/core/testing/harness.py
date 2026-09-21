"""Drive a component (or a scripted base) through a real policy chain.

These build the *same* chain the container builds, so a test exercises real
retry/timeout/breaker/cache behaviour - just on an instant clock, without a
running system.
"""

from collections.abc import Mapping
from typing import Any, cast

from warpweft.core.axes import GLOBAL_SCOPE, AxisRegistry, ScopeKey
from warpweft.core.clock import Clock
from warpweft.core.component import AComponent, describe
from warpweft.core.component.descriptor import BUILTIN_LINK_MODELS
from warpweft.core.component.policy import EffectivePolicy
from warpweft.core.component.settings import build_config_model
from warpweft.core.composition.endpoint import endpoint_axis, use_endpoint
from warpweft.core.composition.wiring import (
    degradation_interceptor,
    link_settings,
    make_base,
    method_factories,
    validate_degradation,
)
from warpweft.core.context import InvocationContext, use_context
from warpweft.core.errors import ConfigurationError, DefaultErrorClassifier, ErrorClassifier
from warpweft.core.outcome import Outcome
from warpweft.core.pipeline.builtin.degradation import DegradationInterceptor, DegradationSettings, StubProvider
from warpweft.core.pipeline.chain import DEFAULT_ORDER, build_chain, compose
from warpweft.core.pipeline.interceptor import Next
from warpweft.core.pipeline.state import InMemoryStateStore

from .clock import InstantClock

_POLICY_ONLY_MODEL = build_config_model("harness", None, DEFAULT_ORDER, BUILTIN_LINK_MODELS)


async def _run(
    policy: Mapping[str, Any],
    effective: EffectivePolicy,
    base: Next,
    *,
    clock: Clock | None,
    classifier: ErrorClassifier | None,
    arguments: Mapping[str, Any] | None,
    scope_key: ScopeKey,
    endpoint: str,
    component: AComponent[Any, Any, Any] | None = None,
    stub: StubProvider | None = None,
) -> Outcome[Any]:
    clock = clock or InstantClock()
    resolved = classifier or DefaultErrorClassifier()
    config = _POLICY_ONLY_MODEL.model_validate({"policy": dict(policy)})
    factories = method_factories(config, effective, clock, resolved)
    axes = AxisRegistry()
    axes.register(endpoint_axis())
    chain = build_chain(factories, InMemoryStateStore(), axes, base)
    degradation: DegradationInterceptor | None = None
    if component is not None:
        # drive(): mirror the container's build-time gating and wiring.
        validate_degradation(component.name, type(component), config)
        degradation = degradation_interceptor(config, effective, component, resolved)
    else:
        # drive_policy(): a synthetic stub opts degradation in; the key without
        # a stub is an error, mirroring the build contract's fail-loud rule.
        settings = link_settings(config, "degradation", {})
        if settings is not None and stub is None:
            raise ConfigurationError(
                "policy configures degradation but drive_policy was given no stub= to fall back to"
            )
        if settings is not None and stub is not None:
            degradation = DegradationInterceptor(cast(DegradationSettings, settings), stub, resolved)
    if degradation is not None:
        chain = compose((degradation,), chain)
    ctx = InvocationContext(
        operation="harness",
        correlation_id="test",
        arguments=dict(arguments or {}),
        scope_key=scope_key,
        clock=clock,
    )
    with use_endpoint(endpoint), use_context(ctx):
        return await chain(ctx)


async def drive(
    component: AComponent[Any, Any, Any],
    method: str,
    *,
    config: Mapping[str, Any] | None = None,
    clock: Clock | None = None,
    classifier: ErrorClassifier | None = None,
    scope_key: ScopeKey = GLOBAL_SCOPE,
    **arguments: Any,
) -> Outcome[Any]:
    """Run one of a component's invocables through its policy chain.

    The component is used as-is (construct it with test settings and inject any
    fakes first). ``config`` is the deployment config whose ``policy`` section
    turns links on; the method's own policy still restricts the chain. Backoff
    does not wait for real - an `InstantClock` is used by default.
    """
    spec = describe(type(component)).invocables[method]
    policy = dict((config or {}).get("policy", {}))
    endpoint = component.endpoint() or component.identity.uid
    base = make_base(component, spec)
    return await _run(
        policy,
        spec.policy,
        base,
        clock=clock,
        classifier=classifier,
        arguments=arguments,
        scope_key=scope_key,
        endpoint=endpoint,
        component=component,
    )


async def drive_policy(
    policy: Mapping[str, Any],
    base: Next,
    *,
    clock: Clock | None = None,
    classifier: ErrorClassifier | None = None,
    arguments: Mapping[str, Any] | None = None,
    scope_key: ScopeKey = GLOBAL_SCOPE,
    endpoint: str = "test",
    stub: StubProvider | None = None,
) -> Outcome[Any]:
    """Run a scripted base call through a chain built from a ``policy`` dict.

    All configured links are active (the full default order); use the
    `~warpweft.core.testing.scenarios` builders for ``base`` to check how a
    policy behaves against a misbehaving service. Pass ``stub`` to activate
    degradation when ``policy`` includes a ``degradation`` block - the stub is
    the fallback; the block without a stub is a `ConfigurationError`.
    """
    effective = EffectivePolicy(chain=DEFAULT_ORDER, overrides={})
    return await _run(
        policy,
        effective,
        base,
        clock=clock,
        classifier=classifier,
        arguments=arguments,
        scope_key=scope_key,
        endpoint=endpoint,
        stub=stub,
    )
