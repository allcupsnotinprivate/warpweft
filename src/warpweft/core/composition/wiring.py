"""Building a method's link chain from its config and effective policy.

Shared by the container and the testing helpers, so a test drives the *same*
chain production does. The order comes from the config's chain, the method's
effective policy restricts which links are allowed, and a link is active only
when it is implemented and has settings.
"""

from collections.abc import Mapping
import inspect
from typing import Any, cast, get_type_hints

from pydantic import BaseModel

from warpweft.core.clock import Clock
from warpweft.core.component.component import AComponent, defines_stub
from warpweft.core.component.invocable import InvocableSpec
from warpweft.core.component.policy import Criticality, EffectivePolicy
from warpweft.core.component.settings import POLICY_FIELD
from warpweft.core.context import InvocationContext
from warpweft.core.errors import ConfigurationError, ErrorClassifier
from warpweft.core.outcome import Outcome
from warpweft.core.pipeline.builtin.degradation import DegradationInterceptor, DegradationSettings
from warpweft.core.pipeline.chain import DEFAULT_ORDER
from warpweft.core.pipeline.interceptor import InterceptorFactory, Next

from .links import BUILTIN_LINK_BUILDERS


def link_settings(config: BaseModel, link: str, override: Mapping[str, Any]) -> BaseModel | None:
    """Resolve a link's settings from the config, applying a per-method override."""
    policy = getattr(config, POLICY_FIELD, None)
    base: BaseModel | None = getattr(policy, link, None) if policy is not None else None
    if base is None:
        return None
    if not override:
        return base
    return type(base).model_validate({**base.model_dump(), **override})


def active_links(config: BaseModel, policy: EffectivePolicy) -> list[tuple[str, BaseModel]]:
    """The ``(link name, settings)`` pairs active for a method, outermost first."""
    config_policy = getattr(config, POLICY_FIELD, None)
    config_chain = getattr(config_policy, "chain", None) or DEFAULT_ORDER
    allowed = set(policy.chain)
    result: list[tuple[str, BaseModel]] = []
    for link in config_chain:
        if link not in allowed or BUILTIN_LINK_BUILDERS.get(link) is None:
            continue
        settings = link_settings(config, link, policy.overrides.get(link, {}))
        if settings is None:
            continue
        result.append((link, settings))
    return result


def method_factories(
    config: BaseModel, policy: EffectivePolicy, clock: Clock, classifier: ErrorClassifier
) -> list[InterceptorFactory]:
    """The link factories for a method, outermost first."""
    return [BUILTIN_LINK_BUILDERS[link](settings, clock, classifier) for link, settings in active_links(config, policy)]


def validate_degradation(component: str, cls: type[AComponent[Any, Any, Any]], config: BaseModel) -> None:
    """Reject a ``policy.degradation`` block that contradicts the component.

    Called eagerly at ``Container.build`` (and re-run when per-slice overrides
    are assembled, and by the test harness): degradation on a required
    component, or without a ``stub()`` to fall back to, is a configuration
    error, not a silent no-op.
    """
    if link_settings(config, "degradation", {}) is None:
        return
    if cls.criticality is not Criticality.OPTIONAL:
        raise ConfigurationError(
            f"component '{component}' configures policy.degradation but its criticality "
            f"is 'required': only an optional component may serve a stub"
        )
    if not defines_stub(cls):
        raise ConfigurationError(f"component '{component}' configures policy.degradation but defines no stub() method")


def degradation_interceptor(
    config: BaseModel,
    policy: EffectivePolicy,
    instance: AComponent[Any, Any, Any],
    classifier: ErrorClassifier,
) -> DegradationInterceptor | None:
    """The fixed outer degradation link for one method, or ``None`` when inactive.

    Deliberately outside the ordered chain (never in DEFAULT_ORDER or the link
    store): a stub must not be cached or retried, and its stub is bound per
    instance - like telemetry, its position is fixed, not listed.
    """
    cls = type(instance)
    if cls.criticality is not Criticality.OPTIONAL or not defines_stub(cls):
        return None
    settings = link_settings(config, "degradation", policy.overrides.get("degradation", {}))
    if settings is None:
        return None
    return DegradationInterceptor(cast(DegradationSettings, settings), instance.stub, classifier)


def make_base(instance: object, spec: InvocableSpec) -> Next:
    """Wrap a bound method as a base call: bind arguments, inject context, wrap result.

    The invocable's ``arg_binder`` maps ``ctx.arguments`` (the caller-facing input
    fields) to the method's keyword arguments - the default passes them through,
    a custom binding rebuilds a richer shape. If the method declares an
    ``InvocationContext`` parameter it also receives the context, and a raw return
    value is wrapped in an `Outcome`.
    """
    method = getattr(instance, spec.method_name)
    hints = get_type_hints(method)
    ctx_param = next(
        (name for name in inspect.signature(method).parameters if hints.get(name) is InvocationContext),
        None,
    )
    bind = spec.arg_binder

    async def base(ctx: InvocationContext) -> Outcome[Any]:
        kwargs = bind(ctx.arguments or {})
        if ctx_param is not None:
            kwargs[ctx_param] = ctx
        result = await method(**kwargs)
        return result if isinstance(result, Outcome) else Outcome(value=result)

    return base
