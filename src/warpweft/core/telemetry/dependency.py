"""Dependency-call telemetry: the proxy injected in place of a bare instance.

When the container wires a dependency into a component it does not hand over
the raw instance: it hands over this proxy. The proxy is transparent for
everything except the dependency's ``@invocable`` methods, which it wraps in
the standard `instrument` contract - one span named
``<component>.<method>``, the ``warpweft.calls`` / ``warpweft.call.duration``
metrics, the axis cardinality policy and the host ``span_enricher``.

Deliberately **telemetry only**: no policy links run here. The dependency's
retry, breaker or cache would double up with the caller's own chain, so a raw
dependency call stays raw - it just becomes visible. Calls a component makes
to itself (``self.method()``) are not routed through a proxy and stay
uninstrumented; telemetry sits on the component boundary.

``isinstance`` checks against the dependency's class keep working: the proxy
reports the wrapped instance's class as its ``__class__`` (the same mechanism
``unittest.mock`` uses). Subclassing the proxy is not supported.
"""

from collections.abc import Iterable
import functools
import inspect
from typing import Any, Final
import uuid

from warpweft.core.axes import ScopeKey
from warpweft.core.clock import Clock
from warpweft.core.context import InvocationContext, current_context, current_correlation_id, use_context
from warpweft.core.errors import ErrorClassifier
from warpweft.core.outcome import Outcome
from warpweft.core.pipeline.interceptor import Next

from .instrument import DEFAULT_CONFIG, SpanEnricher, TelemetryConfig, instrument

#: Bag key carrying the raw ``(args, kwargs)`` of a call to the base link, so
#: the method is invoked exactly as the caller wrote it (positionals intact).
_CALL_KEY: Final = "warpweft.dependency.call"


class DependencyTelemetryProxy:
    """Transparent proxy adding the telemetry contract to invocable calls.

    Attribute access and assignment pass straight through to the wrapped
    instance; only the listed ``methods`` are substituted with instrumented
    wrappers (built lazily, cached per method). A wrapper returns the method's
    raw value, so ``await dep.fetch()`` reads exactly like a raw call.
    """

    def __init__(
        self,
        instance: object,
        *,
        component: str,
        methods: Iterable[str],
        scope_key: ScopeKey,
        clock: Clock | None = None,
        tracer_provider: Any = None,
        meter_provider: Any = None,
        classifier: ErrorClassifier | None = None,
        config: TelemetryConfig = DEFAULT_CONFIG,
        span_enricher: SpanEnricher | None = None,
    ) -> None:
        # Own state bypasses __setattr__, which forwards to the instance.
        object.__setattr__(self, "_ww_instance", instance)
        object.__setattr__(self, "_ww_component", component)
        object.__setattr__(self, "_ww_methods", frozenset(methods))
        object.__setattr__(self, "_ww_scope_key", scope_key)
        object.__setattr__(self, "_ww_clock", clock)
        object.__setattr__(
            self,
            "_ww_instrument_kwargs",
            {
                "clock": clock,
                "tracer_provider": tracer_provider,
                "meter_provider": meter_provider,
                "classifier": classifier,
                "config": config,
                "span_enricher": span_enricher,
            },
        )
        object.__setattr__(self, "_ww_wrappers", {})

    @property  # type: ignore[misc]
    def __class__(self) -> type:
        # isinstance() falls back to ``__class__`` when type() does not match,
        # so checks against the dependency's class hold on the proxy too.
        return type(self._ww_instance)

    def __getattr__(self, item: str) -> Any:
        if item in self._ww_methods:
            wrappers: dict[str, Any] = self._ww_wrappers
            wrapper = wrappers.get(item)
            if wrapper is None:
                wrapper = _wrap_method(
                    getattr(self._ww_instance, item),
                    operation=f"{self._ww_component}.{item}",
                    scope_key=self._ww_scope_key,
                    clock=self._ww_clock,
                    instrument_kwargs=self._ww_instrument_kwargs,
                )
                wrappers[item] = wrapper
            return wrapper
        return getattr(self._ww_instance, item)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._ww_instance, name, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # A callable dependency (an Action) routes its call through its own
        # invoker and full chain - instrumented there; no proxy telemetry here.
        return self._ww_instance(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<telemetry proxy for {self._ww_instance!r}>"


def _wrap_method(
    method: Any,
    *,
    operation: str,
    scope_key: ScopeKey,
    clock: Clock | None,
    instrument_kwargs: dict[str, Any],
) -> Any:
    """Build one instrumented wrapper around a bound invocable method."""
    signature = inspect.signature(method)

    async def base(ctx: InvocationContext) -> Outcome[Any]:
        args, kwargs = ctx.bag.pop(_CALL_KEY)
        # Always wrap, even an Outcome-returning method: the caller-facing
        # wrapper returns ``outcome.value``, so the raw result is preserved.
        return Outcome(value=await method(*args, **kwargs))

    instrumented: Next = instrument(base, **instrument_kwargs)

    @functools.wraps(method)
    async def call(*args: Any, **kwargs: Any) -> Any:
        try:
            arguments = dict(signature.bind(*args, **kwargs).arguments)
        except TypeError:
            arguments = dict(kwargs)  # let the method itself raise the real error
        parent = current_context()
        ctx = InvocationContext(
            operation=operation,
            correlation_id=(parent.correlation_id if parent is not None else None)
            or current_correlation_id()
            or uuid.uuid4().hex,
            deadline=parent.deadline if parent is not None else None,
            scope_key=scope_key,
            arguments=arguments,
            clock=clock or (parent.clock if parent is not None else None),
            bag={_CALL_KEY: (args, kwargs)},
        )
        with use_context(ctx):
            outcome = await instrumented(ctx)
        return outcome.value

    return call
