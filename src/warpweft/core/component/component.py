"""AComponent: the base every component subclasses.

A plain class, deliberately **not** a ``BaseModel``: it holds live state -
clients, pools, link references. Settings are a field on the instance, not a
base class, and their type is the first generic parameter, so ``self.settings``
is precisely typed rather than ``BaseModel | None``. The lifecycle hooks
default to no-ops so a trivial component writes none of them.

A component declares its remaining metadata as class attributes and marks its
entry points with `invocable`. The derived contract (identity, config
model, per-method schemas and effective policies) is produced by ``describe``
and cached on the class; the settings model is recovered from the generic
argument, so it is declared exactly once.
"""

from collections.abc import Awaitable, Callable, Mapping
from enum import StrEnum
import re
from typing import Any, ClassVar, Generic, TypeVar, get_args, get_origin, get_type_hints

from pydantic import BaseModel

from warpweft.core.axes import EMPTY_SCOPE, ScopeSpec
from warpweft.core.context import InvocationContext
from warpweft.core.outcome import Outcome
from warpweft.core.telemetry.component import NOOP_TELEMETRY, ComponentTelemetry
from warpweft.core.unit import Identity

from .health import HealthStatus
from .policy import Criticality, Policy

TSettings = TypeVar("TSettings", bound=BaseModel)
TIn = TypeVar("TIn")
TOut = TypeVar("TOut")

_CAMEL_BOUNDARY_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_BOUNDARY_2 = re.compile(r"([a-z0-9])([A-Z])")


def _to_snake_case(name: str) -> str:
    name = _CAMEL_BOUNDARY_1.sub(r"\1_\2", name)
    return _CAMEL_BOUNDARY_2.sub(r"\1_\2", name).lower()


class Lifetime(StrEnum):
    """How many instances of a component the container keeps.

    ``PROCESS`` - a single instance for the whole process, created at startup.
    ``SCOPED`` - one instance per axis key (see `AComponent.scope`),
    created lazily on first use and evicted by LRU.
    """

    PROCESS = "process"
    SCOPED = "scoped"


class EmptySettings(BaseModel):
    """Settings model for components that need no configuration of their own."""


class AComponent(Generic[TSettings, TIn, TOut]):
    """Base class for components.

    Parameterised as ``AComponent[Settings, In, Out]``. The `name`
    defaults to the snake_cased class name; set the class attribute to
    override. Other optional class attributes: `version`, `policy`
    (component-wide default), `dependencies` and `criticality`. The
    settings model comes from the ``Settings`` type argument - use
    `EmptySettings` for a component that needs none.

    Dependencies can be declared two ways: the `dependencies` name tuple
    (accessed via `dependency`), or a class-level annotation whose type
    is another component - the container then assigns the resolved instance to
    that attribute, fully typed::

        class Search(AComponent[SearchSettings, str, list[Doc]]):
            embedder: Embedder  # dependency; self.embedder after start
    """

    #: The component's stable name; derived from the class name if not set.
    name: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "name" not in cls.__dict__:
            cls.name = _to_snake_case(cls.__name__)

    #: Optional version; part of the identity uid.
    version: ClassVar[str] = "0"
    #: Component-wide default policy; a method may override it.
    policy: ClassVar[Policy | None] = None
    #: Names of components this one depends on.
    dependencies: ClassVar[tuple[str, ...]] = ()
    #: Author-provided default config, merged under the deployment config.
    defaults: ClassVar[Mapping[str, Any]] = {}
    #: Whether the system may run without this component.
    criticality: ClassVar[Criticality] = Criticality.REQUIRED
    #: How many instances the container keeps.
    lifetime: ClassVar[Lifetime] = Lifetime.PROCESS
    #: Axes a scoped component is sliced along. Must be empty for PROCESS and
    #: non-empty for SCOPED (enforced by ``describe``).
    scope: ClassVar[ScopeSpec] = EMPTY_SCOPE
    #: Whether this component is an external entry point. Most components are;
    #: an infrastructure-only one sets this ``False`` - it is never called from
    #: outside and may therefore declare no invocables.
    entrypoint: ClassVar[bool] = True

    #: Bound by the container at instantiation; ``None`` outside a container.
    _ww_telemetry: ComponentTelemetry | None = None

    def __init__(self, settings: TSettings) -> None:
        self.settings: TSettings = settings
        self._deps: Mapping[str, AComponent[Any, Any, Any]] = {}

    @property
    def telemetry(self) -> ComponentTelemetry:
        """Channel for the component's own metrics (see `ComponentTelemetry`).

        Inside a container this records through the container's meter provider
        with ``warpweft.component`` and the instance's axis pairs attached
        automatically. A component constructed directly (unit tests) gets a
        no-op that accepts every call and records nothing.
        """
        bound = self._ww_telemetry
        return bound if bound is not None else NOOP_TELEMETRY

    def bind_dependencies(self, deps: Mapping[str, "AComponent[Any, Any, Any]"]) -> None:
        """Install resolved dependencies (called by the container before start).

        Annotation-declared dependencies are additionally assigned to their
        attributes, so ``self.embedder`` is the live instance.
        """
        self._deps = dict(deps)
        for attr, dep_cls in dependency_annotations(type(self)).items():
            instance = self._deps.get(dep_cls.name)
            if instance is not None:
                setattr(self, attr, instance)

    def dependency(self, name: str) -> "AComponent[Any, Any, Any]":
        """Return a declared dependency's instance."""
        return self._deps[name]

    def bind_invoker(self, invoke: Callable[..., Awaitable[Outcome[Any]]]) -> None:
        """Receive a bound invoker for this instance (called by the container).

        The invoker routes ``invoke(method, **kwargs)`` through this component's
        own policy chain. The default ignores it; a subclass may override it to
        run its own operations guarded (retry, breaker, telemetry) instead of raw.
        """

    @property
    def identity(self) -> Identity:
        return Identity.of(self.name, self.version)

    def endpoint(self) -> str | None:
        """Identity of the external system this instance talks to.

        Drives ``[endpoint]``-sliced link state (breaker, concurrency): two
        instances sharing an endpoint share that state. Default ``None`` means
        the instance is its own endpoint (per-instance link state). Override to
        return the resolved host.
        """
        return None

    def stub(self, ctx: InvocationContext) -> Any:
        """Fallback value for a degraded call (optional; define it to enable degradation).

        Wired only when the component's ``criticality`` is ``optional`` **and**
        ``policy.degradation`` is configured. One stub per component: switch on
        ``ctx.operation`` / ``ctx.arguments`` for per-method values.

        Deliberately synchronous: a stub must be cheap and local (a constant, a
        last-known-good value). A stub that does I/O is a second dependency, not
        a fallback. If the stub itself raises, that exception propagates (with
        the original failure as context) - a broken stub must be loud.
        """
        raise NotImplementedError(f"component '{self.name}' defines no stub()")

    @property
    def settings_model(self) -> type[BaseModel] | None:
        """The component's own settings model (for Unit conformance)."""
        return settings_model_of(type(self))

    async def start(self) -> None:
        """Acquire resources. Default: nothing."""

    async def stop(self) -> None:
        """Release resources. Default: nothing."""

    async def health(self) -> HealthStatus:
        """Report health. Default: healthy."""
        return HealthStatus.ok()


_annotation_cache: dict[type, Mapping[str, type["AComponent[Any, Any, Any]"]]] = {}


def dependency_annotations(cls: type["AComponent[Any, Any, Any]"]) -> Mapping[str, type["AComponent[Any, Any, Any]"]]:
    """Attribute -> component type for annotation-declared dependencies.

    A class-level annotation counts as a dependency when its type is a strict
    ``AComponent`` subclass and it is not a ``ClassVar``. Base-class annotations
    are inherited; a redeclared attribute takes the most-derived type.
    """
    cached = _annotation_cache.get(cls)
    if cached is not None:
        return cached
    result: dict[str, type[AComponent[Any, Any, Any]]] = {}
    hints = get_type_hints(cls)
    for attr, annotation in hints.items():
        if get_origin(annotation) is ClassVar:
            continue
        if isinstance(annotation, type) and annotation is not AComponent and issubclass(annotation, AComponent):
            result[attr] = annotation
    _annotation_cache[cls] = result
    return result


def component_dependencies(cls: type["AComponent[Any, Any, Any]"]) -> tuple[str, ...]:
    """All declared dependency names: the explicit tuple plus annotated ones."""
    names = list(cls.dependencies)
    for dep_cls in dependency_annotations(cls).values():
        if dep_cls.name not in names:
            names.append(dep_cls.name)
    return tuple(names)


def defines_stub(cls: type["AComponent[Any, Any, Any]"]) -> bool:
    """Whether the class overrides ``AComponent.stub``, enabling degradation."""
    return cls.stub is not AComponent.stub


def settings_model_of(cls: type["AComponent[Any, Any, Any]"]) -> type[BaseModel] | None:
    """Recover a component's settings model from its ``AComponent[...]`` argument.

    Inspects the class's generic bases and returns the first type argument when
    it is a ``BaseModel`` subclass. Returns ``None`` for a component that never
    parameterised its settings.
    """
    for base in getattr(cls, "__orig_bases__", ()):
        origin = get_origin(base)
        if isinstance(origin, type) and issubclass(origin, AComponent):
            args = get_args(base)
            if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                return args[0]
    return None
