"""``invocable``: mark a method as a pipeline entry point.

The decorator records an optional policy override and nothing else at
definition time. The input/output schemas are derived later, when the
descriptor is built, from the method's type annotations via pydantic - the
same schemas that describe its inputs and outputs to callers.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import inspect
from typing import Any, TypeVar, get_type_hints, overload

from pydantic import BaseModel, TypeAdapter, create_model

from warpweft.core.context import InvocationContext

from .policy import EffectivePolicy, Policy

F = TypeVar("F", bound=Callable[..., Any])

#: Attribute the decorator stamps on the function object.
_MARK = "__warpweft_invocable__"

#: Attribute a custom `InputBinding` is stamped under.
_INPUT_BINDING = "__warpweft_input_binding__"


@overload
def invocable(fn: F) -> F: ...
@overload
def invocable(*, policy: Policy | None = ...) -> Callable[[F], F]: ...
def invocable(fn: F | None = None, *, policy: Policy | None = None) -> F | Callable[[F], F]:
    """Mark a method as an invocable, optionally overriding its policy.

    Usable bare (``@invocable``) or with arguments (``@invocable(policy=...)``).
    """

    def stamp(func: F) -> F:
        setattr(func, _MARK, policy)
        return func

    return stamp if fn is None else stamp(fn)


def is_invocable(obj: object) -> bool:
    return callable(obj) and hasattr(obj, _MARK)


def policy_override(obj: object) -> Policy | None:
    return getattr(obj, _MARK, None)


def _by_parameter(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Default argument binder: caller-facing fields become method kwargs 1:1."""
    return dict(arguments)


@dataclass(frozen=True)
class InputBinding:
    """A custom shape for an invocable's caller-facing input.

    ``model`` is the input contract used for the schema and validation; ``bind``
    maps the caller-facing arguments to the keyword arguments the method is
    actually invoked with. A higher layer attaches this when an invocable's input
    is not the usual one-field-per-parameter form; absent otherwise. Core only
    consumes it - it never depends on why a layer chose a particular shape.

    ``caller_view`` is the inverse of ``bind``: it maps a method's bound
    arguments back to the flat, caller-facing fields. Only the raw
    dependency-call path needs it, to report ``ctx.arguments`` in the same shape
    as the guarded path (a ``span_enricher`` then reads the same keys either
    way). ``None`` when the binding has no computable inverse; the caller then
    falls back to a generic per-parameter view.
    """

    model: type[BaseModel]
    bind: Callable[[Mapping[str, Any]], dict[str, Any]]
    caller_view: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None


def set_input_binding(fn: F, binding: InputBinding) -> F:
    """Attach a custom `InputBinding` to an invocable method; return the method."""
    setattr(fn, _INPUT_BINDING, binding)
    return fn


def input_binding_of(fn: object) -> InputBinding | None:
    """Return the `InputBinding` attached to ``fn``, or ``None`` if it has none."""
    return getattr(fn, _INPUT_BINDING, None)


@dataclass(frozen=True)
class InvocableSpec:
    """The derived contract of one invocable method."""

    method_name: str
    input_model: type[BaseModel]
    output_adapter: TypeAdapter[Any]
    policy: EffectivePolicy
    #: Maps the caller-facing arguments to the method's call kwargs. The default
    #: passes them through unchanged; a custom binder (from an `InputBinding`)
    #: rebuilds a richer shape, e.g. a single model parameter.
    arg_binder: Callable[[Mapping[str, Any]], dict[str, Any]] = _by_parameter
    #: Inverse of ``arg_binder`` for the raw dependency-call path: maps a method's
    #: bound arguments back to the flat caller-facing fields so that path reports
    #: ``ctx.arguments`` in the same shape as the guarded path. ``None`` (the
    #: default-binder case) leaves the dependency path to a generic view.
    caller_view: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None

    def input_json_schema(self) -> dict[str, Any]:
        return self.input_model.model_json_schema()

    def output_json_schema(self) -> dict[str, Any]:
        return self.output_adapter.json_schema()


def build_input_model(owner: str, method_name: str, fn: Callable[..., Any]) -> type[BaseModel]:
    """Derive a pydantic model of the method's inputs.

    ``self`` and any parameter typed as `InvocationContext` are dropped -
    they are plumbing, not part of the caller-facing contract. ``Annotated``
    metadata is preserved, so field formats (`warpweft.core.formats`) reach
    the model and its schema.
    """
    hints = get_type_hints(fn, include_extras=True)
    sig = inspect.signature(fn)
    fields: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        annotation = hints.get(name, Any)
        if annotation is InvocationContext:
            continue
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[name] = (annotation, default)
    return create_model(f"{owner}_{method_name}_input", **fields)


def build_output_adapter(fn: Callable[..., Any]) -> TypeAdapter[Any]:
    """Derive a schema adapter for the method's return annotation."""
    hints = get_type_hints(fn, include_extras=True)
    return TypeAdapter(hints.get("return", Any))
