"""``Action``: a component that is a single, callable entry point.

An action has one operation, ``execute(self, params)``, taking its whole input
as one pydantic model. Calling the instance runs that operation **through the
component's policy chain** (retry, breaker, cache, telemetry) and returns the
value::

    class Summarize(Action[EmptySettings, Digest, Digest]):
        description = "Summarise a document."

        async def execute(self, params: Summarise) -> Digest: ...


    digest = await summarize(text="...")  # guarded, via the chain
    digest = await Summarize(settings).execute(Summarise(text="..."))  # pure, for tests

Wiring happens once, at class creation: ``execute`` is marked ``@invocable`` and
"boxed" (its flat fields become the tool/schema contract, rebuilt into the model
on the way in), and - unless ``entrypoint`` is ``False`` - it is tagged as an MCP
tool from the class-level metadata below. Nothing here imports the mcp SDK, so
actions carry tool metadata whether or not the ``mcp`` extra is installed; the
optional `warpweft.mcp` layer collects them when a server is built.
"""

from collections.abc import Awaitable, Callable, Collection, Mapping
import inspect
from typing import Any, ClassVar, TypeVar, cast, get_type_hints

from pydantic import BaseModel

from warpweft.core.component import AComponent
from warpweft.core.component.invocable import InputBinding, invocable, set_input_binding
from warpweft.core.context import InvocationContext
from warpweft.core.errors import ConfigurationError
from warpweft.core.outcome import Outcome
from warpweft.mcp.tool import tool

TSettings = TypeVar("TSettings", bound=BaseModel)
TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


def _params_of(execute: Any, owner: str) -> tuple[str, type[BaseModel]]:
    """The single input parameter of ``execute``: its name and pydantic model."""
    hints = get_type_hints(execute)
    inputs = [
        name
        for name, p in inspect.signature(execute).parameters.items()
        if name != "self"
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        and hints.get(name) is not InvocationContext
    ]
    if len(inputs) != 1:
        raise TypeError(
            f"action '{owner}.execute' must take exactly one input parameter (its params model); got {inputs or 'none'}"
        )
    name = inputs[0]
    model = hints.get(name)
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise TypeError(
            f"action '{owner}.execute' parameter '{name}' must be annotated with a pydantic BaseModel; got {model!r}"
        )
    return name, model


def _box_input(execute: Any, param: str, model: type[BaseModel]) -> None:
    """Give ``execute`` a single-model input binding.

    The action's caller-facing contract is ``model``'s flat fields; on the way in
    those fields are rebuilt into the one ``param`` model the method declares. This
    is the only place that knows an action is "boxed" - core just runs the binder.
    """

    def bind(arguments: Mapping[str, Any]) -> dict[str, Any]:
        return {param: model.model_validate(dict(arguments))}

    set_input_binding(execute, InputBinding(model=model, bind=bind))


def _flatten(params: Any, fields: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise a call's input to the params model's flat fields."""
    if params is None:
        return dict(fields)
    if fields:
        raise TypeError("pass either a params object/dict or keyword fields, not both")
    if isinstance(params, BaseModel):
        return params.model_dump()
    if isinstance(params, Mapping):
        return dict(params)
    raise TypeError(f"action params must be a BaseModel, a mapping or keyword fields; got {type(params)!r}")


class Action(AComponent[TSettings, TIn, TOut]):
    """Base class for actions. Parameterised as ``Action[Settings, In, Out]``."""

    #: Tool description; falls back to the ``execute`` docstring.
    description: ClassVar[str | None] = None
    #: Override the derived MCP tool name (``<component>__execute`` otherwise).
    tool_name: ClassVar[str | None] = None
    #: Human-friendly tool title.
    title: ClassVar[str | None] = None
    #: MCP behaviour hints (left unset unless the action states them).
    read_only: ClassVar[bool | None] = None
    destructive: ClassVar[bool | None] = None
    idempotent: ClassVar[bool | None] = None
    open_world: ClassVar[bool | None] = None
    #: Tool tags, for serving different tool sets from one app.
    tags: ClassVar[Collection[str] | None] = None
    #: Expose as a non-blocking background tool (submit + poll); see ``@tool``.
    background: ClassVar[bool] = False

    _invoker: Callable[..., Awaitable[Outcome[Any]]] | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        execute = cls.__dict__.get("execute")
        if execute is None:  # an intermediate base that leaves execute abstract
            return
        param, model = _params_of(execute, cls.__qualname__)
        invocable(execute)
        _box_input(execute, param, model)
        if cls.entrypoint:
            tool(
                name=cls.tool_name,
                title=cls.title,
                description=cls.description,
                read_only=cls.read_only,
                destructive=cls.destructive,
                idempotent=cls.idempotent,
                open_world=cls.open_world,
                tags=cls.tags,
                background=cls.background,
            )(execute)

    async def execute(self, params: TIn) -> TOut:
        """The action's single operation; implement it in a subclass."""
        raise NotImplementedError

    def bind_invoker(self, invoke: Callable[..., Awaitable[Outcome[Any]]]) -> None:
        self._invoker = invoke

    async def __call__(self, params: TIn | Mapping[str, Any] | None = None, /, **fields: Any) -> TOut:
        """Run ``execute`` through the policy chain; return the outcome's value.

        Accepts the params model, a mapping, or keyword fields. Requires a
        running app (the container binds the invoker); outside one, call
        ``execute`` directly - that is the pure, chain-free path for unit tests.
        """
        if self._invoker is None:
            raise ConfigurationError(
                f"action '{self.name}' is not bound to a running app; invoke it via "
                f"app.proxy/app.invoke, or call '.execute(...)' directly in unit tests"
            )
        outcome = await self._invoker("execute", **_flatten(params, fields))
        return cast("TOut", outcome.value)
