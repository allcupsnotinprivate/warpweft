"""Build an MCP server from an app's ``@tool`` invocables.

Discovery, the ``list_tools`` / ``call_tool`` handlers and the transport
adapters live here. Tool logic is transport-agnostic: `build_server`
returns a wired server; `run_stdio` drives it over stdio (a local
subprocess host such as Claude Desktop or an IDE).

Each tool is one ``@tool``-marked invocable. A call is routed through
``container.invoke``, so it runs the component's full policy chain and
telemetry; the result is serialized against the invocable's output schema
(secrets masked) and returned as both structured and text content.
"""

from collections.abc import Callable, Collection, Mapping
import contextlib
from dataclasses import dataclass
from fnmatch import fnmatchcase
import inspect
import json
from typing import Any

from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as mt
from pydantic import ValidationError

from warpweft.core.component import InvocableSpec, describe
from warpweft.core.context import use_progress_sink
from warpweft.core.errors import FrameworkError
from warpweft.runtime import App, AxisHandle

from .errors import error_result
from .schema import tool_input_schema
from .tasks import DEFAULT_TTL_MS, TERMINAL, TaskRunner, TaskStatusView, status_view, task_runner
from .tool import ToolMeta, is_tool, tool_meta

#: Derives an axis value (e.g. tenant) from the incoming MCP call - the request
#: context (``ctx``: session, headers, ``_meta``) and the call params (name,
#: arguments). Returning ``None`` leaves the axis unbound (its default/required
#: rule then applies). Bound around the call so scoped components resolve.
AxisBinder = Callable[[Any, mt.CallToolRequestParams], str | None]

#: Names of the shared built-in tools that drive background execution. They are
#: reserved: a component tool may not claim one (checked at build time).
_TASK_STATUS = "task_status"
_TASK_RESULT = "task_result"
_TASK_CANCEL = "task_cancel"
_RESERVED_NAMES = frozenset({_TASK_STATUS, _TASK_RESULT, _TASK_CANCEL})

#: A ``@tool(background=True)`` submit returns only the task id.
_SUBMIT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"task_id": {"type": "string"}},
    "required": ["task_id"],
}
#: Every built-in task tool takes a single ``task_id``.
_TASK_ID_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"task_id": {"type": "string"}},
    "required": ["task_id"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ToolBinding:
    """One exposed tool: its MCP name and how to run it."""

    name: str
    component: str
    method: str
    meta: ToolMeta
    spec: InvocableSpec


def _tool_name(component: str, method: str, meta: ToolMeta) -> str:
    # MCP tool names allow [A-Za-z0-9_-] but not '.', so join with '__'.
    return meta.name or f"{component}__{method}"


def _filter_bindings(
    bindings: list[ToolBinding],
    *,
    tags: Collection[str] | None,
    include: Collection[str] | None,
    exclude: Collection[str] | None,
) -> list[ToolBinding]:
    """Narrow the tool set: ``tags`` (any match) -> ``include`` -> ``exclude``.

    Validation is strict so a typo cannot silently expose the wrong set: every
    requested tag must be declared by some tool, every include/exclude pattern
    must match some tool name (both checked against the *unfiltered* set), and
    the surviving set must not be empty.
    """
    if tags is None and include is None and exclude is None:
        return bindings
    all_names = sorted(b.name for b in bindings)
    all_tags = {t for b in bindings for t in b.meta.tags}
    if tags is not None:
        unknown = sorted(set(tags) - all_tags)
        if unknown:
            raise FrameworkError(f"no tool declares tag(s) {unknown}; declared tags: {sorted(all_tags)}")
    for label, patterns in (("include", include), ("exclude", exclude)):
        for pattern in sorted(patterns or ()):
            if not any(fnmatchcase(name, pattern) for name in all_names):
                raise FrameworkError(f"{label} pattern '{pattern}' matches no tool; tools: {all_names}")
    selected = bindings
    if tags is not None:
        wanted = frozenset(tags)
        selected = [b for b in selected if wanted & b.meta.tags]
    if include is not None:
        selected = [b for b in selected if any(fnmatchcase(b.name, p) for p in include)]
    if exclude is not None:
        selected = [b for b in selected if not any(fnmatchcase(b.name, p) for p in exclude)]
    if not selected:
        raise FrameworkError("tool filter leaves no tools to expose")
    return selected


def collect_tools(
    app: App,
    *,
    tags: Collection[str] | None = None,
    include: Collection[str] | None = None,
    exclude: Collection[str] | None = None,
) -> list[ToolBinding]:
    """Find every ``@tool`` invocable across the app's registered components.

    Raises if a method is marked ``@tool`` but is not an ``@invocable`` (a
    tool must be a pipeline entry point), or if two tools resolve to the same
    MCP name.

    The keyword arguments narrow the set, applied in order:

    - ``tags`` - keep tools carrying at least one of these tags. A tool with
      no tags never passes a tag filter, so tagging works as a whitelist.
    - ``include`` - keep only these MCP tool names (``fnmatch`` globs allowed,
      e.g. ``"search__*"``).
    - ``exclude`` - drop these names (globs allowed); wins over ``include``.

    A tag no tool declares, a pattern matching no tool, or a filter leaving
    nothing to expose is a `FrameworkError` - a typo should fail loudly, not
    quietly serve the wrong tools.
    """
    bindings: list[ToolBinding] = []
    seen: dict[str, str] = {}
    for component in sorted(app.registry.names()):
        cls = app.registry.get(component)
        descriptor = describe(cls)
        tools = inspect.getmembers(cls, predicate=is_tool)
        if not cls.entrypoint:
            # A non-entry-point component (infrastructure) is never a tool.
            if tools:
                raise FrameworkError(f"component '{component}' is not an entry point but declares @tool methods")
            continue
        for method_name, member in tools:
            if method_name not in descriptor.invocables:
                raise FrameworkError(f"'{component}.{method_name}' is marked @tool but is not @invocable")
            meta = tool_meta(member)
            name = _tool_name(component, method_name, meta)
            if name in seen:
                raise FrameworkError(f"duplicate tool name '{name}' from {seen[name]} and {component}.{method_name}")
            seen[name] = f"{component}.{method_name}"
            bindings.append(
                ToolBinding(
                    name=name,
                    component=component,
                    method=method_name,
                    meta=meta,
                    spec=descriptor.invocables[method_name],
                )
            )
    return _filter_bindings(bindings, tags=tags, include=include, exclude=exclude)


def _annotations(meta: ToolMeta) -> mt.ToolAnnotations | None:
    hints = {
        "title": meta.title,
        "read_only_hint": meta.read_only,
        "destructive_hint": meta.destructive,
        "idempotent_hint": meta.idempotent,
        "open_world_hint": meta.open_world,
    }
    present = {k: v for k, v in hints.items() if v is not None}
    return mt.ToolAnnotations(**present) if present else None


def _output_contract(binding: ToolBinding) -> tuple[dict[str, Any], bool]:
    """The advertised output schema, and whether values get result-wrapped.

    MCP output schemas must be object schemas. A non-object return is wrapped
    in ``{"result": ...}`` so every tool advertises a schema and returns
    structured content. ``$defs`` are hoisted to the wrapper root so ``$ref``
    pointers inside the nested schema stay valid.
    """
    schema = binding.spec.output_json_schema()
    if schema.get("type") == "object":
        return schema, False
    inner = dict(schema)
    defs = inner.pop("$defs", None)
    wrapper: dict[str, Any] = {"type": "object", "properties": {"result": inner}, "required": ["result"]}
    if defs:
        wrapper["$defs"] = defs
    return wrapper, True


def _describe_tool(cls: type, binding: ToolBinding) -> mt.Tool:
    method = getattr(cls, binding.method)
    description = binding.meta.description or (inspect.getdoc(method) or None)
    if binding.meta.background:
        # A background tool is a *submit*: it advertises the op's args as input
        # but returns only a task id; the result is fetched later via task_result.
        note = "Starts a background task and returns a task_id; poll task_status / task_result."
        description = f"{description} {note}" if description else note
        output_schema: dict[str, Any] = _SUBMIT_OUTPUT_SCHEMA
    else:
        output_schema, _ = _output_contract(binding)
    return mt.Tool(
        name=binding.name,
        title=binding.meta.title,
        description=description,
        input_schema=tool_input_schema(binding.spec.input_model),
        output_schema=output_schema,
        annotations=_annotations(binding.meta),
    )


def _rewrite_refs(node: Any, remap: dict[str, str]) -> Any:
    """Return ``node`` with every ``$ref`` string rewritten per ``remap``."""
    if isinstance(node, dict):
        return {
            key: (remap.get(value, value) if key == "$ref" and isinstance(value, str) else _rewrite_refs(value, remap))
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_rewrite_refs(item, remap) for item in node]
    return node


def _namespaced_defs(binding: ToolBinding) -> tuple[dict[str, Any], dict[str, Any]]:
    """A background op's result schema and ``$defs``, namespaced by the tool name.

    ``$ref`` pointers are root-relative (``#/$defs/Foo``), so hoisting several
    ops' ``$defs`` into one root would let two different models that share a
    name overwrite each other. Prefixing each op's defs with its (unique) tool
    name and rewriting that op's refs to match keeps the union collision-free.
    """
    schema = dict(binding.spec.output_json_schema())
    member_defs = schema.pop("$defs", None)
    if not member_defs:
        return schema, {}
    prefix = f"{binding.name}."
    remap = {f"#/$defs/{key}": f"#/$defs/{prefix}{key}" for key in member_defs}
    member = _rewrite_refs(schema, remap)
    renamed = {f"{prefix}{key}": _rewrite_refs(value, remap) for key, value in member_defs.items()}
    return member, renamed


def _task_result_output_schema(bindings: list[ToolBinding]) -> dict[str, Any]:
    """The ``task_result`` output schema: a union of every background op's result.

    Each background result is stored wrapped as ``{"result": <value>}`` (see
    `_result`), so the advertised schema is that wrapper with the inner value
    constrained to the ``oneOf`` of the ops' raw output schemas.
    """
    members: list[dict[str, Any]] = []
    all_defs: dict[str, Any] = {}
    if len(bindings) == 1:
        # A single op cannot collide with itself, so keep its defs unprefixed.
        only = dict(bindings[0].spec.output_json_schema())
        defs = only.pop("$defs", None)
        members.append(only)
        if defs:
            all_defs = defs
    else:
        for binding in bindings:
            member, renamed = _namespaced_defs(binding)
            members.append(member)
            all_defs.update(renamed)  # keys prefixed by the unique tool name
    result_schema = members[0] if len(members) == 1 else {"oneOf": members}
    wrapper: dict[str, Any] = {
        "type": "object",
        "properties": {"result": result_schema},
        "required": ["result"],
    }
    if all_defs:
        wrapper["$defs"] = all_defs
    return wrapper


def _task_tool_descriptors(background: list[ToolBinding]) -> list[mt.Tool]:
    """Descriptors for the shared built-in tools (present iff background tools exist)."""
    read_only = mt.ToolAnnotations(read_only_hint=True)
    return [
        mt.Tool(
            name=_TASK_STATUS,
            description="Poll a background task's status.",
            input_schema=_TASK_ID_INPUT_SCHEMA,
            output_schema=TaskStatusView.model_json_schema(),
            annotations=read_only,
        ),
        mt.Tool(
            name=_TASK_RESULT,
            description="Fetch a finished background task's result (error if it is not done yet).",
            input_schema=_TASK_ID_INPUT_SCHEMA,
            output_schema=_task_result_output_schema(background),
            annotations=read_only,
        ),
        mt.Tool(
            name=_TASK_CANCEL,
            description="Request cancellation of a background task.",
            input_schema=_TASK_ID_INPUT_SCHEMA,
            output_schema=TaskStatusView.model_json_schema(),
        ),
    ]


def _serialize(binding: ToolBinding, value: Any) -> Any:
    return binding.spec.output_adapter.dump_python(value, mode="json")


def _result(binding: ToolBinding, outcome: Any, *, wrap: bool) -> mt.CallToolResult:
    """Serialize an invoke outcome into a ``CallToolResult``.

    Text stays the raw serialization (readable for humans). ``wrap`` controls
    the structured content: an inline call wraps only when its advertised schema
    is the ``{"result": ...}`` wrapper (a non-object return), while a background
    result always wraps so ``task_result`` has one uniform union schema.
    """
    serialized = _serialize(binding, outcome.value)
    text = serialized if isinstance(serialized, str) else json.dumps(serialized)
    return mt.CallToolResult(
        content=[mt.TextContent(type="text", text=text)],
        structured_content={"result": serialized} if wrap else serialized,
        meta={"warpweft.source": outcome.source, "warpweft.degraded": outcome.degraded},
    )


def _success_result(binding: ToolBinding, outcome: Any) -> mt.CallToolResult:
    """Build the inline success result; the wrap decision follows the advertised
    schema, not the runtime value, so structured content always conforms."""
    _, wrapped = _output_contract(binding)
    return _result(binding, outcome, wrap=wrapped)


def _status_result(view: TaskStatusView) -> mt.CallToolResult:
    dumped = view.model_dump(mode="json")
    return mt.CallToolResult(
        content=[mt.TextContent(type="text", text=json.dumps(dumped))],
        structured_content=dumped,
    )


_ELICITATION = mt.ClientCapabilities(elicitation=mt.ElicitationCapability())


def _refusal(text: str, *, code: str, retryable: bool = False) -> mt.CallToolResult:
    meta = {"warpweft.error": code, "warpweft.retryable": retryable}
    return mt.CallToolResult(content=[mt.TextContent(type="text", text=text)], is_error=True, meta=meta)


async def _confirm_destructive(ctx: Any, binding: ToolBinding) -> mt.CallToolResult | None:
    """Ask the user to confirm a destructive call; ``None`` means proceed.

    Fails closed: when the operator demanded confirmation, a client that
    cannot elicit gets an error, never an unconfirmed execution. The decision
    is the elicitation ``action`` itself, so the form requests no fields.
    """
    if not ctx.session.check_client_capability(_ELICITATION):
        return _refusal(
            f"tool '{binding.name}' requires user confirmation, but the client does not support elicitation",
            code="confirmation_unsupported",
        )
    label = binding.meta.title or binding.name
    result = await ctx.session.elicit_form(
        f"Confirm running '{label}'. This action is marked destructive.",
        {"type": "object", "properties": {}},
        related_request_id=ctx.request_id,
    )
    if result.action != "accept":
        return _refusal(f"the user declined to run '{binding.name}'", code="declined")
    return None


def _bind_axes(
    stack: contextlib.ExitStack,
    binders: Mapping[AxisHandle, AxisBinder],
    ctx: Any,
    params: mt.CallToolRequestParams,
) -> None:
    """Enter each axis binding whose binder yields a value for this call."""
    for handle, binder in binders.items():
        value = binder(ctx, params)
        if value is not None:
            stack.enter_context(handle.use(value))


def build_server(
    app: App,
    *,
    name: str = "warpweft",
    version: str = "0",
    tags: Collection[str] | None = None,
    include: Collection[str] | None = None,
    exclude: Collection[str] | None = None,
    confirm_destructive: bool = False,
    runner: TaskRunner | None = None,
    axis_binders: Mapping[AxisHandle, AxisBinder] | None = None,
) -> Server[Any]:
    """Wire an MCP server exposing the app's tools. The app must be started.

    ``tags``/``include``/``exclude`` narrow which tools are served (see
    `collect_tools`), so one app can back several servers with different
    tool sets - e.g. ``tags={"public"}`` for an assistant, no filter for an
    operator console.

    ``confirm_destructive=True`` gates every ``destructive=True`` tool behind
    an MCP elicitation: the user must accept before the call runs. Decline
    (or a client that cannot elicit) is a tool error; the call never happens.

    ``runner`` enables ``@tool(background=True)`` tools: such a tool becomes a
    non-blocking *submit* (returns a ``task_id``), and the server also exposes
    the shared ``task_status`` / ``task_result`` / ``task_cancel`` tools. A
    background tool without a ``runner`` is a `FrameworkError` - the advertised
    behaviour must be real. Everything here is plain ``tools/call``, so it needs
    no special transport.

    ``axis_binders`` maps a state-slicing `AxisHandle` (from ``app.axis(...)``)
    to a function that derives its value from each call - e.g. a ``tenant`` axis
    bound from an ``x-tenant`` header or the auth context. Each call binds the
    axis (via the handle's contextvar) around ``invoke`` so scoped components
    resolve; for a background submit the binding spans ``runner.start`` so the
    job inherits it. A binder returning ``None`` leaves the axis to its
    default/required rule. Without this, an axis value must be bound by the host
    (e.g. ASGI middleware) instead.
    """
    binders = dict(axis_binders or {})
    bindings = {b.name: b for b in collect_tools(app, tags=tags, include=include, exclude=exclude)}
    classes = {name: app.registry.get(name) for name in app.registry.names()}

    collisions = sorted(bindings.keys() & _RESERVED_NAMES)
    if collisions:
        raise FrameworkError(f"tool name(s) {collisions} are reserved for background task tools")

    background = [b for b in bindings.values() if b.meta.background]
    if background and runner is None:
        names = sorted(b.name for b in background)
        raise FrameworkError(
            f"tool(s) {names} are @tool(background=True) but the server has no task runner; "
            "pass build_server(..., runner=...) (see warpweft.mcp.tasks.task_runner)"
        )
    task_tools = _task_tool_descriptors(background) if background else []

    async def on_list_tools(ctx: Any, params: Any) -> mt.ListToolsResult:
        tools = [_describe_tool(classes[b.component], b) for b in bindings.values()]
        return mt.ListToolsResult(tools=tools + task_tools)

    async def on_call_tool(ctx: Any, params: mt.CallToolRequestParams) -> mt.CallToolResult:
        if params.name in _RESERVED_NAMES:
            return await _handle_task_tool(runner, params)

        binding = bindings.get(params.name)
        if binding is None:
            unknown = mt.TextContent(type="text", text=f"unknown tool '{params.name}'")
            return mt.CallToolResult(content=[unknown], is_error=True)
        # Validate/coerce the LLM-supplied arguments against the invocable's
        # input model (this is where field formats are enforced), then pass the
        # coerced values on. The container itself does not re-validate.
        try:
            model = binding.spec.input_model.model_validate(params.arguments or {})
        except ValidationError as exc:
            return error_result(exc)
        kwargs = {field: getattr(model, field) for field in type(model).model_fields}

        # Confirmation comes after validation (no point confirming a call that
        # would fail anyway) and before any execution - including a background
        # submit, so the task never starts without consent.
        if confirm_destructive and binding.meta.destructive:
            denial = await _confirm_destructive(ctx, binding)
            if denial is not None:
                return denial

        # Bind request-derived axis values (e.g. tenant) for the duration of the
        # call. For a background submit the binding must span runner.start so
        # start_soon captures it into the job's context; for an inline call it
        # must span the invoke. A binder that raises is a tool error, never a
        # transport-level failure.
        with contextlib.ExitStack() as axes:
            try:
                _bind_axes(axes, binders, ctx, params)
            except Exception as exc:
                return error_result(exc)

            if binding.meta.background and runner is not None:  # runner presence guaranteed at build time
                return await _submit(app, runner, binding, kwargs)

            async def forward_progress(progress: float, total: float | None, message: str | None) -> None:
                # Best-effort: a failed notification must never fail the call.
                # The session no-ops by itself when the client sent no token.
                with contextlib.suppress(Exception):
                    await ctx.session.report_progress(progress, total, message)

            # Any failure - framework or user code - becomes a tool error with
            # retry guidance, never a transport-level failure. Cancellation (a
            # BaseException) still propagates: the SDK cancels this handler's
            # anyio scope on notifications/cancelled, unwinding the policy chain.
            try:
                with use_progress_sink(forward_progress):
                    outcome = await app.container.invoke(binding.component, binding.method, **kwargs)
            except Exception as exc:
                return error_result(exc)

            return _success_result(binding, outcome)

    return Server(name, version=version, on_list_tools=on_list_tools, on_call_tool=on_call_tool)


async def _submit(app: App, runner: TaskRunner, binding: ToolBinding, kwargs: dict[str, Any]) -> mt.CallToolResult:
    """Start a background job and return its task id immediately."""

    async def job(task_id: str) -> mt.CallToolResult:
        # Progress becomes status updates on the task record, since the submit
        # request has already been answered with the id.
        async def to_status(progress: float, total: float | None, message: str | None) -> None:
            with contextlib.suppress(Exception):
                await runner.store.update(task_id, status="working", status_message=message)

        try:
            with use_progress_sink(to_status):
                outcome = await app.container.invoke(binding.component, binding.method, **kwargs)
        except Exception as exc:
            return error_result(exc)
        # Always wrap: task_result advertises one union schema over {"result": ...}.
        return _result(binding, outcome, wrap=True)

    record = await runner.start(binding.name, job, ttl_ms=DEFAULT_TTL_MS)
    task_id = {"task_id": record.task_id}
    return mt.CallToolResult(
        content=[mt.TextContent(type="text", text=json.dumps(task_id))],
        structured_content=task_id,
    )


async def _handle_task_tool(runner: TaskRunner | None, params: mt.CallToolRequestParams) -> mt.CallToolResult:
    """Serve task_status / task_result / task_cancel against the runner's store."""
    if runner is None:
        unknown = mt.TextContent(type="text", text=f"unknown tool '{params.name}'")
        return mt.CallToolResult(content=[unknown], is_error=True)

    task_id = (params.arguments or {}).get("task_id")
    if not isinstance(task_id, str) or not task_id:
        return _refusal("task_id is required", code="invalid_arguments")

    record = await runner.store.get(task_id)
    if record is None:
        return _refusal(f"unknown task '{task_id}'", code="unknown_task")

    if params.name == _TASK_STATUS:
        return _status_result(status_view(record))

    if params.name == _TASK_CANCEL:
        await runner.cancel(task_id)
        # Report the state as it stands; the job settles to ``cancelled``
        # asynchronously as its scope unwinds.
        fresh = await runner.store.get(task_id)
        return _status_result(status_view(fresh or record))

    # _TASK_RESULT: the stored payload is already typed and secret-masked.
    if record.result is not None:
        return record.result
    if record.status in TERMINAL:
        # Terminal but no payload (cancelled, or a job that failed before
        # producing a result): a distinct, non-retryable code - never the
        # retry-implying not_ready, which would make a poller loop forever.
        return _refusal(f"task '{task_id}' ended without a result (status: {record.status})", code=record.status)
    # Still running: the result will appear, so invite the caller to poll again.
    return _refusal(f"task '{task_id}' is not finished yet (status: {record.status})", code="not_ready", retryable=True)


async def run_stdio(
    app: App,
    *,
    name: str = "warpweft",
    version: str = "0",
    tags: Collection[str] | None = None,
    include: Collection[str] | None = None,
    exclude: Collection[str] | None = None,
    confirm_destructive: bool = False,
    background: bool = True,
    axis_binders: Mapping[AxisHandle, AxisBinder] | None = None,
) -> None:  # pragma: no cover - needs real stdio
    """Start the app and serve its tools over stdio until the stream closes.

    ``background=True`` (the default) opens an in-memory task runner so
    ``@tool(background=True)`` tools can run in the background; set it to
    ``False`` to serve inline-only (such a tool then fails loudly at build).
    ``axis_binders`` is passed through to `build_server`.
    """
    async with app.run(), _optional_runner(background) as runner:
        server = build_server(
            app,
            name=name,
            version=version,
            tags=tags,
            include=include,
            exclude=exclude,
            confirm_destructive=confirm_destructive,
            runner=runner,
            axis_binders=axis_binders,
        )
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())


@contextlib.asynccontextmanager
async def _optional_runner(enabled: bool) -> Any:
    """Yield a `TaskRunner` when ``enabled``, else ``None`` (no nursery opened)."""
    if not enabled:
        yield None
        return
    async with task_runner() as runner:
        yield runner
