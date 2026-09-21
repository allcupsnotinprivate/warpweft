# MCP tools

The `mcp` extra (`pip install warpweft[mcp]`) exposes a component's invocables as [Model Context
Protocol](https://modelcontextprotocol.io) tools. It is a projection, not a new
component type: a tool **is** an `@invocable` method that you additionally mark
with `@tool`. Every tool call is routed through `container.invoke`, so it runs
the component's full policy chain (retry, cache, breaker) and telemetry for
free. `@tool` lives in this package, so the core never learns about MCP.

## Marking a tool

```python
from warpweft import AComponent, invocable
from warpweft.mcp import tool


class Weather(AComponent[WeatherSettings, str, Forecast]):
    @tool(description="Get the forecast for a city.", read_only=True)
    @invocable
    async def forecast(self, city: str) -> Forecast: ...

    @invocable  # no @tool -> never exposed to an LLM
    async def _refresh(self) -> None: ...
```

Exposure is opt-in: only `@tool`-marked invocables become tools, so health
checks and internal helpers stay private. `@tool` accepts a `name`/`title`
override, a `description` (else the method docstring is used), the MCP
annotation hints `read_only`, `destructive`, `idempotent`, `open_world`,
free-form `tags` used to [filter](#filtering) which tools a server exposes, and
`background=True` for [long-running tools](#background-long-running-tools). A
`@tool` on a method that is not `@invocable` is rejected when the server is
built.

An [action](actions.md) is marked automatically - its `execute` is a tool by
default (opt out with `entrypoint = False`), from class-level metadata - so you
rarely write `@tool` yourself.

## Serving

```python
from warpweft.mcp import run_stdio

anyio.run(run_stdio, app)  # local host (Claude Desktop, an IDE)
```

`run_stdio` starts the app, serves its tools over stdio, and stops the app when
the stream closes. For programmatic use (or a future HTTP transport),
`build_server(app)` returns a transport-agnostic MCP server.

## Filtering

One app can back several servers with different tool sets - a read-only server
for an assistant, the full set for an operator console. `collect_tools`,
`build_server` and `run_stdio` take three narrowing arguments, applied in
order:

```python
build_server(app, tags={"public"})  # any-match on @tool tags
build_server(app, include={"search__*"})  # only these names (fnmatch globs)
build_server(app, exclude={"billing__refund"})  # drop these names; wins over include
```

A tool with no tags never passes a `tags` filter, so tagging works as a
whitelist: a forgotten tag keeps a tool private rather than exposing it. The
filter is validated strictly - a tag no tool declares, a pattern that matches
no tool, or a filter that leaves nothing to expose raises a `FrameworkError`
instead of quietly serving the wrong set.

## The tool contract

Per tool, derived from the invocable's descriptor:

- **name** - `component__method` (MCP names disallow `.`) or the `name` override.
- **description** - the `@tool` description, else the method docstring.
- **inputSchema** - the method's input JSON Schema, with read-only/computed
  fields dropped (a model does not fill those in). Parameters annotated with a
  field format (`warpweft.formats`, e.g. `host: Ipv4`) carry the `format`
  keyword, and the arguments an LLM supplies are validated against the input
  model before the call - an invalid value comes back as a tool error.
- **outputSchema** - the return type's JSON Schema. MCP requires an object
  schema, so a non-object return (`str`, `list[Doc]`, ...) is advertised
  wrapped: `{"result": <schema>}`. Every tool has an output schema.
- **annotations** - the hint flags above.

## Results

A tool call returns the invocable's `Outcome`, serialized against the output
schema in JSON mode - so `SecretStr` fields are masked and enums/dates become
primitives. Every call returns `structuredContent` conforming to the
advertised schema - a non-object value arrives as `{"result": ...}` - plus a
text content block that stays the *raw* serialization (a `str` result reads
as plain text, not JSON). The `Outcome`'s `source` and `degraded` are
reported in the result `meta`.

## Errors

Any failure - framework or user code - becomes a tool error (`isError=true`),
never a transport-level failure. The error tells the model what to do next: a
one-sentence hint is appended to the message, and the result `meta` carries
machine-readable guidance:

| meta key | Meaning |
| --- | --- |
| `warpweft.error` | a stable code: `invalid_arguments`, `circuit_open`, `timeout`, `retry_exhausted`, `unavailable`, `transient`, `permanent`, `error` - plus `declined` / `confirmation_unsupported` from the [destructive-tool gate](#confirming-destructive-tools) |
| `warpweft.retryable` | whether calling again can help |
| `warpweft.retry_after_s` | for `circuit_open`: seconds until the breaker admits a probe |
| `warpweft.attempts` | for `retry_exhausted`: attempts already spent |

The codes mirror the [error taxonomy](composition.md): transient failures
(timeouts, an open breaker, a degraded component) are retryable, permanent
ones are not, and an unclassified exception is reported as non-retryable
`error` - the same stance the retry link takes.

## Progress and cancellation

A long-running invocable reports progress without knowing what transport
drives it:

```python
from warpweft import report_progress


class Indexer(AComponent[IndexerSettings, str, Report]):
    @tool(description="Rebuild the search index.")
    @invocable
    async def rebuild(self) -> Report:
        for step, shard in enumerate(self._shards, start=1):
            await self._index(shard)
            await report_progress(step, total=len(self._shards), message=shard)
        ...
```

`report_progress` is a no-op unless the transport installed a sink
(`warpweft.core.context.use_progress_sink`). The MCP server installs one per
call, so reports become `notifications/progress` exactly when the client sent
a `progressToken`. Reports are fire-and-forget - a failed notification never
fails the call.

Cancellation needs no code at all: when the client cancels an MCP request,
the SDK cancels the handler's anyio scope, the cancellation unwinds the
policy chain, and the component's cleanup (`finally` blocks, context
managers) runs as usual.

## Background (long-running) tools

A tool that runs for minutes should not hold its `tools/call` open that long -
many clients and proxies time out. Mark it `background=True` and it becomes a
non-blocking **submit**: the call returns a `task_id` at once and the work runs
in the background. The caller then polls with the shared built-in tools.

```python
class Reports(AComponent[ReportSettings, str, Report]):
    @tool(background=True, description="Generate a monthly report.")
    @invocable
    async def report(self, month: str) -> Report:
        await report_progress(0.5, message="crunching")
        return Report(...)
```

When at least one background tool is served, three shared tools appear:

- **`task_status(task_id)`** - `working` / `completed` / `failed` / `cancelled`,
  plus the latest `report_progress` message.
- **`task_result(task_id)`** - the finished result, or a `not_ready` tool error
  while it is still running. The payload is the origin op's return value, fully
  typed and secret-masked, wrapped as `{"result": ...}`. Its `outputSchema` is
  the `oneOf` union of every background op's result schema, so a model still
  knows the shapes to expect.
- **`task_cancel(task_id)`** - requests cancellation; the job's policy chain
  unwinds and the task settles to `cancelled`.

This is all plain `tools/call` returning `CallToolResult`, so it needs no
special transport - it works over stdio today. Everything else about a tool
still applies: arguments are validated before submit, `report_progress` becomes
task status (not client notifications, since the request already returned), and
`confirm_destructive` gates the submit so a task never starts without consent.

`run_stdio` opens an in-memory task runner by default (`background=False` serves
inline-only). For programmatic use, pass your own store and runner:

```python
from warpweft.mcp import build_server, task_runner

async with task_runner(store=my_store) as runner:  # your store survives restarts
    server = build_server(app, runner=runner)
    ...
```

The names `task_status` / `task_result` / `task_cancel` are reserved; a
component tool claiming one, or a `background=True` tool served without a
runner, raises a `FrameworkError` at build time. The default in-memory store
keeps task state and results for the process lifetime only - a deployment that
must survive a restart supplies its own `TaskStore` (Redis, Postgres, ...).

## Confirming destructive tools

```python
build_server(app, confirm_destructive=True)
```

puts a human in the loop for every tool marked `destructive=True`: before
executing, the server sends an MCP elicitation that the client presents to
the user. Only an explicit accept runs the tool - decline or cancel returns a
tool error (`warpweft.error: "declined"`) without executing anything. The
gate fails closed: a client that does not support elicitation gets
`confirmation_unsupported` instead of an unconfirmed execution. Confirmation
happens after argument validation and never applies to non-destructive
tools.

Off by default - the `destructive` annotation alone only *hints* to the host
UI; this flag turns it into an enforced contract.
