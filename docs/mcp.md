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
free-form `tags` used to [filter](#filtering) which tools a server exposes,
`background=True` for [long-running tools](#background-long-running-tools), and
`icons` clients show next to the tool (see [server metadata](#server-metadata)).
A `@tool` on a method that is not `@invocable` is rejected when the server is
built.

An [action](actions.md) is marked automatically - its `execute` is a tool by
default (opt out with `entrypoint = False`), from class-level metadata - so you
rarely write `@tool` yourself.

## Serving

The same tools can be served over two transports. Tool behaviour does not
depend on the transport: both run the same handlers, policy chain and
telemetry.

### stdio

```python
from warpweft.mcp import run_stdio

anyio.run(run_stdio, app)  # local host (Claude Desktop, an IDE)
```

`run_stdio` starts the app, serves its tools over stdio, and stops the app when
the stream closes. This is the transport for a host that launches the server as
a subprocess.

### Streamable HTTP

For a remote or shared deployment, serve the tools over HTTP:

```python
from warpweft.mcp import run_http

anyio.run(run_http, app)  # http://127.0.0.1:8000/mcp, served by uvicorn
```

`run_http` is a thin wrapper for the common case. `asgi_app(app)` returns the
underlying ASGI application, for any ASGI server (uvicorn, hypercorn) or for
mounting into an existing Starlette/FastAPI app:

```python
from warpweft.mcp import asgi_app

mcp = asgi_app(app, path="/mcp")

api = FastAPI(lifespan=mcp.lifespan)  # starts the app - use instead of app.lifespan
api.mount("/", mcp)
```

Its lifespan starts the app, the background task runner and the MCP session
manager together, so the app runs exactly while requests can reach it. Starlette
does not run the lifespan of a mounted app, which is why the host must take
`mcp.lifespan`. A request that arrives outside the lifespan gets `503`.

The HTTP plumbing comes from the mcp SDK and is configured through `asgi_app`:

| Option | Effect |
| --- | --- |
| `stateless=True` | no per-client session: every request stands alone, so it scales across workers |
| `json_response=True` | a single JSON body per response instead of an SSE stream |
| `host` | the address clients use; for a loopback host DNS-rebinding protection turns on automatically (`transport_security` configures it explicitly) |
| `auth`, `token_verifier` | bearer-token auth as an OAuth resource server |
| `session_idle_timeout`, `max_sessions` | limits for stateful sessions |

Both the `initialize` handshake protocol and the 2026-07-28 per-request
protocol are served; the client chooses (the mcp SDK client picks 2026-07-28).
On the handshake protocol, a server-initiated request mid-call needs a
stateful session with SSE responses; the 2026-07-28 protocol asks the client by
multi-round-trip instead and works in every mode. Today this matters for
[destructive-tool confirmation](#confirming-destructive-tools).

`run_http` runs on asyncio only, because uvicorn does. Under trio, serve
`asgi_app(app)` with hypercorn.

### Server metadata

Every entry point takes the same `ServerOptions` keywords. Besides the tool
filters they carry the server's identity, as a client shows it:

```python
from warpweft.mcp import Icon, asgi_app

mcp = asgi_app(
    app,
    name="billing",
    version="2.3.0",
    title="Billing",
    description="Invoices and refunds for one tenant.",
    instructions="Look an account up with billing__lookup before any refund.",
    website_url="https://example.com/billing",
    icons=[Icon("https://example.com/billing.svg", mime_type="image/svg+xml", sizes=["any"])],
)
```

| Option | What the client sees |
| --- | --- |
| `name`, `version` | `serverInfo.name` / `.version` (defaults `warpweft` / `0`) |
| `title`, `description` | a human-friendly name and a one-line description in the UI |
| `instructions` | guidance for the model, returned with `initialize` and `server/discover` |
| `website_url` | a link to the server's website |
| `icons` | the server's icons |

`instructions` is the one field written for the model: clients typically put it
in the system prompt. Say when to use which tool, in which order, and what to
avoid; do not repeat the tool descriptions, the model has those too.

An `Icon` takes an `https:` URL or a `data:` URI with the image inline, an
optional `mime_type`, `sizes` (`"48x48"`, or `"any"` for a scalable image) and a
`theme` (`"light"` or `"dark"`). The MCP spec has clients reject every other
scheme, `http:` included, so `Icon` rejects one when it is defined. On the
2026-07-28 protocol the SDK stamps `serverInfo`, icons included, into `_meta` of
every response: give the server `https:` URLs and keep `data:` URIs for tool
icons, which travel once per `tools/list`. `@tool(icons=[...])` and an action's
`icons` attribute put icons on a tool.

### Command line

The `warpweft` command serves an app without a launcher script:

```
warpweft serve --app myapp.main:app                       # stdio
warpweft serve --app myapp.main:app --transport http --port 8000
warpweft serve --app myapp.main:app --tag public --confirm-destructive
warpweft serve --app myapp.main:app --title Billing --instructions-file mcp-instructions.md
```

`--tag`, `--include` and `--exclude` take one value each and can be repeated.
`--name`, `--title` and `--instructions-file` set the
[server metadata](#server-metadata); the instructions come from a file because
they are usually several lines.
`warpweft serve --help` lists the HTTP options.

### Custom transports

`build_server(app)` returns a transport-agnostic MCP server. `open_server(app)`
wraps it in the shared lifecycle - start the app, open the task runner, build
the server - for a transport adapter of your own.

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
| `warpweft.error` | a stable code: `invalid_arguments`, `circuit_open`, `timeout`, `retry_exhausted`, `unavailable`, `transient`, `permanent`, `error` - plus `declined` / `confirmation_unsupported` from the [destructive-tool gate](#confirming-destructive-tools), and `unknown_task` / `not_ready` / `cancelled` / `failed` from the [background task tools](#background-long-running-tools) (`not_ready` is the only retryable one) |
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

This is all plain `tools/call` returning `CallToolResult`, so it works with
every client and transport. Everything else about a tool still applies:
arguments are validated before submit, `report_progress` becomes task status
(not client notifications, since the request already returned), and
`confirm_destructive` gates the submit so a task never starts without consent.

### Native MCP tasks

A client that supports the [MCP Tasks extension](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
(`io.modelcontextprotocol/tasks`, on the 2026-07-28 protocol) gets the same
background tools natively, with no change to the component. The server
advertises the extension whenever a background tool is served, and for a client
that declares it:

- `tools/list` shows a background tool as an ordinary tool with its real output
  schema; the `task_*` tools are left out.
- `tools/call` answers with a task handle (`resultType: "task"`) instead of a
  `task_id` payload.
- The client polls `tasks/get` and receives the tool's own result once the task
  completes - exactly what an inline call would have returned. `tasks/cancel`
  cancels the job.

Native tasks and the `task_*` tools share one runner and store, so a task
started on either surface can be read on the other, each in its own advertised
shape. A running task reports `ttlMs: null`, since it never expires; a settled
one reports its run time plus the retention. A tool error
completes the task with an `isError` result, as the extension requires;
`failed` is reserved for a job that crashed without a result. Two optional
parts of the extension are not implemented: `input_required` tasks (a job never
asks for input mid-flight, so `tasks/update` accepts and ignores responses) and
`notifications/tasks` push updates (clients poll).

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
runner, raises a `FrameworkError` at build time. A task never expires while it
runs, however long it takes; once it settles, its state and result are retained
for five minutes (`DEFAULT_TTL_MS`). The default in-memory store keeps task
state and results for the process lifetime only - a deployment that
must survive a restart supplies its own `TaskStore` (Redis, Postgres, ...).
The same applies to an HTTP deployment with several workers: each worker has
its own in-memory store, so a poll can land on a worker that never saw the
task. Run one worker, or supply a shared store.

## Per-request axes (tenancy)

A [scoped component](composition.md) slices its state along [axes](runtime.md)
(a `tenant`, a region, ...). Over MCP those axis values have to come from the
call. `build_server`/`run_stdio` take `axis_binders` - a map from an
`AxisHandle` (returned by `app.axis(...)`) to a function that derives its value
from each call:

```python
tenant = app.axis("tenant")  # or app.axis("tenant", default="public")

server = build_server(
    app,
    axis_binders={tenant: lambda ctx, params: (params.arguments or {}).get("tenant")},
)
```

Over HTTP, the usual source is a request header. `header_binder` reads one, case
insensitively, and yields `None` on stdio, where there are no headers:

```python
from warpweft.mcp import asgi_app, header_binder

asgi_app(app, axis_binders={tenant: header_binder("x-tenant")})
```

The value is bound (via the axis's contextvar) around the call, so a scoped
component resolves the right slice; returning `None` leaves the axis to its
default (or, for a required axis, fails the call). The binder receives the
request context `ctx` and the call `params` (tool name and arguments), so
tenancy can come from a header, the auth context, or a tool argument. Over
HTTP, `ctx.request` is the Starlette request; on stdio it is `None`. With
`token_verifier` set, the verified token is available through the SDK's
`get_access_token()`. A background submit binds the axis across `runner.start`, so the
[background job](#background-long-running-tools) inherits the submit-time value.
A binder that raises becomes a tool error, never a transport failure.

Without `axis_binders`, an axis value must instead be bound by the host around
the call - e.g. ASGI middleware that reads the tenant and enters
`tenant.use(...)` before dispatching to the MCP handler. `axis_binders` just
moves that wiring into the server so stdio and non-ASGI hosts get it too.

## Confirming destructive tools

```python
build_server(app, confirm_destructive=True)
```

puts a human in the loop for every tool marked `destructive=True`: before
executing, the server sends an MCP elicitation that the client presents to
the user. Only an explicit accept runs the tool - decline or cancel returns a
tool error (`warpweft.error: "declined"`) without executing anything.
Confirmation happens after argument validation - and before a
[background](#background-long-running-tools) submit - and never applies to
non-destructive tools.

The question travels by whichever mechanism the protocol has:

| Protocol | How the user is asked | Where it works |
| --- | --- | --- |
| 2026-07-28 | the call returns an `InputRequiredResult`; the client asks the user and retries the call with the answer | every transport and mode |
| handshake (`initialize`) | the server sends `elicitation/create` mid-call | stdio, and HTTP with stateful sessions and SSE responses |

The gate fails closed: a client that does not support elicitation, or a
handshake connection that cannot carry the request (stateless HTTP, JSON
responses), gets `confirmation_unsupported` instead of an unconfirmed
execution.

Off by default - the `destructive` annotation alone only *hints* to the host
UI; this flag turns it into an enforced contract.
