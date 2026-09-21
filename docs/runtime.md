# Runtime

The application layer over the core: one `App` object owns the registry, the
configuration and the container lifecycle. The core stays explicit and
instance-scoped; the runtime adds the convenient defaults an application wants.

## Declaring components

```python
from warpweft import App, component


@component  # module-level: registers into the default registry
class Weather(AComponent[WeatherSettings, str, str]):
    @invocable
    async def forecast(self, city: str) -> str: ...
```

The module-level `@component` writes to a process-wide **default registry**
(the convenient path, like Celery's `shared_task`). For full isolation -
several independent apps in one process, hermetic tests - give the app its own
registry and use `@app.component`:

```python
app = App(registry=Registry())


@app.component
class Weather(...): ...
```

## Autodiscovery

Registration stays explicit; discovery only removes the manual import list:

```python
app.autodiscover("myapp.components")  # imports the package's modules
```

Modules with a leading underscore are skipped. There is deliberately no
metaclass auto-registration: subclassing is not intent to register.

## Configuration

`App` merges config layers field by field, each overriding the previous: a
**config file**, the **programmatic config**, a **`.env` file**, then the
**environment**.

```python
app = App(
    config_file="warpweft.toml",  # .toml / .json / .yaml (yaml via the extra)
    config={"weather": {"policy": {"retry": {...}}}},
    dotenv=".env",  # optional
    env_prefix="MYAPP",  # highest layer
)
```

Reading is done by **pydantic-settings** (each source read into a dict);
merging, provenance and validation stay in the core, so a bad value is still
reported with its field path *and* which source it came from. A config file's
shape is the same `{component: {field: value}}` mapping.

Environment convention: `<PREFIX>_<COMPONENT>__<FIELD>[__<NESTED>...]`:

```
MYAPP_WEATHER__CITY_DEFAULT=reykjavik
MYAPP_WEATHER__POLICY__RETRY__ATTEMPTS=3
```

Env values arrive as strings and are coerced by the core's validation. Env vars
mapping to no component field are ignored. Every registered component is
included in the container - an absent config section means "all defaults".

YAML support needs the extra: `pip install warpweft[yaml]`.

All `Container.build` options (classifier, telemetry providers,
`framework_defaults`, a `SettingsResolver`, timeouts) pass through `App(...)`.
Third-party components can be pulled in with `app.load_entry_points()`.

## Axes and tenancy

`app.axis` registers a contextvar-backed axis and returns a handle to bind it
per request/task - the one-liner behind scoped components and `[axis]`-sliced
state:

```python
tenant = app.axis("tenant", default="public")  # optional; omit default to require it

with tenant.use("acme"):  # in a middleware/dependency
    data = await app.proxy(Reports).daily()  # scoped instances resolve to "acme"
```

Every axis carries a soft cap, `max_cardinality` (default 1000;
`app.axis("tenant", max_cardinality=50)`). The registry counts the distinct
values it resolves per axis and logs a single warning (logger
`warpweft.core.axes`) the first time a new value would exceed the cap -
resolution itself is never blocked. Treat the warning as a leak detector: axis
values feed per-slice state (scoped instances, breaker/concurrency slices) and
error-counter metric attributes, so unbounded values (user ids, request ids)
are a memory and cardinality leak.

## Correlation and budgets

```python
with app.correlation(request_id):  # every call in the block inherits it
    await app.invoke("weather", "forecast", city="oslo", budget=2.0)  # overall deadline

fast = app.proxy(Weather, budget=1.5)  # budget bound to every proxy call
```

`budget` is an overall deadline in seconds (retries included); the per-attempt
timeout link still bounds each attempt. `correlation_id` defaults to the
ambient one, then a fresh id.

## Lifecycle

```python
async with app.run() as container:
    outcome = await app.invoke("weather", "forecast", city="oslo")
    weather = app.proxy(Weather)  # typed facade through the chain
    raw = await app.get(Weather)  # raw instance - bypasses the chain!
```

### Embedding into a host

`app.lifespan` is a framework-agnostic lifespan: start on enter, stop (with
the drain) on exit. It accepts and ignores whatever the host passes, so the
bound method plugs into anything that takes a lifespan callable:

```python
app = App(env_prefix="MYAPP")
app.autodiscover("myapp.components")

api = FastAPI(lifespan=app.lifespan)  # or Starlette(lifespan=app.lifespan)
```

Nothing is published into the host - handlers reach the running system through
the `app` object they already have (`app.proxy(...)`, `app.invoke(...)`,
`app.container`). For hosts with startup/shutdown callback pairs instead of a
lifespan (e.g. aiohttp), call `await app.start()` / `await app.stop()` from
those callbacks; standalone scripts can simply do `async with app.lifespan():`
or `async with app.run() as container:`.

### Worker processes

For a process with no web host, `app.serve()` starts the app and runs until
`SIGINT`/`SIGTERM`, then drains and stops:

```python
async def main() -> None:
    app = App(env_prefix="MYWORKER")
    app.autodiscover("myworker.components")
    await app.serve()  # returns when a shutdown signal arrives


anyio.run(main)
```

Pass a `shutdown` event to drive it from your own code instead of signals.

## Command line

Warpweft provides a `warpweft` command that inspects and
validates an app **offline** - it builds and validates the container but never
starts it, so no component, client or pool is touched. Point it at your App by
import path (`module:attribute`, like uvicorn), or set `WARPWEFT_APP`:

```
warpweft check   --app myapp.main:app     # validate config + graph (exit 1 on error) - for CI
warpweft list    --app myapp.main:app     # registered components
warpweft describe [component] --app ...   # metadata, invocables, effective chains
warpweft explain <component> <method>     # a method's chain + where each setting came from
warpweft config  <component> --app ...    # resolved config, secrets masked
warpweft schema  <component> --app ...    # JSON Schema of the config model (editor autocomplete)
```

`--json` gives machine-readable output. `warpweft check` is the CI gate: it
fails with a non-zero exit and a field-path-and-source error on bad config.
