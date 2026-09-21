# Composition

The composition layer turns component *types* into a running system: a registry
of what exists, and a container of what is configured and running.

## Registry vs container

Two distinct things, deliberately not merged:

- **`Registry`** - which component types exist. Registration is explicit (the
  `@registry.register` decorator or `registry.register(cls)`), never an import
  side effect. Third-party components are discovered through entry points
  (group `warpweft.components`). Registering builds and caches the descriptor, so
  a malformed component fails at registration. Component names derive from the
  class name (`SearchService` → `search_service`) unless set explicitly, and
  dependencies may be declared as typed class annotations (see
  [first-component.md](first-component.md)).
- **`Container`** - what is configured and running in this process, built from
  configuration. It holds no global state, so a process can run several
  independent containers (tests included).

```python
container = Container.build(
    registry,
    {
        "embedder": {},
        "search": {"index": "primary", "policy": {"retry": {"attempts": 3, "base_delay": 0.0, "max_delay": 0.1}}},
    },
)
await container.start()
outcome = await container.invoke("search", "query", text="warpweft")
await container.stop()
```

The `configs` mapping is the deployment layer (see below); each assembled config
is validated against that type's dynamic config model (own fields + `policy`).

## Instance access

Three ways to reach a component, in order of preference:

- `container.invoke("name", "method", **args)` → full chain, returns `Outcome`
  (value + source/degraded/attempts).
- `container.proxy(Type)` → a typed facade; each `@invocable` method routes
  through `invoke` (chain included) but keeps the component's signatures for
  the type checker. Keyword arguments only; returns the outcome's *value*.
- `await container.get(Type)` → the raw live instance (scoped ones resolve for
  the current axis values). ⚠️ Direct method calls bypass the chain entirely -
  no retry, breaker or telemetry. Meant for advanced wiring, not for calls.

## Dependency graph

Validated at build time, before anything starts, with errors naming the pair:

- missing dependencies (a declared dependency is not configured);
- cycles (the error prints the cycle path);
- **scope leak** - a `process`-lifetime component cannot depend on a `scoped`
  one, or it would capture one scope's instance for everyone.

## Lifetime and scope

- `Lifetime.PROCESS` (default) - a single instance, started at `start()` in
  dependency order (independent branches in parallel), each with an init
  timeout. A required component failing aborts startup; an optional one is
  marked degraded and the system continues.
- `Lifetime.SCOPED` - one instance per axis key (the component's `scope`),
  created lazily on first use and evicted by LRU (calling `stop()`).

`stop()` drains in-flight calls, then stops everything in reverse order.

## The chain per method

Each invocable is wired through its links: the order comes from the config's
`policy.chain`, the method's effective policy restricts which links are allowed
(so a `health` method pinned to `("timeout",)` is never retried), and a link is
active only when it also has settings. Telemetry wraps the whole thing from the
outside. The base call binds the method, passing arguments from the context and
injecting the context itself if the method declares an `InvocationContext`
parameter.

## The endpoint axis (axes in action)

The built-in `endpoint` axis reflects which external system the current
*instance* talks to (`AComponent.endpoint()`), bound per invocation. Because a
single link store backs the container, `[endpoint]`-sliced state (circuit
breaker, concurrency's outer limit) is shared by instances on the same endpoint
and separated across different ones - with no component declaring it. Other axes
(e.g. a tenant axis for scoped components) are registered by the application.

## Layered configuration

A component's config is merged from four sources, each overriding the previous
**field by field** (never replacing a whole object):

1. framework defaults (`Container.build(framework_defaults=...)`),
2. the component author's defaults (`AComponent.defaults`),
3. the deployment config (the `configs` mapping),
4. the per-slice override, from a swappable `SettingsResolver` (production reads
   a database, tests read a dict via `DictSettingsResolver`).

The deployment layer is validated eagerly at build; per-slice overrides are
applied and re-validated per instance, and their results are cached per
`(component, scope_key)`. A slice override *tunes* existing fields - required
fields must come from the deployment config.

Merging records every leaf value's **provenance** (which source set it). A
validation error therefore names the component, the full field path, and the
source: `invalid config for component 'search' at 'policy.retry.attempts' (from
slice override): Input should be greater than 0`.

## Introspection

The container answers the questions that otherwise mean reading the core:

- `explain(component, method)` - the effective link chain for a method (after
  config order and the method's policy filter) plus the provenance of every
  config value.
- `resolved_settings(component, scope_key=...)` - an instance's fully merged
  config as a mapping.
- `snapshot()` - live runtime state: each breaker's state, each concurrency
  link's occupancy, and the live slice keys of scoped components.

Circuit breakers can also be driven by hand (mirrored on `App`):

- `force_open_breakers(endpoint=None)` / `reset_breakers(endpoint=None)` - open
  or close live breakers, either all of them or only those guarding one
  `endpoint`. Force-open re-arms `reset_timeout`; reset closes and clears the
  window. Both return the number touched - breakers are created lazily on first
  guarded call, so `0` means none are live yet. Manual changes are logged but
  emit no transition metric. Best-effort under concurrency: a probe that
  finishes just after `force_open` sees the open state and records nothing, and
  a failure racing `reset` lands in the fresh window - both harmless.

## Health

Liveness and readiness are separate:

- **Liveness** - is the process up? Dependencies are never consulted.
- **Readiness** - can the system serve? Every *required* component must be
  healthy; a degraded *optional* one is reported but does not break readiness.
  Health aggregates bottom-up: a component is downgraded when a *required*
  dependency is unhealthy, but an unhealthy *optional* dependency leaves it
  alone. A component's health check runs through a timeout only, never the full
  chain.
