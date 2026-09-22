# Telemetry

Warpweft instruments every invocation with OpenTelemetry: one span per call,
child spans per retry attempt, and five metrics. This page is the **stability
contract**: dashboards and alerts are built on these names, so renaming any
span, event, attribute or metric listed here is a breaking change.

## Design rules

- **Not a link. Always outside, always on.** Telemetry wraps the assembled
  chain from the outside (`warpweft.core.telemetry.instrument.instrument`) and
  is not part of the policy chain. A link inside the chain could not see a
  circuit-breaker rejection the call never reached, nor measure total latency
  with retries included. There is no setting to turn it off.
- **Instrumented is not collecting.** The core depends on `opentelemetry-api`
  only. Until the host application configures an SDK (or passes explicit
  providers), every span and metric is a no-op costing microseconds. The
  collection switch belongs to the application, not to component settings.
- **The pipeline is OTel-free.** Links emit through the neutral
  `warpweft.core.observe.Observer` seam found in the context `bag`; only the
  `warpweft.core.telemetry` package imports OpenTelemetry.

## Enabling collection

```python
from warpweft.core.telemetry.instrument import instrument

chain = build_chain([...], store, registry, base)
wrapped = instrument(chain)  # uses OTel globals
wrapped = instrument(
    chain,
    tracer_provider=tp,  # or explicit providers
    meter_provider=mp,
)
```

Configure an SDK the usual OTel way (globals or explicit providers). Without
one, `instrument` is a transparent pass-through. See
`examples/telemetry_console.py` for a runnable console setup.

## Spans

| Span | Name | Attributes |
|---|---|---|
| Invocation | `ctx.operation` | start: `warpweft.operation`, `warpweft.correlation_id`, `warpweft.axis.<name>` per scope pair; end: `warpweft.source`, `warpweft.degraded`, `warpweft.attempts`, `warpweft.cache` (if the cache link ran); on error: `warpweft.error.class` |
| Attempt | `warpweft.attempt` | `warpweft.attempt.number` (numbering starts at 1) |

A successful span keeps status `UNSET` (per OTel spec, instrumentation does
not set `OK`). A failing span gets status `ERROR` and an `exception` event -
including each failed attempt span, not just the invocation.

## Events

| Event | Emitted by | Attributes |
|---|---|---|
| `warpweft.retry.backoff` | retry, right before the backoff sleep (lands on the invocation span) | `warpweft.backoff.delay` (seconds), `warpweft.attempt.number` (upcoming attempt) |
| `warpweft.circuit_breaker.rejected` | circuit breaker, on rejecting a call | `warpweft.circuit_breaker.state` = `open` \| `half_open` |
| `warpweft.circuit_breaker.transition` | circuit breaker, on every state change while processing a call | `warpweft.circuit_breaker.state.from`, `warpweft.circuit_breaker.state.to` (each `closed` \| `open` \| `half_open`) |

## Metrics

| Metric | Instrument | Unit | Attributes | Axis values |
|---|---|---|---|---|
| `warpweft.calls` | Counter | `{call}` | `warpweft.operation`, `warpweft.status` (`ok`/`error`), `warpweft.source`, `warpweft.degraded`; `warpweft.error.class` when status=`error` | when status=`error`, or allowlisted |
| `warpweft.call.duration` | Histogram | `s` | `warpweft.operation`, `warpweft.status` | only allowlisted |
| `warpweft.degradations` | Counter | `{call}` | `warpweft.operation` | always |
| `warpweft.circuit_breaker.rejections` | Counter | `{rejection}` | `warpweft.operation`, `warpweft.circuit_breaker.state` | always |
| `warpweft.circuit_breaker.transitions` | Counter | `{transition}` | `warpweft.operation`, `warpweft.circuit_breaker.state.from`, `warpweft.circuit_breaker.state.to` | always |

`warpweft.circuit_breaker.rejections` counts every rejection **at the moment
the breaker rejects**, not when an exception escapes: an outer retry may
recover from a rejection, and a degradation stub may swallow it - the counter
still moves.

`warpweft.circuit_breaker.transitions` counts the four organic state changes
that happen while processing a call: `closed`→`open`, `open`→`half_open`,
`half_open`→`closed` and `half_open`→`open` (at most four series per
operation/axes - bounded). **Manual** transitions via
`Container.force_open_breakers` / `reset_breakers` happen outside any
invocation and are logged only - no metric point.

## Dependency calls

Telemetry is end to end: when the container injects a dependency into a
component, it injects a lightweight telemetry proxy, not the bare instance.
Every call to one of the dependency's `@invocable` methods then carries the
full observability contract above - a `<component>.<method>` span (a child of
the caller's invocation span, inheriting its correlation id), the
`warpweft.calls` and `warpweft.call.duration` metrics with the same attributes
and axis cardinality policy, and the `span_enricher` hook. Like the rest of
telemetry, this is always on and free of cost until an SDK is configured.

What the proxy deliberately does **not** do is run policy links: the
dependency's retry, breaker or cache would double up with the caller's own
chain, so a raw dependency call stays raw - it only becomes visible. An error
in the dependency propagates unchanged (recorded as an error point with its
`warpweft.error.class`); recovering from it is the caller's chain's business.
Guarded invocations (`invoke` / `proxy`) are untouched: their chains are built
on the bare instance, so nothing is counted twice.

Boundaries to know:

- Calls a component makes to **itself** (`self.method()`) are not proxied and
  stay uninstrumented - telemetry sits on the component boundary.
- `container.get(...)` returns the bare instance; only *injected* dependencies
  are proxied.
- `isinstance` checks against the dependency's class keep working on the proxy
  (it reports the wrapped instance's class as its `__class__`), but
  `type(dep) is DepClass` does not, and the proxy cannot be subclassed.
- A callable dependency (an `Action`) called as `self.dep(...)` routes through
  its own invoker and full chain - instrumented there, not by the proxy.

## Component metrics

Components record their own domain metrics - tokens consumed, rows synced -
through `self.telemetry`, with no meter plumbing:

```python
class Llm(AComponent[LlmSettings, Prompt, str]):
    @invocable
    async def complete(self, prompt: str) -> str:
        answer, used = await self._client.complete(prompt)
        self.telemetry.counter("o2.llm.tokens", unit="{token}").add(used, {"model": self._model})
        return answer
```

`counter(name, *, unit="", description="")` and
`histogram(name, *, unit="", description="")` return cached-by-name
instruments with `.add(value, attributes=None)` / `.record(value,
attributes=None)`. Metric names are entirely yours - no prefix is imposed.

Every recorded point automatically carries:

- `warpweft.component` - the component's name;
- `warpweft.axis.<name>` for the instance's slice (scoped components), subject
  to the same `axis_allowlist` cardinality policy as the framework metrics -
  values outside the allowlist are dropped.

User attributes merge underneath the automatic ones and cannot overwrite them.

The container binds a live channel to every instance it creates (a scoped
component gets one per slice, so per-tenant breakdown needs no code), recording
through the container's meter provider under the instrumentation scope
`warpweft.component` - separate from the `warpweft` scope, whose metric names
are the framework's stability contract. A component constructed directly (unit
tests) gets a process-wide no-op: every call is accepted, nothing is recorded,
nothing fails.

## Enriching the invocation span

The framework attaches only its own vendor-neutral attributes. To record
application data - a call's input and output, a tenant's plan, a tracing
vendor's conventions (Langfuse, etc.) - pass a `span_enricher` host hook. It is
a callable

```python
(span, ctx, outcome, exc) -> None
```

invoked **once at the end** of every invocation:

- on success as `(span, ctx, outcome, None)` - after the framework's own
  attributes are set, right before the outcome is returned;
- on error as `(span, ctx, None, exc)` - after `warpweft.error.class` is set,
  right before the exception propagates.

Map `ctx.arguments` (bound input) and `outcome.value` (output) onto whatever
attributes your tracing backend expects; warpweft stays vendor-neutral. The
hook's failures are **swallowed** (logged at `debug`) so enrichment can never
break a call, and **cancellation bypasses it**, just as it bypasses the
metrics.

Pass it wherever the wrapper is configured - directly to `instrument`, or
through the container / app, which forward it to `instrument`:

```python
def enrich(span, ctx, outcome, exc):
    if outcome is not None:
        span.set_attribute("app.output", outcome.value)


wrapped = instrument(chain, span_enricher=enrich)
# or, end to end:
app = App(span_enricher=enrich)  # forwarded to Container.build
container = Container.build(registry, configs, span_enricher=enrich)
```

## Cardinality policy

Axis values as metric attributes are the classic way to explode a time-series
database, so:

- **Error, rejection and transition counters carry axes always** - failures
  are where the per-slice breakdown pays off, and their volume is expected to
  be low.
- **The duration histogram (and ok-status call counts) never carry axes by
  default.**
- `TelemetryConfig(axis_allowlist=frozenset({...}))` grants full breakdown to
  specific axis *values*: matching pairs are then attached to the histogram
  and ok-status counts too. The allowlist is value-based; a value shared by
  two different axes (e.g. `"prod"`) unlocks both pairs.

## Semantics worth knowing

- **Cancellation** is not an error: a cancelled call closes its span with
  status `UNSET` and records no metric point.
- **Coalesced cache waiters** (single-flight) get their own invocation span
  that shows only the wait; the real work happened under the leader's spans.
- Duration is measured with the framework `Clock` (explicit parameter, else
  `ctx.clock`, else the system clock), so tests never sleep for real.
