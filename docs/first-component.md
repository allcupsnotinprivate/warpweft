# Write your first component

A component wraps one integration - an HTTP API, a database, a queue. You write
the *call*; the framework wraps every call with retry, timeout, circuit
breaking, caching and metrics, driven by config. This guide takes you from
nothing to a tested component with retry, cache and metrics.

## 1. Write the component

Subclass `AComponent[Settings, In, Out]`, declare your settings as a pydantic
model, and mark entry points with `@invocable`. Put resource setup in `start` /
`stop`; the method itself just makes the call.

```python
from pydantic import BaseModel
from warpweft import AComponent, invocable


class ProfilesSettings(BaseModel):
    base_url: str


class Profiles(AComponent[ProfilesSettings, str, dict]):
    async def start(self) -> None:
        self._client = make_client(self.settings.base_url)  # your real client

    async def stop(self) -> None:
        await self._client.aclose()

    @invocable
    async def get(self, user_id: str) -> dict:
        resp = await self._client.get(f"/users/{user_id}")
        resp.raise_for_status()
        return resp.json()
```

That is the whole component: it says nothing about retries or caching, and even
the name is derived (`Profiles` → `"profiles"`; set `name = "..."` to override).

Everything else is **opt-in**, added only when you need it:

- `endpoint()` - return the resolved host when several instances talk to the
  same system and should *share* breaker/concurrency state; by default each
  instance keeps its own.
- `stub(ctx)` - a synchronous fallback value served during an outage of an
  `optional` component, activated by a `policy.degradation` config block; see
  [composition.md](composition.md).
- a typed dependency - annotate an attribute with another component's type and
  the container injects the live instance:

  ```python
  class Search(AComponent[SearchSettings, str, list[dict]]):
      profiles: Profiles  # dependency, injected at start

      @invocable
      async def find(self, user_id: str) -> list[dict]:
          owner = await self.profiles.get(user_id)  # typed, no strings
          ...
  ```
- typed **field formats** - annotate a string parameter with a format so its
  schema says what shape it is and the value is validated:

  ```python
  from warpweft.formats import Ipv4, Uuid


  @invocable
  async def lookup(self, host: Ipv4, ticket: Uuid) -> Report: ...
  ```

  The input schema then carries `"format": "ipv4"` (an LLM or config author
  sees the intent), and a bad value is rejected. Only the standard JSON Schema
  string formats ship (date, email, uri, uuid, ipv4/6, hostname, regex, ...);
  define your own with `Annotated[str, Format("my-format", "...", validator)]` -
  there is no registry.
- `criticality`, `lifetime`/`scope`, `defaults`, `version` - see
  [composition.md](composition.md).

## 2. Turn on resilience with config

Resilience is deployment config, not code. Each link is on when you give it
settings:

```python
config = {
    "profiles": {
        "base_url": "https://api.example.com",
        "policy": {
            "retry": {"attempts": 3, "base_delay": 0.1, "max_delay": 1.0},
            "cache": {"ttl": 30.0, "max_entries": 1000},
        },
    },
}
```

## 3. Run it

```python
from warpweft import Container, Registry

registry = Registry()
registry.register(Profiles)

container = Container.build(registry, config)
await container.start()

outcome = await container.invoke("profiles", "get", user_id="42")
print(outcome.value, outcome.source)  # second identical call -> source == "cache"

# Typed alternative to string-based invoke - same chain underneath:
profiles = container.proxy(Profiles)
data = await profiles.get(user_id="42")  # keyword args; returns the value

# Raw instance (no retry/breaker/telemetry!) - for advanced wiring only:
raw = await container.get(Profiles)

await container.stop()
```

**Metrics and traces** are already there: the container instruments every call.
Configure an OpenTelemetry SDK (or pass providers to `Container.build`) and you
get a span per call, child spans per retry, and the `warpweft.*` metrics - see
[telemetry.md](telemetry.md). With no SDK it is a free no-op.

**Errors** map cleanly: retry repeats only transient failures; permanent ones
propagate. Raise `TransientError` / `PermanentError` from your method, or pass a
`classifier=` to `Container.build` that maps your client library's exceptions.

## 4. Test it - without a running system

`warpweft.testing.drive` runs your real component through its real chain on
an instant clock, so backoff never actually waits. Inject a fake client that
misbehaves and assert the outcome:

```python
import pytest
from warpweft.testing import drive


class FlakyClient:
    def __init__(self, fail: int, payload: dict) -> None:
        self.fail, self.payload, self.calls = fail, payload, 0

    async def get(self, path: str) -> "FlakyClient":
        self.calls += 1
        if self.calls <= self.fail:
            raise TransientError("upstream warming up")
        return self

    def raise_for_status(self) -> None: ...
    def json(self) -> dict:
        return self.payload


@pytest.mark.anyio
async def test_profiles_recovers_from_two_failures() -> None:
    profiles = Profiles(ProfilesSettings(base_url="https://api.example.com"))
    profiles._client = FlakyClient(fail=2, payload={"id": "42"})

    outcome = await drive(
        profiles,
        "get",
        config={"policy": {"retry": {"attempts": 3, "base_delay": 1.0, "max_delay": 5.0}}},
        user_id="42",
    )
    assert outcome.value == {"id": "42"}
    assert outcome.attempts == 3  # recovered on the 3rd try, zero real delay
```

To check a *policy* against a misbehaving service without writing a component,
use `drive_policy` with a scenario:

```python
from warpweft.testing import drive_policy, fails_then_succeeds

outcome = await drive_policy(
    {"retry": {"attempts": 5, "base_delay": 1.0, "max_delay": 8.0}},
    fails_then_succeeds(2, value="ok"),
)
assert outcome.attempts == 3
```

The testing package also exports `ManualClock`, `InstantClock`,
`FakeSettingsResolver`, an in-memory state store, and scenarios
(`always_transient`, `always_permanent`, `always_times_out`, `hangs`).

Installing warpweft also registers a small pytest plugin, so two fixtures are
available with no setup - `instant_clock` and `manual_clock`, each a fresh
clock per test:

```python
async def test_get_recovers(instant_clock):
    outcome = await drive(profiles, "get", clock=instant_clock, user_id="42")
    assert outcome.attempts == 3
    assert instant_clock.slept  # virtual backoff, zero real time
```

## Where next

- [composition.md](composition.md) - dependencies, lifecycle, per-tenant slices,
  health, layered config and introspection.
- [telemetry.md](telemetry.md) - the metrics and tracing contract.
