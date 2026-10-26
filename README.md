# warpweft

<p align="center">
  <em>A declarative framework for resilient, observable service components in async Python.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/warpweft/"><img src="https://img.shields.io/pypi/v/warpweft.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/warpweft/"><img src="https://img.shields.io/pypi/pyversions/warpweft.svg" alt="Supported Python versions"></a>
  <a href="https://github.com/allcupsnotinprivate/warpweft/actions/workflows/ci.yml"><img src="https://github.com/allcupsnotinprivate/warpweft/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://allcupsnotinprivate.github.io/warpweft/"><img src="https://img.shields.io/badge/docs-online-4c9aff.svg" alt="Documentation"></a>
  <a href="https://opensource.org/license/apache-2-0"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache-2.0"></a>
</p>

## The idea

In weaving, the **warp** threads are strung taut on the loom and the **weft** is
woven across them - together they make the cloth. warpweft applies that metaphor
to services:

> **You write the warp** - your component's business calls.
> **The framework weaves the weft** - retry, timeout, circuit breaking, caching,
> concurrency limiting, graceful degradation and telemetry - across *every* call.

The weave is declared in **configuration**, not scattered through your code. You
write a plain async method; warpweft turns it into a guarded, observable,
composable unit driven by policy. Change the resilience behaviour of a call by
editing config - environment, TOML or YAML - never the call site.

## Why warpweft

- **Resilience, declared** - retry, timeout, circuit breaker, caching,
  concurrency limiting and graceful degradation wrap every call, driven by config
  rather than boilerplate.
- **Observable by default** - OpenTelemetry spans and metrics emitted with
  consistent semantic conventions, no manual instrumentation.
- **Composition & lifecycle** - a component registry, dependency-injection
  container, dependency graph, ordered startup/shutdown and health checks.
- **Config-driven** - policies resolved from environment, TOML or YAML; the same
  component runs differently per deployment with zero code change.
- **Ergonomic units** - model a single callable operation as an `Action`, or
  shared infrastructure as a `Provider`, both built on the same core.
- **LLM-ready** - expose invocables as [Model Context Protocol](https://modelcontextprotocol.io/)
  tools with flat, typed schemas, over stdio or Streamable HTTP.
- **Async & typed** - built on [anyio](https://anyio.readthedocs.io/) (runs on
  asyncio and trio) and checked with `mypy --strict` throughout.

## Install

```bash
pip install warpweft          # core + runtime
pip install warpweft[mcp]     # + expose components as Model Context Protocol tools
pip install warpweft[yaml]    # + YAML config files
```

## Quickstart

Write a component's calls; declare their resilience in config:

```python
from pydantic import BaseModel

from warpweft import AComponent, App, component, invocable


class WeatherSettings(BaseModel):
    base_url: str


@component
class Weather(AComponent[WeatherSettings, str, dict]):
    @invocable
    async def forecast(self, city: str) -> dict:
        resp = await self._client.get(f"/forecast/{city}")
        resp.raise_for_status()
        return resp.json()


app = App(
    env_prefix="MYAPP",
    # retry is declared, not coded: 3 attempts with backoff, applied to every call
    config={"weather": {"policy": {"retry": {"attempts": 3, "base_delay": 0.1, "max_delay": 1.0}}}},
)

async with app.run():
    forecast = await app.proxy(Weather).forecast(city="oslo")  # runs through the chain
```

Everything useful is importable from `warpweft`; focused surfaces have their own
module - `warpweft.formats` (field formats), `warpweft.testing` (test helpers) and
`warpweft.mcp` (MCP tools, with the extra). The internal layers remain available
as `warpweft.core.*` and `warpweft.runtime.*`.

## License

Apache-2.0 - see [`LICENSE`](LICENSE). Contributions are welcome; see
[`CONTRIBUTING.md`](CONTRIBUTING.md).
