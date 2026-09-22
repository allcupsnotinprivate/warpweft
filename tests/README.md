# Tests

The suite mirrors the `warpweft` package tree, so the tests for a module live at
the matching path:

```
tests/
  _support/              # shared helpers/factories/fakes - NOT collected
    otel.py              # in-memory OTel: tracing(), metering(), read(), by_name(), ...
    containers.py        # registry_of(), running_container()
  sample_app/            # a small app used as a fixture (not tests)
  core/                  # warpweft.core.*
    component/           #   .component  (test_component, test_invocable, test_descriptor, ...)
    composition/         #   .composition (test_container, test_registry, test_graph, ...)
    pipeline/            #   .pipeline   (test_chain, test_state)
      builtin/           #   .pipeline.builtin (test_retry, test_circuit_breaker, ...)
    telemetry/           #   .telemetry  (test_instrument, test_metrics, test_component, test_dependency)
    testing/             #   .testing    (test_testing, test_pytest_plugin)
    test_axes.py test_context.py test_clock.py test_errors.py ...
  actions/               # warpweft.actions
  mcp/                   # warpweft.mcp  (test_server, test_tasks, test_axes, test_schema)
  runtime/               # warpweft.runtime (test_app/test_runtime, test_cli)
  test_*.py              # cross-cutting suites (integration, smoke, public_api, ergonomics, introspection)
```

Test files are named `test_<module>.py` to match `<module>.py`. Suites that span
several packages (end-to-end smoke, public-API surface, ergonomics) stay at the
`tests/` root. `--import-mode=importlib` (set in `pyproject.toml`) lets the same
basename appear in different packages (e.g. `core/test_axes.py` and
`mcp/test_axes.py`).

## Tiers (unit / integration / e2e)

Every test carries a tier marker. You rarely write it by hand: the auto-marker in
`conftest.py` (`pytest_collection_modifyitems`) applies one unless the file/test
sets it explicitly:

- an **explicit** `pytestmark` / decorator always wins;
- otherwise the tier is **inferred from fixtures** - a test that requests
  `container`, `app`, `running_app` or `connect` is `integration`, anything else
  is `unit`;
- any `async def` test also gets the `anyio` marker automatically.

Rule of thumb for writing tests:

- **unit** - exercises one class/function with fakes; builds no container/app.
- **integration** - builds a `Container`/`App`/MCP session and checks real wiring.
- **e2e** - drives the MCP server over a client session, or the CLI as a process.

Prefer letting inference do the work: request the `container` fixture instead of
setting a manual marker. A file-level `pytestmark` is still fine when a whole
module is one tier.

## Running

```bash
uv run pytest                      # everything (both anyio backends: asyncio + trio)
uv run pytest -m unit              # fast, no I/O
uv run pytest -m integration       # container/app wiring
uv run pytest tests/core/telemetry # one package
uv run pytest --cov=warpweft --cov-report=term   # with the 90% coverage gate
```

CI runs the unit tier first (fast feedback, gate disabled), then the
integration+e2e tier with `--cov-append` so the 90% gate applies to the union.

## Shared support (`_support/`)

Import helpers from `_support` (it is on `sys.path` via the root `conftest.py`);
nothing under `_support/` is collected as a test.

- **`_support/otel.py`** - isolated in-memory OpenTelemetry: `tracing()`,
  `metering()`, `read()`, `scoped_metrics()`, `sole_point()`,
  `points_by_operation()`, `by_name()`. Never touch the OTel globals.
- **`_support/containers.py`** - `registry_of(*classes)` and the
  `running_container(*classes, config=..., **build_kwargs)` context manager.

Fixtures (in `conftest.py`):

- **`container`** - a factory; requesting it also marks the test `integration`:
  ```python
  async def test_it(container):
      provider, reader = metering()
      async with container(MyComponent, config={"my": {}}, meter_provider=provider) as c:
          await c.invoke("my", "work")
      # assert on reader after the container has stopped
  ```
- **`connect`** - yields the `connected(app)` context manager for MCP client tests.

`tests/core/telemetry/test_component.py` is the worked example of the `container`
fixture; `tests/core/telemetry/test_dependency.py` shows `registry_of`.

## Adding a test

1. Put it at the path mirroring the module under test (`test_<module>.py`).
2. Use fakes for a unit test; request `container`/`app`/`connect` for integration.
3. Don't set `anyio`/tier markers unless a whole file needs an explicit tier -
   the auto-marker handles the common case.
4. Reuse `_support` helpers instead of re-declaring providers/registries.
