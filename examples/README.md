# Warpweft usage examples

- `runtime_app.py` - the App: `@component`, config (programmatic + env), a typed proxy, introspection.
- `actions.py` - an `Action` as a callable unit and as an MCP tool, with a `Provider` dependency (with the `mcp` extra).
- `testing_a_component.py` - testing a component with `warpweft.testing` (drive + failure scenarios).
- `mcp_tools.py` - exposing `@tool` invocables over MCP (with the `mcp` extra).
- `telemetry_console.py` - OTel SDK console exporters around an App: spans + metrics on stdout (needs `opentelemetry-sdk`).
