"""The ``warpweft`` command line: inspect, validate and serve an app.

Every command loads an `~warpweft.runtime.app.App` by import path
(``module:attribute``, like uvicorn). The inspection commands read it offline -
the container is built and validated but never started, so no component,
client or pool is touched. That makes ``warpweft check`` safe in CI and the
inspection commands safe against production config. Only ``serve`` starts the
app, to expose its tools over MCP (needs the ``mcp`` extra).

    warpweft check   --app myapp.main:app        # validate config + dependency graph
    warpweft list    --app myapp.main:app        # registered components
    warpweft describe [component] --app ...       # metadata, invocables, chains
    warpweft explain <component> <method> --app ...   # effective chain + provenance
    warpweft config  <component> --app ...        # resolved config (secrets masked)
    warpweft schema  <component> --app ...        # JSON Schema of the config model
    warpweft serve   --app ... [--transport http]  # serve @tool invocables over MCP

The ``--app`` value may also come from the ``WARPWEFT_APP`` environment variable.
Pass ``--json`` for machine-readable output.
"""

import argparse
import functools
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import anyio

from warpweft.core.errors import FrameworkError

from .app import App


def _load_app(spec: str) -> App:
    """Import ``module:attribute`` and return the App it names."""
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise SystemExit(f"error: --app must be 'module:attribute', got {spec!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f"error: cannot import '{module_name}': {exc}") from exc
    try:
        app = getattr(module, attr)
    except AttributeError:
        raise SystemExit(f"error: module '{module_name}' has no attribute '{attr}'") from None
    if not isinstance(app, App):
        raise SystemExit(f"error: '{spec}' is {type(app).__name__}, not a warpweft App")
    return app


def _emit(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def _cmd_check(app: App, args: argparse.Namespace) -> int:
    try:
        app.build()
    except (FrameworkError, ValueError) as exc:
        if args.json:
            _emit({"ok": False, "error": str(exc)})
        else:
            print(f"invalid: {exc}")
        return 1
    if args.json:
        _emit({"ok": True, "components": sorted(app.registry.names())})
    else:
        print(f"ok: {len(app.registry.names())} component(s) configured")
    return 0


def _cmd_list(app: App, args: argparse.Namespace) -> int:
    rows = []
    for name in sorted(app.registry.names()):
        d = app.registry.descriptor(name)
        rows.append(
            {
                "name": name,
                "version": d.identity.version,
                "criticality": d.criticality.value,
                "lifetime": d.lifetime.value,
                "dependencies": list(d.dependencies),
            }
        )
    if args.json:
        _emit(rows)
        return 0
    for row in rows:
        deps = f" -> {', '.join(row['dependencies'])}" if row["dependencies"] else ""
        print(f"{row['name']} (v{row['version']}, {row['criticality']}, {row['lifetime']}){deps}")
    return 0


def _describe_one(app: App, name: str) -> dict[str, Any]:
    d = app.registry.descriptor(name)
    container = app.build()
    invocables = {}
    for method, spec in d.invocables.items():
        invocables[method] = {
            "chain": list(container.explain(name, method).chain),
            "input_schema": spec.input_json_schema(),
            "output_schema": spec.output_json_schema(),
        }
    return {
        "name": name,
        "uid": d.identity.uid,
        "criticality": d.criticality.value,
        "lifetime": d.lifetime.value,
        "scope": list(d.scope.axes),
        "dependencies": list(d.dependencies),
        "invocables": invocables,
    }


def _cmd_describe(app: App, args: argparse.Namespace) -> int:
    names = [args.component] if args.component else sorted(app.registry.names())
    for name in names:
        if name not in app.registry:
            raise SystemExit(f"error: component '{name}' is not registered")
    reports = [_describe_one(app, name) for name in names]
    if args.json:
        _emit(reports if args.component is None else reports[0])
        return 0
    for report in reports:
        print(f"{report['name']} ({report['uid']}, {report['criticality']}, {report['lifetime']})")
        if report["scope"]:
            print(f"  scope: {', '.join(report['scope'])}")
        if report["dependencies"]:
            print(f"  dependencies: {', '.join(report['dependencies'])}")
        for method, info in report["invocables"].items():
            chain = " -> ".join(info["chain"]) or "(none)"
            print(f"  {method}(): {chain}")
    return 0


def _cmd_explain(app: App, args: argparse.Namespace) -> int:
    container = app.build()
    explanation = container.explain(args.component, args.method)
    if args.json:
        _emit({"chain": list(explanation.chain), "provenance": dict(explanation.provenance)})
        return 0
    print(f"{args.component}.{args.method}: {' -> '.join(explanation.chain) or '(no links)'}")
    if explanation.provenance:
        print("settings:")
        for path, source in sorted(explanation.provenance.items()):
            print(f"  {path} <- {source}")
    return 0


def _cmd_config(app: App, args: argparse.Namespace) -> int:
    container = app.build()
    config = container.config_json(args.component)  # secrets already masked
    if args.json:
        _emit(config)
        return 0
    print(json.dumps(config, indent=2, sort_keys=True))
    return 0


def _cmd_schema(app: App, args: argparse.Namespace) -> int:
    schema = app.registry.descriptor(args.component).config_json_schema()
    print(json.dumps(schema, indent=2, sort_keys=True))
    return 0


#: ``serve`` flags that only mean something for ``--transport http``, with their
#: defaults. argparse leaves them ``None``, so a flag the user passed is visible.
_HTTP_DEFAULTS: dict[str, Any] = {
    "host": "127.0.0.1",
    "port": 8000,
    "path": "/mcp",
    "stateless": False,
    "json_response": False,
}


def _cmd_serve(app: App, args: argparse.Namespace) -> int:
    given = [name for name in _HTTP_DEFAULTS if getattr(args, name) is not None]
    if args.transport == "stdio" and given:
        flags = ", ".join("--" + name.replace("_", "-") for name in given)
        raise SystemExit(f"error: {flags} only apply to --transport http")
    options: dict[str, Any] = {
        "tags": args.tag,
        "include": args.include,
        "exclude": args.exclude,
        "confirm_destructive": args.confirm_destructive,
        "background": not args.no_background,
        **_identity_options(args),
    }
    if args.transport == "stdio":
        # stdout is the protocol channel: nothing else may be printed to it.
        anyio.run(functools.partial(_mcp_entry_point("run_stdio"), app, **options))
    else:
        http = _HTTP_DEFAULTS | {name: getattr(args, name) for name in given}
        anyio.run(functools.partial(_mcp_entry_point("run_http"), app, **http, **options))
    return 0


def _identity_options(args: argparse.Namespace) -> dict[str, Any]:
    """The server-identity options the user set; an unset one keeps the library default."""
    options: dict[str, Any] = {k: v for k in ("name", "title") if (v := getattr(args, k)) is not None}
    if args.instructions_file is not None:
        try:
            options["instructions"] = Path(args.instructions_file).read_text(encoding="utf-8")
        except OSError as exc:
            raise SystemExit(f"error: cannot read --instructions-file: {exc}") from exc
    return options


def _mcp_entry_point(name: str) -> Any:
    """``warpweft.mcp.<name>``, importing only that transport - or exit asking for the extra."""
    try:
        import warpweft.mcp

        return getattr(warpweft.mcp, name)
    except ModuleNotFoundError as exc:
        raise SystemExit(f"error: 'serve' needs the mcp extra (pip install 'warpweft[mcp]'): {exc}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="warpweft", description="Inspect and validate a warpweft app.")
    parser.add_argument(
        "--app",
        default=os.environ.get("WARPWEFT_APP"),
        help="App import path 'module:attribute' (or set WARPWEFT_APP).",
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Validate config and the dependency graph, without starting.")
    sub.add_parser("list", help="List registered components.")

    describe = sub.add_parser("describe", help="Show component metadata, invocables and chains.")
    describe.add_argument("component", nargs="?", help="A component name; omit for all.")

    explain = sub.add_parser("explain", help="Show a method's effective chain and setting provenance.")
    explain.add_argument("component")
    explain.add_argument("method")

    config = sub.add_parser("config", help="Show a component's resolved config (secrets masked).")
    config.add_argument("component")

    schema = sub.add_parser("schema", help="Print the JSON Schema of a component's config.")
    schema.add_argument("component")

    serve = sub.add_parser("serve", help="Start the app and serve its @tool invocables over MCP.")
    serve.add_argument("--transport", choices=["stdio", "http"], default="stdio", help="Default: stdio.")
    serve.add_argument("--host", help="HTTP bind address. Default: 127.0.0.1.")
    serve.add_argument("--port", type=int, help="HTTP port. Default: 8000.")
    serve.add_argument("--path", help="HTTP endpoint path. Default: /mcp.")
    serve.add_argument("--stateless", action="store_true", default=None, help="HTTP: no per-client sessions.")
    serve.add_argument(
        "--json-response", action="store_true", default=None, help="HTTP: JSON bodies instead of SSE streams."
    )
    serve.add_argument("--tag", action="append", help="Serve only tools with this tag (repeatable).")
    serve.add_argument("--include", action="append", help="Serve only tools matching this glob (repeatable).")
    serve.add_argument("--exclude", action="append", help="Drop tools matching this glob (repeatable).")
    serve.add_argument(
        "--confirm-destructive", action="store_true", help="Ask the user before running a destructive tool."
    )
    serve.add_argument("--no-background", action="store_true", help="Serve without a background task runner.")
    serve.add_argument("--name", help="Server name reported to clients. Default: warpweft.")
    serve.add_argument("--title", help="Human-friendly server name for the client's UI.")
    serve.add_argument(
        "--instructions-file", metavar="PATH", help="File with guidance for the model (server instructions)."
    )
    return parser


_COMMANDS = {
    "check": _cmd_check,
    "list": _cmd_list,
    "describe": _cmd_describe,
    "explain": _cmd_explain,
    "config": _cmd_config,
    "schema": _cmd_schema,
    "serve": _cmd_serve,
}


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``warpweft`` command."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.app:
        parser.error("--app is required (or set WARPWEFT_APP)")
    app = _load_app(args.app)
    try:
        return _COMMANDS[args.command](app, args)
    except (FrameworkError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
