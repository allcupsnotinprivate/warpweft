"""The warpweft CLI: app loading, each command, JSON output, error paths."""

import json
from pathlib import Path

import pytest

from warpweft.runtime.cli import main

pytestmark = pytest.mark.unit

APP = "sample_app.cli_target:app"


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# --- app loading -------------------------------------------------------------


def test_missing_app_argument_errors(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["check"])  # no --app, no WARPWEFT_APP


def test_app_from_env(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WARPWEFT_APP", APP)
    code, out, _ = run(capsys, "check")
    assert code == 0
    assert "ok" in out


def test_bad_app_spec(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="module:attribute"):
        main(["--app", "nocolon", "check"])


def test_unimportable_module(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="cannot import"):
        main(["--app", "no.such.module:app", "check"])


def test_missing_attribute(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="no attribute"):
        main(["--app", "sample_app.cli_target:ghost", "check"])


def test_attribute_is_not_an_app(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="not a warpweft App"):
        main(["--app", "sample_app.cli_target:not_an_app", "check"])


# --- check -------------------------------------------------------------------


def test_check_ok(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--app", APP, "check")
    assert code == 0
    assert "ok:" in out


def test_check_json(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--app", APP, "--json", "check")
    assert code == 0
    assert json.loads(out) == {"ok": True, "components": ["cli-api", "cli-reports"]}


def test_check_invalid_config_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--app", "sample_app.cli_bad:app", "check")
    assert code == 1
    assert "invalid" in out


def test_check_invalid_config_json(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--app", "sample_app.cli_bad:app", "--json", "check")
    assert code == 1
    assert json.loads(out)["ok"] is False


# --- list / describe ---------------------------------------------------------


def test_list(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--app", APP, "list")
    assert code == 0
    assert "cli-api" in out and "optional" in out


def test_list_json(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--app", APP, "--json", "list")
    rows = {row["name"]: row for row in json.loads(out)}
    assert rows["cli-api"]["criticality"] == "optional"
    assert rows["cli-reports"]["dependencies"] == ["cli-api"]


def test_describe_all(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--app", APP, "describe")
    assert code == 0
    assert "cli-api" in out
    assert "fetch()" in out


def test_describe_one_json_shows_chain_and_schema(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--app", APP, "--json", "describe", "cli-api")
    report = json.loads(out)
    assert report["name"] == "cli-api"
    assert report["invocables"]["fetch"]["chain"] == ["retry"]
    assert "path" in report["invocables"]["fetch"]["input_schema"]["properties"]


def test_describe_unknown_component(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="not registered"):
        main(["--app", APP, "describe", "ghost"])


def test_describe_shows_scope_and_dependencies(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--app", APP, "describe", "cli-reports")
    assert code == 0
    assert "scope: tenant" in out
    assert "dependencies: cli-api" in out


# --- explain / config / schema -----------------------------------------------


def test_explain(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--app", APP, "explain", "cli-api", "fetch")
    assert code == 0
    assert "retry" in out
    assert "policy.retry.attempts" in out


def test_explain_json(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--app", APP, "--json", "explain", "cli-api", "fetch")
    data = json.loads(out)
    assert data["chain"] == ["retry"]
    assert data["provenance"]["policy.retry.attempts"] == "deployment config"


def test_explain_with_no_configured_links(capsys: pytest.CaptureFixture[str]) -> None:
    # cli-reports has no policy config: empty chain, no provenance block.
    code, out, _ = run(capsys, "--app", APP, "explain", "cli-reports", "daily")
    assert code == 0
    assert "(no links)" in out
    assert "settings:" not in out


def test_config_masks_secrets(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--app", APP, "config", "cli-api")
    assert code == 0
    data = json.loads(out)
    assert data["api_key"] == "**********"  # never the real secret
    assert data["base_url"] == "https://api.example.com"


def test_config_json_flag(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--app", APP, "--json", "config", "cli-api")
    assert json.loads(out)["api_key"] == "**********"


def test_schema(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--app", APP, "schema", "cli-api")
    assert code == 0
    schema = json.loads(out)
    assert "base_url" in schema["properties"]
    assert "policy" in schema["properties"]


def test_command_error_is_reported(capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run(capsys, "--app", APP, "explain", "cli-api", "nope")
    assert code == 1
    assert "error:" in err


# --- serve ---------------------------------------------------------------------


def _record_calls(monkeypatch: pytest.MonkeyPatch, target: str) -> list[dict[str, object]]:
    """Replace ``warpweft.mcp.<module>.<function>`` with a recorder of its keyword arguments."""
    module, _, func = target.rpartition(".")
    calls: list[dict[str, object]] = []

    async def fake(app: object, **kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(f"warpweft.mcp.{module}.{func}", fake)
    return calls


def test_serve_stdio_is_the_default(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls = _record_calls(monkeypatch, "server.run_stdio")
    code, out, _ = run(capsys, "--app", APP, "serve", "--tag", "public", "--confirm-destructive")
    assert code == 0
    assert out == ""  # stdout is the protocol channel
    assert calls == [
        {"tags": ["public"], "include": None, "exclude": None, "confirm_destructive": True, "background": True}
    ]


def test_serve_http_passes_transport_options(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _record_calls(monkeypatch, "http.run_http")
    argv = ["serve", "--transport", "http", "--host", "0.0.0.0", "--port", "9000", "--path", "/tools"]  # noqa: S104
    code, _, _ = run(capsys, "--app", APP, *argv, "--stateless", "--json-response", "--no-background")
    assert code == 0
    assert calls == [
        {
            "host": "0.0.0.0",  # noqa: S104
            "port": 9000,
            "path": "/tools",
            "stateless": True,
            "json_response": True,
            "tags": None,
            "include": None,
            "exclude": None,
            "confirm_destructive": False,
            "background": False,
        }
    ]


def test_serve_reports_a_bad_tool_filter(capsys: pytest.CaptureFixture[str]) -> None:
    # The filter is checked before the app starts or stdio is touched.
    code, _, err = run(capsys, "--app", APP, "serve", "--tag", "no-such-tag")
    assert code == 1
    assert "error:" in err


def test_serve_without_the_mcp_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import warpweft.mcp

    def missing(name: str) -> object:
        raise ModuleNotFoundError("No module named 'mcp'")

    monkeypatch.setattr(warpweft.mcp, "__getattr__", missing)
    with pytest.raises(SystemExit, match=r"needs the mcp extra"):
        main(["--app", APP, "serve"])


def test_serve_rejects_http_flags_on_stdio() -> None:
    # --port without --transport http would silently start a stdio server.
    with pytest.raises(SystemExit, match=r"--port, --stateless only apply to --transport http"):
        main(["--app", APP, "serve", "--port", "9000", "--stateless"])


def test_serve_http_fills_unset_flags_with_defaults(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _record_calls(monkeypatch, "http.run_http")
    code, _, _ = run(capsys, "--app", APP, "serve", "--transport", "http")
    assert code == 0
    [call] = calls
    assert {k: call[k] for k in ("host", "port", "path", "stateless", "json_response")} == {
        "host": "127.0.0.1",
        "port": 8000,
        "path": "/mcp",
        "stateless": False,
        "json_response": False,
    }


def test_serve_passes_server_metadata(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    instructions = tmp_path / "instructions.md"
    instructions.write_text("Look the account up first.\n", encoding="utf-8")
    calls = _record_calls(monkeypatch, "server.run_stdio")
    argv = ["serve", "--name", "billing", "--title", "Billing", "--instructions-file", str(instructions)]
    code, _, _ = run(capsys, "--app", APP, *argv)
    assert code == 0
    [call] = calls
    assert {k: call[k] for k in ("name", "title", "instructions")} == {
        "name": "billing",
        "title": "Billing",
        "instructions": "Look the account up first.\n",
    }


def test_serve_reports_an_unreadable_instructions_file(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match=r"cannot read --instructions-file"):
        main(["--app", APP, "serve", "--instructions-file", str(tmp_path / "missing.md")])
