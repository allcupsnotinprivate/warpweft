"""warpweft.mcp.http: serving tools over Streamable HTTP.

Most tests drive the ASGI app in-process through httpx's ASGI transport and the
SDK client - no socket. That transport buffers a response body, so an SSE
stream cannot carry a server-initiated request mid-call; the one test that
needs that (elicitation) runs against a real uvicorn instead.
"""

from collections.abc import AsyncIterator
import contextlib
import socket
from typing import Any

import anyio
import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
import mcp.types as mt
import pytest
from starlette.applications import Starlette
from starlette.routing import Mount

from warpweft.core.component import AComponent, EmptySettings, invocable
from warpweft.core.composition import Registry
from warpweft.core.context import report_progress
from warpweft.core.errors import ConfigurationError, FrameworkError
from warpweft.mcp import MCPHttpApp, asgi_app, header_binder, run_http, tool
from warpweft.runtime import App, AxisHandle

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

#: DNS-rebinding protection (on for the default loopback host) accepts only a
#: Host header with a port, so the in-process base URL must carry one.
BASE_URL = "http://localhost:8000"


class Echo(AComponent[EmptySettings, str, str]):
    name = "echo"

    @tool(description="Echo the text.", read_only=True)
    @invocable
    async def say(self, text: str) -> str:
        await report_progress(1, total=1, message="echoed")
        return f"echo:{text}"

    @tool(description="Delete everything.", destructive=True)
    @invocable
    async def wipe(self) -> str:
        return "wiped"

    @tool(background=True, description="Slow job.")
    @invocable
    async def job(self, n: int) -> int:
        return n * 2


def echo_app() -> App:
    reg = Registry()
    reg.register(Echo)
    return App(registry=reg)


def tenant_app() -> tuple[App, AxisHandle]:
    """An app whose one tool returns the current ``tenant`` axis value."""
    holder: dict[str, AxisHandle] = {}

    class Svc(AComponent[EmptySettings, None, str]):
        name = "svc"

        @tool
        @invocable
        async def whoami(self) -> str:
            return holder["tenant"].current() or "unset"

    reg = Registry()
    reg.register(Svc)
    app = App(registry=reg)
    holder["tenant"] = app.axis("tenant", default="unset")
    return app, holder["tenant"]


async def accept(context: Any, params: Any) -> mt.ElicitResult:
    return mt.ElicitResult(action="accept", content={})


@contextlib.asynccontextmanager
async def http_client(
    asgi: Any,
    *,
    url: str = f"{BASE_URL}/mcp",
    mode: str = "auto",
    headers: dict[str, str] | None = None,
    elicitation_callback: Any = None,
) -> AsyncIterator[Client]:
    """A connected SDK client talking to ``asgi`` in-process (lifespan not entered here)."""
    transport = httpx2.ASGITransport(app=asgi)
    async with httpx2.AsyncClient(transport=transport, base_url=BASE_URL, headers=headers or {}) as http:
        mcp_transport = streamable_http_client(url, http_client=http)
        async with Client(mcp_transport, mode=mode, elicitation_callback=elicitation_callback) as client:
            yield client


@contextlib.asynccontextmanager
async def serving(mcp: MCPHttpApp, **client_kwargs: Any) -> AsyncIterator[Client]:
    """Enter the app's lifespan and yield a client connected to it."""
    async with mcp.lifespan(), http_client(mcp, **client_kwargs) as client:
        yield client


# --- tool calls over every transport shape ------------------------------------

TRANSPORTS = [
    pytest.param({}, id="stateful-sse"),
    pytest.param({"json_response": True}, id="stateful-json"),
    pytest.param({"stateless": True, "json_response": True}, id="stateless-json"),
]
PROTOCOLS = [pytest.param("legacy", id="handshake"), pytest.param("2026-07-28", id="per-request")]


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_lists_and_calls_tools(transport: dict[str, Any], protocol: str) -> None:
    async with serving(asgi_app(echo_app(), **transport), mode=protocol) as client:
        tools = await client.list_tools()
        assert {t.name for t in tools.tools} >= {"echo__say", "echo__wipe", "echo__job"}

        result = await client.call_tool("echo__say", {"text": "hi"})
        assert not result.is_error
        assert result.structured_content == {"result": "echo:hi"}


async def test_server_options_pass_through() -> None:
    async with serving(asgi_app(echo_app(), include={"echo__say"}, background=False)) as client:
        tools = await client.list_tools()
        assert [t.name for t in tools.tools] == ["echo__say"]


async def test_tool_errors_keep_their_guidance() -> None:
    async with serving(asgi_app(echo_app())) as client:
        result = await client.call_tool("echo__say", {})  # missing 'text'
        assert result.is_error
        assert result.meta is not None
        assert result.meta["warpweft.error"] == "invalid_arguments"


async def test_progress_reaches_the_client() -> None:
    seen: list[tuple[float, str | None]] = []

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        seen.append((progress, message))

    async with serving(asgi_app(echo_app()), mode="legacy") as client:
        await client.call_tool("echo__say", {"text": "hi"}, progress_callback=on_progress)
    assert seen == [(1, "echoed")]


async def test_background_tools_submit_and_poll() -> None:
    async with serving(asgi_app(echo_app())) as client:
        submitted = await client.call_tool("echo__job", {"n": 21})
        assert submitted.structured_content is not None
        task_id = submitted.structured_content["task_id"]

        with anyio.fail_after(5):
            while True:
                status = await client.call_tool("task_status", {"task_id": task_id})
                assert status.structured_content is not None
                if status.structured_content["status"] != "working":
                    break
                await anyio.sleep(0.01)
        result = await client.call_tool("task_result", {"task_id": task_id})
        assert result.structured_content == {"result": 42}


# --- per-request axes from headers ---------------------------------------------


@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_header_binder_binds_the_axis(transport: dict[str, Any]) -> None:
    app, tenant = tenant_app()
    mcp = asgi_app(app, axis_binders={tenant: header_binder("X-Tenant")}, **transport)
    async with serving(mcp, headers={"x-tenant": "acme"}) as client:
        result = await client.call_tool("svc__whoami", {})
    assert result.structured_content == {"result": "acme"}


async def test_header_binder_missing_header_leaves_the_default() -> None:
    app, tenant = tenant_app()
    async with serving(asgi_app(app, axis_binders={tenant: header_binder("x-tenant")})) as client:
        result = await client.call_tool("svc__whoami", {})
    assert result.structured_content == {"result": "unset"}


def test_header_binder_without_an_http_request_yields_none() -> None:
    bind = header_binder("x-tenant")
    params = mt.CallToolRequestParams(name="t", arguments={})

    class StdioCtx:  # what a stdio call looks like: no HTTP request
        request = None

    assert bind(StdioCtx(), params) is None
    assert bind(object(), params) is None


# --- bearer-token auth -----------------------------------------------------------


class TenantTokens:
    """A token verifier: ``good`` is a valid token whose ``tenant`` claim is ``acme``."""

    async def verify_token(self, token: str) -> AccessToken | None:
        if token != "good":  # noqa: S105 - a test fixture token
            return None
        return AccessToken(token=token, client_id="client", scopes=[], claims={"tenant": "acme"})


def tenant_from_token(ctx: Any, params: mt.CallToolRequestParams) -> str | None:
    token = get_access_token()
    return (token.claims or {}).get("tenant") if token else None


def authed(app: App, **kwargs: Any) -> MCPHttpApp:
    settings = AuthSettings(issuer_url="https://auth.example.com", resource_server_url=None)
    return asgi_app(app, auth=settings, token_verifier=TenantTokens(), **kwargs)


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_verified_token_reaches_the_axis_binder(transport: dict[str, Any], protocol: str) -> None:
    app, tenant = tenant_app()
    mcp = authed(app, axis_binders={tenant: tenant_from_token}, **transport)
    async with serving(mcp, mode=protocol, headers={"authorization": "Bearer good"}) as client:
        result = await client.call_tool("svc__whoami", {})
    assert result.structured_content == {"result": "acme"}


async def test_an_invalid_token_is_rejected() -> None:
    app, _ = tenant_app()
    mcp = authed(app)
    transport = httpx2.ASGITransport(app=mcp)
    async with mcp.lifespan(), httpx2.AsyncClient(transport=transport, base_url=BASE_URL) as http:
        response = await http.post("/mcp", json={}, headers={"authorization": "Bearer forged"})
    assert response.status_code == 401


# --- destructive-tool confirmation ---------------------------------------------


@pytest.mark.parametrize(
    ("transport", "protocol"),
    [
        pytest.param({"stateless": True}, "legacy", id="stateless"),
        pytest.param({"json_response": True}, "legacy", id="json-response"),
        pytest.param({}, "2026-07-28", id="per-request-protocol"),
    ],
)
async def test_confirmation_without_a_back_channel_fails_closed(transport: dict[str, Any], protocol: str) -> None:
    # The client can elicit, but the connection cannot carry the request: the
    # tool is refused as a tool error - never run, never a transport failure.
    mcp = asgi_app(echo_app(), confirm_destructive=True, **transport)
    async with serving(mcp, mode=protocol, elicitation_callback=accept) as client:
        result = await client.call_tool("echo__wipe", {})
    assert result.is_error
    assert result.meta is not None
    assert result.meta["warpweft.error"] == "confirmation_unsupported"


@pytest.mark.e2e
@pytest.mark.parametrize("anyio_backend", ["asyncio"])  # uvicorn is asyncio-only
async def test_confirmation_over_a_real_server(anyio_backend: str) -> None:
    import uvicorn

    mcp = asgi_app(echo_app(), confirm_destructive=True)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(mcp, log_level="warning", lifespan="on"))
    async with anyio.create_task_group() as tg:
        tg.start_soon(server.serve, [sock])  # drives MCPHttpApp's ASGI lifespan
        try:
            with anyio.fail_after(10):
                while not server.started:  # noqa: ASYNC110 - uvicorn exposes a flag, not an event
                    await anyio.sleep(0.01)
                url = f"http://127.0.0.1:{port}/mcp"
                async with Client(url, mode="legacy", elicitation_callback=accept) as client:
                    result = await client.call_tool("echo__wipe", {})
        finally:
            server.should_exit = True
    assert not result.is_error
    assert result.structured_content == {"result": "wiped"}


# --- lifecycle ------------------------------------------------------------------


async def test_requests_before_the_lifespan_get_503() -> None:
    mcp = asgi_app(echo_app())
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=mcp), base_url=BASE_URL) as http:
        response = await http.post("/mcp", json={})
    assert response.status_code == 503


async def test_lifespan_runs_the_app_only_inside_the_block() -> None:
    app = echo_app()
    mcp = asgi_app(app)
    async with mcp.lifespan():
        assert app.container is not None
    with pytest.raises(ConfigurationError):
        _ = app.container
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=mcp), base_url=BASE_URL) as http:
        assert (await http.post("/mcp", json={})).status_code == 503


async def test_one_instance_serves_one_lifespan_at_a_time() -> None:
    mcp = asgi_app(echo_app())
    async with mcp.lifespan():
        with pytest.raises(FrameworkError, match="already running"):
            async with mcp.lifespan():
                pass


async def test_mounted_into_a_host_app() -> None:
    mcp = asgi_app(echo_app())
    host = Starlette(routes=[Mount("/api", app=mcp)], lifespan=mcp.lifespan)  # the host runs the MCP lifespan
    async with host.router.lifespan_context(host), http_client(host, url=f"{BASE_URL}/api/mcp") as client:
        result = await client.call_tool("echo__say", {"text": "mounted"})
    assert result.structured_content == {"result": "echo:mounted"}


@pytest.mark.parametrize("anyio_backend", ["trio"])
async def test_run_http_refuses_trio(anyio_backend: str) -> None:
    with pytest.raises(FrameworkError, match="asyncio"):
        await run_http(echo_app())
