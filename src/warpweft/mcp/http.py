"""Serve an app's MCP tools over Streamable HTTP.

The HTTP counterpart of `~warpweft.mcp.server.run_stdio`, for a remote or
shared deployment rather than a local subprocess host. Tool behaviour is
identical on both transports - the same `build_server` handlers run - so this
module only adds the HTTP plumbing:

- `asgi_app` returns an ASGI application. Serve it with any ASGI server
  (uvicorn, hypercorn) or mount it into an existing Starlette/FastAPI app.
- `run_http` serves it with uvicorn, mirroring ``run_stdio``.
- `header_binder` derives a per-request axis value (e.g. a tenant) from an
  HTTP request header.

Sessions, the Streamable HTTP wire protocol, DNS-rebinding protection, the
request-size limit and bearer-token auth all come from the mcp SDK; the app,
its task runner and the SDK's session manager share one lifespan, so the app
is running exactly while requests can reach it.
"""

from collections.abc import AsyncIterator
import contextlib
from contextlib import AbstractAsyncContextManager
from typing import Any, Unpack

from mcp.server.auth.provider import TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.streamable_http_manager import DEFAULT_MAX_SESSIONS, DEFAULT_SESSION_IDLE_TIMEOUT
from mcp.server.transport_security import TransportSecuritySettings
import mcp.types as mt
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

from warpweft.core.errors import FrameworkError
from warpweft.runtime import App

from .server import AxisBinder, ServerOptions, open_server

__all__ = ["MCPHttpApp", "asgi_app", "header_binder", "run_http"]


class MCPHttpApp:
    """An ASGI application serving an app's MCP tools at one path.

    Standalone, it is a complete ASGI app: an ASGI server drives `lifespan`
    through the ASGI lifespan protocol. Mounted into a Starlette/FastAPI host,
    the host must run it, because Starlette does not run the lifespan of
    mounted apps. It starts the app too, so it replaces ``app.lifespan``::

        mcp = asgi_app(app)
        api = FastAPI(lifespan=mcp.lifespan)  # not app.lifespan: this one starts the app
        api.mount("/", mcp)  # the endpoint is asgi_app's ``path`` under the mount

    Requests that arrive while the lifespan is not active get ``503``. One
    instance serves one lifespan at a time.
    """

    def __init__(self, app: App, *, background: bool, options: ServerOptions, http: dict[str, Any]) -> None:
        self._app = app
        self._background = background
        self._options = options
        self._http = http
        self._inner: Starlette | None = None
        # Starlette implements the ASGI lifespan protocol for the standalone case.
        self._asgi = Starlette(routes=[Mount("", app=self._dispatch)], lifespan=self.lifespan)

    def lifespan(self, *_: Any) -> AbstractAsyncContextManager[None]:
        """Run the app, its task runner and the MCP session manager for the block.

        Like ``App.lifespan`` it accepts and ignores whatever a host passes, so
        the bound method plugs into ``FastAPI(lifespan=...)`` directly.
        """
        return self._lifespan()

    @contextlib.asynccontextmanager
    async def _lifespan(self) -> AsyncIterator[None]:
        if self._inner is not None:
            raise FrameworkError("this MCP HTTP app is already running; one instance serves one lifespan at a time")
        async with open_server(self._app, background=self._background, **self._options) as server:
            inner = server.streamable_http_app(**self._http)
            async with inner.router.lifespan_context(inner):
                self._inner = inner
                try:
                    yield
                finally:
                    self._inner = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self._asgi(scope, receive, send)

    async def _dispatch(self, scope: Scope, receive: Receive, send: Send) -> None:
        inner = self._inner
        if inner is None:
            await PlainTextResponse("MCP server is not running", status_code=503)(scope, receive, send)
            return
        await inner(scope, receive, send)


def asgi_app(
    app: App,
    *,
    path: str = "/mcp",
    stateless: bool = False,
    json_response: bool = False,
    host: str = "127.0.0.1",
    transport_security: TransportSecuritySettings | None = None,
    auth: AuthSettings | None = None,
    token_verifier: TokenVerifier | None = None,
    session_idle_timeout: float | None = DEFAULT_SESSION_IDLE_TIMEOUT,
    max_sessions: int | None = DEFAULT_MAX_SESSIONS,
    background: bool = True,
    **options: Unpack[ServerOptions],
) -> MCPHttpApp:
    """Build an ASGI application serving the app's MCP tools at ``path``.

    The app is started by the returned application's lifespan, not here.

    Transport (passed to the mcp SDK):

    - ``stateless`` - no ``Mcp-Session-Id`` session per client: every request
      stands alone, so it scales across workers but the server cannot send
      requests back to the client mid-call (no elicitation).
    - ``json_response`` - answer with a single JSON body instead of an SSE
      stream. Same limitation: no server-initiated requests.
    - ``host`` - the address the server is reached on. For a loopback host
      (the default) DNS-rebinding protection is switched on automatically;
      pass ``transport_security`` to configure it explicitly.
    - ``auth`` / ``token_verifier`` - bearer-token auth as an OAuth resource
      server; the verified token is then visible to an `AxisBinder` through
      the SDK's ``get_access_token()``.
    - ``session_idle_timeout`` / ``max_sessions`` - stateful-session limits.

    ``background`` opens an in-memory task runner for
    ``@tool(background=True)`` tools (see `open_server`); with several
    workers each has its own, so either run one worker or keep background
    tools on a dedicated server. The remaining keyword arguments are
    `ServerOptions`, passed through to `build_server`.

    The client's ``initialize`` handshake and the 2026-07-28 per-request
    protocol are both served. Server-initiated requests - the elicitation behind
    ``confirm_destructive`` - need a stateful session with SSE responses on the
    handshake protocol; elsewhere a destructive tool is refused with
    ``confirmation_unsupported`` rather than run unconfirmed.
    """
    http: dict[str, Any] = {
        "streamable_http_path": path,
        "stateless_http": stateless,
        "json_response": json_response,
        "host": host,
        "transport_security": transport_security,
        "auth": auth,
        "token_verifier": token_verifier,
        "session_idle_timeout": session_idle_timeout,
        "max_sessions": max_sessions,
    }
    return MCPHttpApp(app, background=background, options=options, http=http)


async def run_http(
    app: App,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    log_level: str = "info",
    path: str = "/mcp",
    stateless: bool = False,
    json_response: bool = False,
    background: bool = True,
    **options: Unpack[ServerOptions],
) -> None:  # pragma: no cover - binds a real socket until shut down
    """Serve the app's MCP tools over HTTP with uvicorn until interrupted.

    A thin wrapper over `asgi_app` for the common case; for TLS, several
    workers, auth or a different ASGI server, build the app with `asgi_app` and
    serve it yourself. uvicorn runs on asyncio only: under trio, serve
    `asgi_app` with hypercorn instead.
    """
    import uvicorn

    _require_asyncio()
    asgi = asgi_app(
        app,
        path=path,
        stateless=stateless,
        json_response=json_response,
        host=host,
        background=background,
        **options,
    )
    server = uvicorn.Server(uvicorn.Config(asgi, host=host, port=port, log_level=log_level, lifespan="on"))
    await server.serve()
    if not server.started:
        # uvicorn logs a startup failure (e.g. the app failed to start) and returns.
        raise FrameworkError("the MCP HTTP server failed to start; see the log above")


def _require_asyncio() -> None:
    """Fail clearly when called outside an asyncio event loop (e.g. under trio)."""
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        raise FrameworkError(
            "run_http needs an asyncio event loop (uvicorn does not support trio); "
            "under trio, serve asgi_app(...) with hypercorn"
        ) from None


def header_binder(header: str) -> AxisBinder:
    """An `AxisBinder` that reads an axis value from an HTTP request header.

    ::

        tenant = app.axis("tenant")
        asgi_app(app, axis_binders={tenant: header_binder("x-tenant")})

    The lookup is case-insensitive. A missing header - or a transport without
    HTTP requests, such as stdio - yields ``None``, leaving the axis to its
    default (or failing the call when the axis is required).
    """

    def bind(ctx: Any, params: mt.CallToolRequestParams) -> str | None:
        request = getattr(ctx, "request", None)
        headers = getattr(request, "headers", None)
        if headers is None:
            return None
        value: str | None = headers.get(header)
        return value

    return bind
