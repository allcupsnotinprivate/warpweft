"""Runnable demo: serve @tool invocables over Streamable HTTP.

``asgi_app(app)`` is an ASGI application: its lifespan starts the app, and
requests reach the same tool handlers as over stdio. A ``tenant`` axis is bound
per request from the ``X-Tenant`` header with ``header_binder``. To keep the
demo self-contained it serves on a free local port with uvicorn and connects
an MCP client to it; a real deployment runs ``run_http(app)`` or
``warpweft serve --app myapp.main:app --transport http``.

Run:
    uv run python examples/mcp_http.py
"""

import socket

import anyio
import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
import uvicorn

from warpweft import AComponent, App, EmptySettings, Registry, invocable
from warpweft.mcp import asgi_app, header_binder, tool


async def main() -> None:
    registry = Registry()
    app = App(registry=registry)
    tenant = app.axis("tenant", default="public")

    class Greeter(AComponent[EmptySettings, None, str]):
        name = "greeter"

        @tool(description="Greet the caller's tenant.", read_only=True)
        @invocable
        async def hello(self, name: str) -> str:
            return f"hello {name} from tenant {tenant.current()}"

    registry.register(Greeter)
    mcp = asgi_app(app, axis_binders={tenant: header_binder("x-tenant")})

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    url = f"http://127.0.0.1:{sock.getsockname()[1]}/mcp"
    server = uvicorn.Server(uvicorn.Config(mcp, log_level="warning"))

    async with anyio.create_task_group() as tg:
        tg.start_soon(server.serve, [sock])
        while not server.started:  # noqa: ASYNC110 - uvicorn exposes a flag, not an event
            await anyio.sleep(0.01)

        for who in (None, "acme"):
            headers = {"x-tenant": who} if who else {}
            async with (
                httpx2.AsyncClient(headers=headers) as http,
                Client(streamable_http_client(url, http_client=http)) as client,
            ):
                result = await client.call_tool("greeter__hello", {"name": "Ada"})
                print(f"X-Tenant={who!r:8} -> {result.structured_content}")

        server.should_exit = True


if __name__ == "__main__":
    anyio.run(main, backend="asyncio")  # uvicorn runs on asyncio
