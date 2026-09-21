"""Shared fixtures: both anyio backends, sample package, and an MCP client session."""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
import sys
from typing import Any

import anyio
from mcp import ClientSession
from mcp.client.session import ElicitationFnT
from mcp.shared.memory import create_client_server_memory_streams
import pytest

from warpweft.mcp import build_server
from warpweft.runtime import App

#: Enables the ``pytester`` fixture used by tests/test_pytest_plugin.py.
pytest_plugins = ["pytester"]

sys.path.insert(0, str(Path(__file__).parent))  # makes `sample_app` importable


@pytest.fixture(params=["asyncio", "trio"])
def anyio_backend(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def connect() -> Callable[..., AbstractAsyncContextManager[ClientSession]]:
    """Return the ``connected(app)`` context manager for a test to use."""
    return connected


@asynccontextmanager
async def connected(
    app: App, *, elicitation_callback: ElicitationFnT | None = None, **server_kwargs: Any
) -> AsyncIterator[ClientSession]:
    """Start the app, run its MCP server, yield a connected client session.

    Keyword arguments are passed through to ``build_server`` (e.g. ``tags``).
    ``elicitation_callback`` configures the client side; passing one makes the
    client advertise the elicitation capability.
    """
    async with app.run(), create_client_server_memory_streams() as (client_streams, server_streams):
        server = build_server(app, **server_kwargs)
        server_read, server_write = server_streams
        client_read, client_write = client_streams

        async def run_server() -> None:
            await server.run(server_read, server_write, server.create_initialization_options(), raise_exceptions=True)

        async with anyio.create_task_group() as tg:
            tg.start_soon(run_server)
            async with ClientSession(client_read, client_write, elicitation_callback=elicitation_callback) as client:
                await client.initialize()
                yield client
            tg.cancel_scope.cancel()
