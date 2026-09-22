"""Shared fixtures: both anyio backends, sample package, and an MCP client session.

Also carries the tier auto-marker (see ``pytest_collection_modifyitems``): every
test is tagged ``unit``/``integration``/``e2e`` and ``anyio`` automatically, so
new files need no boilerplate and ``-m <tier>`` selection is always complete.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
import inspect
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

sys.path.insert(0, str(Path(__file__).parent))  # makes `sample_app` / `_support` importable

#: A test is ``integration`` (not ``unit``) when it pulls in one of these fixtures.
_INTEGRATION_FIXTURES = frozenset({"connect", "container", "app", "running_app"})
_TIER_MARKERS = ("unit", "integration", "e2e")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-apply the ``anyio`` and tier markers so files need no boilerplate.

    An explicit ``pytestmark``/decorator always wins. Otherwise the tier is
    inferred from the fixtures a test uses (integration if it builds a
    container/app/MCP session, unit otherwise), and any ``async def`` test gets
    the ``anyio`` marker.
    """
    for item in items:
        func = getattr(item, "obj", None)
        if inspect.iscoroutinefunction(func) and item.get_closest_marker("anyio") is None:
            item.add_marker(pytest.mark.anyio)
        if not any(item.get_closest_marker(name) for name in _TIER_MARKERS):
            fixtures = set(getattr(item, "fixturenames", ()))
            tier = "integration" if fixtures & _INTEGRATION_FIXTURES else "unit"
            item.add_marker(getattr(pytest.mark, tier))


@pytest.fixture(params=["asyncio", "trio"])
def anyio_backend(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def connect() -> Callable[..., AbstractAsyncContextManager[ClientSession]]:
    """Return the ``connected(app)`` context manager for a test to use."""
    return connected


@pytest.fixture
def container() -> Callable[..., AbstractAsyncContextManager[Any]]:
    """A container factory: ``async with container(A, B, config=...) as c: ...``.

    Requesting it marks the test ``integration`` (see the tier auto-marker) and
    folds away the build/start/stop boilerplate.
    """
    from _support.containers import running_container

    return running_container


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
