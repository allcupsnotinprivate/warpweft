"""warpweft.mcp axis_binders: derive a state-slicing axis (e.g. tenant) from the
incoming MCP call and bind it around the invoke.

All plain ``tools/call``; the tool reports the axis value it saw so the tests
can assert the binder reached the component through the axis contextvar.
"""

from typing import Any

import anyio
import mcp.types as mt
import pytest

from warpweft.core.component import AComponent, EmptySettings, invocable
from warpweft.core.composition import Registry
from warpweft.mcp import task_runner, tool
from warpweft.runtime import App, AxisHandle

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


def _tenant_app(*, background: bool = False) -> tuple[App, AxisHandle]:
    """An app whose one tool returns the current ``tenant`` axis value."""
    holder: dict[str, AxisHandle] = {}

    class Svc(AComponent[EmptySettings, None, str]):
        name = "svc"

        @tool(background=background)
        @invocable
        async def whoami(self) -> str:
            return holder["handle"].current() or "unset"

    reg = Registry()
    reg.register(Svc)
    app = App(registry=reg)
    holder["handle"] = app.axis("tenant", default="unset")
    return app, holder["handle"]


async def test_binder_binds_the_axis_per_call(connect) -> None:
    app, tenant = _tenant_app()
    seen = {"value": "acme"}  # stands in for a per-request header / auth lookup

    def binder(ctx: Any, params: mt.CallToolRequestParams) -> str | None:
        return seen["value"]

    async with connect(app, axis_binders={tenant: binder}) as client:
        first = await client.call_tool("svc__whoami", {})
        assert first.structured_content == {"result": "acme"}

        seen["value"] = "beta"  # a different tenant on the next call
        second = await client.call_tool("svc__whoami", {})
        assert second.structured_content == {"result": "beta"}


async def test_binder_returning_none_leaves_the_default(connect) -> None:
    app, tenant = _tenant_app()

    async with connect(app, axis_binders={tenant: lambda ctx, params: None}) as client:
        result = await client.call_tool("svc__whoami", {})
    assert result.structured_content == {"result": "unset"}  # the axis default


async def test_no_binder_leaves_the_default(connect) -> None:
    app, _ = _tenant_app()
    async with connect(app) as client:  # no axis_binders at all
        result = await client.call_tool("svc__whoami", {})
    assert result.structured_content == {"result": "unset"}


async def test_binder_can_read_the_call_params(connect) -> None:
    app, tenant = _tenant_app()

    def from_args(ctx: Any, params: mt.CallToolRequestParams) -> str | None:
        return (params.arguments or {}).get("as_tenant")

    async with connect(app, axis_binders={tenant: from_args}) as client:
        # A binder-only argument the tool itself does not declare is ignored by
        # the tool's input model but still visible to the binder.
        result = await client.call_tool("svc__whoami", {"as_tenant": "gamma"})
    assert result.structured_content == {"result": "gamma"}


async def test_a_raising_binder_is_a_tool_error(connect) -> None:
    app, tenant = _tenant_app()

    def boom(ctx: Any, params: mt.CallToolRequestParams) -> str | None:
        raise RuntimeError("no tenant in request")

    async with connect(app, axis_binders={tenant: boom}) as client:
        result = await client.call_tool("svc__whoami", {})
    assert result.is_error is True
    assert result.meta is not None
    assert "warpweft.error" in result.meta


async def test_background_job_inherits_the_bound_axis(connect) -> None:
    # The axis is bound during the submit, and start_soon captures that context,
    # so the background job (which runs later) still sees the submit-time tenant.
    app, tenant = _tenant_app(background=True)

    binders = {tenant: lambda ctx, params: "acme"}
    async with task_runner() as runner, connect(app, runner=runner, axis_binders=binders) as client:
        submitted = await client.call_tool("svc__whoami", {})
        task_id = submitted.structured_content["task_id"]

        with anyio.fail_after(2):
            while True:
                status = await client.call_tool("task_status", {"task_id": task_id})
                if status.structured_content and status.structured_content["status"] == "completed":
                    break
                await anyio.sleep(0.01)

        result = await client.call_tool("task_result", {"task_id": task_id})
    assert result.structured_content == {"result": "acme"}
