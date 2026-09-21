"""Runnable demo: a long-running @tool(background=True) over MCP.

A background tool does not block its call: it returns a ``task_id`` at once and
runs in the background. The caller polls the shared ``task_status`` tool and
collects the answer from ``task_result``. It is all plain ``tools/call``, so it
works over stdio with no special transport - this demo drives an in-memory MCP
client to keep it self-contained (a real host connects via ``run_stdio(app)``).

Run:
    uv run python examples/mcp_background.py
"""

import anyio
from mcp import ClientSession
from mcp.shared.memory import create_client_server_memory_streams
from pydantic import BaseModel

from warpweft import AComponent, App, EmptySettings, Registry, invocable, report_progress
from warpweft.mcp import build_server, task_runner, tool


class Report(BaseModel):
    month: str
    total: int


class Reports(AComponent[EmptySettings, str, Report]):
    name = "reports"

    @tool(background=True, description="Generate a monthly report (slow).")
    @invocable
    async def report(self, month: str) -> Report:
        for step in range(1, 4):
            await anyio.sleep(0.1)
            await report_progress(step / 3, message=f"step {step}/3")
        return Report(month=month, total=42)


async def main() -> None:
    registry = Registry()
    registry.register(Reports)
    app = App(registry=registry)

    async with (
        app.run(),
        task_runner() as runner,  # in-memory store; a real deployment brings its own
        create_client_server_memory_streams() as (client_streams, server_streams),
    ):
        server = build_server(app, runner=runner)
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                lambda: server.run(*server_streams, server.create_initialization_options(), raise_exceptions=True)
            )
            async with ClientSession(*client_streams) as client:
                await client.initialize()

                names = [t.name for t in (await client.list_tools()).tools]
                print(f"tools: {names}")  # reports__report + task_status/result/cancel

                submitted = await client.call_tool("reports__report", {"month": "june"})
                task_id = submitted.structured_content["task_id"]
                print(f"\nsubmitted -> {task_id}")

                while True:
                    status = await client.call_tool("task_status", {"task_id": task_id})
                    sc = status.structured_content
                    print(f"  status: {sc['status']:<9} message: {sc['message']}")
                    if sc["status"] != "working":
                        break
                    await anyio.sleep(0.1)

                result = await client.call_tool("task_result", {"task_id": task_id})
                print(f"\nresult: {result.structured_content}")
            tg.cancel_scope.cancel()


if __name__ == "__main__":
    anyio.run(main)
