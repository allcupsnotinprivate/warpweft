"""warpweft.mcp: tool discovery, schemas, and end-to-end call routing."""

from typing import Any

import anyio
import mcp.types as mt
from pydantic import BaseModel, SecretStr
import pytest

from warpweft.core.component import AComponent, EmptySettings, invocable
from warpweft.core.composition import Registry
from warpweft.core.context import report_progress
from warpweft.core.errors import (
    CircuitOpen,
    ComponentUnavailable,
    DeadlineExceeded,
    FrameworkError,
    PermanentError,
    RetryExhausted,
    TransientError,
)
from warpweft.core.formats import Ipv4
from warpweft.mcp import collect_tools, tool
from warpweft.mcp.errors import error_result
from warpweft.runtime import App

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


class Doc(BaseModel):
    id: str
    score: float


class Search(AComponent[EmptySettings, str, list[Doc]]):
    name = "search"

    @tool(description="Search the index.", read_only=True)
    @invocable
    async def query(self, text: str, limit: int = 10) -> list[Doc]:
        return [Doc(id=f"{text}-{i}", score=1.0 / (i + 1)) for i in range(1)]

    @invocable
    async def _refresh(self) -> None:  # not a tool
        return None


class Profile(BaseModel):
    name: str
    token: SecretStr


class Accounts(AComponent[EmptySettings, str, Profile]):
    name = "accounts"

    @tool  # bare; description from the docstring
    @invocable
    async def get(self, user_id: str) -> Profile:
        """Return a user profile."""
        return Profile(name=f"user-{user_id}", token=SecretStr("s3cr3t"))


def app_with(*classes: type[AComponent[Any, Any, Any]]) -> App:
    reg = Registry()
    for cls in classes:
        reg.register(cls)
    return App(registry=reg)


# --- discovery ---------------------------------------------------------------


def test_collect_finds_only_tool_marked_invocables() -> None:
    bindings = collect_tools(app_with(Search))
    assert [(b.name, b.component, b.method) for b in bindings] == [("search__query", "search", "query")]


def test_collect_uses_name_override() -> None:
    class Named(AComponent[EmptySettings, None, str]):
        name = "svc"

        @tool(name="do_the_thing")
        @invocable
        async def run(self) -> str:
            return "ok"

    (binding,) = collect_tools(app_with(Named))
    assert binding.name == "do_the_thing"


def test_tool_on_non_invocable_is_rejected() -> None:
    class Bad(AComponent[EmptySettings, None, str]):
        name = "bad"

        @invocable
        async def real(self) -> str:
            return "ok"

        @tool  # marked tool but NOT invocable
        async def fake(self) -> str:
            return "no"

    with pytest.raises(FrameworkError, match="marked @tool but is not @invocable"):
        collect_tools(app_with(Bad))


def test_duplicate_tool_names_are_rejected() -> None:
    class A(AComponent[EmptySettings, None, str]):
        name = "a"

        @tool(name="dup")
        @invocable
        async def x(self) -> str:
            return "1"

    class B(AComponent[EmptySettings, None, str]):
        name = "b"

        @tool(name="dup")
        @invocable
        async def y(self) -> str:
            return "2"

    with pytest.raises(FrameworkError, match="duplicate tool name 'dup'"):
        collect_tools(app_with(A, B))


# --- list_tools --------------------------------------------------------------


async def test_list_tools_exposes_schema_and_annotations(connect) -> None:
    async with connect(app_with(Search)) as client:
        result = await client.list_tools()
    (tool_def,) = result.tools
    assert tool_def.name == "search__query"
    assert tool_def.description == "Search the index."
    assert set(tool_def.input_schema["properties"]) == {"text", "limit"}
    assert tool_def.annotations is not None
    assert tool_def.annotations.read_only_hint is True


async def test_description_falls_back_to_the_docstring(connect) -> None:
    async with connect(app_with(Accounts)) as client:
        result = await client.list_tools()
    (tool_def,) = result.tools
    assert tool_def.description == "Return a user profile."


async def test_object_return_advertises_output_schema(connect) -> None:
    async with connect(app_with(Accounts)) as client:
        result = await client.list_tools()
    (tool_def,) = result.tools
    assert tool_def.output_schema is not None
    assert tool_def.output_schema["type"] == "object"


async def test_non_object_return_advertises_a_wrapped_schema(connect) -> None:
    async with connect(app_with(Search)) as client:
        result = await client.list_tools()
    (tool_def,) = result.tools  # returns a list -> wrapped in {"result": ...}
    schema = tool_def.output_schema
    assert schema is not None
    assert schema["type"] == "object"
    assert schema["required"] == ["result"]
    assert schema["properties"]["result"]["type"] == "array"
    assert "Doc" in schema["$defs"]  # hoisted so nested $refs stay valid


async def test_non_object_call_returns_wrapped_structured_content(connect) -> None:
    async with connect(app_with(Search)) as client:
        result = await client.call_tool("search__query", {"text": "sre"})
    assert result.structured_content == {"result": [{"id": "sre-0", "score": 1.0}]}


async def test_scalar_call_returns_wrapped_structured_content(connect) -> None:
    class Echo(AComponent[EmptySettings, None, str]):
        name = "echo"

        @tool
        @invocable
        async def say(self) -> str:
            return "ok"

    async with connect(app_with(Echo)) as client:
        result = await client.call_tool("echo__say", {})
    assert result.structured_content == {"result": "ok"}
    assert result.content[0].text == "ok"  # the text stays raw, not the wrapper


# --- call_tool ---------------------------------------------------------------


async def test_call_returns_text_content(connect) -> None:
    async with connect(app_with(Search)) as client:
        result = await client.call_tool("search__query", {"text": "sre"})
    assert result.is_error is False
    assert "sre-0" in result.content[0].text


async def test_object_call_returns_structured_content_and_masks_secrets(connect) -> None:
    async with connect(app_with(Accounts)) as client:
        result = await client.call_tool("accounts__get", {"user_id": "42"})
    assert result.structured_content == {"name": "user-42", "token": "**********"}
    assert "s3cr3t" not in result.content[0].text  # secret never leaks


async def test_call_reports_outcome_metadata(connect) -> None:
    async with connect(app_with(Search)) as client:
        result = await client.call_tool("search__query", {"text": "x"})
    assert result.meta is not None
    assert result.meta["warpweft.source"] == "live"
    assert result.meta["warpweft.degraded"] is False


async def test_degraded_call_reports_meta(connect) -> None:
    from warpweft.core.component import Criticality
    from warpweft.core.context import InvocationContext

    class Weather(AComponent[EmptySettings, str, dict]):
        name = "weather"
        criticality = Criticality.OPTIONAL

        def stub(self, ctx: InvocationContext) -> dict:
            return {"temp": None}

        @tool(description="Current weather.", read_only=True)
        @invocable
        async def current(self, city: str) -> dict:
            raise TransientError("upstream down")

    reg = Registry()
    reg.register(Weather)
    app = App(registry=reg, config={"weather": {"policy": {"degradation": {}}}})
    async with connect(app) as client:
        result = await client.call_tool("weather__current", {"city": "paris"})
    assert result.is_error is False
    assert result.meta is not None
    assert result.meta["warpweft.degraded"] is True
    assert result.meta["warpweft.source"] == "stub"


async def test_unknown_tool_is_an_error(connect) -> None:
    async with connect(app_with(Search)) as client:
        result = await client.call_tool("search__ghost", {})
    assert result.is_error is True
    assert "unknown tool" in result.content[0].text


async def test_permanent_error_becomes_a_tool_error(connect) -> None:
    class Boom(AComponent[EmptySettings, None, str]):
        name = "boom"

        @tool
        @invocable
        async def go(self) -> str:
            raise PermanentError("bad request")

    async with connect(app_with(Boom)) as client:
        result = await client.call_tool("boom__go", {})
    assert result.is_error is True
    assert "bad request" in result.content[0].text
    assert result.meta["warpweft.error"] == "permanent"
    assert result.meta["warpweft.retryable"] is False


async def test_unexpected_exception_becomes_a_tool_error(connect) -> None:
    class Oops(AComponent[EmptySettings, None, str]):
        name = "oops"

        @tool
        @invocable
        async def go(self) -> str:
            raise ValueError("not a framework error")

    async with connect(app_with(Oops)) as client:
        result = await client.call_tool("oops__go", {})
    assert result.is_error is True
    assert "not a framework error" in result.content[0].text
    assert result.meta["warpweft.error"] == "error"
    assert result.meta["warpweft.retryable"] is False


class Geo(AComponent[EmptySettings, str, dict]):
    name = "geo"

    @tool(description="Locate an IPv4 address.")
    @invocable
    async def locate(self, host: Ipv4) -> dict[str, str]:
        return {"host": host}


async def test_format_annotated_input_appears_in_the_tool_schema(connect) -> None:
    async with connect(app_with(Geo)) as client:
        result = await client.list_tools()
    (tool_def,) = result.tools
    assert tool_def.input_schema["properties"]["host"]["format"] == "ipv4"


async def test_valid_format_argument_passes(connect) -> None:
    async with connect(app_with(Geo)) as client:
        result = await client.call_tool("geo__locate", {"host": "10.0.0.1"})
    assert result.is_error is False
    assert result.structured_content == {"host": "10.0.0.1"}


async def test_invalid_format_argument_is_a_validation_error(connect) -> None:
    async with connect(app_with(Geo)) as client:
        result = await client.call_tool("geo__locate", {"host": "not-an-ip"})
    assert result.is_error is True
    assert "invalid ipv4" in result.content[0].text


async def test_missing_required_argument_is_a_validation_error(connect) -> None:
    async with connect(app_with(Geo)) as client:
        result = await client.call_tool("geo__locate", {})
    assert result.is_error is True
    assert "host" in result.content[0].text.lower()


async def test_call_runs_through_the_policy_chain(connect) -> None:
    calls = {"n": 0}

    class Flaky(AComponent[EmptySettings, None, str]):
        name = "flaky"

        @tool
        @invocable
        async def fetch(self) -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise TransientError("warming up")
            return "ok"

    reg = Registry()
    reg.register(Flaky)
    app = App(
        registry=reg,
        config={"flaky": {"policy": {"retry": {"attempts": 3, "base_delay": 0.0, "max_delay": 1.0}}}},
    )
    async with connect(app) as client:
        result = await client.call_tool("flaky__fetch", {})
    assert result.is_error is False
    assert calls["n"] == 3  # retry ran under the tool call


# --- filtering ----------------------------------------------------------------


class Billing(AComponent[EmptySettings, str, dict]):
    name = "billing"

    @tool(description="Show an invoice.", read_only=True, tags={"public"})
    @invocable
    async def invoice(self, invoice_id: str) -> dict[str, str]:
        return {"invoice": invoice_id}

    @tool(description="List payments.", read_only=True, tags={"public", "admin"})
    @invocable
    async def payments(self) -> dict[str, str]:
        return {"payments": "[]"}

    @tool(description="Refund a payment.", destructive=True, tags={"admin"})
    @invocable
    async def refund(self, payment_id: str) -> dict[str, str]:
        return {"refunded": payment_id}


def _names(bindings: list[Any]) -> list[str]:
    return [b.name for b in bindings]


def test_no_filter_keeps_every_tool() -> None:
    bindings = collect_tools(app_with(Billing, Search))
    assert _names(bindings) == ["billing__invoice", "billing__payments", "billing__refund", "search__query"]


def test_tags_filter_keeps_tools_with_any_requested_tag() -> None:
    bindings = collect_tools(app_with(Billing), tags={"public"})
    assert _names(bindings) == ["billing__invoice", "billing__payments"]


def test_untagged_tool_never_passes_a_tag_filter() -> None:
    # Search.query declares no tags, so a tag filter works as a whitelist.
    bindings = collect_tools(app_with(Billing, Search), tags={"public", "admin"})
    assert "search__query" not in _names(bindings)


def test_include_supports_globs() -> None:
    bindings = collect_tools(app_with(Billing, Search), include={"billing__*"})
    assert _names(bindings) == ["billing__invoice", "billing__payments", "billing__refund"]


def test_exclude_wins_over_include() -> None:
    bindings = collect_tools(app_with(Billing), include={"billing__*"}, exclude={"billing__refund"})
    assert _names(bindings) == ["billing__invoice", "billing__payments"]


def test_unknown_tag_is_rejected() -> None:
    with pytest.raises(FrameworkError, match="no tool declares tag"):
        collect_tools(app_with(Billing), tags={"ops"})


def test_pattern_matching_no_tool_is_rejected() -> None:
    with pytest.raises(FrameworkError, match="include pattern .* matches no tool"):
        collect_tools(app_with(Billing), include={"billng__*"})
    with pytest.raises(FrameworkError, match="exclude pattern .* matches no tool"):
        collect_tools(app_with(Billing), exclude={"billing__ghost"})


def test_filter_leaving_no_tools_is_rejected() -> None:
    with pytest.raises(FrameworkError, match="leaves no tools"):
        collect_tools(app_with(Billing), tags={"admin"}, exclude={"billing__payments", "billing__refund"})


async def test_served_tool_set_respects_the_tag_filter(connect) -> None:
    async with connect(app_with(Billing, Search), tags={"public"}) as client:
        result = await client.list_tools()
    assert [t.name for t in result.tools] == ["billing__invoice", "billing__payments"]


# --- error mapping ------------------------------------------------------------


def test_error_results_carry_retry_guidance() -> None:
    cases = [
        (PermanentError("bad key"), "permanent", False),
        (TransientError("blip"), "transient", True),
        (DeadlineExceeded("out of time"), "timeout", True),
        (ComponentUnavailable("weather is degraded"), "unavailable", True),
        (ValueError("bug"), "error", False),
    ]
    for exc, code, retryable in cases:
        result = error_result(exc)
        assert result.is_error is True
        assert result.meta["warpweft.error"] == code
        assert result.meta["warpweft.retryable"] is retryable
        assert str(exc) in result.content[0].text


def test_circuit_open_reports_retry_after() -> None:
    result = error_result(CircuitOpen("circuit for 'op' is open", retry_after=4.2))
    assert result.meta["warpweft.error"] == "circuit_open"
    assert result.meta["warpweft.retry_after_s"] == 4.2
    assert "4.2" in result.content[0].text  # the hint names the wait


def test_retry_exhausted_reports_attempts() -> None:
    err = RetryExhausted("gave up", attempts=3, last_error=TransientError("blip"))
    result = error_result(err)
    assert result.meta["warpweft.error"] == "retry_exhausted"
    assert result.meta["warpweft.attempts"] == 3


async def test_validation_error_meta_marks_invalid_arguments(connect) -> None:
    async with connect(app_with(Geo)) as client:
        result = await client.call_tool("geo__locate", {"host": "not-an-ip"})
    assert result.meta["warpweft.error"] == "invalid_arguments"
    assert result.meta["warpweft.retryable"] is True


# --- progress & cancellation ---------------------------------------------------


async def test_progress_reports_reach_the_client(connect) -> None:
    class Cruncher(AComponent[EmptySettings, None, str]):
        name = "cruncher"

        @tool
        @invocable
        async def crunch(self) -> str:
            await report_progress(0.5, total=1.0, message="halfway")
            await report_progress(1.0, total=1.0, message="done")
            return "ok"

    seen: list[tuple[float, float | None, str | None]] = []

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        seen.append((progress, total, message))

    async with connect(app_with(Cruncher)) as client:
        result = await client.call_tool("cruncher__crunch", {}, progress_callback=on_progress)
    assert result.is_error is False
    assert seen == [(0.5, 1.0, "halfway"), (1.0, 1.0, "done")]


async def test_progress_without_a_client_token_is_dropped(connect) -> None:
    class Quiet(AComponent[EmptySettings, None, str]):
        name = "quiet"

        @tool
        @invocable
        async def crunch(self) -> str:
            await report_progress(0.5)  # no token -> the session no-ops
            return "ok"

    async with connect(app_with(Quiet)) as client:
        result = await client.call_tool("quiet__crunch", {})
    assert result.is_error is False


async def test_client_cancellation_reaches_the_invocable(connect) -> None:
    started = anyio.Event()
    cancelled = anyio.Event()

    class Slow(AComponent[EmptySettings, None, str]):
        name = "slow"

        @tool
        @invocable
        async def wait(self) -> str:
            started.set()
            try:
                await anyio.sleep(60)
            except anyio.get_cancelled_exc_class():
                cancelled.set()
                raise
            return "never"

    async with connect(app_with(Slow)) as client:
        async with anyio.create_task_group() as tg:

            async def call() -> None:
                await client.call_tool("slow__wait", {})

            tg.start_soon(call)
            await started.wait()
            tg.cancel_scope.cancel()  # abandoning the request sends notifications/cancelled

        with anyio.fail_after(2):
            await cancelled.wait()  # the SDK interrupted the handler's scope


# --- destructive confirmation ---------------------------------------------------


def purge_component() -> tuple[type[AComponent[Any, Any, Any]], dict[str, int]]:
    ran = {"n": 0}

    class Purge(AComponent[EmptySettings, None, str]):
        name = "purge"

        @tool(description="Drop everything.", destructive=True)
        @invocable
        async def run(self) -> str:
            ran["n"] += 1
            return "purged"

    return Purge, ran


async def _accept(context: Any, params: Any) -> mt.ElicitResult:
    return mt.ElicitResult(action="accept", content={})


async def _decline(context: Any, params: Any) -> mt.ElicitResult:
    return mt.ElicitResult(action="decline")


async def test_destructive_tool_runs_after_acceptance(connect) -> None:
    Purge, ran = purge_component()
    async with connect(app_with(Purge), elicitation_callback=_accept, confirm_destructive=True) as client:
        result = await client.call_tool("purge__run", {})
    assert result.is_error is False
    assert ran["n"] == 1


async def test_declined_destructive_tool_is_not_executed(connect) -> None:
    Purge, ran = purge_component()
    async with connect(app_with(Purge), elicitation_callback=_decline, confirm_destructive=True) as client:
        result = await client.call_tool("purge__run", {})
    assert result.is_error is True
    assert result.meta["warpweft.error"] == "declined"
    assert ran["n"] == 0


async def test_destructive_confirmation_fails_closed_without_capability(connect) -> None:
    Purge, ran = purge_component()
    # no elicitation_callback -> the client does not advertise the capability
    async with connect(app_with(Purge), confirm_destructive=True) as client:
        result = await client.call_tool("purge__run", {})
    assert result.is_error is True
    assert result.meta["warpweft.error"] == "confirmation_unsupported"
    assert ran["n"] == 0


async def test_destructive_tool_without_the_flag_runs_unprompted(connect) -> None:
    Purge, ran = purge_component()
    async with connect(app_with(Purge)) as client:  # confirm_destructive defaults to off
        result = await client.call_tool("purge__run", {})
    assert result.is_error is False
    assert ran["n"] == 1


async def test_non_destructive_tool_is_never_confirmed(connect) -> None:
    # confirm_destructive on, but the tool is not destructive and the client
    # cannot elicit: the call must still go through.
    async with connect(app_with(Search), confirm_destructive=True) as client:
        result = await client.call_tool("search__query", {"text": "x"})
    assert result.is_error is False
