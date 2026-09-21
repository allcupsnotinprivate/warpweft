"""Actions and providers: boxing, pure vs guarded calls, and tool exposure."""

from typing import Any

from pydantic import BaseModel, SecretStr
import pytest

from warpweft import Action, ActionParams, App, EmptySettings, Provider
from warpweft.core.component import AComponent, describe, invocable
from warpweft.core.composition import Registry
from warpweft.core.errors import ConfigurationError, FrameworkError, TransientError
from warpweft.mcp import collect_tools, tool

pytestmark = [pytest.mark.unit, pytest.mark.anyio]


class Summarise(ActionParams):
    text: str
    max_len: int = 200


class Digest(BaseModel):
    summary: str
    length: int


class Summarize(Action[EmptySettings, Summarise, Digest]):
    description = "Summarise a document."
    read_only = True

    async def execute(self, params: Summarise) -> Digest:
        clip = params.text[: params.max_len]
        return Digest(summary=clip, length=len(clip))


class Ping(ActionParams):
    value: int = 0


def app_with(*classes: type[AComponent[Any, Any, Any]], config: dict[str, Any] | None = None) -> App:
    reg = Registry()
    for cls in classes:
        reg.register(cls)
    return App(registry=reg, config=config)


# --- boxing / schema ---------------------------------------------------------


def test_execute_is_boxed_with_the_params_model_as_input() -> None:
    spec = describe(Summarize).invocables["execute"]
    assert spec.input_model is Summarise
    # the caller-facing flat fields are rebuilt into the single ``params`` model
    assert spec.arg_binder({"text": "hi", "max_len": 2}) == {"params": Summarise(text="hi", max_len=2)}


def test_tool_input_schema_is_flat() -> None:
    spec = describe(Summarize).invocables["execute"]
    assert set(spec.input_json_schema()["properties"]) == {"text", "max_len"}


# --- execute vs __call__ -----------------------------------------------------


async def test_execute_runs_without_a_container() -> None:
    out = await Summarize(EmptySettings()).execute(Summarise(text="hello", max_len=3))
    assert out == Digest(summary="hel", length=3)


async def test_call_without_a_running_app_raises() -> None:
    with pytest.raises(ConfigurationError):
        await Summarize(EmptySettings())(text="x")


async def test_call_accepts_model_mapping_and_kwargs() -> None:
    async with app_with(Summarize).run() as container:
        action = await container.get(Summarize)
        assert (await action(Summarise(text="abcdef", max_len=3))).summary == "abc"
        assert (await action({"text": "abcdef", "max_len": 2})).summary == "ab"
        assert (await action(text="abcdef", max_len=4)).summary == "abcd"


async def test_call_rejects_both_params_and_kwargs() -> None:
    async with app_with(Summarize).run() as container:
        action = await container.get(Summarize)
        with pytest.raises(TypeError):
            await action(Summarise(text="x"), max_len=1)


async def test_call_rejects_an_unsupported_params_type() -> None:
    async with app_with(Summarize).run() as container:
        action = await container.get(Summarize)
        with pytest.raises(TypeError):
            await action(123)


async def test_intermediate_base_without_execute_stays_abstract() -> None:
    class Mid(Action[EmptySettings, Ping, int]):
        """An intermediate base that leaves execute unimplemented."""

    with pytest.raises(NotImplementedError):
        await Mid(EmptySettings()).execute(Ping())


async def test_call_runs_through_the_policy_chain() -> None:
    attempts = {"n": 0}

    class Flaky(Action[EmptySettings, Ping, str]):
        async def execute(self, params: Ping) -> str:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise TransientError("warming up")
            return "ok"

    app = app_with(Flaky, config={"flaky": {"policy": {"retry": {"attempts": 3, "base_delay": 0.0, "max_delay": 1.0}}}})
    async with app.run() as container:
        action = await container.get(Flaky)
        assert await action(value=1) == "ok"
    assert attempts["n"] == 3  # retry ran under __call__


async def test_action_injected_as_dependency_is_guarded() -> None:
    seen = {"n": 0}

    class Inner(Action[EmptySettings, Ping, str]):
        async def execute(self, params: Ping) -> str:
            seen["n"] += 1
            if seen["n"] < 2:
                raise TransientError("retry me")
            return "inner-ok"

    class Outer(Action[EmptySettings, Ping, str]):
        inner: Inner

        async def execute(self, params: Ping) -> str:
            return await self.inner(value=params.value)

    app = app_with(
        Inner, Outer, config={"inner": {"policy": {"retry": {"attempts": 2, "base_delay": 0.0, "max_delay": 1.0}}}}
    )
    async with app.run() as container:
        outer = await container.get(Outer)
        assert await outer(value=1) == "inner-ok"
    assert seen["n"] == 2  # inner retried -> self.inner(...) went through inner's chain


# --- class-time validation ---------------------------------------------------


def test_execute_must_take_exactly_one_model_parameter() -> None:
    with pytest.raises(TypeError):

        class NoParam(Action[EmptySettings, Ping, int]):
            async def execute(self) -> int:
                return 0

    with pytest.raises(TypeError):

        class TwoParams(Action[EmptySettings, Ping, int]):
            async def execute(self, a: Ping, b: int) -> int:
                return b

    with pytest.raises(TypeError):

        class NotAModel(Action[EmptySettings, Ping, int]):
            async def execute(self, params: int) -> int:
                return params


# --- exposure ----------------------------------------------------------------


def test_action_is_exposed_as_a_tool_by_default() -> None:
    (binding,) = collect_tools(app_with(Summarize))
    assert (binding.name, binding.component, binding.method) == ("summarize__execute", "summarize", "execute")


def test_action_class_tags_reach_the_tool_filter() -> None:
    class Audit(Action[EmptySettings, Ping, int]):
        tags = {"admin"}

        async def execute(self, params: Ping) -> int:
            return params.value

    (binding,) = collect_tools(app_with(Audit, Summarize), tags={"admin"})
    assert binding.component == "audit"
    assert binding.meta.tags == frozenset({"admin"})


def test_action_background_flag_reaches_the_tool_meta() -> None:
    class Crunch(Action[EmptySettings, Ping, int]):
        background = True

        async def execute(self, params: Ping) -> int:
            return params.value

    (binding,) = collect_tools(app_with(Crunch))
    assert binding.meta.background is True


def test_action_is_foreground_by_default() -> None:
    (binding,) = collect_tools(app_with(Summarize))
    assert binding.meta.background is False


def test_action_can_opt_out_of_exposure() -> None:
    class Hidden(Action[EmptySettings, Ping, int]):
        entrypoint = False

        async def execute(self, params: Ping) -> int:
            return params.value

    assert collect_tools(app_with(Hidden)) == []


# --- providers ---------------------------------------------------------------


def test_provider_is_not_exposed_and_may_have_no_invocables() -> None:
    class Pool(Provider[EmptySettings]):
        def acquire(self) -> str:
            return "conn"

    assert Pool.entrypoint is False
    assert dict(describe(Pool).invocables) == {}


def test_provider_is_never_collected_as_a_tool() -> None:
    class Cache(Provider[EmptySettings]):
        @invocable
        async def get(self, key: str) -> str:
            return key

    components = {b.component for b in collect_tools(app_with(Cache, Summarize))}
    assert components == {"summarize"}


def test_tool_on_a_provider_is_rejected() -> None:
    class Bad(Provider[EmptySettings]):
        @tool
        @invocable
        async def leak(self, x: str) -> str:
            return x

    with pytest.raises(FrameworkError):
        collect_tools(app_with(Bad))


async def test_provider_starts_and_injects_into_an_action() -> None:
    class Vocabulary(Provider[EmptySettings]):
        def stopwords(self) -> set[str]:
            return {"the", "a"}

    class Clean(Action[EmptySettings, Ping, list[str]]):
        vocab: Vocabulary

        async def execute(self, params: Ping) -> list[str]:
            return sorted(self.vocab.stopwords())

    async with app_with(Vocabulary, Clean).run() as container:
        action = await container.get(Clean)
        assert await action(value=1) == ["a", "the"]


# --- served as an MCP tool ---------------------------------------------------


async def test_action_is_served_with_a_flat_schema(connect: Any) -> None:
    async with connect(app_with(Summarize)) as client:
        listed = await client.list_tools()
        (tool_def,) = listed.tools
        assert tool_def.name == "summarize__execute"
        assert set(tool_def.input_schema["properties"]) == {"text", "max_len"}
        assert tool_def.annotations.read_only_hint is True
        result = await client.call_tool("summarize__execute", {"text": "hello world", "max_len": 5})
    assert result.is_error is False
    assert result.structured_content == {"summary": "hello", "length": 5}


async def test_action_tool_masks_secrets(connect: Any) -> None:
    class Token(BaseModel):
        name: str
        secret: SecretStr

    class Issue(Action[EmptySettings, Ping, Token]):
        async def execute(self, params: Ping) -> Token:
            return Token(name="k", secret=SecretStr("s3cr3t"))

    async with connect(app_with(Issue)) as client:
        result = await client.call_tool("issue__execute", {"value": 1})
    assert result.structured_content["secret"] == "**********"  # noqa: S105 - the masked value, not a secret
