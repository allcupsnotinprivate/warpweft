"""Per-scope configuration: slice overrides, provenance, and the process asymmetry.

A slice override (``SettingsResolver`` keyed by ``(name, ScopeKey)``) tunes a
scoped instance per tenant. For a PROCESS component the chain is built once under
``GLOBAL_SCOPE``, so only a ``GLOBAL_SCOPE``-keyed override reaches it - an
override keyed by the per-invoke ``(("component", name),)`` scope is silently
ignored. That asymmetry is pinned as a characterization test.
"""

from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from typing import Any

from _support.axes import context_axes
from pydantic import BaseModel
import pytest

from warpweft.core.axes import GLOBAL_SCOPE, ScopeKey, ScopeSpec
from warpweft.core.component import AComponent, Lifetime, invocable
from warpweft.core.composition import Container, DictSettingsResolver
from warpweft.core.composition.config import SOURCE_SLICE
from warpweft.core.errors import TransientError

pytestmark = pytest.mark.anyio

Make = Callable[..., AbstractAsyncContextManager[Container]]

ACME = (("tenant", "acme"),)
GLOBEX = (("tenant", "globex"),)


class _Prefix(BaseModel):
    prefix: str = "-"


class ScopedEcho(AComponent[_Prefix, str, str]):
    name = "scoped-echo"
    lifetime = Lifetime.SCOPED
    scope = ScopeSpec(("tenant",))

    @invocable
    async def echo(self, text: str) -> str:
        return f"{self.settings.prefix}{text}"


class ProcEcho(AComponent[_Prefix, str, str]):
    name = "proc-echo"

    @invocable
    async def echo(self, text: str) -> str:
        return f"{self.settings.prefix}{text}"


async def test_slice_override_applies_per_tenant(container: Make) -> None:
    axes, handles = context_axes("tenant")
    resolver = DictSettingsResolver({("scoped-echo", ACME): {"prefix": "A:"}})
    async with container(ScopedEcho, config={"scoped-echo": {"prefix": "D:"}}, axes=axes, resolver=resolver) as c:
        with handles["tenant"].use("acme"):
            assert (await c.invoke("scoped-echo", "echo", text="x")).value == "A:x"  # slice override
        with handles["tenant"].use("globex"):
            assert (await c.invoke("scoped-echo", "echo", text="x")).value == "D:x"  # deployment default


async def test_slice_override_provenance_is_slice(container: Make) -> None:
    axes, _ = context_axes("tenant")
    resolver = DictSettingsResolver({("scoped-echo", ACME): {"prefix": "A:"}})
    async with container(ScopedEcho, config={"scoped-echo": {"prefix": "D:"}}, axes=axes, resolver=resolver) as c:
        explanation = c.explain("scoped-echo", "echo", scope_key=ACME)
        assert explanation.provenance["prefix"] == SOURCE_SLICE  # first positive coverage of the slice source


@pytest.mark.characterization
async def test_process_component_slice_override_is_silently_ignored(container: Make) -> None:
    # CHARACTERIZATION: a process chain is built under GLOBAL_SCOPE, but invoke
    # carries scope_key=(("component", name),). A slice override keyed by that
    # component tuple never reaches the running process component, even though
    # resolved_settings(scope_key=that_key) would report it.
    component_key = (("component", "proc-echo"),)
    resolver = DictSettingsResolver({("proc-echo", component_key): {"prefix": "C:"}})
    async with container(ProcEcho, config={"proc-echo": {"prefix": "D:"}}, resolver=resolver) as c:
        assert (await c.invoke("proc-echo", "echo", text="x")).value == "D:x"  # override ignored at runtime
        assert c.resolved_settings("proc-echo", scope_key=component_key)["prefix"] == "C:"  # yet assemble sees it


async def test_global_scope_override_applies_to_process_component(container: Make) -> None:
    resolver = DictSettingsResolver({("proc-echo", GLOBAL_SCOPE): {"prefix": "G:"}})
    async with container(ProcEcho, config={"proc-echo": {"prefix": "D:"}}, resolver=resolver) as c:
        assert (await c.invoke("proc-echo", "echo", text="x")).value == "G:x"  # the only working process-slice path


class _Flaky(AComponent[_Prefix, None, str]):
    name = "flaky"
    lifetime = Lifetime.SCOPED
    scope = ScopeSpec(("tenant",))

    def __init__(self, settings: _Prefix) -> None:
        super().__init__(settings)
        self.calls = 0

    @invocable
    async def go(self) -> str:
        self.calls += 1
        if self.calls < 2:
            raise TransientError("first fails")
        return "ok"


class _CountingResolver:
    """Returns a distinct override each time a scope_key is resolved, and counts calls."""

    def __init__(self) -> None:
        self.calls: dict[ScopeKey, int] = {}

    def resolve(self, component: str, scope_key: ScopeKey) -> Mapping[str, Any]:
        self.calls[scope_key] = self.calls.get(scope_key, 0) + 1
        return {"prefix": f"n{self.calls[scope_key]}:"}


async def test_config_cache_pins_first_resolution(container: Make) -> None:
    axes, handles = context_axes("tenant")
    resolver = _CountingResolver()
    async with container(ScopedEcho, config={"scoped-echo": {"prefix": "D:"}}, axes=axes, resolver=resolver) as c:
        with handles["tenant"].use("acme"):
            assert (await c.invoke("scoped-echo", "echo", text="x")).value == "n1:x"
            assert (
                await c.invoke("scoped-echo", "echo", text="x")
            ).value == "n1:x"  # cached, resolver not re-consulted
        with handles["tenant"].use("globex"):
            assert (
                await c.invoke("scoped-echo", "echo", text="x")
            ).value == "n1:x"  # fresh scope, its own first resolve
    assert resolver.calls[ACME] == 1  # resolved exactly once per scope_key
    assert resolver.calls[GLOBEX] == 1


async def test_multi_axis_slice_key_is_canonical(container: Make) -> None:
    axes, handles = context_axes("region", "tenant")

    class MultiEcho(AComponent[_Prefix, str, str]):
        name = "multi-echo"
        lifetime = Lifetime.SCOPED
        scope = ScopeSpec(("tenant", "region"))  # declared tenant-first

        @invocable
        async def echo(self, text: str) -> str:
            return f"{self.settings.prefix}{text}"

    sorted_key = (("region", "eu"), ("tenant", "acme"))  # override keyed by the canonical (sorted) form
    resolver = DictSettingsResolver({("multi-echo", sorted_key): {"prefix": "S:"}})
    async with container(MultiEcho, config={"multi-echo": {"prefix": "D:"}}, axes=axes, resolver=resolver) as c:
        with handles["region"].use("eu"), handles["tenant"].use("acme"):
            assert (await c.invoke("multi-echo", "echo", text="x")).value == "S:x"  # applies despite declaration order


async def test_slice_override_can_add_a_policy_block(container: Make) -> None:
    axes, handles = context_axes("tenant")
    # retry added only for acme, purely via the slice override.
    resolver = DictSettingsResolver(
        {("flaky", ACME): {"policy": {"retry": {"attempts": 3, "base_delay": 0.0, "max_delay": 0.1}}}}
    )
    async with container(_Flaky, config={"flaky": {}}, axes=axes, resolver=resolver) as c:
        with handles["tenant"].use("acme"):
            outcome = await c.invoke("flaky", "go")
            assert outcome.value == "ok" and outcome.attempts == 2  # acme's slice-added retry recovered
        with handles["tenant"].use("globex"), pytest.raises(TransientError):
            await c.invoke("flaky", "go")  # globex has no retry
