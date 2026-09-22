"""Per-scope configuration: slice overrides, provenance, and the process asymmetry.

A slice override (``SettingsResolver`` keyed by ``(name, ScopeKey)``) tunes a
scoped instance per tenant. For a PROCESS component the chain is built once under
``GLOBAL_SCOPE``, so only a ``GLOBAL_SCOPE``-keyed override reaches it - an
override keyed by the per-invoke ``(("component", name),)`` scope is silently
ignored. That asymmetry is pinned as a characterization test.
"""

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

from _support.axes import context_axes
from pydantic import BaseModel
import pytest

from warpweft.core.axes import GLOBAL_SCOPE, ScopeSpec
from warpweft.core.component import AComponent, Lifetime, invocable
from warpweft.core.composition import Container, DictSettingsResolver
from warpweft.core.composition.config import SOURCE_SLICE

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
