"""Cross-mode dependencies: a scoped caller reaching scoped / process deps.

A scoped dependency slices by its OWN scope spec, independent of the caller's.
Because a scoped caller binds its dependencies once (in the store factory), the
dependency set is snapshotted at the caller's creation - a later axis change is
not reflected on the already-cached caller. Both facts are pinned here; the
snapshot one as a characterization test.
"""

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

from _support.axes import context_axes
import pytest

from warpweft.core.axes import ScopeSpec
from warpweft.core.component import AComponent, Criticality, EmptySettings, Lifetime, invocable
from warpweft.core.composition import Container
from warpweft.runtime.tenancy import AxisHandle

pytestmark = pytest.mark.anyio

Make = Callable[..., AbstractAsyncContextManager[Container]]


def _region_dep_and_caller(region: AxisHandle) -> tuple[type, type]:
    """A region-scoped dep (captures its region at creation) and a tenant-scoped caller."""

    class RegionDep(AComponent[EmptySettings, None, str]):
        name = "region-dep"
        lifetime = Lifetime.SCOPED
        scope = ScopeSpec(("region",))

        def __init__(self, settings: EmptySettings) -> None:
            super().__init__(settings)
            self.region = region.current()  # captured at its own creation

        @invocable
        async def where(self) -> str:
            return self.region or "?"

    class TenantCaller(AComponent[EmptySettings, None, str]):
        name = "tenant-caller"
        lifetime = Lifetime.SCOPED
        scope = ScopeSpec(("tenant",))
        dependencies = ("region-dep",)

        @invocable
        async def ask(self) -> str:
            return await self.dependency("region-dep").where()

    return RegionDep, TenantCaller


async def test_scoped_dep_with_different_axis_slices_by_its_own_spec(container: Make) -> None:
    axes, handles = context_axes("tenant", "region")
    region_dep, caller = _region_dep_and_caller(handles["region"])
    async with container(region_dep, caller, config={"region-dep": {}, "tenant-caller": {}}, axes=axes) as c:
        with handles["tenant"].use("acme"), handles["region"].use("eu"):
            assert (await c.invoke("tenant-caller", "ask")).value == "eu"
        with handles["tenant"].use("beta"), handles["region"].use("eu"):
            assert (await c.invoke("tenant-caller", "ask")).value == "eu"
        # Two tenants in one region share a single region-dep slice.
        assert c.snapshot().live_slices["region-dep"] == ((("region", "eu"),),)


@pytest.mark.characterization
async def test_scoped_dep_is_snapshotted_at_caller_creation(container: Make) -> None:
    # CHARACTERIZATION: a scoped caller binds its deps once, in the store factory.
    # After the caller is cached, changing the dependency's axis does NOT rebind
    # it - the cached caller keeps calling the dep slice it was created with.
    axes, handles = context_axes("tenant", "region")
    region_dep, caller = _region_dep_and_caller(handles["region"])
    async with container(region_dep, caller, config={"region-dep": {}, "tenant-caller": {}}, axes=axes) as c:
        with handles["tenant"].use("acme"), handles["region"].use("eu"):
            assert (await c.invoke("tenant-caller", "ask")).value == "eu"  # caller created under region=eu
        with handles["tenant"].use("acme"), handles["region"].use("us"):
            assert (await c.invoke("tenant-caller", "ask")).value == "eu"  # cached caller still calls the eu dep


@pytest.mark.characterization
async def test_degraded_optional_process_dep_raises_keyerror_at_use(container: Make) -> None:
    # CHARACTERIZATION: a degraded (start-failed) OPTIONAL process dependency is
    # silently omitted from the dependent's dep map; the dependent starts fine but
    # self.dependency(name) raises KeyError at call time (not ComponentUnavailable).
    axes, handles = context_axes("tenant")

    class OptProc(AComponent[EmptySettings, None, str]):
        name = "opt-proc"
        criticality = Criticality.OPTIONAL

        async def start(self) -> None:
            raise RuntimeError("backend down")

        @invocable
        async def ping(self) -> str:
            return "pong"

    class Caller(AComponent[EmptySettings, None, str]):
        name = "caller"
        lifetime = Lifetime.SCOPED
        scope = ScopeSpec(("tenant",))
        dependencies = ("opt-proc",)

        @invocable
        async def use_dep(self) -> str:
            return await self.dependency("opt-proc").ping()

    async with container(OptProc, Caller, config={"opt-proc": {}, "caller": {}}, axes=axes) as c:
        assert c.is_degraded("opt-proc") is True
        with handles["tenant"].use("acme"), pytest.raises(KeyError, match="opt-proc"):
            await c.invoke("caller", "use_dep")
