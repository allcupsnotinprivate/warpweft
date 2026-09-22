"""Multi-tenant end-to-end: multi-axis scopes and per-tenant state isolation.

Cross-cutting scenarios spanning the container, the axis machinery and the
policy chain. Cache state is isolated per tenant automatically (the slice key is
part of the cache key); breaker/concurrency state is sliced by the ``endpoint``
axis, so a scoped component only gets a per-tenant breaker if its ``endpoint()``
varies by tenant - otherwise the breaker is shared across tenants (pinned as a
characterization).
"""

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

from _support.axes import context_axes
from _support.components import scoped_recorder
import anyio
import pytest

from warpweft.core.axes import ScopeSpec
from warpweft.core.component import AComponent, Criticality, EmptySettings, Lifetime, invocable
from warpweft.core.composition import Container
from warpweft.core.errors import CircuitOpen, ConfigurationError, TransientError

pytestmark = pytest.mark.anyio

Make = Callable[..., AbstractAsyncContextManager[Container]]

_BREAKER = {"policy": {"circuit_breaker": {"window": 1, "failure_threshold": 1, "reset_timeout": 60.0}}}


async def test_multi_axis_scope_creates_one_instance_per_combo(container: Make) -> None:
    axes, handles = context_axes("region", "tenant")
    recorder = scoped_recorder(
        "rec", scope=("region", "tenant"), events=[], value_of=lambda: handles["tenant"].current()
    )
    async with container(recorder, config={"rec": {}}, axes=axes) as c:
        for region in ("eu", "us"):
            for tenant in ("acme", "globex"):
                with handles["region"].use(region), handles["tenant"].use(tenant):
                    await c.invoke("rec", "whoami")
        slices = c.snapshot().live_slices["rec"]
        assert len(slices) == 4  # one instance per (region, tenant) combination
        # repeats reuse, not recreate
        with handles["region"].use("eu"), handles["tenant"].use("acme"):
            await c.invoke("rec", "whoami")
        assert len(c.snapshot().live_slices["rec"]) == 4


async def test_multi_axis_declaration_order_is_canonical(container: Make) -> None:
    axes, handles = context_axes("region", "tenant")
    # scope declared tenant-first; the resolved key must be sorted (region-first).
    recorder = scoped_recorder(
        "rec", scope=("tenant", "region"), events=[], value_of=lambda: handles["tenant"].current()
    )
    async with container(recorder, config={"rec": {}}, axes=axes) as c:
        with handles["region"].use("eu"), handles["tenant"].use("acme"):
            await c.invoke("rec", "whoami")
        (key,) = c.snapshot().live_slices["rec"]
        assert key == (("region", "eu"), ("tenant", "acme"))  # canonical (sorted) regardless of declaration order


async def test_cache_is_isolated_per_tenant(container: Make) -> None:
    axes, handles = context_axes("tenant")
    hits: dict[str, int] = {"acme": 0, "globex": 0}

    class Svc(AComponent[EmptySettings, str, str]):
        name = "svc"
        lifetime = Lifetime.SCOPED
        scope = ScopeSpec(("tenant",))

        @invocable
        async def compute(self, key: str) -> str:
            tenant = handles["tenant"].current() or "?"
            hits[tenant] += 1
            return f"{tenant}:{key}:{hits[tenant]}"

    config = {"svc": {"policy": {"cache": {"ttl": 60.0, "max_entries": 100}}}}
    async with container(Svc, config=config, axes=axes) as c:
        with handles["tenant"].use("acme"):
            assert (await c.invoke("svc", "compute", key="k")).value == "acme:k:1"
            assert (await c.invoke("svc", "compute", key="k")).value == "acme:k:1"  # cache hit
        with handles["tenant"].use("globex"):
            assert (await c.invoke("svc", "compute", key="k")).value == "globex:k:1"  # not acme's entry
        assert hits == {"acme": 1, "globex": 1}


async def test_breaker_isolated_per_tenant_when_endpoint_varies(container: Make) -> None:
    axes, handles = context_axes("tenant")

    class Svc(AComponent[EmptySettings, None, str]):
        name = "svc"
        lifetime = Lifetime.SCOPED
        scope = ScopeSpec(("tenant",))

        def __init__(self, settings: EmptySettings) -> None:
            super().__init__(settings)
            self._tenant = handles["tenant"].current()

        def endpoint(self) -> str | None:
            return self._tenant  # per-tenant endpoint => per-tenant breaker slice

        @invocable
        async def go(self) -> str:
            raise TransientError("down")

    async with container(Svc, config={"svc": _BREAKER}, axes=axes) as c:
        with handles["tenant"].use("acme"):
            with pytest.raises(TransientError):
                await c.invoke("svc", "go")  # trips acme's breaker
            with pytest.raises(CircuitOpen):
                await c.invoke("svc", "go")  # acme is open
        with handles["tenant"].use("globex"), pytest.raises(TransientError):
            await c.invoke("svc", "go")  # globex's own breaker is still closed


@pytest.mark.characterization
async def test_breaker_is_shared_across_tenants_by_default(container: Make) -> None:
    # CHARACTERIZATION: without a per-tenant endpoint(), the breaker slices by
    # endpoint=identity.uid, which is per (name, version) - the SAME for every
    # tenant slice. One tenant tripping the breaker opens it for all tenants.
    axes, handles = context_axes("tenant")

    class Svc(AComponent[EmptySettings, None, str]):
        name = "svc"
        lifetime = Lifetime.SCOPED
        scope = ScopeSpec(("tenant",))

        @invocable
        async def go(self) -> str:
            raise TransientError("down")

    async with container(Svc, config={"svc": _BREAKER}, axes=axes) as c:
        with handles["tenant"].use("acme"), pytest.raises(TransientError):
            await c.invoke("svc", "go")  # trips the shared breaker
        with handles["tenant"].use("globex"), pytest.raises(CircuitOpen):
            await c.invoke("svc", "go")  # globex sees acme's open breaker


async def test_concurrent_tenants_keep_state_isolated(container: Make) -> None:
    axes, handles = context_axes("tenant")
    gate = anyio.Event()
    results: dict[str, str] = {}

    class Svc(AComponent[EmptySettings, None, str]):
        name = "svc"
        lifetime = Lifetime.SCOPED
        scope = ScopeSpec(("tenant",))

        @invocable
        async def go(self) -> str:
            tenant = handles["tenant"].current() or "?"
            await gate.wait()  # park until both calls are in flight
            return tenant

    async with container(Svc, config={"svc": {}}, axes=axes) as c:

        async def call(tenant: str) -> None:
            with handles["tenant"].use(tenant):
                results[tenant] = (await c.invoke("svc", "go")).value

        async with anyio.create_task_group() as tg:
            tg.start_soon(call, "acme")
            tg.start_soon(call, "globex")
            await anyio.sleep(0.05)  # both parked in go()
            gate.set()
        assert results == {"acme": "acme", "globex": "globex"}  # no cross-tenant leakage


async def test_concurrency_limits_do_not_couple_tenants(container: Make) -> None:
    axes, handles = context_axes("tenant")
    gate = anyio.Event()
    running: list[str] = []

    class Svc(AComponent[EmptySettings, None, str]):
        name = "svc"
        lifetime = Lifetime.SCOPED
        scope = ScopeSpec(("tenant",))

        @invocable
        async def go(self) -> str:
            tenant = handles["tenant"].current() or "?"
            running.append(tenant)
            await gate.wait()
            return tenant

    # inner_limit=1 => one in-flight call per tenant slice; outer_limit is generous.
    config = {"svc": {"policy": {"concurrency": {"inner_limit": 1, "outer_limit": 10}}}}
    async with container(Svc, config=config, axes=axes) as c:

        async def call(tenant: str) -> None:
            with handles["tenant"].use(tenant):
                await c.invoke("svc", "go")

        async with anyio.create_task_group() as tg:
            tg.start_soon(call, "acme")  # enters, holds acme's inner slot
            tg.start_soon(call, "acme")  # blocked by acme's inner_limit=1
            await anyio.sleep(0.03)
            tg.start_soon(call, "globex")  # acme's saturation must not block globex
            await anyio.sleep(0.03)
            # exactly one acme call is running, and globex got through.
            assert running.count("acme") == 1
            assert "globex" in running
            gate.set()


async def test_degradation_is_per_tenant_slice(container: Make) -> None:
    axes, handles = context_axes("tenant")

    class Svc(AComponent[EmptySettings, None, str]):
        name = "svc"
        lifetime = Lifetime.SCOPED
        scope = ScopeSpec(("tenant",))
        criticality = Criticality.OPTIONAL

        def stub(self, ctx: object) -> str:
            return "STUB"

        @invocable
        async def go(self) -> str:
            if handles["tenant"].current() == "acme":
                raise TransientError("acme down")
            return "LIVE"

    async with container(Svc, config={"svc": {"policy": {"degradation": {}}}}, axes=axes) as c:
        with handles["tenant"].use("acme"):
            acme = await c.invoke("svc", "go")
            assert (acme.value, acme.source, acme.degraded) == ("STUB", "stub", True)
        with handles["tenant"].use("globex"):
            globex = await c.invoke("svc", "go")
            assert (globex.value, globex.source, globex.degraded) == ("LIVE", "live", False)


async def test_missing_required_axis_fails_scoped_invoke(container: Make) -> None:
    axes, _ = context_axes("tenant")  # required (no default)

    class Svc(AComponent[EmptySettings, None, str]):
        name = "svc"
        lifetime = Lifetime.SCOPED
        scope = ScopeSpec(("tenant",))

        @invocable
        async def go(self) -> str:
            return "ok"

    async with container(Svc, config={"svc": {}}, axes=axes) as c:
        with pytest.raises(ConfigurationError, match="tenant"):
            await c.invoke("svc", "go")  # no tenant bound
