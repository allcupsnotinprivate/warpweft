"""Scoped-component lifecycle: creation, start-failure, stop, and eviction.

A scoped instance starts lazily inside the store, but that lazy start gets the
same treatment a process start does: it is bounded by ``init_timeout`` and its
failure is routed by criticality (a REQUIRED slice surfaces ``StartupError``, an
OPTIONAL one degrades the component and reports ``ComponentUnavailable``).
"""

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

from _support.axes import context_axes
from _support.components import failing_start, hanging_start, scoped_recorder
import anyio
import pytest

from warpweft.core.axes import ScopeSpec
from warpweft.core.component import AComponent, Criticality, EmptySettings, Lifetime, invocable
from warpweft.core.composition import Container
from warpweft.core.errors import ComponentUnavailable, ConfigurationError, StartupError

pytestmark = pytest.mark.anyio

Make = Callable[..., AbstractAsyncContextManager[Container]]


async def test_scoped_start_failure_raises_startup_error(container: Make) -> None:
    # A REQUIRED scoped instance whose start() raises surfaces StartupError (with
    # the raw failure as __cause__), like a process component - not the raw error.
    axes, handles = context_axes("tenant")
    failing = failing_start("f", exc=RuntimeError("boom"))
    async with container(failing, config={"f": {}}, axes=axes) as c:
        with handles["tenant"].use("acme"), pytest.raises(StartupError, match="failed to start") as info:
            await c.invoke("f", "go")
        assert isinstance(info.value.__cause__, RuntimeError) and str(info.value.__cause__) == "boom"


async def test_scoped_start_failure_is_not_cached(container: Make) -> None:
    starts: list[str] = []
    axes, handles = context_axes("tenant")
    failing = failing_start("f", exc=RuntimeError("boom"))
    original_start = failing.start

    async def counting_start(self: object) -> None:
        starts.append("start")
        await original_start(self)  # type: ignore[arg-type]

    failing.start = counting_start  # type: ignore[method-assign]

    async with container(failing, config={"f": {}}, axes=axes) as c:
        for _ in range(2):
            with handles["tenant"].use("acme"), pytest.raises(StartupError):
                await c.invoke("f", "go")
            assert c.snapshot().live_slices.get("f") == ()  # nothing cached between attempts
        assert len(starts) == 2  # each invoke re-attempts creation


async def test_scoped_start_honors_init_timeout(container: Make) -> None:
    # init_timeout bounds the lazy scoped start too: a hanging start is cut at the
    # deadline and surfaces StartupError, rather than blocking until an outer
    # cancellation (the 0.15s guard scope) fires.
    axes, handles = context_axes("tenant")
    hanging = hanging_start("h")
    async with container(hanging, config={"h": {}}, axes=axes, init_timeout=0.01) as c:
        with anyio.move_on_after(0.15) as scope, handles["tenant"].use("acme"), pytest.raises(StartupError):
            await c.invoke("h", "go")
        assert not scope.cancelled_caught  # the 0.01s init_timeout fired first


async def test_scoped_start_failure_degrades_optional(container: Make) -> None:
    # An OPTIONAL scoped component's start failure degrades the component (like a
    # process one) and reports ComponentUnavailable instead of raising raw.
    axes, handles = context_axes("tenant")
    failing = failing_start("f", exc=RuntimeError("nope"), criticality=Criticality.OPTIONAL)
    async with container(failing, config={"f": {}}, axes=axes) as c:
        with handles["tenant"].use("acme"), pytest.raises(ComponentUnavailable):
            await c.invoke("f", "go")
        assert c.is_degraded("f") is True


async def test_container_stop_stops_all_live_scoped_slices(container: Make) -> None:
    events: list[str] = []
    axes, handles = context_axes("tenant")
    recorder = scoped_recorder("rec", events=events, value_of=lambda: handles["tenant"].current())
    async with container(recorder, config={"rec": {}}, axes=axes) as c:
        for tenant in ("acme", "globex"):
            with handles["tenant"].use(tenant):
                await c.invoke("rec", "whoami")
    # stop() ran on the fixture's exit; every live slice was stopped in order.
    assert events == ["start:acme", "start:globex", "stop:acme", "stop:globex"]


async def test_invoke_after_stop_reports_not_started(container: Make) -> None:
    axes, handles = context_axes("tenant")
    recorder = scoped_recorder("rec", events=[], value_of=lambda: handles["tenant"].current())
    async with container(recorder, config={"rec": {}}, axes=axes) as c:
        pass  # container is stopped on exit
    with handles["tenant"].use("acme"), pytest.raises(ConfigurationError, match="not started"):
        await c.invoke("rec", "whoami")


@pytest.mark.characterization
async def test_lru_eviction_stops_an_in_flight_instance(container: Make) -> None:
    # CHARACTERIZATION: the scoped store's LRU stops an evicted instance even
    # while a call is still running on it (the store lock does not cover the
    # in-flight call). The parked call nonetheless completes.
    axes, handles = context_axes("tenant")
    stopped: list[str] = []
    gate = anyio.Event()

    class Svc(AComponent[EmptySettings, None, str]):
        name = "svc"
        lifetime = Lifetime.SCOPED
        scope = ScopeSpec(("tenant",))

        def __init__(self, settings: EmptySettings) -> None:
            super().__init__(settings)
            self._tenant = handles["tenant"].current()

        async def stop(self) -> None:
            stopped.append(self._tenant or "?")

        @invocable
        async def go(self) -> str:
            if self._tenant == "acme":
                await gate.wait()  # park acme mid-call
            return self._tenant or "?"

    result: dict[str, str] = {}

    async def call(tenant: str) -> None:
        with handles["tenant"].use(tenant):
            result[tenant] = (await c.invoke("svc", "go")).value

    async with container(Svc, config={"svc": {}}, axes=axes, scoped_max_entries=1) as c:
        async with anyio.create_task_group() as tg:
            tg.start_soon(call, "acme")
            await anyio.sleep(0.02)  # acme now parked in go()
            tg.start_soon(call, "globex")  # evicts acme (max_entries=1) and stops it
            await anyio.sleep(0.02)
            assert stopped == ["acme"]  # acme was stopped while still running
            gate.set()
        assert result["acme"] == "acme"  # the parked call completed anyway
