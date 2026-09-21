"""Container config layering and the introspection API."""

from contextvars import ContextVar

import pytest

from warpweft.core.axes import Axis, AxisRegistry, ScopeSpec
from warpweft.core.component import AComponent, EmptySettings, Lifetime, Policy, invocable
from warpweft.core.composition import Container, DictSettingsResolver, Registry
from warpweft.core.composition.config import SOURCE_COMPONENT, SOURCE_DEPLOYMENT, SOURCE_FRAMEWORK, SOURCE_SLICE
from warpweft.core.errors import CircuitOpen, ConfigurationError

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

_tenant: ContextVar[str | None] = ContextVar("intro_tenant", default=None)
ACME = (("tenant", "acme"),)
GLOBEX = (("tenant", "globex"),)


class Svc(AComponent[EmptySettings, None, str]):
    name = "svc"
    defaults = {"policy": {"timeout": {"seconds": 2.0}}}  # author default

    @invocable
    async def go(self) -> str:
        return "ok"

    @invocable(policy=Policy(chain=("timeout",)))
    async def health_probe(self) -> bool:
        return True


class Guarded(AComponent[EmptySettings, None, str]):
    name = "guarded"

    def endpoint(self) -> str | None:
        return "host-x"

    @invocable
    async def go(self) -> str:
        return "ok"


class TenantSvc(AComponent[EmptySettings, None, int]):
    name = "tenant-svc"
    lifetime = Lifetime.SCOPED
    scope = ScopeSpec(("tenant",))

    @invocable
    async def rate(self) -> int:
        return 0


# --- config layering through the container -----------------------------------


async def test_component_defaults_merge_under_deployment() -> None:
    reg = Registry()
    reg.register(Svc)
    config = {"svc": {"policy": {"retry": {"attempts": 3, "base_delay": 0.0, "max_delay": 1.0}}}}
    container = Container.build(reg, config)
    await container.start()
    settings = container.resolved_settings("svc")
    assert settings["policy"]["retry"]["attempts"] == 3  # from deployment
    assert settings["policy"]["timeout"]["seconds"] == 2.0  # from the author default
    await container.stop()


async def test_framework_defaults_apply() -> None:
    reg = Registry()
    reg.register(Svc)
    container = Container.build(
        reg,
        {"svc": {}},
        framework_defaults={"policy": {"retry": {"attempts": 7, "base_delay": 0.0, "max_delay": 1.0}}},
    )
    await container.start()
    explanation = container.explain("svc", "go")
    assert explanation.provenance["policy.retry.attempts"] == SOURCE_FRAMEWORK
    await container.stop()


async def test_slice_override_wins_per_field() -> None:
    reg = Registry()
    reg.register(TenantSvc)
    axes = AxisRegistry()
    axes.register(Axis(name="tenant", resolver=_tenant.get))
    resolver = DictSettingsResolver(
        {("tenant-svc", ACME): {"policy": {"retry": {"attempts": 9}}}},
    )
    deployment = {"tenant-svc": {"policy": {"retry": {"attempts": 1, "base_delay": 0.0, "max_delay": 1.0}}}}
    container = Container.build(reg, deployment, axes=axes, resolver=resolver)
    await container.start()

    acme = container.resolved_settings("tenant-svc", scope_key=ACME)
    globex = container.resolved_settings("tenant-svc", scope_key=GLOBEX)
    assert acme["policy"]["retry"]["attempts"] == 9  # slice override
    assert acme["policy"]["retry"]["max_delay"] == 1.0  # deployment field survives
    assert globex["policy"]["retry"]["attempts"] == 1  # no override for globex
    await container.stop()


async def test_bad_slice_override_names_the_slice_source() -> None:
    reg = Registry()
    reg.register(TenantSvc)
    axes = AxisRegistry()
    axes.register(Axis(name="tenant", resolver=_tenant.get))
    resolver = DictSettingsResolver({("tenant-svc", ACME): {"policy": {"retry": {"attempts": -1}}}})
    deployment = {"tenant-svc": {"policy": {"retry": {"attempts": 1, "base_delay": 0.0, "max_delay": 1.0}}}}
    container = Container.build(reg, deployment, axes=axes, resolver=resolver)
    await container.start()
    with pytest.raises(ConfigurationError) as excinfo:
        container.resolved_settings("tenant-svc", scope_key=ACME)
    assert SOURCE_SLICE in str(excinfo.value)
    await container.stop()


# --- explain -----------------------------------------------------------------


async def test_explain_shows_the_effective_chain_and_provenance() -> None:
    reg = Registry()
    reg.register(Svc)
    config = {"svc": {"policy": {"retry": {"attempts": 3, "base_delay": 0.0, "max_delay": 1.0}}}}
    container = Container.build(reg, config)
    await container.start()
    explanation = container.explain("svc", "go")
    assert explanation.chain == ("retry", "timeout")  # retry (deployment) + timeout (default)
    assert explanation.provenance["policy.retry.attempts"] == SOURCE_DEPLOYMENT
    assert explanation.provenance["policy.timeout.seconds"] == SOURCE_COMPONENT
    await container.stop()


async def test_explain_respects_a_method_policy() -> None:
    reg = Registry()
    reg.register(Svc)
    config = {"svc": {"policy": {"retry": {"attempts": 3, "base_delay": 0.0, "max_delay": 1.0}}}}
    container = Container.build(reg, config)
    await container.start()
    # health_probe is pinned to ("timeout",): retry must not appear.
    assert container.explain("svc", "health_probe").chain == ("timeout",)
    await container.stop()


async def test_explain_rejects_unknown_component_or_method() -> None:
    reg = Registry()
    reg.register(Svc)
    container = Container.build(reg, {"svc": {}})
    await container.start()
    with pytest.raises(ConfigurationError, match="not configured"):
        container.explain("ghost", "go")
    with pytest.raises(ConfigurationError, match="no invocable"):
        container.explain("svc", "missing")
    with pytest.raises(ConfigurationError, match="not configured"):
        container.resolved_settings("ghost")
    await container.stop()


# --- runtime snapshot --------------------------------------------------------


async def test_snapshot_reports_breaker_and_concurrency_state() -> None:
    reg = Registry()
    reg.register(Guarded)
    config = {
        "guarded": {
            "policy": {
                "circuit_breaker": {"window": 5, "failure_threshold": 5, "reset_timeout": 10.0},
                "concurrency": {"inner_limit": 2, "outer_limit": 3},
            }
        }
    }
    container = Container.build(reg, config)
    await container.start()
    await container.invoke("guarded", "go")  # materialises the link instances

    snap = container.snapshot()
    assert len(snap.breakers) == 1
    breaker = snap.breakers[0]
    assert breaker.unit.startswith("circuit_breaker")
    assert breaker.slice == (("endpoint", "host-x"),)
    assert breaker.state == "closed"

    assert len(snap.concurrency) == 1
    conc = snap.concurrency[0]
    assert conc.outer_limit == 3
    assert conc.outer_available == 3  # the call completed, slots released
    assert conc.inner_slices >= 1
    await container.stop()


def _guarded_breaker_config() -> dict[str, object]:
    return {"guarded": {"policy": {"circuit_breaker": {"window": 5, "failure_threshold": 5, "reset_timeout": 10.0}}}}


async def test_force_open_and_reset_breakers() -> None:
    reg = Registry()
    reg.register(Guarded)
    container = Container.build(reg, _guarded_breaker_config())
    await container.start()
    await container.invoke("guarded", "go")  # materialise the breaker

    assert await container.force_open_breakers(endpoint="host-x") == 1
    assert container.snapshot().breakers[0].state == "open"
    with pytest.raises(CircuitOpen):
        await container.invoke("guarded", "go")  # open breaker rejects

    assert await container.reset_breakers() == 1  # no filter -> every live breaker
    assert container.snapshot().breakers[0].state == "closed"
    assert (await container.invoke("guarded", "go")).value == "ok"  # closed again
    await container.stop()


async def test_breaker_controls_return_zero_when_nothing_matches() -> None:
    reg = Registry()
    reg.register(Guarded)
    container = Container.build(reg, _guarded_breaker_config())
    await container.start()

    # No call yet: the breaker is created lazily, so nothing is live.
    assert await container.force_open_breakers() == 0
    assert await container.reset_breakers() == 0

    await container.invoke("guarded", "go")  # now the breaker exists
    assert await container.force_open_breakers(endpoint="other") == 0  # wrong slice
    await container.stop()


async def test_snapshot_ignores_links_that_are_neither_breaker_nor_concurrency() -> None:
    reg = Registry()
    reg.register(Svc)
    # Only cache configured: the link store holds a cache interceptor, which the
    # snapshot walks past without reporting.
    config = {"svc": {"policy": {"cache": {"ttl": 100.0, "max_entries": 10}}}}
    container = Container.build(reg, config)
    await container.start()
    await container.invoke("svc", "go")

    snap = container.snapshot()
    assert snap.breakers == ()
    assert snap.concurrency == ()
    await container.stop()


async def test_snapshot_lists_live_scoped_slices() -> None:
    reg = Registry()
    reg.register(TenantSvc)
    axes = AxisRegistry()
    axes.register(Axis(name="tenant", resolver=_tenant.get))
    container = Container.build(reg, {"tenant-svc": {}}, axes=axes)
    await container.start()

    _tenant.set("acme")
    await container.invoke("tenant-svc", "rate")
    _tenant.set("globex")
    await container.invoke("tenant-svc", "rate")

    live = container.snapshot().live_slices["tenant-svc"]
    assert set(live) == {ACME, GLOBEX}
    await container.stop()
