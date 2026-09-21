"""App facade: registration, autodiscovery, env config, lifecycle, lifespan."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
from pydantic import BaseModel
import pytest

from warpweft.core.axes import AxisRegistry, ScopeSpec
from warpweft.core.component import AComponent, EmptySettings, Lifetime, invocable
from warpweft.core.composition import Registry
from warpweft.core.errors import ConfigurationError, DeadlineExceeded, TransientError
from warpweft.runtime import App, component, default_registry

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


class GreeterSettings(BaseModel):
    greeting: str = "hello"
    volume: int = 1


class Greeter(AComponent[GreeterSettings, str, str]):
    @invocable
    async def greet(self, whom: str) -> str:
        return f"{self.settings.greeting} {whom} x{self.settings.volume}"


class Flaky(AComponent[EmptySettings, None, str]):
    @invocable
    async def fetch(self) -> str:
        raise TransientError("always down")


class PerTenant(AComponent[EmptySettings, None, str]):
    lifetime = Lifetime.SCOPED
    scope = ScopeSpec(("tenant",))

    @invocable
    async def go(self) -> str:
        return "ok"


def fresh_app(**kwargs: Any) -> App:
    registry = Registry()
    registry.register(Greeter)
    return App(registry=registry, **kwargs)


# --- registration ------------------------------------------------------------


def test_module_level_component_registers_into_the_default_registry() -> None:
    @component
    class DefaultRegistered(AComponent[EmptySettings, None, None]):
        @invocable
        async def go(self) -> None: ...

    assert default_registry().get("default_registered") is DefaultRegistered


def test_app_component_decorator_targets_the_apps_own_registry() -> None:
    registry = Registry()
    app = App(registry=registry)

    @app.component
    class Isolated(AComponent[EmptySettings, None, None]):
        @invocable
        async def go(self) -> None: ...

    assert registry.get("isolated") is Isolated
    assert app.registry is registry
    assert "isolated" not in default_registry()


def test_autodiscover_imports_modules_and_skips_private_ones() -> None:
    app = App()  # default registry: the fixture package registers into it
    imported = app.autodiscover("sample_app")
    assert "sample_app.widgets" in imported
    assert all("_private" not in name for name in imported)
    assert "sample_widget" in default_registry()


def test_autodiscover_accepts_a_plain_module() -> None:
    app = App()
    assert app.autodiscover("sample_app.widgets") == ("sample_app.widgets",)


# --- configuration -----------------------------------------------------------


async def test_env_values_override_config_per_field(monkeypatch: pytest.MonkeyPatch) -> None:
    # Env values arrive as strings; the core coerces them at validation.
    monkeypatch.setenv("TESTAPP_GREETER__VOLUME", "5")
    monkeypatch.setenv("TESTAPP_GREETER__GREETING", "env-hi")
    app = fresh_app(config={"greeter": {"greeting": "cfg-hi", "volume": 2}}, env_prefix="TESTAPP")
    async with app.run():
        settings = app.container.resolved_settings("greeter")
        assert settings["greeting"] == "env-hi"  # env overrode config
        assert settings["volume"] == 5  # env overrode config, coerced to int


async def test_env_reads_nested_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MYAPP_GREETER__VOLUME", "4")
    app = fresh_app(env_prefix="MYAPP")
    async with app.run():
        assert app.container.resolved_settings("greeter")["volume"] == 4


def test_env_for_unknown_component_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    # pydantic-settings ignores env vars that map to no component field.
    monkeypatch.setenv("MYAPP_GHOST__X", "1")
    app = fresh_app(env_prefix="MYAPP")
    assert app.config_mapping() == {"greeter": {}}


def test_registered_but_unconfigured_component_defaults_to_empty_section() -> None:
    app = fresh_app()
    assert app.config_mapping() == {"greeter": {}}


async def test_dotenv_file_is_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("MYAPP_GREETER__GREETING=from-dotenv\n")
    monkeypatch.setenv("MYAPP_GREETER__VOLUME", "6")  # real env still overrides the .env
    app = fresh_app(env_prefix="MYAPP", dotenv=dotenv)
    async with app.run():
        settings = app.container.resolved_settings("greeter")
        assert settings["greeting"] == "from-dotenv"
        assert settings["volume"] == 6


# --- lifecycle ---------------------------------------------------------------


async def test_run_starts_serves_and_stops() -> None:
    app = fresh_app(config={"greeter": {"greeting": "privet"}})
    async with app.run() as container:
        outcome = await app.invoke("greeter", "greet", whom="mir")
        assert outcome.value == "privet mir x1"
        assert container.started
    assert not container.started  # stopped on exit


async def test_typed_passthroughs() -> None:
    app = fresh_app()
    async with app.run():
        greeter = await app.get(Greeter)
        assert isinstance(greeter, Greeter)
        assert await app.proxy(Greeter).greet(whom="you") == "hello you x1"


async def test_container_property_requires_start() -> None:
    app = fresh_app()
    with pytest.raises(ConfigurationError, match="not started"):
        _ = app.container
    async with app.run():
        assert app.container.started
    await app.stop()  # second stop: no-op


async def test_start_is_idempotent() -> None:
    app = fresh_app()
    first = await app.start()
    second = await app.start()
    assert first is second
    await app.stop()


async def test_env_config_reaches_the_running_component(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RTAPP_GREETER__GREETING", "salve")
    app = fresh_app(env_prefix="RTAPP")
    async with app.run():
        assert await app.proxy(Greeter).greet(whom="munde") == "salve munde x1"


async def test_build_options_pass_through_to_the_container() -> None:
    app = fresh_app(framework_defaults={"volume": 7})
    async with app.run():
        assert app.container.resolved_settings("greeter")["volume"] == 7


# --- universal lifespan --------------------------------------------------------


async def test_lifespan_used_by_an_asgi_style_host() -> None:
    app = fresh_app()
    host = SimpleNamespace()  # whatever the framework passes is ignored

    async with app.lifespan(host):  # e.g. FastAPI(lifespan=app.lifespan)
        container = app.container
        assert container.started
        assert (await app.invoke("greeter", "greet", whom="web")).value == "hello web x1"
    assert not container.started  # stopped on shutdown


async def test_lifespan_works_standalone_without_a_host() -> None:
    app = fresh_app()
    async with app.lifespan():
        assert app.container.started
    with pytest.raises(ConfigurationError, match="not started"):
        _ = app.container


async def test_lifespan_stops_even_when_the_host_body_raises() -> None:
    app = fresh_app()
    with pytest.raises(RuntimeError, match="host blew up"):
        async with app.lifespan():
            saved = app.container
            raise RuntimeError("host blew up")
    assert not saved.started


# --- config file -------------------------------------------------------------


async def test_config_file_is_the_base_layer_under_config_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "warpweft.toml"
    cfg.write_text('[greeter]\ngreeting = "file-hi"\nvolume = 9\n')
    monkeypatch.setenv("FA_GREETER__VOLUME", "5")  # env overrides the file's volume
    app = App(
        registry=_greeter_registry(),
        config_file=cfg,
        config={"greeter": {"greeting": "config-hi"}},  # config overrides the file's greeting
        env_prefix="FA",
    )
    async with app.run():
        settings = app.container.resolved_settings("greeter")
        assert settings == {"greeting": "config-hi", "volume": 5, "policy": settings["policy"]}


def test_json_config_file(tmp_path: Path) -> None:
    cfg = tmp_path / "warpweft.json"
    cfg.write_text('{"greeter": {"volume": 3}}')
    app = App(registry=_greeter_registry(), config_file=cfg)
    assert app.config_mapping()["greeter"] == {"volume": 3}


def test_yaml_config_file(tmp_path: Path) -> None:
    cfg = tmp_path / "warpweft.yaml"
    cfg.write_text("greeter:\n  greeting: yaml-hi\n  volume: 8\n")
    app = App(registry=_greeter_registry(), config_file=cfg)
    assert app.config_mapping()["greeter"] == {"greeting": "yaml-hi", "volume": 8}


def test_unsupported_config_file_type_is_rejected(tmp_path: Path) -> None:
    cfg = tmp_path / "warpweft.ini"
    cfg.write_text("[greeter]\n")
    app = App(registry=_greeter_registry(), config_file=cfg)
    with pytest.raises(ConfigurationError, match="unsupported config file"):
        app.config_mapping()


def _greeter_registry() -> Registry:
    registry = Registry()
    registry.register(Greeter)
    return registry


# --- axes / tenancy ----------------------------------------------------------


async def test_axis_helper_drives_scoped_components() -> None:
    registry = Registry()
    registry.register(PerTenant)
    app = App(registry=registry)
    tenant = app.axis("tenant", default="public")

    async with app.run() as container:
        assert tenant.current() == "public"
        with tenant.use("acme"):
            assert tenant.current() == "acme"
            acme = await app.get(PerTenant)
        with tenant.use("globex"):
            globex = await app.get(PerTenant)
        assert acme is not globex  # one instance per tenant slice
        assert container.snapshot().live_slices["per_tenant"]  # slices are live


async def test_required_axis_without_a_value_errors_on_use() -> None:
    registry = Registry()
    registry.register(PerTenant)
    app = App(registry=registry)
    app.axis("tenant")  # no default -> required
    async with app.run():
        with pytest.raises(ConfigurationError, match="required but has no value"):
            await app.get(PerTenant)


def test_app_axis_passes_max_cardinality() -> None:
    axes = AxisRegistry()
    app = App(registry=Registry(), axes=axes)
    app.axis("tenant", default="public", max_cardinality=5)
    assert axes.get("tenant").max_cardinality == 5


# --- budget & correlation ----------------------------------------------------


async def test_budget_passthrough_on_invoke_and_proxy() -> None:
    registry = Registry()
    registry.register(Flaky)
    config = {"flaky": {"policy": {"retry": {"attempts": 5, "base_delay": 1.0, "max_delay": 1.0}}}}
    app = App(registry=registry, config=config)
    async with app.run():
        with pytest.raises(DeadlineExceeded):
            await app.invoke("flaky", "fetch", budget=0.05)
        with pytest.raises(DeadlineExceeded):
            await app.proxy(Flaky, budget=0.05).fetch()


async def test_correlation_helper_binds_the_ambient_id() -> None:
    from warpweft.core.context import InvocationContext

    class TypedEcho(AComponent[EmptySettings, None, str]):
        name = "echo"

        @invocable
        async def whoami(self, ctx: InvocationContext) -> str:
            return ctx.correlation_id

    registry = Registry()
    registry.register(TypedEcho)
    app = App(registry=registry)
    async with app.run():
        with app.correlation("req-99"):
            assert (await app.invoke("echo", "whoami")).value == "req-99"


# --- entry points ------------------------------------------------------------


def test_load_entry_points_delegates_to_the_registry() -> None:
    app = fresh_app()
    assert app.load_entry_points("nonexistent.group.for.tests") == {}


# --- serve -------------------------------------------------------------------


async def test_serve_runs_until_the_shutdown_event() -> None:
    app = fresh_app()
    shutdown = anyio.Event()

    async def run_serve() -> None:
        await app.serve(shutdown=shutdown)

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_serve)
        while app._container is None or not app._container.started:
            await anyio.lowlevel.checkpoint()
        assert (await app.invoke("greeter", "greet", whom="worker")).value == "hello worker x1"
        shutdown.set()

    with pytest.raises(ConfigurationError, match="not started"):
        _ = app.container
