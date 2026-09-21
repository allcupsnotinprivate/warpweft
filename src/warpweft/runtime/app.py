"""The App facade: registry + configuration + container lifecycle in one object.

``App`` owns a type registry and builds/starts/stops the core ``Container``.
Configuration merges the programmatic mapping with environment variables
(environment wins, field by field). Registration stays explicit - the
``@component`` decorator - while `warpweft.runtime.discovery.autodiscover`
removes the manual import list.

The module-level `component` decorator registers into a process-wide
default registry (the convenient path, like Celery's ``shared_task``). For
full isolation - several independent apps in one process, hermetic tests -
give each ``App`` its own ``Registry`` and use ``@app.component`` instead.
"""

from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager, AbstractContextManager, asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
import signal
from typing import Any, TypeVar

import anyio

from warpweft.core.axes import Axis, AxisRegistry
from warpweft.core.component import AComponent
from warpweft.core.composition import Container, Registry
from warpweft.core.composition.config import deep_merge
from warpweft.core.composition.registry import ENTRY_POINT_GROUP
from warpweft.core.context import use_correlation_id
from warpweft.core.errors import ConfigurationError
from warpweft.core.outcome import Outcome

from .discovery import autodiscover
from .sources import read_dotenv, read_env, read_file
from .tenancy import AxisHandle

C = TypeVar("C", bound=AComponent[Any, Any, Any])
T = TypeVar("T", bound=type[AComponent[Any, Any, Any]])

_SOURCE_ENV = "environment"
_SOURCE_DOTENV = "dotenv file"
_SOURCE_CONFIG = "app config"
_SOURCE_FILE = "config file"

_default_registry = Registry()


def default_registry() -> Registry:
    """The process-wide registry the module-level decorator writes to."""
    return _default_registry


def component(cls: T) -> T:
    """Register a component type in the default registry. Use as a decorator."""
    return _default_registry.register(cls)


class App:
    """An application: registered components, their config, one container."""

    def __init__(
        self,
        *,
        env_prefix: str | None = None,
        config: Mapping[str, Mapping[str, Any]] | None = None,
        config_file: str | Path | None = None,
        dotenv: str | Path | None = None,
        registry: Registry | None = None,
        axes: AxisRegistry | None = None,
        **build_options: Any,
    ) -> None:
        self._registry = registry if registry is not None else _default_registry
        self._axes = axes if axes is not None else AxisRegistry()
        self._env_prefix = env_prefix
        self._config = {name: dict(section) for name, section in (config or {}).items()}
        self._config_file = config_file
        self._dotenv = dotenv
        self._build_options = build_options
        self._container: Container | None = None

    @property
    def registry(self) -> Registry:
        return self._registry

    def component(self, cls: T) -> T:
        """Register a component type in this app's registry. Use as a decorator."""
        return self._registry.register(cls)

    def autodiscover(self, *packages: str) -> tuple[str, ...]:
        """Import the packages' modules so their ``@component`` decorators fire."""
        return autodiscover(*packages)

    def load_entry_points(self, group: str = ENTRY_POINT_GROUP) -> Mapping[str, type[AComponent[Any, Any, Any]]]:
        """Discover and register third-party components advertised under ``group``."""
        return self._registry.load_entry_points(group)

    def axis(self, name: str, *, default: str | None = None, max_cardinality: int = 1000) -> AxisHandle:
        """Register a contextvar-backed axis and return a handle to bind it.

        With a ``default`` the axis is optional (that value when unbound);
        without one it is required and resolving it while unbound is an error.
        ``max_cardinality`` is a soft cap (default 1000): the registry logs a
        single warning (logger ``warpweft.core.axes``) the first time a new
        value would exceed it, but never blocks resolution.
        """
        var: ContextVar[str | None] = ContextVar(f"warpweft_axis_{name}", default=default)
        axis = (
            Axis(name=name, resolver=var.get, on_missing="default", default=default, max_cardinality=max_cardinality)
            if default is not None
            else Axis(name=name, resolver=var.get, on_missing="required", max_cardinality=max_cardinality)
        )
        self._axes.register(axis)
        return AxisHandle(name, var)

    def config_mapping(self) -> dict[str, dict[str, Any]]:
        """The deployment config for every registered component.

        Layered field by field, each overriding the previous: config file, the
        programmatic config, a ``.env`` file, then the environment. An absent
        section means "all defaults". Reading uses pydantic-settings; merging
        and validation stay in the core, so provenance and source-named errors
        are preserved.
        """
        config_models = {name: self._registry.descriptor(name).config_model for name in self._registry.names()}
        layers: list[tuple[str, Mapping[str, Any]]] = []
        if self._config_file is not None:
            layers.append((_SOURCE_FILE, read_file(config_models, self._config_file)))
        layers.append((_SOURCE_CONFIG, self._config))
        if self._dotenv is not None:
            layers.append((_SOURCE_DOTENV, read_dotenv(config_models, self._env_prefix or "", self._dotenv)))
        if self._env_prefix is not None:
            layers.append((_SOURCE_ENV, read_env(config_models, self._env_prefix)))

        merged: dict[str, dict[str, Any]] = {}
        for name in sorted(self._registry.names()):
            section, _ = deep_merge([(source, dict(layer.get(name, {}))) for source, layer in layers])
            merged[name] = section
        return merged

    # --- lifecycle -------------------------------------------------------

    def build(self) -> Container:
        """Build and validate the container without starting it.

        Merges the config, validates it against every component's model and
        validates the dependency graph - but instantiates nothing and touches
        no resources. Useful for offline checks (CI) and introspection.
        """
        return Container.build(self._registry, self.config_mapping(), axes=self._axes, **self._build_options)

    async def start(self) -> Container:
        """Build the container from the merged config and start it."""
        if self._container is not None:
            return self._container
        container = self.build()
        await container.start()
        self._container = container
        return container

    async def serve(self, *, shutdown: anyio.Event | None = None) -> None:
        """Start, then run until a shutdown signal (or ``shutdown`` event), then stop.

        For worker processes with no web host: waits for SIGINT/SIGTERM by
        default, draining and stopping on either. Pass ``shutdown`` to drive it
        from your own event instead.
        """
        async with self.run():
            await self._await_shutdown(shutdown)

    async def _await_shutdown(self, shutdown: anyio.Event | None) -> None:
        if shutdown is not None:
            await shutdown.wait()
            return
        with anyio.open_signal_receiver(signal.SIGINT, signal.SIGTERM) as signals:  # pragma: no cover
            async for _ in signals:  # pragma: no cover - OS signals are not unit-testable portably
                return

    async def stop(self) -> None:
        if self._container is None:
            return
        await self._container.stop()
        self._container = None

    @asynccontextmanager
    async def run(self) -> AsyncIterator[Container]:
        """``async with app.run() as container:`` - start now, stop on exit."""
        container = await self.start()
        try:
            yield container
        finally:
            await self.stop()

    @property
    def container(self) -> Container:
        if self._container is None:
            raise ConfigurationError("app is not started")
        return self._container

    # --- passthroughs ------------------------------------------------------

    async def invoke(
        self,
        component_name: str,
        method: str,
        *,
        correlation_id: str | None = None,
        budget: float | None = None,
        **arguments: Any,
    ) -> Outcome[Any]:
        return await self.container.invoke(
            component_name, method, correlation_id=correlation_id, budget=budget, **arguments
        )

    async def get(self, ref: type[C] | str) -> C:
        return await self.container.get(ref)

    def proxy(self, ref: type[C] | str, *, budget: float | None = None) -> C:
        return self.container.proxy(ref, budget=budget)

    def correlation(self, correlation_id: str) -> AbstractContextManager[str]:
        """Bind a correlation id for calls made in the block (``with app.correlation(id):``)."""
        return use_correlation_id(correlation_id)

    async def force_open_breakers(self, *, endpoint: str | None = None) -> int:
        """Manually open live circuit breakers; return how many were flipped."""
        return await self.container.force_open_breakers(endpoint=endpoint)

    async def reset_breakers(self, *, endpoint: str | None = None) -> int:
        """Manually close live circuit breakers; return how many were reset."""
        return await self.container.reset_breakers(endpoint=endpoint)

    # --- embedding into a host ------------------------------------------------

    def lifespan(self, *_: Any) -> AbstractAsyncContextManager[None]:
        """A framework-agnostic lifespan: start on enter, stop (drained) on exit.

        Accepts and ignores whatever a host passes (ASGI frameworks pass their
        application object), so the bound method plugs in directly::

            FastAPI(lifespan=app.lifespan)      # or Starlette(lifespan=...)
            async with app.lifespan():          # or standalone

        Nothing is published into the host: reach the running system through
        this object (``app.container``, ``app.proxy(...)``, ``app.invoke(...)``).
        Hosts with startup/shutdown callback pairs instead of a lifespan can
        call `start` and `stop` directly.
        """
        return self._lifespan()

    @asynccontextmanager
    async def _lifespan(self) -> AsyncIterator[None]:
        await self.start()
        try:
            yield
        finally:
            await self.stop()
