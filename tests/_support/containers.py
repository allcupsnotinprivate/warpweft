"""Container/registry factories shared across integration suites.

``registry_of`` replaces the per-file ``fresh_registry`` copies; ``running_container``
folds the ``build -> start -> ... -> stop`` boilerplate into one context manager.
The ``container`` fixture in ``conftest`` exposes ``running_container`` and, by
being requested, tags the test ``integration`` (see the tier auto-marker).
"""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

from warpweft.core.component import AComponent
from warpweft.core.composition import Container, Registry


def registry_of(*classes: type[AComponent[Any, Any, Any]]) -> Registry:
    """A fresh registry with ``classes`` registered, in order."""
    registry = Registry()
    for cls in classes:
        registry.register(cls)
    return registry


@asynccontextmanager
async def running_container(
    *classes: type[AComponent[Any, Any, Any]],
    config: Mapping[str, Mapping[str, Any]] | None = None,
    **build_kwargs: Any,
) -> AsyncIterator[Container]:
    """Build a container from ``classes``, start it, yield it, stop on exit.

    ``config`` is the per-component deployment config; any other keyword
    (``tracer_provider``, ``meter_provider``, ``axes``, ``telemetry``,
    ``span_enricher`` …) is forwarded to ``Container.build``.
    """
    container = Container.build(registry_of(*classes), dict(config or {}), **build_kwargs)
    await container.start()
    try:
        yield container
    finally:
        await container.stop()
