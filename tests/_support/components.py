"""Component factories for mode/tenancy tests.

Replaces the per-file re-declarations of ``TenantThing``-style components. Each
factory returns a fresh ``AComponent`` subclass configured for the test; pass
``lifetime``/``scope``/``criticality`` to shape its mode.
"""

from collections.abc import Callable, Sequence

from warpweft.core.axes import EMPTY_SCOPE, ScopeSpec
from warpweft.core.component import AComponent, Criticality, EmptySettings, HealthStatus, Lifetime, invocable


def _scope_of(axes: Sequence[str]) -> ScopeSpec:
    return ScopeSpec(tuple(axes)) if axes else EMPTY_SCOPE


def scoped_recorder(
    name: str,
    *,
    scope: Sequence[str] = ("tenant",),
    events: list[str],
    value_of: Callable[[], str | None],
    lifetime: Lifetime = Lifetime.SCOPED,
    criticality: Criticality = Criticality.REQUIRED,
    dependencies: Sequence[str] = (),
) -> type[AComponent[EmptySettings, None, str]]:
    """A component that logs ``start:<slice>`` / ``stop:<slice>`` and echoes its slice.

    ``value_of`` reads the current slice value (e.g. the bound tenant); it is
    captured at ``start``/``stop`` and returned by ``whoami``.
    """

    class _Recorder(AComponent[EmptySettings, None, str]):
        pass

    _Recorder.name = name
    _Recorder.lifetime = lifetime
    _Recorder.scope = _scope_of(scope)
    _Recorder.criticality = criticality
    _Recorder.dependencies = tuple(dependencies)

    # Capture the slice at start (the instance is created under its bound axis);
    # stop() runs at container close after the contextvar is unbound, so it must
    # use the captured value, not read value_of() live.
    async def start(self: AComponent[EmptySettings, None, str]) -> None:
        self._slice = value_of()  # type: ignore[attr-defined]
        events.append(f"start:{self._slice}")  # type: ignore[attr-defined]

    async def stop(self: AComponent[EmptySettings, None, str]) -> None:
        events.append(f"stop:{getattr(self, '_slice', None)}")

    @invocable
    async def whoami(self: AComponent[EmptySettings, None, str]) -> str:
        return value_of() or "?"

    _Recorder.start = start  # type: ignore[method-assign]
    _Recorder.stop = stop  # type: ignore[method-assign]
    _Recorder.whoami = whoami  # type: ignore[attr-defined]
    return _Recorder


def failing_start(
    name: str,
    *,
    exc: BaseException | None = None,
    lifetime: Lifetime = Lifetime.SCOPED,
    scope: Sequence[str] = ("tenant",),
    criticality: Criticality = Criticality.REQUIRED,
) -> type[AComponent[EmptySettings, None, str]]:
    """A component whose ``start()`` raises ``exc`` (default ``RuntimeError('boom')``)."""
    error = exc if exc is not None else RuntimeError("boom")

    class _Failing(AComponent[EmptySettings, None, str]):
        pass

    _Failing.name = name
    _Failing.lifetime = lifetime
    _Failing.scope = _scope_of(scope) if lifetime is Lifetime.SCOPED else EMPTY_SCOPE
    _Failing.criticality = criticality

    async def start(self: AComponent[EmptySettings, None, str]) -> None:
        raise error

    @invocable
    async def go(self: AComponent[EmptySettings, None, str]) -> str:
        return "ok"

    _Failing.start = start  # type: ignore[method-assign]
    _Failing.go = go  # type: ignore[attr-defined]
    return _Failing


def hanging_start(
    name: str,
    *,
    lifetime: Lifetime = Lifetime.SCOPED,
    scope: Sequence[str] = ("tenant",),
) -> type[AComponent[EmptySettings, None, str]]:
    """A component whose ``start()`` never returns (blocks on an event)."""
    import anyio

    class _Hanging(AComponent[EmptySettings, None, str]):
        pass

    _Hanging.name = name
    _Hanging.lifetime = lifetime
    _Hanging.scope = _scope_of(scope) if lifetime is Lifetime.SCOPED else EMPTY_SCOPE

    async def start(self: AComponent[EmptySettings, None, str]) -> None:
        await anyio.Event().wait()

    @invocable
    async def go(self: AComponent[EmptySettings, None, str]) -> str:
        return "ok"

    _Hanging.start = start  # type: ignore[method-assign]
    _Hanging.go = go  # type: ignore[attr-defined]
    return _Hanging


def domain_metric_emitter(
    name: str,
    *,
    lifetime: Lifetime = Lifetime.PROCESS,
    scope: Sequence[str] = (),
    metric: str = "test.hits",
) -> type[AComponent[EmptySettings, None, str]]:
    """A component whose invocable bumps ``self.telemetry.counter(metric)`` by 1."""

    class _Emitter(AComponent[EmptySettings, None, str]):
        pass

    _Emitter.name = name
    _Emitter.lifetime = lifetime
    _Emitter.scope = _scope_of(scope) if lifetime is Lifetime.SCOPED else EMPTY_SCOPE

    @invocable
    async def work(self: AComponent[EmptySettings, None, str]) -> str:
        self.telemetry.counter(metric).add(1)
        return "ok"

    _Emitter.work = work  # type: ignore[attr-defined]
    return _Emitter


def health_reporter(
    name: str,
    *,
    healthy: Callable[[], bool],
    lifetime: Lifetime = Lifetime.SCOPED,
    scope: Sequence[str] = ("tenant",),
    on_check: Callable[[], None] | None = None,
) -> type[AComponent[EmptySettings, None, str]]:
    """A component whose ``health()`` reflects ``healthy()`` and notes each poll."""

    class _Reporter(AComponent[EmptySettings, None, str]):
        pass

    _Reporter.name = name
    _Reporter.lifetime = lifetime
    _Reporter.scope = _scope_of(scope) if lifetime is Lifetime.SCOPED else EMPTY_SCOPE

    async def health(self: AComponent[EmptySettings, None, str]) -> HealthStatus:
        if on_check is not None:
            on_check()
        return HealthStatus.ok() if healthy() else HealthStatus.unhealthy("reporter says no")

    @invocable
    async def go(self: AComponent[EmptySettings, None, str]) -> str:
        return "ok"

    _Reporter.health = health  # type: ignore[method-assign]
    _Reporter.go = go  # type: ignore[attr-defined]
    return _Reporter


__all__ = [
    "domain_metric_emitter",
    "failing_start",
    "hanging_start",
    "health_reporter",
    "scoped_recorder",
]
