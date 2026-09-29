"""The built-in ``endpoint`` axis.

Unlike an ordinary axis, ``endpoint`` does not read a request-scoped value: it
reflects which external system the *current instance* talks to. The container
sets it around each invocation from the instance's `AComponent.endpoint`,
so ``[endpoint]``-sliced link state (breaker, concurrency) is shared by all
instances hitting the same endpoint and separated otherwise - without any
component declaring it.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from warpweft.core.axes import Axis, ScopeKey

#: Name of the built-in endpoint axis.
ENDPOINT_AXIS = "endpoint"


def default_endpoint(uid: str, scope_key: ScopeKey) -> str:
    """Fallback endpoint for an instance that declares no ``endpoint()``.

    An instance's identity uid is per ``(name, version)`` - shared by every
    slice of a scoped component. Using it bare would slice ``[endpoint]`` link
    state (breaker, concurrency) by component, so one tenant tripping the
    breaker would open it for all tenants. Folding the instance's slice into
    the fallback isolates that state per slice by default (as cache already is);
    an explicit ``endpoint()`` still wins and can deliberately share state.
    """
    if not scope_key:
        return uid
    slice_repr = ",".join(f"{axis}={value}" for axis, value in scope_key)
    return f"{uid}[{slice_repr}]"


_current_endpoint: ContextVar[str | None] = ContextVar("warpweft_current_endpoint", default=None)


@contextmanager
def use_endpoint(value: str) -> Iterator[None]:
    """Bind the current endpoint for the duration of an invocation."""
    token = _current_endpoint.set(value)
    try:
        yield
    finally:
        _current_endpoint.reset(token)


def endpoint_axis() -> Axis:
    """The endpoint axis: resolves to the endpoint bound by `use_endpoint`."""
    return Axis(name=ENDPOINT_AXIS, resolver=_current_endpoint.get)
