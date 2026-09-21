"""Building link factories from validated settings.

Maps a link name to a builder that turns its settings model into a factory.
Degradation is deliberately absent from the builders: it is not an ordered
link but is wired at a fixed position around the whole chain by
``warpweft.core.composition.wiring.degradation_interceptor`` (gated on
criticality + a ``stub()`` method + ``policy.degradation`` config).
"""

from collections.abc import Callable, Mapping
from typing import cast

from pydantic import BaseModel

from warpweft.core.clock import Clock
from warpweft.core.errors import ErrorClassifier
from warpweft.core.pipeline.builtin.cache import CacheFactory, CacheSettings
from warpweft.core.pipeline.builtin.circuit_breaker import CircuitBreakerFactory, CircuitBreakerSettings
from warpweft.core.pipeline.builtin.concurrency import ConcurrencyFactory, ConcurrencySettings
from warpweft.core.pipeline.builtin.retry import RetryFactory, RetrySettings
from warpweft.core.pipeline.builtin.timeout import TimeoutFactory, TimeoutSettings
from warpweft.core.pipeline.interceptor import InterceptorFactory

#: A builder turns a settings model + clock + classifier into a link factory.
LinkBuilder = Callable[[BaseModel, Clock, ErrorClassifier], InterceptorFactory]


def _retry(settings: BaseModel, clock: Clock, classifier: ErrorClassifier) -> InterceptorFactory:
    return RetryFactory(cast(RetrySettings, settings), clock, classifier=classifier)


def _timeout(settings: BaseModel, clock: Clock, classifier: ErrorClassifier) -> InterceptorFactory:
    return TimeoutFactory(cast(TimeoutSettings, settings), clock)


def _circuit_breaker(settings: BaseModel, clock: Clock, classifier: ErrorClassifier) -> InterceptorFactory:
    return CircuitBreakerFactory(cast(CircuitBreakerSettings, settings), clock, classifier=classifier)


def _concurrency(settings: BaseModel, clock: Clock, classifier: ErrorClassifier) -> InterceptorFactory:
    return ConcurrencyFactory(cast(ConcurrencySettings, settings), clock)


def _cache(settings: BaseModel, clock: Clock, classifier: ErrorClassifier) -> InterceptorFactory:
    return CacheFactory(cast(CacheSettings, settings), clock)


#: The built-in links that can appear in a policy chain, by name.
BUILTIN_LINK_BUILDERS: Mapping[str, LinkBuilder] = {
    "retry": _retry,
    "timeout": _timeout,
    "circuit_breaker": _circuit_breaker,
    "concurrency": _concurrency,
    "cache": _cache,
}
