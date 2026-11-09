"""Chain composition: ordering, isolation of links, per-scope instances."""

from typing import Any

import pytest

from warpweft.core.axes import Axis, AxisRegistry, ScopeKey, ScopeSpec
from warpweft.core.context import InvocationContext
from warpweft.core.outcome import Outcome
from warpweft.core.pipeline.chain import DEFAULT_ORDER, build_chain, compose
from warpweft.core.pipeline.interceptor import Interceptor, Next
from warpweft.core.pipeline.state import InMemoryStateStore
from warpweft.core.unit import Identity

pytestmark = [pytest.mark.unit, pytest.mark.anyio]


def ctx() -> InvocationContext:
    return InvocationContext(operation="op", correlation_id="cid")


class Recorder:
    """Link that records when the call passes through it, in both directions."""

    def __init__(self, name: str, log: list[str]) -> None:
        self.identity = Identity.of(name)
        self._log = log

    async def call(self, next: Next, ctx: InvocationContext) -> Outcome[Any]:
        self._log.append(f">{self.identity.name}")
        outcome = await next(ctx)
        self._log.append(f"<{self.identity.name}")
        return outcome


async def base(ctx: InvocationContext) -> Outcome[Any]:
    return Outcome(value="result")


async def test_first_link_is_outermost() -> None:
    log: list[str] = []

    async def logging_base(c: InvocationContext) -> Outcome[Any]:
        log.append("base")
        return Outcome(value="ok")

    chain = compose([Recorder("outer", log), Recorder("inner", log)], logging_base)
    outcome = await chain(ctx())

    assert outcome.value == "ok"
    assert log == [">outer", ">inner", "base", "<inner", "<outer"]


async def test_empty_chain_is_base() -> None:
    chain = compose([], base)
    assert (await chain(ctx())).value == "result"


async def test_links_communicate_only_through_bag() -> None:
    class Writer:
        identity = Identity.of("writer")

        async def call(self, next: Next, ctx: InvocationContext) -> Outcome[Any]:
            ctx.bag["fact"] = 42
            return await next(ctx)

    class Reader:
        identity = Identity.of("reader")
        seen: int | None = None

        async def call(self, next: Next, ctx: InvocationContext) -> Outcome[Any]:
            self.seen = ctx.bag.get("fact")
            return await next(ctx)

    reader = Reader()
    chain = compose([Writer(), reader], base)
    await chain(ctx())
    assert reader.seen == 42


def test_default_order_is_fixed() -> None:
    assert DEFAULT_ORDER == ("concurrency", "cache", "circuit_breaker", "retry", "timeout")


class RecorderFactory:
    """Factory producing Recorder instances, counting creations per key."""

    def __init__(self, name: str, log: list[str], scope: ScopeSpec = ScopeSpec()) -> None:
        self.identity = Identity.of(name)
        self.state_scope = scope
        self._log = log
        self.created: list[ScopeKey] = []

    def create(self, key: ScopeKey) -> Interceptor:
        self.created.append(key)
        return Recorder(self.identity.name, self._log)


async def test_build_chain_reuses_instance_per_key() -> None:
    log: list[str] = []
    factory = RecorderFactory("r", log)
    chain = build_chain([factory], InMemoryStateStore(), AxisRegistry(), base)

    await chain(ctx())
    await chain(ctx())

    assert factory.created == [()]  # one instance, empty scope


async def test_empty_scope_key_includes_operation_and_canonical_invocation_scope() -> None:
    log: list[str] = []
    factory = RecorderFactory("r", log)
    store = InMemoryStateStore()
    chain = build_chain([factory], store, AxisRegistry(), base)
    unsorted_scope: ScopeKey = (("tenant", "acme"), ("region", "eu"))
    canonical_scope: ScopeKey = (("region", "eu"), ("tenant", "acme"))

    await chain(ctx().child(scope_key=unsorted_scope))
    await chain(ctx().child(operation="other", scope_key=canonical_scope))
    await chain(ctx().child(scope_key=(("tenant", "globex"),)))
    await chain(ctx().child(scope_key=canonical_scope))  # same operation and scope: reuse

    unit = ("__unit__", factory.identity.uid)
    assert store.keys() == [
        (unit, ("__operation__", "other"), *canonical_scope),
        (unit, ("__operation__", "op"), ("tenant", "globex")),
        (unit, ("__operation__", "op"), *canonical_scope),
    ]
    assert factory.created == [(), (), ()]


async def test_empty_scope_key_is_stable_across_factory_instances() -> None:
    log: list[str] = []
    store = InMemoryStateStore()
    first = RecorderFactory("same", log)
    second = RecorderFactory("same", log)

    await build_chain([first], store, AxisRegistry(), base)(ctx())
    await build_chain([second], store, AxisRegistry(), base)(ctx())
    await build_chain([second], store, AxisRegistry(), base)(ctx().child(operation="other"))

    assert first.created == [()]
    assert second.created == [()]  # only the different operation creates a new link


async def test_non_empty_scope_key_excludes_operation_and_invocation_scope() -> None:
    log: list[str] = []
    registry = AxisRegistry()
    registry.register(Axis(name="endpoint", resolver=lambda: "shared"))
    factory = RecorderFactory("r", log, scope=ScopeSpec(["endpoint"]))
    store = InMemoryStateStore()
    chain = build_chain([factory], store, registry, base)

    await chain(ctx().child(scope_key=(("tenant", "acme"),)))
    await chain(ctx().child(operation="other", scope_key=(("tenant", "globex"),)))

    resolved: ScopeKey = (("endpoint", "shared"),)
    assert store.keys() == [(("__unit__", factory.identity.uid), *resolved)]
    assert factory.created == [resolved]


async def test_build_chain_resolves_scope_on_every_call() -> None:
    """Changing an axis value between calls must yield a different instance."""
    log: list[str] = []
    current = {"tenant": "a"}

    registry = AxisRegistry()
    registry.register(Axis(name="tenant", resolver=lambda: current["tenant"]))

    factory = RecorderFactory("r", log, scope=ScopeSpec(["tenant"]))
    chain = build_chain([factory], InMemoryStateStore(), registry, base)

    await chain(ctx())
    current["tenant"] = "b"
    await chain(ctx())
    current["tenant"] = "a"
    await chain(ctx())

    assert factory.created == [(("tenant", "a"),), (("tenant", "b"),)]


async def test_links_with_identical_scope_do_not_collide_in_shared_store() -> None:
    log: list[str] = []
    first = RecorderFactory("first", log)
    second = RecorderFactory("second", log)
    chain = build_chain([first, second], InMemoryStateStore(), AxisRegistry(), base)

    await chain(ctx())

    assert first.created == [()]
    assert second.created == [()]
    assert log == [">first", ">second", "<second", "<first"]


async def test_build_chain_order_matches_list() -> None:
    log: list[str] = []
    factories = [RecorderFactory("a", log), RecorderFactory("b", log), RecorderFactory("c", log)]
    chain = build_chain(factories, InMemoryStateStore(), AxisRegistry(), base)

    await chain(ctx())

    assert log == [">a", ">b", ">c", "<c", "<b", "<a"]
