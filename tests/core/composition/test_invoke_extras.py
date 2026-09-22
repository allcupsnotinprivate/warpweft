"""invoke/proxy extras: overall budget and ambient correlation id."""

from typing import Any

import pytest

from warpweft.core.component import AComponent, EmptySettings, invocable
from warpweft.core.composition import Container, Registry
from warpweft.core.context import InvocationContext, use_correlation_id
from warpweft.core.errors import DeadlineExceeded, TransientError

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


class Flaky(AComponent[EmptySettings, None, str]):
    def __init__(self, settings: EmptySettings) -> None:
        super().__init__(settings)
        self.calls = 0

    @invocable
    async def fetch(self) -> str:
        self.calls += 1
        raise TransientError(f"attempt {self.calls}")


class Echo(AComponent[EmptySettings, None, str]):
    @invocable
    async def whoami(self, ctx: InvocationContext) -> str:
        return ctx.correlation_id


def container_with(*classes: type[AComponent[Any, Any, Any]], config: Any = None) -> Container:
    reg = Registry()
    for cls in classes:
        reg.register(cls)
    return Container.build(reg, config or {name: {} for name in (c.name for c in classes)})


# --- budget ------------------------------------------------------------------


async def test_budget_sets_an_overall_deadline() -> None:
    # A tiny budget with a 1s backoff: the first failure cannot afford a retry.
    config = {"flaky": {"policy": {"retry": {"attempts": 5, "base_delay": 1.0, "max_delay": 1.0, "jitter": False}}}}
    container = container_with(Flaky, config=config)
    await container.start()
    flaky = await container.get(Flaky)
    with pytest.raises(DeadlineExceeded):
        await container.invoke("flaky", "fetch", budget=0.05)
    assert flaky.calls == 1  # stopped by the deadline, not by exhausting attempts
    await container.stop()


async def test_no_budget_means_no_deadline() -> None:
    container = container_with(Echo)
    await container.start()
    outcome = await container.invoke("echo", "whoami")  # would raise if a bad deadline slipped in
    assert outcome.value  # a generated correlation id
    await container.stop()


async def test_proxy_budget_applies_to_every_call() -> None:
    config = {"flaky": {"policy": {"retry": {"attempts": 5, "base_delay": 1.0, "max_delay": 1.0, "jitter": False}}}}
    container = container_with(Flaky, config=config)
    await container.start()
    flaky = container.proxy(Flaky, budget=0.05)
    with pytest.raises(DeadlineExceeded):
        await flaky.fetch()
    await container.stop()


# --- correlation id ----------------------------------------------------------


async def test_correlation_id_defaults_to_the_ambient_one() -> None:
    container = container_with(Echo)
    await container.start()
    with use_correlation_id("req-42"):
        assert (await container.invoke("echo", "whoami")).value == "req-42"
        assert (await container.proxy(Echo).whoami()) == "req-42"
    await container.stop()


async def test_explicit_correlation_id_wins_over_ambient() -> None:
    container = container_with(Echo)
    await container.start()
    with use_correlation_id("ambient"):
        assert (await container.invoke("echo", "whoami", correlation_id="explicit")).value == "explicit"
    await container.stop()


async def test_correlation_id_is_generated_without_an_ambient_one() -> None:
    container = container_with(Echo)
    await container.start()
    first = (await container.invoke("echo", "whoami")).value
    second = (await container.invoke("echo", "whoami")).value
    assert first and second and first != second  # fresh ids
    await container.stop()
