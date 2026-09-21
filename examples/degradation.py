"""Runnable demo: graceful degradation of an optional component.

An *optional* component that defines a ``stub()`` answers from that fallback
when its real call is unavailable, instead of failing the caller. Turn it on
with a ``policy.degradation`` config block. Here retry exhausts against a
persistent outage, then the stub substitutes: the outcome carries
``source="stub"`` and ``degraded=True``. When the upstream recovers the call is
live again, with no config change.

Run:
    uv run python examples/degradation.py
"""

import anyio
from pydantic import BaseModel

from warpweft import AComponent, App, Criticality, InvocationContext, Registry, TransientError, invocable


class WeatherSettings(BaseModel):
    city: str = "paris"


class Weather(AComponent[WeatherSettings, str, dict]):
    name = "weather"
    criticality = Criticality.OPTIONAL  # required for degradation

    def __init__(self, settings: WeatherSettings) -> None:
        super().__init__(settings)
        self.online = False

    def stub(self, ctx: InvocationContext) -> dict:
        # Synchronous, local: a last-known-good value, not a second call.
        return {"city": ctx.arguments["city"], "temp": None, "stale": True}

    @invocable
    async def current(self, city: str) -> dict:
        if not self.online:
            raise TransientError("weather upstream is down")
        return {"city": city, "temp": 18, "stale": False}


async def main() -> None:
    registry = Registry()
    registry.register(Weather)
    app = App(
        registry=registry,
        config={
            "weather": {
                "policy": {
                    "degradation": {},
                    "retry": {"attempts": 3, "base_delay": 0.01, "max_delay": 0.1},
                }
            }
        },
    )

    async with app.run():
        # Outage: retry exhausts, then the stub answers.
        degraded = await app.invoke("weather", "current", city="paris")
        print("outage  ->", degraded.value, f"(source={degraded.source}, degraded={degraded.degraded})")

        # Recover the upstream; the same call is live again.
        weather: Weather = await app.get(Weather)
        weather.online = True
        live = await app.invoke("weather", "current", city="paris")
        print("recover ->", live.value, f"(source={live.source}, degraded={live.degraded})")


if __name__ == "__main__":
    anyio.run(main)
