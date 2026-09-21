"""Runnable demo: OpenTelemetry console exporters around a warpweft App.

Warpweft always instruments through the OTel *API*; nothing is collected until
the application configures an SDK. This example wires the SDK's console
exporters as explicit providers passed through ``App(...)`` (any
``Container.build`` option passes through), then drives a flaky component so the
console shows the invocation span, a retry ``warpweft.attempt`` child span, the
``warpweft.retry.backoff`` event and the metrics from the docs/telemetry.md
contract.

The OTel SDK is not a warpweft dependency; install it to run this example:
    pip install opentelemetry-sdk

Run:
    uv run python examples/telemetry_console.py
"""

import anyio
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from pydantic import BaseModel

from warpweft import AComponent, App, Registry, TransientError, component, invocable


class QuotesSettings(BaseModel):
    text: str = "the loom holds the warp taut"


@component  # registered under the derived name "quotes"
class Quotes(AComponent[QuotesSettings, str, str]):
    def __init__(self, settings: QuotesSettings) -> None:
        super().__init__(settings)
        self._attempts = 0

    @invocable
    async def fetch(self) -> str:
        self._attempts += 1
        if self._attempts == 1:
            raise TransientError("upstream warming up")  # retry recovers this
        return self.settings.text


async def main() -> None:
    # Configure the OTel SDK the usual way; warpweft collects nothing without it.
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    meter_provider = MeterProvider(
        metric_readers=[PeriodicExportingMetricReader(ConsoleMetricExporter(), export_interval_millis=60_000)]
    )

    # A private registry keeps this demo out of the process-wide default registry.
    registry = Registry()
    registry.register(Quotes)
    app = App(
        registry=registry,
        config={"quotes": {"policy": {"retry": {"attempts": 3, "base_delay": 0.05, "max_delay": 0.5}}}},
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )

    async with app.run():
        quotes = app.proxy(Quotes)  # runs through the full chain, instrumented
        print("fetch() =", await quotes.fetch(), "\n")  # first attempt fails, retry succeeds

    # Flush: the span processor exports on end, the meter reader on shutdown.
    tracer_provider.shutdown()
    meter_provider.shutdown()


if __name__ == "__main__":
    anyio.run(main)
