# Logging

Warpweft follows the library convention: it **emits** through named loggers and
never **configures** logging. There is no `basicConfig`, no handler, no level
set by default - that is the application's job. A `NullHandler` is attached to
the `warpweft` logger so records drop silently until you opt in.

Logging is deliberately sparse and complements telemetry rather than
duplicating it: per-call detail lives in OpenTelemetry spans/metrics (see
[telemetry.md](telemetry.md)); logs cover coarse lifecycle and diagnostic
events (container start/stop, a component degrading at startup, a circuit
breaker opening/closing, a call degrading to a stub).

## Logger names

Loggers follow the module path under the `warpweft` hierarchy, e.g.
`warpweft.core.composition.container`, `warpweft.core.pipeline.builtin.circuit_breaker`.
Tune any branch independently.

## Turning it on

Standard `logging`:

```python
import logging

logging.getLogger("warpweft").setLevel(logging.INFO)
logging.basicConfig()  # your handler/format

# or per subsystem
logging.getLogger("warpweft.core.pipeline.builtin.circuit_breaker").setLevel(logging.WARNING)
```

Records propagate to whatever handlers your application attaches to the root
(or to `warpweft`) - the library plugs into your setup automatically, with no
coupling.

## Turning it off

Off is the default (NullHandler, no level). To silence it even when the root is
configured:

```python
logging.getLogger("warpweft").setLevel(logging.CRITICAL + 1)
# or
logging.getLogger("warpweft").disabled = True
```

## Correlation id in logs (opt-in)

Attach `CorrelationIdFilter` to your handler to surface the ambient correlation
id (set by the runtime per request, or via `use_correlation_id`) in your
format string:

```python
from warpweft.core.logging import CorrelationIdFilter

handler = logging.StreamHandler()
handler.addFilter(CorrelationIdFilter())  # adds record.correlation_id
handler.setFormatter(logging.Formatter("%(correlation_id)s %(name)s: %(message)s"))
logging.getLogger("warpweft").addHandler(handler)
```

Warpweft does not install the filter for you - it is a plain `logging.Filter`.

## Don't log secrets

The same caution as elsewhere: never log resolved settings verbatim - they may
contain API keys or DSNs. Use `pydantic.SecretStr` for secret fields.
