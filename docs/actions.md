# Actions & providers

Actions and providers are two ergonomic shapes over a component. An **action** is
a component with a single operation, `execute`, callable as the object itself and
auto-exposed as a tool. A **provider** is its internal sibling: infrastructure
wired in as a dependency, never an entry point.

They are a convention, not a new runtime: an action's `execute` is an ordinary
`@invocable`, so it goes through the same policy chain, descriptor, schema and
MCP machinery as any component - you just write less.

## Writing an action

An action takes its whole input as one pydantic model and returns a value:

```python
from warpweft import Action, ActionParams, EmptySettings


class Summarise(ActionParams):
    text: str
    max_words: int = 20


class Summarize(Action[EmptySettings, Summarise, Digest]):
    description = "Summarise a document."
    read_only = True

    async def execute(self, params: Summarise) -> Digest: ...
```

`execute` is wired once, at class creation: it is marked `@invocable` and
"boxed" (its model's fields are the flat contract, rebuilt into `params` on the
way in), so you never write `@invocable` or `@tool` by hand. The input model is
read from the parameter's annotation - not a generic - so it is exactly the
model whose validators and field formats apply.

## Calling it

Calling the instance runs `execute` **through the policy chain** (retry, breaker,
cache, telemetry) and returns the value. It accepts the model, a mapping, or
keyword fields:

```python
summarize = await container.get(Summarize)  # or an injected dependency
digest = await summarize(text="...", max_words=8)
digest = await summarize(Summarise(text="..."))
```

`__call__` needs a running app - the container binds the invoker when it resolves
the instance. Outside one, call `execute` directly: that is the pure, chain-free
path, ideal for unit tests.

```python
digest = await Summarize(EmptySettings()).execute(Summarise(text="..."))
```

An action injected as a dependency (`self.my_action`) is the same resolved
instance, so `await self.my_action(...)` also runs guarded, through its own
chain - unlike a raw dependency method call, which bypasses it.

## Exposure as a tool

An action is a tool by default. It carries its tool metadata unconditionally
(the marker is pure - no MCP SDK needed), and the optional [`warpweft.mcp`](mcp.md)
layer collects every action when a server is built - no per-class SDK detection.
The metadata comes from class attributes:

| Attribute | Meaning |
| --- | --- |
| `description` | tool description (else the `execute` docstring) |
| `tool_name` | override the derived `<component>__execute` name |
| `title` | human-friendly title |
| `icons` | icons clients show next to the tool ([server metadata](mcp.md#server-metadata)) |
| `read_only`, `destructive`, `idempotent`, `open_world` | MCP behaviour hints |
| `tags` | labels for [tool filtering](mcp.md#filtering) when a server is built |

Set `entrypoint = False` on an action to keep it internal (registered and
callable, but never a tool).

The tool's input schema is **flat** - the params model's fields, not a nested
`{params: {...}}` - and identical to what `warpweft schema <action>:execute` and
`invoke` validate against.

## Providers

A provider is a component used only as a dependency - a pool, a client, a
gateway. It has `entrypoint = False`, so it is never collected as a tool, and,
unlike every other component, it may declare **no** invocables at all (it is
reached through raw dependency access):

```python
from warpweft import Provider, EmptySettings


class Vocabulary(Provider[EmptySettings]):
    def stopwords(self) -> set[str]:
        return {"the", "a", "an"}
```

Its methods may still be `@invocable` when you want them to run through a policy
chain via `container.invoke`. Attempting to `@tool` a provider method is rejected
when the server is built - the "internal only" boundary is enforced, not just a
convention.

## When to reach for which

- A unit of work an LLM or a caller invokes by name, with typed input/output →
  **action**.
- Shared infrastructure other components depend on → **provider**.
- A component with several distinct entry points, or one that does not fit the
  single-`execute` shape → a plain [`AComponent`](first-component.md) with
  `@invocable` methods (and `@tool` where you want exposure).
```
