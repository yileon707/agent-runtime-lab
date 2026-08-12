# P0.2A — Canonical Model / Provider Contract

> **Date**: 2026-08-12
> **Prerequisite**: [P0.1 — Baseline Audit](./00-baseline-audit.md)
> **Followed by**: P0.2B (adapters + legacy bridge)

---

## Problem

The current Runtime (s01–s17) has **no abstraction** between application logic and the Anthropic SDK. Every lesson file:

1. Imports `from anthropic import Anthropic`
2. Calls `client.messages.create(...)`
3. Iterates `response.content` checking `block.type == "tool_use"`
4. Constructs `{"type": "tool_result", "tool_use_id": block.id, ...}` in Anthropic message format

The Anthropic message schema **is** the internal runtime representation. This means:

- Adding DeepSeek (OpenAI-compatible `POST /chat/completions`) requires touching **30+ files**.
- Context compaction depends on `assistant(tool_use) → user(tool_result)` message adjacency.
- Tool call representation is `block.name`, `block.input` — Anthropic SDK attributes.
- Loop control is `response.stop_reason == "tool_use"` — an Anthropic string.
- Token usage is read as `response.usage.input_tokens` — Anthropic attribute names.

The P0.1 audit confirmed this coupling is **pervasive and has zero abstraction**.

---

## Decision

Introduce a **provider-agnostic Canonical Model Contract** (`agent_runtime/model/`).

The contract defines the vocabulary the Runtime uses to describe conversations, tool calls, and model responses — without importing any vendor SDK.

---

## Boundary

```
┌──────────────────────────────────────────────┐
│  Agent Runtime (s15, compaction, teams, …)   │
│                                               │
│         uses agent_runtime.model types        │
├──────────────────────────────────────────────┤
│  Canonical Model Contract (contracts.py)      │
│  ModelProvider Protocol (provider.py)         │
├──────────────────────────────────────────────┤
│  Provider Adapters (future: P0.2B)            │
│  ├── AnthropicAdapter                         │
│  ├── DeepSeekAdapter                          │
│  └── …                                        │
├──────────────────────────────────────────────┤
│  Vendor SDKs / HTTP APIs                      │
│  ├── Anthropic Messages API                   │
│  ├── OpenAI Chat Completions API              │
│  └── DeepSeek (OpenAI-compatible)             │
└──────────────────────────────────────────────┘
```

The Runtime **never** touches vendor SDKs directly. It only calls `ModelProvider.complete(request) → ModelResponse`.

---

## Canonical Types

### MessageRole

```
SYSTEM | USER | ASSISTANT | TOOL
```

### FinishReason

```
STOP          — model chose to end the turn
TOOL_CALLS    — model emitted one or more tool calls
LENGTH        — output hit max limit
CONTENT_FILTER — safety filter blocked output
UNKNOWN       — unrecognised reason
```

**IMPORTANT**: There is **no** ``ERROR`` finish reason. Provider failures are raised as ``ProviderError`` exceptions — this ensures a single error channel. ``FinishReason`` describes *successful* completions only.

These are **normalised** — every adapter translates vendor-specific strings (`"tool_use"`, `"function_call"`, `"max_tokens"`) into these canonical values.

### ToolDefinition

```python
ToolDefinition(
    name="bash",
    description="Run a shell command",
    parameters={...},   # ← provider-neutral JSON Schema (NOT "input_schema")
)
```

### ToolCall — with Malformed-Argument Tolerance

```python
ToolCall(
    id="call_01",
    name="bash",
    arguments={"command": "ls"},   # parsed dict, or None if parse failed
    raw_arguments='{"command": "ls"}',  # original provider string
    argument_error=None,                 # error message if parse failed
)
```

The `ToolCall.from_raw_json()` and `ToolCall.from_arguments()` factories are the **two canonical entry points** for constructing tool calls:

- `from_arguments(call_id, name, arguments: dict)` — for providers that natively return structured dict input (e.g. Anthropic). The dict is **defensively copied** via `copy.deepcopy()` so provider-side mutation cannot affect Runtime state.
- `from_raw_json(call_id, name, raw_arguments: str | None)` — for providers that return JSON string arguments (e.g. OpenAI / DeepSeek).

- Valid JSON → `arguments` = parsed dict, `argument_error` = None
- Invalid JSON → `arguments` = None, `raw_arguments` preserved, `argument_error` set
- Non-dict JSON (arrays, strings, numbers) → same graceful failure path
- `raw_arguments=None` (no arguments) → `arguments = {}`, no error

**Rationale**: Provider APIs sometimes return malformed JSON in tool call arguments. The Runtime must not crash. By preserving `raw_arguments` and `argument_error`, the Runtime can later retry or report the failure without losing data.

### ProviderState — Opaque Replay State

```python
ProviderState(
    provider="deepseek",
    data={"session_id": "abc123", ...},  # opaque, JSON-serialisable
)
```

**Invariant**:

1. Runtime MAY store, copy, serialise, and hand this object back to the provider unchanged.
2. Runtime MUST NEVER interpret or depend on any key inside `data`.
3. `data` MUST be JSON-serialisable (enforced at construction).

**Semantic vs Protocol State**: ``ProviderState`` carries **protocol-level** provider state — opaque data the provider needs for replay continuity (multi-turn conversation IDs, cached-prompt identifiers, reasoning tokens). This is distinct from **semantic state** (conversation messages, tool call/result pairs) which lives in ``ModelMessage``. Protocol state is provider-specific and opaque; semantic state is canonical and transparent. Keeping them separate prevents vendor concepts from leaking into Runtime logic.

**Role constraint**: ``provider_state`` is only valid on ``ASSISTANT`` messages. Providers attach replay-critical state to assistant turns (e.g., Anthropic's ``message.id`` or DeepSeek's conversation session identifier). Constructing a ``USER``, ``TOOL``, or ``SYSTEM`` message with ``provider_state`` raises ``ValueError``. This ensures protocol state is always anchored to the turn that produced it.

### ModelMessage

```python
ModelMessage(
    role=MessageRole.ASSISTANT,
    content="I'll run that command.",       # text, or None for tool-call-only
    tool_calls=[tool_call],                  # ONLY on ASSISTANT
    tool_call_id=None,                       # REQUIRED when role=TOOL
    provider_state=None,                     # opaque provider data
)
```

**Constraints enforced at construction**:
- `role=TOOL` → `tool_call_id` must be set → enables tool result correlation.
- `role != ASSISTANT` → `tool_calls` must be empty/None.
- `role != ASSISTANT` → `provider_state` must be None (protocol state is anchored to assistant turns only).
- Convenience constructors: `ModelMessage.system()`, `.user()`, `.assistant()`, `.tool()`.

### TokenUsage

```python
TokenUsage(
    input_tokens=150,           # non-negative
    output_tokens=50,           # non-negative
    total_tokens=200,           # optional; derived from input+output if absent
    reasoning_tokens=None,      # optional, non-negative
    cache_read_tokens=None,     # optional, non-negative — tokens read from cache
    cache_creation_tokens=None, # optional, non-negative — tokens written to cache
    cache_miss_tokens=None,     # optional, non-negative — cache lookups that missed
    provider_details={},        # dict — extra usage metadata (must be JSON-serialisable)
)
```

All numeric fields are validated non-negative at construction. This prevents different providers' usage schemas from leaking into Runtime accounting.

**Cache semantics**: ``cache_read_tokens``, ``cache_creation_tokens``, and ``cache_miss_tokens`` are **independent dimensions**. A cache miss is NOT automatically a cache write — providers can report cache misses without writing to cache (e.g., Anthropic's cache hit/miss breakdown vs OpenAI's prompt caching that auto-caches on miss). Adapters must map provider-specific cache fields without conflating these concepts.

**provider_details**: Carries usage metadata that cannot be reliably canonicalised across providers (e.g., Anthropic's ``cache_read_input_tokens`` breakdown, OpenAI's ``prompt_tokens_details``). The Runtime MAY store, trace, or serialise it, but MUST NOT depend on specific keys for core decision-making.

### ModelRequest / ModelResponse

```python
ModelRequest(
    model="deepseek-v4-pro",
    messages=[...],           # list[ModelMessage]
    tools=[...],              # optional list[ToolDefinition]
    max_output_tokens=8000,
    # NOTE: api_key and base_url are NOT here — they belong to provider config
)

ModelResponse(
    message=ModelMessage(...),  # MUST be ASSISTANT
    finish_reason=FinishReason.TOOL_CALLS,
    usage=TokenUsage(...),
    provider="deepseek",
    response_id="resp-42",
)
```

---

## Provider Protocol

```python
class ModelProvider(Protocol):
    @property
    def name(self) -> str: ...

    def complete(self, request: ModelRequest) -> ModelResponse: ...
```

`complete()` is **synchronous** — matching the current Runtime. Async and streaming are deferred to future phases.

### ProviderError

```python
ProviderError(
    kind=ProviderErrorKind.RATE_LIMIT,
    provider="deepseek",
    message="Too many requests",
    status_code=429,
    provider_code="rate_limit_exceeded",
    retryable=True,          # adapter recommendation; Runtime may override
)
```

**ProviderErrorKind values**:

```
INVALID_REQUEST | AUTH | BILLING | RATE_LIMIT | CONTEXT_LIMIT |
SERVER | OVERLOADED | INVALID_RESPONSE | UNKNOWN
```

The Runtime's recovery logic operates on `kind`, **not** on vendor-specific HTTP status codes or error strings. The adapter is responsible for normalisation.

Default retryable kinds: `RATE_LIMIT`, `SERVER`, `OVERLOADED`.

---

## Why Normalise Finish Reason

Anthropic: `stop_reason == "tool_use"`, `"end_turn"`, `"max_tokens"`
OpenAI / DeepSeek: `finish_reason == "tool_calls"`, `"stop"`, `"length"`

Without normalisation, every piece of Runtime logic that checks "should we stop or continue?" must contain vendor-specific string comparisons. With `FinishReason`, the Runtime checks `FinishReason.TOOL_CALLS` — the adapter handles translation.

---

## Why Normalise Token Usage Now

Different providers report token counts under different attribute names:

| Concept | Anthropic | OpenAI/DeepSeek |
|---|---|---|
| Input tokens | `usage.input_tokens` | `usage.prompt_tokens` |
| Output tokens | `usage.output_tokens` | `usage.completion_tokens` |
| Cache read | `usage.cache_read_input_tokens` | `usage.prompt_tokens_details.cached_tokens` |
| Cache creation | `usage.cache_creation_input_tokens` | auto-cached (implicit) |
| Cache miss | `cache miss = prompt - cache_read - cache_creation` | N/A (auto-caching) |
| Reasoning | N/A | `usage.completion_tokens_details.reasoning_tokens` |
| Provider metadata | `provider_details` (non-canonical) | `provider_details` (non-canonical) |

By normalising into `TokenUsage` now, future Runtime subsystems (cost tracking, context budget, observability) can consume a single vocabulary.

---

## Migration Strategy

| Phase | Scope |
|---|---|
| **P0.2A** (this phase) | Contract definition + tests only. No Runtime changes. |
| **P0.2B** | Provider adapters (Anthropic, DeepSeek), legacy s15 bridge. |
| **P0.2C** | Real DeepSeek validation — live model calls through the contract. |

After P0.2B, s15 can be incrementally migrated to use `ModelProvider.complete()` instead of `client.messages.create()`. Until then, s15 runs unchanged.

---

## Non-Goals (This Phase)

- No Anthropic adapter
- No DeepSeek adapter / real DeepSeek calls
- No s15/s16/s17 migration
- No streaming
- No async API
- No provider capability registry
- No tracing or eval framework
- **DeepSeek is NOT proven runnable yet** — this phase establishes the contract, not the integration.

---

## Test Coverage

`tests/test_model_provider_contract.py` — 54 tests covering:

1. Architectural constraint: `agent_runtime.model` does not import `anthropic` or `openai`
2. Basic conversation construction (system → user → assistant)
3. Tool call → tool result correlation via `tool_call_id`
4. Multiple tool calls in one assistant response
5. Valid tool argument JSON parsing
6. Malformed tool arguments (invalid JSON, non-dict, None) — graceful handling
7. `ToolCall.from_arguments()` with defensive copy (shallow + deep nested)
8. `ProviderState` round-trip serialisation
9. `provider_state` ASSISTANT-only: allowed on ASSISTANT, rejected on USER/TOOL/SYSTEM
10. `TOOL` role without `tool_call_id` → rejected
11. `USER` message with `tool_calls` → rejected
12. `ModelResponse.message` not `ASSISTANT` → rejected
13. `TokenUsage` negative values → rejected (all numeric fields)
14. `TokenUsage.total` derivation when absent
15. `TokenUsage` cache fields: `cache_creation_tokens`, `cache_miss_tokens` independent
16. `TokenUsage.provider_details` — stores extra metadata, JSON-serialisable guard
17. `FinishReason` values are vendor-neutral
18. `FinishReason.ERROR` absent — provider failures use `ProviderError` only
19. `ProviderError` retryable semantics
20. Full `ModelRequest` → `ModelResponse` round-trip
21. `ToolDefinition`, `ModelRequest`, `ToolCall` validation constraints

---

## Files

```
agent_runtime/
  __init__.py
  model/
    __init__.py
    contracts.py              # canonical types
    provider.py               # ModelProvider protocol + ProviderError

tests/
  test_model_provider_contract.py   # 54 tests

docs/interview/
  01-model-provider-contract.md     # this document
```
