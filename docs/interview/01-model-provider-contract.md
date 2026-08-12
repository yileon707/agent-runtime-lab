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
ERROR         — provider returned an error
UNKNOWN       — unrecognised reason
```

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

The `ToolCall.from_raw_json()` factory is the **single canonical entry point** for constructing tool calls. It guarantees:

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

**Rationale**: Some providers require replay-critical protocol state (multi-turn conversation IDs, cached-prompt identifiers, reasoning tokens). This state must survive serialisation (e.g., durable task storage) and be returned to the provider on subsequent turns — but it must never leak into Runtime logic. `ProviderState` is a sealed box.

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
- Convenience constructors: `ModelMessage.system()`, `.user()`, `.assistant()`, `.tool()`.

### TokenUsage

```python
TokenUsage(
    input_tokens=150,        # non-negative
    output_tokens=50,        # non-negative
    total_tokens=200,        # optional; derived from input+output if absent
    cache_read_tokens=None,  # optional, non-negative
    cache_write_tokens=None, # optional, non-negative
    reasoning_tokens=None,   # optional, non-negative
)
```

All fields are validated non-negative at construction. This prevents different providers' usage schemas from leaking into Runtime accounting.

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
| Reasoning | not available | `usage.completion_tokens_details.reasoning_tokens` |

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

`tests/test_model_provider_contract.py` — 35 tests covering:

1. Architectural constraint: `agent_runtime.model` does not import `anthropic` or `openai`
2. Basic conversation construction (system → user → assistant)
3. Tool call → tool result correlation via `tool_call_id`
4. Multiple tool calls in one assistant response
5. Valid tool argument JSON parsing
6. Malformed tool arguments (invalid JSON, non-dict) — graceful handling
7. `ProviderState` round-trip serialisation
8. `TOOL` role without `tool_call_id` → rejected
9. `USER` message with `tool_calls` → rejected
10. `ModelResponse.message` not `ASSISTANT` → rejected
11. `TokenUsage` negative values → rejected
12. `TokenUsage.total` derivation when absent
13. `FinishReason` values are vendor-neutral
14. `ProviderError` retryable semantics
15. Full `ModelRequest` → `ModelResponse` round-trip
16. `ToolDefinition`, `ModelRequest`, `ToolCall` validation constraints

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
  test_model_provider_contract.py   # 35 tests

docs/interview/
  01-model-provider-contract.md     # this document
```
