# P0.2B1 — Provider Adapters

> **Date**: 2026-08-12
> **Prerequisite**: [P0.2A.1 — Canonical Contract Hardening](./01-model-provider-contract.md)
> **Followed by**: P0.2B2 (legacy s15 bridge) or P0.2C (real DeepSeek validation)

---

## Problem

The canonical model contract (`agent_runtime/model/`) defines a provider-agnostic vocabulary, but without concrete adapters it is abstract. To prove that the contract is sufficient, we need **real** providers that:

1. Encode canonical `ModelRequest` into vendor-specific wire formats.
2. Call the vendor API (or a mock, for testing).
3. Decode vendor responses into canonical `ModelResponse`.
4. Normalize vendor errors into `ProviderError`.

This phase delivers two production-ready adapters: **AnthropicProvider** and **DeepSeekProvider**.

---

## Decision

Implement two `ModelProvider` adapters with **no shared abstraction layer**:

```
agent_runtime/
  providers/
    __init__.py
    anthropic.py       # Anthropic Messages API → canonical
    deepseek.py         # DeepSeek (OpenAI-compatible) → canonical
```

Each adapter is self-contained. There is no `BaseProvider`, no generic `OpenAICompatibleProvider`, and no shared encode/decode infrastructure. This keeps the dependency graph explicit and the code readable.

---

## Boundary

```
┌─────────────────────────────────────────────────────┐
│  Agent Runtime (future)                             │
│                                                     │
│         calls ModelProvider.complete()              │
├─────────────────────────────────────────────────────┤
│  Canonical Model Contract (model/)                  │
│  ModelProvider Protocol (provider.py)               │
├──────────────────┬──────────────────────────────────┤
│  AnthropicProvider│  DeepSeekProvider               │
│  anthropic.py     │  deepseek.py                    │
├──────────────────┼──────────────────────────────────┤
│  anthropic SDK    │  openai SDK                     │
│  Messages API     │  POST /chat/completions         │
└──────────────────┴──────────────────────────────────┘
```

---

## Anthropic Protocol Mapping

### Request Encoding

| Canonical | Anthropic |
|---|---|
| Leading `SYSTEM` messages | Top-level `system` parameter (joined with `\n\n`) |
| Late `SYSTEM` after non-system | `ProviderError(INVALID_REQUEST)` — fail closed |
| `USER` | `{"role": "user", "content": text}` |
| `ASSISTANT` (text) | `{"role": "assistant", "content": text}` |
| `ASSISTANT` (tool_calls) | `{"role": "assistant", "content": [{"type": "text", "text": ...}, {"type": "tool_use", ...}]}` |
| `TOOL` (consecutive) | Aggregated into `{"role": "user", "content": [{"type": "tool_result", "tool_use_id": ..., "content": ...}, ...]}` |
| `ToolDefinition` | `{"name": ..., "description": ..., "input_schema": parameters}` |

**Tool call safety**: If a `ToolCall.arguments` is `None` (prior parse failure), the adapter raises `ProviderError(INVALID_REQUEST)` rather than sending garbage to the API.

### Response Decoding

| Anthropic | Canonical |
|---|---|
| `content[].type == "text"` | `ModelMessage.content` |
| `content[].type == "tool_use"` | `ToolCall.from_arguments(id, name, input)` — **no JSON round-trip** |
| `stop_reason == "end_turn"` | `FinishReason.STOP` |
| `stop_reason == "tool_use"` | `FinishReason.TOOL_CALLS` |
| `stop_reason == "max_tokens"` | `FinishReason.LENGTH` |
| `stop_reason == "refusal"` | `FinishReason.CONTENT_FILTER` |
| Unknown stop_reason | `FinishReason.UNKNOWN` |
| `usage.input_tokens` | `TokenUsage.input_tokens` |
| `usage.output_tokens` | `TokenUsage.output_tokens` |
| `usage.cache_read_input_tokens` | `TokenUsage.cache_read_tokens` |
| `usage.cache_creation_input_tokens` | `TokenUsage.cache_creation_tokens` |
| Non-canonicalisable usage fields | `TokenUsage.provider_details` |

---

## DeepSeek Protocol Mapping

### Request Encoding

| Canonical | DeepSeek (OpenAI format) |
|---|---|
| `SYSTEM` | `{"role": "system", "content": ...}` |
| `USER` | `{"role": "user", "content": ...}` |
| `ASSISTANT` (text) | `{"role": "assistant", "content": text}` |
| `ASSISTANT` (tool_calls) | `{"role": "assistant", "content": "" or text, "tool_calls": [...]}` |
| `TOOL` | `{"role": "tool", "tool_call_id": ..., "content": ...}` |
| `ToolDefinition` | `{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}` |
| `ToolCall.arguments` (dict) | `json.dumps(args, sort_keys=True, ensure_ascii=False)` → `function.arguments` |
| `ToolCall.arguments` (None, fallback) | Use `raw_arguments` as-is |

**Content preservation**: When an assistant has tool_calls and `content=None`, the adapter encodes `content=""` — DeepSeek requires the field to be present.

### Thinking / Reasoning

DeepSeek supports extended thinking. The adapter explicitly passes thinking configuration:

```python
extra_body={
    "thinking": {"type": "enabled"},
    "reasoning_effort": "high" | "max",
}
```

Thinking is **enabled by default** (`thinking_enabled=True`) with `reasoning_effort="high"`.

---

## DeepSeek ProviderState — Replay Invariant

This is the **highest-priority correctness invariant** for the DeepSeek adapter.

### Capture

When a DeepSeek response has **both**:
1. Tool calls (`choice.message.tool_calls` is non-empty)
2. Reasoning content (`choice.message.reasoning_content` is not None)

The reasoning content is captured verbatim:

```python
ProviderState(
    provider="deepseek",
    data={"reasoning_content": exact_original_value}
)
```

This is attached to the `ModelMessage.assistant(provider_state=...)`.

### No-Tool Thinking

Ordinary text responses with reasoning but **no tool calls** do NOT persist `ProviderState`. Their reasoning is not replay-critical — it was consumed in producing the final answer.

### Replay

On the **next request**, when encoding an assistant message that:
1. Has tool calls
2. Has `thinking_enabled=True`

The adapter:

1. **Validates** that `provider_state` is present and `provider_state.provider == "deepseek"`.
2. **Replays** the exact `reasoning_content` from `provider_state.data["reasoning_content"]` into the encoded message's `reasoning_content` field.

**No normalization, truncation, summarization, or whitespace modification is allowed.**

### Fail-Closed

If thinking is enabled and a historical assistant tool-call message is **missing** `ProviderState`:

```python
raise ProviderError(
    ProviderErrorKind.INVALID_REQUEST,
    provider="deepseek",
    retryable=False
)
```

This prevents sending a request that the adapter knows violates the DeepSeek protocol. **The check happens before any API call.**

### Why ProviderState (Protocol) ≠ Semantic State

| | Semantic State | Protocol State |
|---|---|---|
| **What it carries** | Conversation meaning | API replay data |
| **Mutability** | Runtime can modify | Must pass through unchanged |
| **Examples** | `ModelMessage.content`, `ToolCall.arguments` | `reasoning_content`, session IDs |
| **Who interprets it** | Agent Runtime | Provider adapter only |
| **Storage in contract** | `ModelMessage.content`, `tool_calls`, etc. | `ProviderState.data` (opaque) |

---

### Response Decoding

| DeepSeek | Canonical |
|---|---|
| `choices[0].message.content` | `ModelMessage.content` |
| `choices[0].message.tool_calls` | `ToolCall.from_raw_json(id, name, function.arguments)` |
| `finish_reason == "stop"` | `FinishReason.STOP` |
| `finish_reason == "tool_calls"` | `FinishReason.TOOL_CALLS` |
| `finish_reason == "length"` | `FinishReason.LENGTH` |
| `finish_reason == "content_filter"` | `FinishReason.CONTENT_FILTER` |
| Unknown finish_reason | `FinishReason.UNKNOWN` |
| `usage.prompt_tokens` | `TokenUsage.input_tokens` |
| `usage.completion_tokens` | `TokenUsage.output_tokens` |
| `usage.total_tokens` | `TokenUsage.total_tokens` |
| `usage.prompt_cache_hit_tokens` | `TokenUsage.cache_read_tokens` |
| `usage.prompt_cache_miss_tokens` | `TokenUsage.cache_miss_tokens` |
| `usage.completion_tokens_details.reasoning_tokens` | `TokenUsage.reasoning_tokens` |
| `cache_creation_tokens` | `None` (DeepSeek doesn't report separately) |
| Non-canonicalisable fields | `TokenUsage.provider_details` |

---

## Error Boundary

Provider adapters **normalize** vendor errors into `ProviderError`. They do NOT:
- Sleep
- Retry
- Back off
- Fall back to another model

These are Runtime policy decisions. The adapter's job is translation, not recovery.

### Anthropic Error Mapping

| Anthropic Exception | ProviderErrorKind | Retryable |
|---|---|---|
| `AuthenticationError` | `AUTH` | No |
| `PermissionDeniedError` | `AUTH` | No |
| `RateLimitError` | `RATE_LIMIT` | Yes |
| `BadRequestError` | `INVALID_REQUEST` | No |
| `InternalServerError` | `SERVER` | Yes |
| `OverloadedError` | `OVERLOADED` | Yes |
| Other `APIStatusError` (4xx) | `INVALID_REQUEST` | No |
| Other `APIStatusError` (5xx) | `SERVER` | Yes |
| Unknown/other | `UNKNOWN` | Default |

### DeepSeek Error Mapping

| HTTP / Exception | ProviderErrorKind | Retryable |
|---|---|---|
| 400 `BadRequestError` | `INVALID_REQUEST` | No |
| 401 `AuthenticationError` | `AUTH` | No |
| 402 `APIStatusError` | `BILLING` | No |
| 422 `UnprocessableEntityError` | `INVALID_REQUEST` | No |
| 429 `RateLimitError` | `RATE_LIMIT` | Yes |
| 500 `InternalServerError` | `SERVER` | Yes |
| 503 `APIStatusError` | `OVERLOADED` | Yes |
| Unknown | `UNKNOWN` | Default |

---

## Non-Goals (This Phase)

- **No Legacy s15 Bridge**: Adapters exist independently of the existing runtime.
- **No Real API Calls**: All tests use fully mocked SDK clients.
- **No Streaming**: Synchronous `complete()` only.
- **No Async**: Sync-only, matching the current protocol.
- **No Retry/Backoff/Fallback**: Runtime policy, not provider concern.
- **No Generic OpenAICompatibleProvider**: Deliberate choice to keep adapters explicit.
- **No Provider Routing**: Runtime selects which provider to use.
- **No Tracing/Eval Framework**: Future phases.
- **No Context Compact modifications**: Unchanged.
- **DeepSeek is NOT proven runnable yet**: Mock tests only. Real validation is P0.2C.

---

## Test Coverage

| Test File | Tests | Type |
|---|---|---|
| `test_anthropic_provider.py` | 23 | Mock-based unit tests |
| `test_deepseek_provider.py` | 37 | Mock-based unit tests |
| `test_provider_cross_contract.py` | 7 | Cross-provider contract tests |
| `test_model_provider_contract.py` | 54 | Canonical contract tests (prior phase) |

### Anthropic Tests Cover

1. System encoding (single, multiple, late-rejection)
2. User text encoding
3. Tool definition: `parameters → input_schema`
4. Assistant text decode
5. Single tool_use decode
6. Multiple tool_use decode
7. `from_arguments()` (no JSON round-trip)
8. Tool result encoding
9. Multiple consecutive TOOL → one user message
10. Finish reason normalization (4 values + unknown + None)
11. Usage normalization
12. Invalid tool call arguments rejected before API
13. Provider error normalization (auth, rate limit, server)
14. `complete()` returns canonical `ModelResponse`
15. `complete()` with tools round-trip

### DeepSeek Tests Cover

1. System/user encoding
2. Tool schema: `function.parameters` format
3. Ordinary assistant decode
4. Single/multiple tool call decode
5. Valid JSON arguments parse
6. Malformed JSON preserved without crash
7. Tool message encoding
8. Finish reason normalization (4 values + unknown + None)
9. Usage: cache hit, miss, reasoning tokens
10. No-tool thinking NOT exposed as ProviderState
11. Thinking + tool call: reasoning captured exactly
12. Next request replays exact reasoning_content
13. Replay preserves Unicode/newlines/whitespace
14. Missing ProviderState → fail before API
15. ProviderState.provider mismatch → fail before network
16. `content=None` encodes as `""` for tool-call turns
17. Error normalization (400, 401, 402, 422, 429, 500, 503, unknown)
18. No internal retry
19. `complete()` returns canonical ModelResponse
20. Thinking config passed to API (enabled/disabled)
21. Invalid reasoning_effort rejected
22. Deterministic JSON serialization
23. Tool call fallback to raw_arguments
24. No arguments or raw → ProviderError

### Cross-Provider Tests

1. Same `ToolDefinition` → `input_schema` (Anthropic) vs `function.parameters` (DeepSeek)
2. Same `ToolCall` → structured dict (Anthropic) vs JSON string (DeepSeek)
3. Different vendor stop reasons → same `FinishReason.TOOL_CALLS`
4. Both factories produce equivalent canonical `ToolCall`
5. Same `ModelRequest` accepted by both providers

---

## Files

```
agent_runtime/
  providers/
    __init__.py                   # package marker
    anthropic.py                  # AnthropicProvider
    deepseek.py                   # DeepSeekProvider

tests/
  test_anthropic_provider.py      # 23 tests
  test_deepseek_provider.py       # 37 tests
  test_provider_cross_contract.py # 7 tests

docs/interview/
  02-provider-adapters.md         # this document
```
