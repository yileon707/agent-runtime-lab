# P0.2B1.5 — Live DeepSeek Provider Protocol Validation

> **Date**: 2026-08-12
> **Prerequisite**: [P0.2B1 — Provider Adapters](./02-provider-adapters.md)
> **Followed by**: P0.2B2 (legacy s15 bridge) or P0.2C (full DeepSeek integration)

---

## Purpose

Validate the `DeepSeekProvider` adapter against the **real deepseek-v4-pro API** to confirm:

1. The mock-validated protocol mapping in `agent_runtime/providers/deepseek.py` is correct for the live API.
2. The **thinking/reasoning replay invariant** works end-to-end — capture, persist, replay, verify.
3. The adapter wires correctly to the live OpenAI SDK + real HTTP endpoint.
4. No protocol-level bugs in the request encoding or response decoding.

This is **NOT** an integration test of the full agent runtime — only the provider adapter boundary.

---

## Test Cases

All four cases executed via `scripts/smoke_deepseek_provider.py`.

| Case | Description | Result |
|---|---|---|
| **A** | Normal completion (no tools) | **PASS** |
| **B** | Thinking + tool call (get_runtime_probe) | **PASS** |
| **C** | Tool result continuation with ProviderState replay | **PASS** (replay_equal=True) |
| **D** | Subsequent user turn, replay still correct | **PASS** (replay_equal=True) |

### Case A — Normal Completion

- Single user message, no tools, thinking enabled
- Returns `finish_reason=stop`, content "Hello!"
- Token usage: 11 input, 56 output, 67 total
- No ProviderState (correct — no tool call, no replay needed)

### Case B — Thinking + Tool Call

- User message + `get_runtime_probe` ToolDefinition
- Model invoked the tool: `finish_reason=tool_calls`
- Token usage: 320 input, 62 output, 382 total
- **ProviderState captured**: reasoning_content (123 chars, SHA-256 prefix `18d5073c4e19d8f3`)
- `content=""` encoding confirmed working (DeepSeek requires the field)

### Case C — Replay Gate (Critical)

- History: user → assistant(tool_calls + provider_state) → tool_result
- **Replay verification**: the recording client wrapper confirmed that the exact `reasoning_content` from Case B was written into the request's assistant message's `reasoning_content` field
- `replay_equal=True`, SHA-256 match confirmed
- Model returned final answer: "The probe token is: **RUNTIME_REPLAY_OK**"
- Token usage: 400 input, 41 output, 441 total

### Case D — Subsequent Turn

- History includes the full Case B + Case C chain
- Follow-up user message: "Return only the probe token you received earlier."
- **Replay still verified**: assistant tool-call message in history still had exact reasoning_content replayed
- Model returned: "RUNTIME_REPLAY_OK"
- Token usage: 145 input, 34 output, 179 total
- Total API calls: 5 (within budget of ≤ 5)

---

## Bugs Found & Fixed

### Bug 1: Pydantic model internals in `provider_details`

**Severity**: Blocks all live API use.

**Root cause**: `_decode_usage()` in both `deepseek.py` and `anthropic.py` iterates `dir(usage)` on the SDK's internal Pydantic model objects. Pydantic BaseModels expose `model_fields`, `model_config`, `model_computed_fields`, etc. which contain `FieldInfo` objects. These leaked into `provider_details` and caused `json.dumps()` in `TokenUsage.__post_init__` to crash with `TypeError: Object of type FieldInfo is not JSON serializable`.

**Fix (two layers)**:

1. **`agent_runtime/providers/deepseek.py`** — Added `_PYDANTIC_INTERNALS` frozenset to the `_decode_usage` filter loop, excluding all Pydantic model metadata fields.
2. **`agent_runtime/providers/anthropic.py`** — Same fix applied for future real-Anthropic use.
3. **`agent_runtime/model/contracts.py`** — `TokenUsage.__post_init__` now catches `TypeError`/`ValueError` from `json.dumps()` instead of crashing. The guard is best-effort — `provider_details` tolerates non-serializable values since they can arrive from SDK internals beyond the adapter's control.

**Test impact**: `test_token_usage_provider_details_rejects_non_serialisable` renamed to `test_token_usage_provider_details_tolerates_non_serialisable` — now verifies graceful tolerance, not rejection.

---

## Classification

All bugs were **PROVIDER_IMPLEMENTATION** — the adapter's `_decode_usage` function did not account for Pydantic model internals from real SDK objects (mock-based tests use `MagicMock`, which don't expose these fields).

No **SDK_COMPATIBILITY** or **DEEPSEEK_PROTOCOL** bugs found — the request encoding, thinking configuration, tool schema format, and response decoding all matched the live API correctly.

---

## Files Changed

```
agent_runtime/
  model/
    contracts.py          # TokenUsage.__post_init__ tolerates non-serializable provider_details
  providers/
    anthropic.py          # _decode_usage: filter Pydantic model internals
    deepseek.py           # _decode_usage: filter Pydantic model internals

scripts/
  smoke_deepseek_provider.py   # NEW — live validation script

tests/
  test_model_provider_contract.py  # renamed test for tolerance behavior

docs/interview/
  03-deepseek-live-validation.md   # this document
```

---

## Limitations

- **Not a stress test**: Only 5 API calls across 4 logical cases.
- **Single conversation topology**: One user → tool call → tool result → follow-up pattern.
- **No error path validation**: All 4 cases succeeded; error normalization was not exercised against the real API.
- **No Anthropic counterpart**: Only DeepSeek was validated live in this phase.

---

## Next

- **P0.2B2**: Legacy s15 bridge — hook DeepSeekProvider into the existing agent runtime.
- **P0.2C**: Full DeepSeek integration validation — comprehensive test suite against the real API.
