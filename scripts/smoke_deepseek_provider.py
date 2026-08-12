#!/usr/bin/env python3
"""P0.2B1.5 — Live DeepSeek Provider Protocol Validation.

Smoke-tests the DeepSeekProvider against the real deepseek-v4-pro API.

Run:  python scripts/smoke_deepseek_provider.py

Requires:
    DEEPSEEK_API_KEY  in .env (or exported environment var)
    DEEPSEEK_BASE_URL (optional, default https://api.deepseek.com)
    DEEPSEEK_MODEL    (optional, default deepseek-v4-pro)

Rules:
    - Never print reasoning_content.
    - Never print API keys or full auth headers.
    - Never log anything that could be committed as a secret.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

# Ensure we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_runtime.model.contracts import (
    FinishReason,
    MessageRole,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolDefinition,
)
from agent_runtime.model.provider import ProviderError
from agent_runtime.providers.deepseek import DeepSeekProvider

# ── secrets ──────────────────────────────────────────────────────────────────

load_dotenv(override=True)

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    print("FATAL: DEEPSEEK_API_KEY not set in environment or .env")
    sys.exit(1)

BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")

# ── recording client wrapper ─────────────────────────────────────────────────

class RecordingClient:
    """Thin wrapper that records the **sanitized** kwargs of the last
    ``chat.completions.create(...)`` call, then delegates to the real client.

    Only ``messages`` and ``model`` are recorded.  Headers and API keys
    are NEVER stored.
    """

    def __init__(self, real_client: OpenAI) -> None:
        self._real = real_client
        self.last_kwargs: dict[str, Any] = {}

    @property
    def chat(self) -> RecordingClient:
        return self

    @property
    def completions(self) -> RecordingClient:
        return self

    def create(self, **kwargs: Any) -> Any:
        # Record sanitized kwargs for replay verification
        self.last_kwargs = {
            "model": kwargs.get("model"),
            "messages": kwargs.get("messages"),
        }
        return self._real.chat.completions.create(**kwargs)


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_hash(text: str) -> str:
    """SHA-256 hash of a string — safe to print."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _check_provider_state(ps) -> dict[str, Any]:
    """Sanitized inspection of a ProviderState."""
    if ps is None:
        return {"present": False}
    reasoning = ps.data.get("reasoning_content")
    return {
        "present": True,
        "provider": ps.provider,
        "reasoning_length": len(reasoning) if isinstance(reasoning, str) else 0,
        "reasoning_hash": _safe_hash(reasoning) if isinstance(reasoning, str) else "N/A",
    }


def _record_case(label: str, response: ModelResponse | None,
                 error: ProviderError | Exception | None,
                 extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a sanitized case result dict."""
    result: dict[str, Any] = {"case": label, "timestamp": datetime.now(timezone.utc).isoformat()}

    if error is not None:
        result["status"] = "FAIL"
        if isinstance(error, ProviderError):
            result["error_kind"] = error.kind.value
            result["error_status_code"] = error.status_code
            result["error_provider_code"] = error.provider_code
            result["error_message"] = error.message_text[:200]
        else:
            result["error_type"] = type(error).__name__
            result["error_message"] = str(error)[:200]
        return result

    if response is None:
        result["status"] = "INCONCLUSIVE"
        return result

    result["status"] = "PASS"
    result["finish_reason"] = response.finish_reason.value
    result["provider"] = response.provider
    result["has_tool_calls"] = response.message.has_tool_calls
    if response.message.has_tool_calls:
        result["tool_call_count"] = len(response.message.tool_calls)  # type: ignore[arg-type]
    result["content_preview"] = (response.message.content or "")[:80]
    result["input_tokens"] = response.usage.input_tokens
    result["output_tokens"] = response.usage.output_tokens
    result["total_tokens"] = response.usage.total

    ps_info = _check_provider_state(response.message.provider_state)
    result["provider_state"] = ps_info

    if extra:
        result.update(extra)

    return result


# ── build provider ───────────────────────────────────────────────────────────

real_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
recording_client = RecordingClient(real_client)

provider = DeepSeekProvider(
    client=recording_client,  # type: ignore[arg-type]
    thinking_enabled=True,
    reasoning_effort="high",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CASE A — Normal Completion (no tools)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("CASE A — Normal Completion")
print("=" * 60)

request_a = ModelRequest(
    model=MODEL,
    messages=[ModelMessage.user("Say hello in exactly one sentence.")],
    max_output_tokens=200,
)

error_a = None
response_a = None
try:
    response_a = provider.complete(request_a)
except ProviderError as exc:
    error_a = exc
except Exception as exc:
    error_a = exc

result_a = _record_case("A — Normal Completion", response_a, error_a)
print(json.dumps(result_a, indent=2, ensure_ascii=False))

# ═══════════════════════════════════════════════════════════════════════════════
# CASE B — Thinking + Tool Call
# ═══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 60)
print("CASE B — Thinking + Tool Call")
print("=" * 60)

probe_tool = ToolDefinition(
    name="get_runtime_probe",
    description=(
        "Call this tool to obtain a probe token that is required to complete "
        "the task. You MUST call this tool first before giving your answer."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
)

request_b = ModelRequest(
    model=MODEL,
    messages=[
        ModelMessage.user(
            "You MUST first call the get_runtime_probe tool to receive a probe "
            "token. After calling the tool and receiving the token, tell me "
            "what the probe token is."
        ),
    ],
    tools=[probe_tool],
    max_output_tokens=400,
)

error_b = None
response_b = None
try:
    response_b = provider.complete(request_b)
except ProviderError as exc:
    error_b = exc
except Exception as exc:
    error_b = exc

extra_b: dict[str, Any] = {}
if response_b is not None and response_b.finish_reason != FinishReason.TOOL_CALLS:
    extra_b["warning"] = "Model did not trigger tool call — MODEL_BEHAVIOR_INCONCLUSIVE"

result_b = _record_case("B — Thinking + Tool Call", response_b, error_b, extra_b)
print(json.dumps(result_b, indent=2, ensure_ascii=False))

if response_b is None or not response_b.message.has_tool_calls:
    print()
    print("CASE B did not produce a tool call — retrying with clearer prompt...")

    request_b2 = ModelRequest(
        model=MODEL,
        messages=[
            ModelMessage.user(
                "IMPORTANT: Before responding, you MUST invoke the "
                "get_runtime_probe tool (no arguments needed). Then tell me "
                "the probe value you received."
            ),
        ],
        tools=[probe_tool],
        max_output_tokens=400,
    )
    try:
        response_b = provider.complete(request_b2)
        error_b = None
    except ProviderError as exc:
        error_b = exc; response_b = None
    except Exception as exc:
        error_b = exc; response_b = None

    extra_b2: dict[str, Any] = {}
    if response_b is not None and response_b.finish_reason != FinishReason.TOOL_CALLS:
        extra_b2["warning"] = "Retry also did not trigger tool call"

    result_b = _record_case("B(retry) — Thinking + Tool Call", response_b, error_b, extra_b2)
    print(json.dumps(result_b, indent=2, ensure_ascii=False))

# ═══════════════════════════════════════════════════════════════════════════════
# CASE C — Tool Result Continuation (REPLAY GATE)
# ═══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 60)
print("CASE C — Tool Result Continuation (Replay Gate)")
print("=" * 60)

response_c_ref = None
if response_b is None or not response_b.message.has_tool_calls:
    print("SKIP: Case B did not produce a tool call — cannot continue.")
    result_c = {"case": "C — Replay Continuation", "status": "SKIP",
                "reason": "Case B did not produce tool call"}
else:
    # Sanity check: ProviderState was captured
    ps_b = response_b.message.provider_state
    captured_reasoning: str | None = None
    if ps_b is not None:
        captured_reasoning = ps_b.data.get("reasoning_content")

    print(f"ProviderState captured: {ps_b is not None}")
    if captured_reasoning is not None:
        print(f"Reasoning length: {len(captured_reasoning)}")
        print(f"Reasoning SHA-256: {_safe_hash(captured_reasoning)}")

    # Build the tool result
    tool_call_id = response_b.message.tool_calls[0].id  # type: ignore[index]

    history_c = [
        ModelMessage.user(
            "You MUST first call the get_runtime_probe tool to receive a probe "
            "token. After calling the tool and receiving the token, tell me "
            "what the probe token is."
        ),
        response_b.message,  # assistant with tool_calls + provider_state
        ModelMessage.tool(tool_call_id=tool_call_id, content="RUNTIME_REPLAY_OK"),
    ]

    request_c = ModelRequest(
        model=MODEL,
        messages=history_c,
        tools=[probe_tool],
        max_output_tokens=400,
    )

    error_c = None
    response_c = None
    try:
        response_c = provider.complete(request_c)
    except ProviderError as exc:
        error_c = exc
    except Exception as exc:
        error_c = exc

    # ── Replay verification ────────────────────────────────────────────
    replay_equal = False
    replay_details: dict[str, Any] = {}

    if error_c is None and response_c is not None:
        # Check the recorded kwargs from the LAST create call
        recorded_messages = recording_client.last_kwargs.get("messages", [])
        if recorded_messages:
            # Find the assistant message in the recorded request
            for msg in recorded_messages:
                if msg.get("role") == "assistant":
                    replayed_reasoning = msg.get("reasoning_content")
                    if replayed_reasoning is not None and captured_reasoning is not None:
                        replay_equal = (replayed_reasoning == captured_reasoning)
                        replay_details["replayed_length"] = len(replayed_reasoning)
                        replay_details["replayed_hash"] = _safe_hash(replayed_reasoning)
                    else:
                        replay_details["replayed_reasoning_present"] = replayed_reasoning is not None
                        replay_details["captured_reasoning_present"] = captured_reasoning is not None
                    break

    print(f"  replay_equal = {replay_equal}")

    extra_c = {"replay_equal": replay_equal, "replay_details": replay_details}
    result_c = _record_case("C — Replay Continuation", response_c, error_c, extra_c)
    print(json.dumps(result_c, indent=2, ensure_ascii=False))

    # Save for Case D
    response_c_ref = response_c

# ═══════════════════════════════════════════════════════════════════════════════
# CASE D — Subsequent User Turn
# ═══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 60)
print("CASE D — Subsequent User Turn")
print("=" * 60)

if response_c_ref is None:
    print("SKIP: Case C did not succeed — cannot continue.")
    result_d = {"case": "D — Subsequent User Turn", "status": "SKIP",
                "reason": "Case C did not succeed"}
else:
    history_d = [
        ModelMessage.user(
            "You MUST first call the get_runtime_probe tool to receive a probe "
            "token. After calling the tool and receiving the token, tell me "
            "what the probe token is."
        ),
        response_b.message,  # assistant with tool_calls + provider_state
        ModelMessage.tool(tool_call_id=tool_call_id, content="RUNTIME_REPLAY_OK"),
        response_c_ref.message,  # final answer from Case C
        ModelMessage.user("Return only the probe token you received earlier."),
    ]

    request_d = ModelRequest(
        model=MODEL,
        messages=history_d,
        max_output_tokens=200,
    )

    error_d = None
    response_d = None
    try:
        response_d = provider.complete(request_d)
    except ProviderError as exc:
        error_d = exc
    except Exception as exc:
        error_d = exc

    # Verify replay still works in this subsequent turn
    replay_d_equal = False
    if error_d is None:
        recorded_messages = recording_client.last_kwargs.get("messages", [])
        for msg in recorded_messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                replayed = msg.get("reasoning_content")
                if replayed is not None and captured_reasoning is not None:
                    replay_d_equal = (replayed == captured_reasoning)
                break

    print(f"  replay_equal (case D) = {replay_d_equal}")

    extra_d = {"replay_equal": replay_d_equal}
    result_d = _record_case("D — Subsequent User Turn", response_d, error_d, extra_d)
    print(json.dumps(result_d, indent=2, ensure_ascii=False))

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)

all_results = [result_a, result_b]
if response_b is not None and response_b.message.has_tool_calls:
    all_results.append(result_c)
    if response_c_ref is not None:
        all_results.append(result_d)

for r in all_results:
    status = r.get("status", "UNKNOWN")
    case = r.get("case", "?")
    fr = r.get("finish_reason", "N/A")
    replay = r.get("replay_equal", "N/A")
    err = r.get("error_kind", "")
    print(f"  [{status}] {case}  finish={fr}  replay_equal={replay}  {err}")

# Environment summary (safe to print)
print()
print("Environment:")
print(f"  Python:     {sys.version.split()[0]}")
import openai
print(f"  openai SDK: {openai.__version__}")
print(f"  model:      {MODEL}")
print(f"  base_url:   {BASE_URL}")
print(f"  thinking:   enabled, reasoning_effort=high")
print(f"  timestamp:  {datetime.now(timezone.utc).isoformat()}")
