"""DeepSeek provider adapter.

Translates between canonical types and DeepSeek's OpenAI-compatible
Chat Completions API via ``openai.OpenAI``.

Key invariants
--------------

* **ProviderState replay**: When ``thinking_enabled=True`` and an assistant
  turn contains tool calls with reasoning content, the reasoning is captured
  in ``ProviderState``.  On the next request, that exact content is replayed
  in the assistant message's ``reasoning_content`` field.
* **No-tool thinking is NOT persisted**: A non-tool-call turn's reasoning is
  not needed for protocol correctness and is discarded.
* **Missing replay state → fail closed**: If the history contains a
  tool-call assistant turn that requires reasoning replay but has no valid
  ProviderState, the adapter raises ``ProviderError(INVALID_REQUEST)``
  before making any API call.
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from agent_runtime.model.contracts import (
    FinishReason,
    MessageRole,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ProviderState,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)
from agent_runtime.model.provider import (
    ModelProvider,
    ProviderError,
    ProviderErrorKind,
)


_REASONING_EFFORTS = ("high", "max")


def _encode_tool_definition(td: ToolDefinition) -> dict[str, Any]:
    """Encode a canonical ToolDefinition into OpenAI function-tool format."""
    return {
        "type": "function",
        "function": {
            "name": td.name,
            "description": td.description,
            "parameters": td.parameters,
        },
    }


def _tool_call_to_dict(tc: ToolCall) -> dict[str, Any]:
    """Convert a canonical ToolCall to an OpenAI tool_calls entry.

    Uses ``arguments`` dict (deterministic JSON serialization).
    Falls back to ``raw_arguments`` if arguments is None.
    Raises ProviderError(INVALID_REQUEST) if neither is available.
    """
    import json

    if tc.arguments is not None:
        args_str = json.dumps(tc.arguments, ensure_ascii=False, sort_keys=True)
    elif tc.raw_arguments is not None:
        args_str = tc.raw_arguments
    else:
        raise ProviderError(
            ProviderErrorKind.INVALID_REQUEST,
            "deepseek",
            f"ToolCall {tc.id} ({tc.name}) has no arguments or raw_arguments",
            retryable=False,
        )

    return {
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.name,
            "arguments": args_str,
        },
    }


def _encode_messages(
    messages: list[ModelMessage],
    thinking_enabled: bool,
) -> list[dict[str, Any]]:
    """Encode canonical messages into OpenAI-format message dicts.

    DeepSeek-specific rules:

    * SYSTEM messages: mapped to ``{"role": "system", "content": ...}``.
    * ASSISTANT with tool_calls: ``content`` field is always present (set
      to ``""`` if the canonical content is None).  If ``thinking_enabled``
      and a ProviderState with reasoning content exists, it is replayed
      as ``reasoning_content``.
    * ASSISTANT with tool_calls + thinking_enabled: MUST have valid
      ProviderState for replay (raises INVALID_REQUEST if missing).
    * TOOL: mapped directly.
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.role

        if role == MessageRole.SYSTEM:
            result.append({"role": "system", "content": msg.content or ""})
            continue

        if role == MessageRole.USER:
            result.append({"role": "user", "content": msg.content or ""})
            continue

        if role == MessageRole.TOOL:
            result.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content or "",
            })
            continue

        if role == MessageRole.ASSISTANT:
            entry: dict[str, Any] = {"role": "assistant"}

            if msg.has_tool_calls:
                # Tool-call assistant turn
                tool_calls = msg.tool_calls  # type: ignore[arg-type]
                entry["tool_calls"] = [_tool_call_to_dict(tc) for tc in tool_calls]
                # content MUST be present; use "" if None
                entry["content"] = msg.content if msg.content is not None else ""

                if thinking_enabled:
                    # Validate and replay reasoning_content from ProviderState
                    if msg.provider_state is None:
                        raise ProviderError(
                            ProviderErrorKind.INVALID_REQUEST,
                            "deepseek",
                            "Thinking is enabled but historical assistant tool-call "
                            "message is missing ProviderState (reasoning_content "
                            "required for protocol correctness)",
                            retryable=False,
                        )
                    if msg.provider_state.provider != "deepseek":
                        raise ProviderError(
                            ProviderErrorKind.INVALID_REQUEST,
                            "deepseek",
                            f"ProviderState.provider is '{msg.provider_state.provider}'"
                            f", expected 'deepseek'",
                            retryable=False,
                        )
                    reasoning = msg.provider_state.data.get("reasoning_content")
                    if reasoning is not None:
                        # Exact replay — no normalization, truncation, or modification
                        entry["reasoning_content"] = reasoning
            else:
                # Plain text assistant turn
                entry["content"] = msg.content if msg.content is not None else ""

            result.append(entry)
            continue

    return result


# ---------------------------------------------------------------------------
# Response decoding
# ---------------------------------------------------------------------------

_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "stop": FinishReason.STOP,
    "tool_calls": FinishReason.TOOL_CALLS,
    "length": FinishReason.LENGTH,
    "content_filter": FinishReason.CONTENT_FILTER,
}


def _decode_finish_reason(finish_reason: str | None) -> FinishReason:
    if finish_reason is None:
        return FinishReason.UNKNOWN
    return _FINISH_REASON_MAP.get(finish_reason, FinishReason.UNKNOWN)


def _decode_tool_calls(choice: Any) -> list[ToolCall]:
    """Extract canonical ToolCalls from an OpenAI choice object.

    Uses ``ToolCall.from_raw_json()`` — DeepSeek returns tool arguments
    as JSON strings.
    """
    tool_calls: list[ToolCall] = []
    raw_calls = getattr(choice.message, "tool_calls", None) or []

    for tc in raw_calls:
        tc_id = getattr(tc, "id", "")
        func = getattr(tc, "function", None) or {}
        name = getattr(func, "name", "")
        raw_args: str | None = getattr(func, "arguments", None)

        canonical = ToolCall.from_raw_json(tc_id, name, raw_args)
        tool_calls.append(canonical)

    return tool_calls


def _build_provider_state(
    choice: Any,
    has_tool_calls: bool,
) -> ProviderState | None:
    """Build ProviderState for tool-call turns with reasoning content.

    Only persists reasoning_content when the turn has tool_calls AND
    non-None reasoning — this is the replay-critical case.  No-tool
    thinking turns are NOT persisted (they don't need replay).
    """
    if not has_tool_calls:
        return None

    reasoning = getattr(choice.message, "reasoning_content", None)
    if reasoning is None:
        return None

    return ProviderState(
        provider="deepseek",
        data={"reasoning_content": reasoning},
    )


def _decode_usage(usage: Any) -> TokenUsage:
    """Map DeepSeek/OpenAI usage to canonical TokenUsage."""
    def _s(v: Any) -> int | None:
        return int(v) if v is not None else None

    input_tokens = _s(getattr(usage, "prompt_tokens", 0)) or 0
    output_tokens = _s(getattr(usage, "completion_tokens", 0)) or 0
    total_tokens = _s(getattr(usage, "total_tokens", None))
    cache_read = _s(getattr(usage, "prompt_cache_hit_tokens", None))
    cache_miss = _s(getattr(usage, "prompt_cache_miss_tokens", None))

    # reasoning tokens from completion_tokens_details
    completion_details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens: int | None = None
    if completion_details is not None:
        reasoning_tokens = _s(getattr(completion_details, "reasoning_tokens", None))

    # Stash non-canonicalisable fields
    provider_details: dict[str, Any] = {}
    for attr in dir(usage):
        if attr.startswith("_"):
            continue
        if attr in (
            "prompt_tokens", "completion_tokens", "total_tokens",
            "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
            "completion_tokens_details",
        ):
            continue
        try:
            val = getattr(usage, attr)
            if not callable(val) and val is not None:
                provider_details[attr] = val
        except Exception:
            pass

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cache_read,
        cache_creation_tokens=None,  # DeepSeek doesn't report cache creation separately
        cache_miss_tokens=cache_miss,
        provider_details=provider_details,
    )


# ---------------------------------------------------------------------------
# Error normalization
# ---------------------------------------------------------------------------

def _normalize_error(exc: Exception) -> ProviderError:
    """Convert an OpenAI SDK exception into a canonical ProviderError."""
    import openai as _openai

    status_code: int | None = getattr(exc, "status_code", None)
    body: dict[str, Any] | None = getattr(exc, "body", None)
    if body is not None and not isinstance(body, dict):
        body = None

    message = str(exc)
    provider_code: str | None = None

    if isinstance(exc, _openai.AuthenticationError):
        kind = ProviderErrorKind.AUTH
    elif isinstance(exc, _openai.RateLimitError):
        kind = ProviderErrorKind.RATE_LIMIT
    elif isinstance(exc, _openai.BadRequestError):
        kind = ProviderErrorKind.INVALID_REQUEST
    elif isinstance(exc, _openai.InternalServerError):
        kind = ProviderErrorKind.SERVER
    elif isinstance(exc, _openai.PermissionDeniedError):
        kind = ProviderErrorKind.AUTH
    elif isinstance(exc, _openai.UnprocessableEntityError):
        kind = ProviderErrorKind.INVALID_REQUEST
    elif isinstance(exc, _openai.APIStatusError):
        # Generic HTTP error — classify by status code
        if status_code is not None:
            normalized = _http_status_to_kind(status_code)
            if normalized is not None:
                kind = normalized
            elif 400 <= status_code < 500:
                kind = ProviderErrorKind.INVALID_REQUEST
            else:
                kind = ProviderErrorKind.SERVER
        else:
            kind = ProviderErrorKind.UNKNOWN
    else:
        kind = ProviderErrorKind.UNKNOWN

    if body:
        error_block = body.get("error", {})
        if isinstance(error_block, dict):
            provider_code = error_block.get("code", None) or error_block.get("type", None)
            inner_msg = error_block.get("message", None)
            if inner_msg:
                message = inner_msg

    return ProviderError(
        kind=kind,
        provider="deepseek",
        message=message,
        status_code=status_code,
        provider_code=provider_code,
    )


def _http_status_to_kind(status: int) -> ProviderErrorKind | None:
    """Classify a bare HTTP status code when no typed exception is available."""
    mapping: dict[int, ProviderErrorKind] = {
        400: ProviderErrorKind.INVALID_REQUEST,
        401: ProviderErrorKind.AUTH,
        402: ProviderErrorKind.BILLING,
        422: ProviderErrorKind.INVALID_REQUEST,
        429: ProviderErrorKind.RATE_LIMIT,
        500: ProviderErrorKind.SERVER,
        503: ProviderErrorKind.OVERLOADED,
    }
    return mapping.get(status)


# ---------------------------------------------------------------------------
# DeepSeekProvider
# ---------------------------------------------------------------------------

class DeepSeekProvider:
    """Provider adapter for DeepSeek's OpenAI-compatible Chat Completions API.

    Parameters
    ----------
    client:
        An ``openai.OpenAI`` instance.  Inject a mock for testing.
    thinking_enabled:
        Enable DeepSeek reasoning/thinking.  Default ``True``.
    reasoning_effort:
        Reasoning effort level: ``"high"`` or ``"max"``.
        Default ``"high"``.
    """

    name: str = "deepseek"

    def __init__(
        self,
        client: OpenAI | None = None,
        *,
        thinking_enabled: bool = True,
        reasoning_effort: str = "high",
    ) -> None:
        if reasoning_effort not in _REASONING_EFFORTS:
            raise ValueError(
                f"reasoning_effort must be one of {_REASONING_EFFORTS}, "
                f"got {reasoning_effort!r}"
            )
        self._client = client if client is not None else OpenAI()
        self.thinking_enabled = thinking_enabled
        self.reasoning_effort = reasoning_effort

    # -- ModelProvider protocol -----------------------------------------------

    def complete(self, request: ModelRequest) -> ModelResponse:
        # 1. Encode
        messages = _encode_messages(request.messages, self.thinking_enabled)
        tools: list[dict[str, Any]] | None = None
        if request.tools:
            tools = [_encode_tool_definition(t) for t in request.tools]

        # 2. Build kwargs with explicit thinking config
        create_kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_output_tokens,
        }
        if tools:
            create_kwargs["tools"] = tools

        # DeepSeek thinking configuration
        if self.thinking_enabled:
            create_kwargs["extra_body"] = {
                "thinking": {
                    "type": "enabled",
                },
                "reasoning_effort": self.reasoning_effort,
            }

        # 3. Vendor call
        try:
            response = self._client.chat.completions.create(**create_kwargs)
        except Exception as exc:
            raise _normalize_error(exc) from exc

        # 4. Decode
        choice = response.choices[0]
        has_tool_calls = bool(getattr(choice.message, "tool_calls", None))
        finish_reason = _decode_finish_reason(
            getattr(choice, "finish_reason", None)
        )
        tool_calls = _decode_tool_calls(choice) if has_tool_calls else []
        usage = _decode_usage(response.usage) if getattr(response, "usage", None) else TokenUsage()
        provider_state = _build_provider_state(choice, has_tool_calls)
        response_id: str | None = getattr(response, "id", None)

        content: str | None = getattr(choice.message, "content", None)

        assistant_msg = ModelMessage.assistant(
            content=content if content else None,
            tool_calls=tool_calls if tool_calls else None,
            provider_state=provider_state,
        )

        return ModelResponse(
            message=assistant_msg,
            finish_reason=finish_reason,
            usage=usage,
            provider="deepseek",
            response_id=response_id,
        )
