"""Anthropic provider adapter.

Translates between the canonical :class:`ModelRequest` / :class:`ModelResponse`
and the Anthropic Messages API via ``anthropic.Anthropic``.
"""

from __future__ import annotations

from typing import Any

from anthropic import Anthropic

from agent_runtime.model.contracts import (
    FinishReason,
    MessageRole,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)
from agent_runtime.model.provider import (
    ModelProvider,
    ProviderError,
    ProviderErrorKind,
)


def _extract_system(messages: list[ModelMessage]) -> str | None:
    """Extract leading SYSTEM messages into the top-level system parameter.

    Raises ProviderError(INVALID_REQUEST) if a SYSTEM message appears after
    any non-SYSTEM message.
    """
    system_parts: list[str] = []
    seen_non_system = False
    for msg in messages:
        if msg.role == MessageRole.SYSTEM:
            if seen_non_system:
                raise ProviderError(
                    ProviderErrorKind.INVALID_REQUEST,
                    "anthropic",
                    "SYSTEM message must appear before any conversation content",
                    retryable=False,
                )
            if msg.content:
                system_parts.append(msg.content)
        else:
            seen_non_system = True
    if not system_parts:
        return None
    return "\n\n".join(system_parts)


def _encode_tool_definition(td: ToolDefinition) -> dict[str, Any]:
    """Encode a canonical ToolDefinition as an Anthropic tool schema."""
    return {
        "name": td.name,
        "description": td.description,
        "input_schema": td.parameters,
    }


def _tool_calls_to_blocks(tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
    """Convert canonical ToolCalls to Anthropic tool_use content blocks.

    Raises ProviderError if any ToolCall has arguments=None
    (indicating a prior parse failure that must not be sent to the API).
    """
    blocks: list[dict[str, Any]] = []
    for tc in tool_calls:
        if tc.arguments is None:
            raise ProviderError(
                ProviderErrorKind.INVALID_REQUEST,
                "anthropic",
                f"ToolCall {tc.id} ({tc.name}) has no parseable arguments "
                f"(argument_error={tc.argument_error}) — refusing to send",
                retryable=False,
            )
        blocks.append({
            "type": "tool_use",
            "id": tc.id,
            "name": tc.name,
            "input": tc.arguments,
        })
    return blocks


def _encode_messages(
    messages: list[ModelMessage],
) -> list[dict[str, Any]]:
    """Encode non-SYSTEM canonical messages into Anthropic API message dicts.

    Consecutive TOOL messages are aggregated into a single user message
    with multiple tool_result blocks.
    """
    result: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg.role == MessageRole.SYSTEM:
            i += 1
            continue

        if msg.role == MessageRole.USER:
            result.append({"role": "user", "content": msg.content or ""})
            i += 1
            continue

        if msg.role == MessageRole.ASSISTANT:
            if msg.has_tool_calls:
                tool_blocks = _tool_calls_to_blocks(msg.tool_calls)  # type: ignore[arg-type]
                content: str | list[dict[str, Any]]
                if msg.content:
                    tool_blocks.insert(0, {"type": "text", "text": msg.content})
                    content = tool_blocks
                else:
                    content = tool_blocks
            else:
                content = msg.content or ""
            result.append({"role": "assistant", "content": content})
            i += 1
            continue

        if msg.role == MessageRole.TOOL:
            tool_result_blocks: list[dict[str, Any]] = []
            while i < len(messages) and messages[i].role == MessageRole.TOOL:
                tm = messages[i]
                # tool_call_id is guaranteed non-empty by ModelMessage.__post_init__
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tm.tool_call_id,
                    "content": tm.content or "",
                })
                i += 1
            result.append({"role": "user", "content": tool_result_blocks})
            continue

        i += 1

    return result


def _decode_content(
    content: list[Any],
) -> tuple[str | None, list[ToolCall]]:
    """Decode Anthropic response content blocks.

    Returns (text_content, tool_calls).  Text content is the concatenated
    text from all text blocks.  Tool calls are constructed via
    ``ToolCall.from_arguments()`` (no JSON round-trip).
    """
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in content:
        block_type = getattr(block, "type", None)

        if block_type == "text":
            text_parts.append(getattr(block, "text", ""))
        elif block_type == "tool_use":
            tc = ToolCall.from_arguments(
                call_id=getattr(block, "id", ""),
                name=getattr(block, "name", ""),
                arguments=getattr(block, "input", {}),
            )
            tool_calls.append(tc)
        # Unknown block types are silently ignored (non-replay-critical).

    text = "\n".join(text_parts) if text_parts else None
    return text, tool_calls


_STOP_REASON_MAP: dict[str, FinishReason] = {
    "end_turn": FinishReason.STOP,
    "tool_use": FinishReason.TOOL_CALLS,
    "max_tokens": FinishReason.LENGTH,
    "refusal": FinishReason.CONTENT_FILTER,
}


def _decode_finish_reason(stop_reason: str | None) -> FinishReason:
    """Map Anthropic stop_reason to canonical FinishReason."""
    if stop_reason is None:
        return FinishReason.UNKNOWN
    return _STOP_REASON_MAP.get(stop_reason, FinishReason.UNKNOWN)


def _decode_usage(usage: Any) -> TokenUsage:
    """Map Anthropic usage object to canonical TokenUsage."""
    provider_details: dict[str, Any] = {}

    def _safe_int(val: Any) -> int | None:
        if val is None:
            return None
        return int(val)

    # Known canonicalisable fields
    input_tokens = _safe_int(getattr(usage, "input_tokens", 0)) or 0
    output_tokens = _safe_int(getattr(usage, "output_tokens", 0)) or 0
    cache_read = _safe_int(getattr(usage, "cache_read_input_tokens", None))
    cache_creation = _safe_int(getattr(usage, "cache_creation_input_tokens", None))

    # Stash any additional fields in provider_details for traceability
    for attr in dir(usage):
        if attr.startswith("_") or attr in (
            "input_tokens", "output_tokens", "total_tokens",
            "cache_read_input_tokens", "cache_creation_input_tokens",
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
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_creation,
        cache_miss_tokens=None,  # Anthropic doesn't report cache miss separately
        provider_details=provider_details,
    )


def _normalize_error(exc: Exception) -> ProviderError:
    """Convert an Anthropic SDK exception into a canonical ProviderError."""
    import anthropic as _anthropic

    status_code: int | None = getattr(exc, "status_code", None)
    body: dict[str, Any] | None = None
    try:
        if hasattr(exc, "body") and exc.body is not None:
            raw_body = exc.body
            if isinstance(raw_body, dict):
                body = raw_body
    except Exception:
        pass

    message = str(exc)
    provider_code = None

    if isinstance(exc, _anthropic.AuthenticationError):
        kind = ProviderErrorKind.AUTH
    elif isinstance(exc, _anthropic.PermissionDeniedError):
        kind = ProviderErrorKind.AUTH
    elif isinstance(exc, _anthropic.RateLimitError):
        kind = ProviderErrorKind.RATE_LIMIT
    elif isinstance(exc, _anthropic.BadRequestError):
        kind = ProviderErrorKind.INVALID_REQUEST
    elif isinstance(exc, _anthropic.InternalServerError):
        kind = ProviderErrorKind.SERVER
    elif isinstance(exc, _anthropic.OverloadedError):
        kind = ProviderErrorKind.OVERLOADED
    elif isinstance(exc, _anthropic.APIStatusError):
        # Catch-all for unrecognized Anthropic status errors
        if status_code is not None:
            if 400 <= status_code < 500:
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
            provider_code = error_block.get("type", None)
            if not message or message == str(exc):
                message = error_block.get("message", message)

    return ProviderError(
        kind=kind,
        provider="anthropic",
        message=message,
        status_code=status_code,
        provider_code=provider_code,
    )


class AnthropicProvider:
    """Provider adapter for the Anthropic Messages API.

    Parameters
    ----------
    client:
        An ``anthropic.Anthropic`` instance.  Inject a mock for testing.
    """

    name: str = "anthropic"

    def __init__(self, client: Anthropic | None = None) -> None:
        self._client = client if client is not None else Anthropic()

    # -- ModelProvider protocol -----------------------------------------------

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Execute a synchronous completion through the Anthropic API."""
        # 1. Encode
        system = _extract_system(request.messages)
        messages = _encode_messages(request.messages)
        tools: list[dict[str, Any]] | None = None
        if request.tools:
            tools = [_encode_tool_definition(t) for t in request.tools]

        # 2. Vendor call
        try:
            response = self._client.messages.create(
                model=request.model,
                system=system or "",
                messages=messages,
                tools=tools,
                max_tokens=request.max_output_tokens,
            )
        except Exception as exc:
            raise _normalize_error(exc) from exc

        # 3. Decode
        text, tool_calls = _decode_content(response.content)
        finish_reason = _decode_finish_reason(response.stop_reason)
        usage = _decode_usage(response.usage)
        response_id: str | None = getattr(response, "id", None)

        assistant_msg = ModelMessage.assistant(
            content=text,
            tool_calls=tool_calls if tool_calls else None,
        )

        return ModelResponse(
            message=assistant_msg,
            finish_reason=finish_reason,
            usage=usage,
            provider="anthropic",
            response_id=response_id,
        )
