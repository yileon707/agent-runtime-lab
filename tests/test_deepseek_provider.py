"""Tests for DeepSeekProvider.

All tests use a fully mocked OpenAI client — no real API calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

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
    ProviderError,
    ProviderErrorKind,
)
from agent_runtime.providers.deepseek import (
    DeepSeekProvider,
    _encode_messages,
    _normalize_error,
    _tool_call_to_dict,
)


# ============================================================================
# Helper factories
# ============================================================================

def _make_openai_response(
    *,
    content: str | None = "Hello.",
    tool_calls: list | None = None,
    finish_reason: str = "stop",
    reasoning_content: str | None = None,
    usage_kwargs: dict | None = None,
    response_id: str = "resp-1",
) -> MagicMock:
    """Build a mock OpenAI chat completion response."""
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message.content = content
    choice.message.tool_calls = tool_calls or []
    choice.message.reasoning_content = reasoning_content

    choices = MagicMock()
    choices.__getitem__ = lambda self, idx: choice
    choices.__len__ = lambda self: 1

    response = MagicMock()
    response.id = response_id
    response.choices = choices

    if usage_kwargs is not None:
        usage = MagicMock()
        for k, v in usage_kwargs.items():
            setattr(usage, k, v)
        response.usage = usage
    else:
        response.usage = None

    return response


def _make_tool_call_openai(idx: str, name: str, args: str) -> MagicMock:
    """Build a mock OpenAI tool call object."""
    func = MagicMock()
    func.name = name
    func.arguments = args

    tc = MagicMock()
    tc.id = idx
    tc.function = func
    tc.type = "function"
    return tc


# ============================================================================
# 1 — System/user encoding
# ============================================================================

def test_system_user_encoding() -> None:
    messages = [
        ModelMessage.system("You are helpful."),
        ModelMessage.user("Hello."),
    ]
    encoded = _encode_messages(messages, thinking_enabled=False)
    assert encoded[0]["role"] == "system"
    assert encoded[0]["content"] == "You are helpful."
    assert encoded[1]["role"] == "user"
    assert encoded[1]["content"] == "Hello."


# ============================================================================
# 2 — Tool schema encoding
# ============================================================================

def test_tool_schema_encoding() -> None:
    from agent_runtime.providers.deepseek import _encode_tool_definition

    td = ToolDefinition(
        name="bash", description="Run command",
        parameters={"type": "object", "properties": {"cmd": {"type": "string"}}},
    )
    result = _encode_tool_definition(td)
    assert result["type"] == "function"
    assert result["function"]["name"] == "bash"
    assert result["function"]["description"] == "Run command"
    assert result["function"]["parameters"] == td.parameters
    assert "input_schema" not in result["function"]


# ============================================================================
# 3 — Ordinary assistant response decode
# ============================================================================

def test_ordinary_assistant_decode() -> None:
    resp = _make_openai_response(content="Hi there!", finish_reason="stop")
    provider = DeepSeekProvider(client=MagicMock(), thinking_enabled=False)
    mock_client = provider._client
    mock_client.chat.completions.create.return_value = resp

    request = ModelRequest(
        model="deepseek-v4-pro",
        messages=[ModelMessage.user("Hi.")],
    )
    result = provider.complete(request)
    assert result.message.content == "Hi there!"
    assert result.finish_reason == FinishReason.STOP
    assert not result.message.has_tool_calls


# ============================================================================
# 4 — Single tool call decode
# ============================================================================

def test_single_tool_call_decode() -> None:
    tool = _make_tool_call_openai("call_1", "bash", '{"command": "ls"}')
    resp = _make_openai_response(
        content=None, tool_calls=[tool], finish_reason="tool_calls",
    )
    provider = DeepSeekProvider(client=MagicMock(), thinking_enabled=False)
    provider._client.chat.completions.create.return_value = resp

    request = ModelRequest(
        model="deepseek-v4-pro",
        messages=[ModelMessage.user("List files.")],
    )
    result = provider.complete(request)
    assert result.message.has_tool_calls
    assert len(result.message.tool_calls) == 1  # type: ignore[arg-type]
    assert result.message.tool_calls[0].id == "call_1"  # type: ignore[index]
    assert result.message.tool_calls[0].name == "bash"  # type: ignore[index]
    assert result.message.tool_calls[0].arguments == {"command": "ls"}  # type: ignore[index]


# ============================================================================
# 5 — Multiple tool calls decode
# ============================================================================

def test_multiple_tool_calls_decode() -> None:
    tc1 = _make_tool_call_openai("c1", "bash", '{"cmd": "ls"}')
    tc2 = _make_tool_call_openai("c2", "read", '{"path": "f.txt"}')
    resp = _make_openai_response(
        content=None, tool_calls=[tc1, tc2], finish_reason="tool_calls",
    )
    provider = DeepSeekProvider(client=MagicMock(), thinking_enabled=False)
    provider._client.chat.completions.create.return_value = resp

    request = ModelRequest(
        model="deepseek-v4-pro",
        messages=[ModelMessage.user("Do things.")],
    )
    result = provider.complete(request)
    assert len(result.message.tool_calls) == 2  # type: ignore[arg-type]
    assert result.message.tool_calls[0].name == "bash"  # type: ignore[index]
    assert result.message.tool_calls[1].name == "read"  # type: ignore[index]


# ============================================================================
# 6 — Valid raw JSON arguments parsed
# ============================================================================

def test_valid_raw_json_arguments() -> None:
    """ToolCall.from_raw_json correctly parses valid JSON string arguments."""
    tc = ToolCall.from_raw_json("id1", "search", '{"query": "hello"}')
    assert tc.arguments == {"query": "hello"}
    assert tc.argument_error is None


# ============================================================================
# 7 — Malformed raw JSON preserved without crash
# ============================================================================

def test_malformed_raw_json_preserved() -> None:
    """Invalid JSON in tool arguments: arguments=None, raw preserved, error set."""
    tc = ToolCall.from_raw_json("id1", "broken", "{not json}")
    assert tc.arguments is None
    assert tc.raw_arguments == "{not json}"
    assert tc.argument_error is not None


# ============================================================================
# 8 — Tool message: role=tool + tool_call_id
# ============================================================================

def test_tool_message_encoding() -> None:
    messages = [
        ModelMessage.assistant(tool_calls=[ToolCall.from_arguments("c1", "bash", {})]),
        ModelMessage.tool(tool_call_id="c1", content="result"),
    ]
    encoded = _encode_messages(messages, thinking_enabled=False)
    assert encoded[1]["role"] == "tool"
    assert encoded[1]["tool_call_id"] == "c1"
    assert encoded[1]["content"] == "result"


# ============================================================================
# 9 — Finish reason normalization
# ============================================================================

@pytest.mark.parametrize("vendor,expected", [
    ("stop", FinishReason.STOP),
    ("tool_calls", FinishReason.TOOL_CALLS),
    ("length", FinishReason.LENGTH),
    ("content_filter", FinishReason.CONTENT_FILTER),
])
def test_finish_reason_normalization(vendor: str, expected: FinishReason) -> None:
    from agent_runtime.providers.deepseek import _decode_finish_reason
    assert _decode_finish_reason(vendor) == expected


def test_unknown_finish_reason_defaults() -> None:
    from agent_runtime.providers.deepseek import _decode_finish_reason
    assert _decode_finish_reason("weird") == FinishReason.UNKNOWN
    assert _decode_finish_reason(None) == FinishReason.UNKNOWN


# ============================================================================
# 10 — Usage normalization: cache hit, cache miss, reasoning tokens
# ============================================================================

def test_usage_full_normalization() -> None:
    from agent_runtime.providers.deepseek import _decode_usage

    completion_details = MagicMock()
    completion_details.reasoning_tokens = 200

    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    usage.total_tokens = 150
    usage.prompt_cache_hit_tokens = 30
    usage.prompt_cache_miss_tokens = 70
    usage.completion_tokens_details = completion_details

    result = _decode_usage(usage)
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.total_tokens == 150
    assert result.cache_read_tokens == 30
    assert result.cache_miss_tokens == 70
    assert result.reasoning_tokens == 200
    assert result.cache_creation_tokens is None  # not guessed from cache miss


# ============================================================================
# 11 — Normal no-tool thinking: reasoning_content NOT exposed as ProviderState
# ============================================================================

def test_no_tool_thinking_not_exposed() -> None:
    """Reasoning content for a non-tool-call turn is discarded."""
    resp = _make_openai_response(
        content="Answer.",
        finish_reason="stop",
        reasoning_content="Let me think about this...",
    )
    provider = DeepSeekProvider(client=MagicMock(), thinking_enabled=True)
    provider._client.chat.completions.create.return_value = resp

    request = ModelRequest(model="deepseek-v4-pro", messages=[ModelMessage.user("Q")])
    result = provider.complete(request)
    # No tool_calls → provider_state must be None
    assert result.message.provider_state is None


# ============================================================================
# 12 — Thinking + tool call: reasoning_content captured exactly into ProviderState
# ============================================================================

def test_thinking_with_tool_call_captures_provider_state() -> None:
    reasoning = "I need to use bash to list files.\nLet me call the tool."
    tool = _make_tool_call_openai("call_1", "bash", '{"command": "ls"}')
    resp = _make_openai_response(
        content=None, tool_calls=[tool], finish_reason="tool_calls",
        reasoning_content=reasoning,
    )
    provider = DeepSeekProvider(client=MagicMock(), thinking_enabled=True)
    provider._client.chat.completions.create.return_value = resp

    request = ModelRequest(model="deepseek-v4-pro", messages=[ModelMessage.user("List files.")])
    result = provider.complete(request)

    assert result.message.provider_state is not None
    assert result.message.provider_state.provider == "deepseek"
    assert result.message.provider_state.data["reasoning_content"] == reasoning


# ============================================================================
# 13 — Next request replays exact reasoning_content
# ============================================================================

def test_reasoning_content_exact_replay() -> None:
    reasoning = "Step 1: check files.\nStep 2: read config.\n€ unicode ✓"
    ps = ProviderState(provider="deepseek", data={"reasoning_content": reasoning})

    tc = ToolCall.from_raw_json("call_1", "bash", '{"command": "ls"}')
    assistant = ModelMessage.assistant(
        tool_calls=[tc], provider_state=ps,
    )

    messages = [
        ModelMessage.user("Do it."),
        assistant,
        ModelMessage.tool(tool_call_id="call_1", content="result"),
    ]
    encoded = _encode_messages(messages, thinking_enabled=True)

    # The assistant message should have the exact reasoning_content replayed
    assert "reasoning_content" in encoded[1]
    assert encoded[1]["reasoning_content"] == reasoning


# ============================================================================
# 14 — Replay preserves Unicode/newlines/whitespace exactly
# ============================================================================

def test_replay_preserves_unicode_and_whitespace() -> None:
    reasoning = "Line 1\n  Line 2  indented\n🎯 emoji & spaces  "
    ps = ProviderState(provider="deepseek", data={"reasoning_content": reasoning})

    tc = ToolCall.from_raw_json("call_1", "bash", '{}')
    assistant = ModelMessage.assistant(tool_calls=[tc], provider_state=ps)

    encoded = _encode_messages([assistant], thinking_enabled=True)
    assert encoded[0]["reasoning_content"] == reasoning
    assert "🎯" in encoded[0]["reasoning_content"]
    assert "  Line 2  indented" in encoded[0]["reasoning_content"]


# ============================================================================
# 15 — Thinking enabled + tool-call history + missing ProviderState: fail before API
# ============================================================================

def test_missing_provider_state_fails_before_api() -> None:
    """When thinking is enabled and an assistant has tool_calls but no
    ProviderState, the adapter must raise INVALID_REQUEST without calling the API."""
    tc = ToolCall.from_raw_json("call_1", "bash", '{"cmd": "ls"}')
    messages = [
        ModelMessage.user("Do it."),
        ModelMessage.assistant(tool_calls=[tc]),  # no provider_state!
        ModelMessage.tool(tool_call_id="call_1", content="result"),
    ]

    with pytest.raises(ProviderError) as exc:
        _encode_messages(messages, thinking_enabled=True)

    assert exc.value.kind == ProviderErrorKind.INVALID_REQUEST
    assert not exc.value.retryable
    assert "missing" in str(exc.value).lower() or "ProviderState" in str(exc.value)


# ============================================================================
# 16 — provider_state.provider mismatch: fail before network
# ============================================================================

def test_provider_state_provider_mismatch() -> None:
    """A ProviderState with provider != 'deepseek' is rejected before API call."""
    ps = ProviderState(provider="openai", data={"reasoning_content": "thinking..."})
    tc = ToolCall.from_raw_json("call_1", "bash", '{}')
    assistant = ModelMessage.assistant(tool_calls=[tc], provider_state=ps)

    with pytest.raises(ProviderError) as exc:
        _encode_messages([assistant], thinking_enabled=True)

    assert exc.value.kind == ProviderErrorKind.INVALID_REQUEST


# ============================================================================
# 17 — Assistant tool call with content=None encodes content=""
# ============================================================================

def test_assistant_tool_call_content_none_encodes_empty_string() -> None:
    tc = ToolCall.from_raw_json("call_1", "bash", '{}')
    assistant = ModelMessage.assistant(tool_calls=[tc], content=None)
    encoded = _encode_messages([assistant], thinking_enabled=False)

    assert encoded[0]["role"] == "assistant"
    assert "content" in encoded[0]
    assert encoded[0]["content"] == ""
    assert "tool_calls" in encoded[0]


# ============================================================================
# 18 — Error normalization (400-503 and unknown)
# ============================================================================

def test_error_400_invalid_request() -> None:
    import openai
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.request = MagicMock()
    err = openai.BadRequestError("bad", response=mock_resp, body={"error": {"message": "Bad"}})
    pe = _normalize_error(err)
    assert pe.kind == ProviderErrorKind.INVALID_REQUEST
    assert not pe.retryable


def test_error_401_auth() -> None:
    import openai
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.request = MagicMock()
    err = openai.AuthenticationError("auth fail", response=mock_resp, body={})
    pe = _normalize_error(err)
    assert pe.kind == ProviderErrorKind.AUTH
    assert not pe.retryable


def test_error_402_billing() -> None:
    import openai
    mock_resp = MagicMock()
    mock_resp.status_code = 402
    mock_resp.request = MagicMock()
    err = openai.APIStatusError("billing", response=mock_resp, body={"error": {"message": "Quota exceeded"}})
    pe = _normalize_error(err)
    assert pe.kind == ProviderErrorKind.BILLING
    assert not pe.retryable


def test_error_422_unprocessable() -> None:
    import openai
    mock_resp = MagicMock()
    mock_resp.status_code = 422
    mock_resp.request = MagicMock()
    err = openai.UnprocessableEntityError("unprocessable", response=mock_resp, body={})
    pe = _normalize_error(err)
    assert pe.kind == ProviderErrorKind.INVALID_REQUEST
    assert not pe.retryable


def test_error_429_rate_limit() -> None:
    import openai
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.request = MagicMock()
    err = openai.RateLimitError("rate limit", response=mock_resp, body={})
    pe = _normalize_error(err)
    assert pe.kind == ProviderErrorKind.RATE_LIMIT
    assert pe.retryable


def test_error_500_server() -> None:
    import openai
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.request = MagicMock()
    err = openai.InternalServerError("server error", response=mock_resp, body={})
    pe = _normalize_error(err)
    assert pe.kind == ProviderErrorKind.SERVER
    assert pe.retryable


def test_error_503_overloaded() -> None:
    import openai
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.request = MagicMock()
    err = openai.APIStatusError("overloaded", response=mock_resp, body={})
    pe = _normalize_error(err)
    assert pe.kind == ProviderErrorKind.OVERLOADED
    assert pe.retryable


def test_unknown_error_normalization() -> None:
    pe = _normalize_error(ValueError("something weird"))
    assert pe.kind == ProviderErrorKind.UNKNOWN
    assert pe.provider == "deepseek"


# ============================================================================
# 19 — Provider does not retry internally
# ============================================================================

def test_provider_does_not_retry() -> None:
    """The adapter itself must not implement retry/sleep/backoff."""
    mock_client = MagicMock()
    import openai
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.request = MagicMock()
    err = openai.RateLimitError("rate limited", response=mock_resp, body={})
    mock_client.chat.completions.create.side_effect = err

    provider = DeepSeekProvider(client=mock_client)
    request = ModelRequest(
        model="deepseek-v4-pro",
        messages=[ModelMessage.user("Hi.")],
    )

    with pytest.raises(ProviderError) as exc:
        provider.complete(request)
    assert exc.value.kind == ProviderErrorKind.RATE_LIMIT
    # create() was called exactly once — no retry loop
    assert mock_client.chat.completions.create.call_count == 1


# ============================================================================
# 20 — complete() returns canonical ModelResponse
# ============================================================================

def test_complete_returns_canonical_model_response() -> None:
    resp = _make_openai_response(content="OK", finish_reason="stop")
    provider = DeepSeekProvider(client=MagicMock(), thinking_enabled=False)
    provider._client.chat.completions.create.return_value = resp

    request = ModelRequest(
        model="deepseek-v4-pro",
        messages=[ModelMessage.user("Test.")],
    )
    result = provider.complete(request)
    assert isinstance(result, ModelResponse)
    assert result.message.role == MessageRole.ASSISTANT
    assert result.provider == "deepseek"
    assert result.response_id is not None


# ============================================================================
# 21 — thinking configuration is passed to API
# ============================================================================

def test_thinking_config_passed_to_api() -> None:
    """Verify extra_body contains thinking config when enabled."""
    mock_client = MagicMock()
    resp = _make_openai_response(content="OK", finish_reason="stop")
    mock_client.chat.completions.create.return_value = resp

    provider = DeepSeekProvider(
        client=mock_client,
        thinking_enabled=True,
        reasoning_effort="max",
    )
    request = ModelRequest(
        model="deepseek-v4-pro",
        messages=[ModelMessage.user("Hi.")],
    )
    provider.complete(request)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "extra_body" in call_kwargs
    assert call_kwargs["extra_body"]["thinking"] == {"type": "enabled"}
    assert call_kwargs["extra_body"]["reasoning_effort"] == "max"


def test_thinking_disabled_omits_config() -> None:
    mock_client = MagicMock()
    resp = _make_openai_response(content="OK", finish_reason="stop")
    mock_client.chat.completions.create.return_value = resp

    provider = DeepSeekProvider(client=mock_client, thinking_enabled=False)
    request = ModelRequest(
        model="deepseek-v4-pro",
        messages=[ModelMessage.user("Hi.")],
    )
    provider.complete(request)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "extra_body" not in call_kwargs


# ============================================================================
# 22 — Invalid reasoning_effort rejected at construction
# ============================================================================

def test_invalid_reasoning_effort_rejected() -> None:
    with pytest.raises(ValueError, match="reasoning_effort"):
        DeepSeekProvider(reasoning_effort="low")


# ============================================================================
# 23 — deterministic JSON serialization for tool arguments
# ============================================================================

def test_deterministic_json_serialization() -> None:
    """ToolCall.arguments is serialized with sort_keys for determinism."""
    import json
    tc = ToolCall.from_arguments("id1", "test", {"z": 1, "a": 2, "m": 3})
    result = _tool_call_to_dict(tc)
    args_str = result["function"]["arguments"]
    # sort_keys=True ensures consistent ordering
    parsed = json.loads(args_str)
    assert parsed == {"a": 2, "m": 3, "z": 1}


def test_tool_call_falls_back_to_raw_arguments() -> None:
    """When arguments is None but raw_arguments is available, use raw_arguments."""
    tc = ToolCall(
        id="id1", name="test",
        arguments=None,
        raw_arguments='{"raw": "fallback"}',
        argument_error="previous parse failed",  # required when arguments=None
    )
    result = _tool_call_to_dict(tc)
    assert result["function"]["arguments"] == '{"raw": "fallback"}'


def test_tool_call_no_args_or_raw_raises() -> None:
    """When BOTH arguments and raw_arguments are unavailable, raise ProviderError."""
    tc = ToolCall(
        id="id1", name="test",
        arguments=None,
        raw_arguments=None,
        argument_error="parse failed",
    )
    with pytest.raises(ProviderError) as exc:
        _tool_call_to_dict(tc)
    assert exc.value.kind == ProviderErrorKind.INVALID_REQUEST
