"""Tests for AnthropicProvider.

All tests use fully mocked Anthropic SDK — no real API calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

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
from agent_runtime.providers.anthropic import (
    AnthropicProvider,
    _decode_content,
    _decode_finish_reason,
    _decode_usage,
    _encode_messages,
    _extract_system,
    _json_safe_boundary,
    _normalize_error,
    _tool_calls_to_blocks,
)


# ============================================================================
# 1 — System encoding
# ============================================================================

def test_system_encoding_single() -> None:
    messages = [ModelMessage.system("You are helpful.")]
    system = _extract_system(messages)
    assert system == "You are helpful."


def test_system_encoding_multiple_leading() -> None:
    messages = [
        ModelMessage.system("First part."),
        ModelMessage.system("Second part."),
        ModelMessage.user("Hello."),
    ]
    system = _extract_system(messages)
    assert "First part." in system
    assert "Second part." in system


def test_system_after_user_rejected() -> None:
    messages = [
        ModelMessage.user("Hello."),
        ModelMessage.system("Late system."),
    ]
    with pytest.raises(ProviderError) as exc:
        _extract_system(messages)
    assert exc.value.kind == ProviderErrorKind.INVALID_REQUEST
    assert not exc.value.retryable


# ============================================================================
# 2 — User text encoding
# ============================================================================

def test_user_encoding() -> None:
    messages = [
        ModelMessage.system("sys"),
        ModelMessage.user("Hello."),
    ]
    encoded = _encode_messages(messages)
    assert len(encoded) == 1
    assert encoded[0]["role"] == "user"
    assert encoded[0]["content"] == "Hello."


# ============================================================================
# 3 — Tool definitions: parameters → input_schema
# ============================================================================

def test_tool_definition_encoding() -> None:
    from agent_runtime.providers.anthropic import _encode_tool_definition

    td = ToolDefinition(name="bash", description="Run command",
                        parameters={"type": "object", "properties": {"cmd": {"type": "string"}}})
    result = _encode_tool_definition(td)
    assert result["name"] == "bash"
    assert result["description"] == "Run command"
    assert result["input_schema"] == td.parameters
    assert "parameters" not in result  # no OpenAI leak


# ============================================================================
# 4 — Assistant text response decode
# ============================================================================

def test_decode_assistant_text() -> None:
    # Simulate Anthropic SDK content blocks
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Hello from Claude."

    text, tool_calls = _decode_content([text_block])
    assert text == "Hello from Claude."
    assert tool_calls == []


# ============================================================================
# 5 — One tool_use decode
# ============================================================================

def test_decode_single_tool_use() -> None:
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "call_01"
    tool_block.name = "bash"
    tool_block.input = {"command": "ls"}

    text, tool_calls = _decode_content([tool_block])
    assert text is None
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call_01"
    assert tool_calls[0].name == "bash"
    assert tool_calls[0].arguments == {"command": "ls"}
    assert tool_calls[0].raw_arguments is None
    assert tool_calls[0].argument_error is None


# ============================================================================
# 6 — Multiple tool_use decode
# ============================================================================

def test_decode_multiple_tool_use() -> None:
    tb1 = MagicMock()
    tb1.type = "tool_use"; tb1.id = "c1"; tb1.name = "bash"; tb1.input = {"cmd": "ls"}
    tb2 = MagicMock()
    tb2.type = "tool_use"; tb2.id = "c2"; tb2.name = "read"; tb2.input = {"path": "f.txt"}

    text, tool_calls = _decode_content([tb1, tb2])
    assert len(tool_calls) == 2
    assert tool_calls[0].id == "c1"
    assert tool_calls[1].id == "c2"


# ============================================================================
# 7 — ToolCall uses from_arguments (no JSON round-trip)
# ============================================================================

def test_tool_call_factory_uses_from_arguments() -> None:
    """Verify that Anthropic tool_use.input goes through from_arguments
    (preserving structured dicts, no json.dumps→json.loads round-trip)."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "call_x"
    tool_block.name = "test"
    tool_block.input = {"nested": {"deep": [1, 2, 3]}}

    _, tool_calls = _decode_content([tool_block])
    tc = tool_calls[0]
    # from_arguments sets raw_arguments=None (NOT a JSON string)
    assert tc.raw_arguments is None
    assert tc.argument_error is None
    assert tc.arguments == {"nested": {"deep": [1, 2, 3]}}


# ============================================================================
# 8 — Canonical tool result → Anthropic user/tool_result
# ============================================================================

def test_tool_result_encoding() -> None:
    messages = [
        ModelMessage.assistant(tool_calls=[ToolCall.from_arguments("c1", "bash", {"cmd": "ls"})]),
        ModelMessage.tool(tool_call_id="c1", content="file1\nfile2"),
    ]
    encoded = _encode_messages(messages)
    assert encoded[0]["role"] == "assistant"
    assert encoded[1]["role"] == "user"
    assert encoded[1]["content"][0]["type"] == "tool_result"
    assert encoded[1]["content"][0]["tool_use_id"] == "c1"
    assert encoded[1]["content"][0]["content"] == "file1\nfile2"


# ============================================================================
# 9 — Multiple consecutive TOOL → one user message
# ============================================================================

def test_multiple_tool_results_aggregated() -> None:
    tc1 = ToolCall.from_arguments("c1", "bash", {"cmd": "ls"})
    tc2 = ToolCall.from_arguments("c2", "read", {"path": "f.txt"})
    messages = [
        ModelMessage.assistant(tool_calls=[tc1, tc2]),
        ModelMessage.tool(tool_call_id="c1", content="result1"),
        ModelMessage.tool(tool_call_id="c2", content="result2"),
    ]
    encoded = _encode_messages(messages)
    assert encoded[0]["role"] == "assistant"
    assert encoded[1]["role"] == "user"
    assert len(encoded[1]["content"]) == 2
    assert encoded[1]["content"][0]["tool_use_id"] == "c1"
    assert encoded[1]["content"][1]["tool_use_id"] == "c2"


# ============================================================================
# 10 — Finish reason normalization
# ============================================================================

@pytest.mark.parametrize("vendor_reason,expected", [
    ("end_turn", FinishReason.STOP),
    ("tool_use", FinishReason.TOOL_CALLS),
    ("max_tokens", FinishReason.LENGTH),
    ("refusal", FinishReason.CONTENT_FILTER),
])
def test_finish_reason_normalization(vendor_reason: str, expected: FinishReason) -> None:
    assert _decode_finish_reason(vendor_reason) == expected


# ============================================================================
# 11 — Usage normalization
# ============================================================================

def test_usage_normalization() -> None:
    usage = MagicMock()
    usage.input_tokens = 100
    usage.output_tokens = 50
    usage.cache_read_input_tokens = 30
    usage.cache_creation_input_tokens = 20
    # Remove other attributes to avoid provider_details
    for attr in dir(usage):
        if attr not in ("input_tokens", "output_tokens",
                        "cache_read_input_tokens", "cache_creation_input_tokens"):
            try:
                val = getattr(usage, attr)
                if not callable(val) and not attr.startswith("_"):
                    delattr(usage, attr)
            except Exception:
                pass
    # Overwrite dir-like behavior
    result = _decode_usage(usage)
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.cache_read_tokens == 30
    assert result.cache_creation_tokens == 20


# ============================================================================
# 12 — Unknown finish reason
# ============================================================================

def test_unknown_finish_reason() -> None:
    assert _decode_finish_reason("something_weird") == FinishReason.UNKNOWN
    assert _decode_finish_reason(None) == FinishReason.UNKNOWN


# ============================================================================
# 13 — Invalid tool call arguments rejected
# ============================================================================

def test_invalid_tool_call_arguments_rejected() -> None:
    """A ToolCall with arguments=None must cause ProviderError before API call."""
    bad_tc = ToolCall(
        id="bad", name="bash",
        arguments=None, raw_arguments="{invalid", argument_error="parse error"
    )
    with pytest.raises(ProviderError) as exc:
        _tool_calls_to_blocks([bad_tc])
    assert exc.value.kind == ProviderErrorKind.INVALID_REQUEST
    assert not exc.value.retryable


# ============================================================================
# 14 — Provider errors normalize
# ============================================================================

def test_auth_error_normalization() -> None:
    import anthropic
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.request = MagicMock()
    err = anthropic.AuthenticationError("bad key", response=mock_resp, body={"error": {"type": "auth_error", "message": "Invalid key"}})
    pe = _normalize_error(err)
    assert pe.kind == ProviderErrorKind.AUTH
    assert pe.provider == "anthropic"


def test_rate_limit_error_normalization() -> None:
    import anthropic
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.request = MagicMock()
    err = anthropic.RateLimitError("rate limit", response=mock_resp, body=None)
    pe = _normalize_error(err)
    assert pe.kind == ProviderErrorKind.RATE_LIMIT


def test_server_error_normalization() -> None:
    import anthropic
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.request = MagicMock()
    err = anthropic.InternalServerError("server error", response=mock_resp, body=None)
    pe = _normalize_error(err)
    assert pe.kind == ProviderErrorKind.SERVER


# ============================================================================
# 15 — complete() returns canonical ModelResponse
# ============================================================================

def test_complete_returns_model_response() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.id = "msg_123"
    mock_response.stop_reason = "end_turn"

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "I am Claude."
    mock_response.content = [text_block]

    usage = MagicMock()
    usage.input_tokens = 10
    usage.output_tokens = 5
    mock_response.usage = usage

    mock_client.messages.create.return_value = mock_response

    provider = AnthropicProvider(client=mock_client)
    request = ModelRequest(
        model="claude-sonnet-5-20251001",
        messages=[ModelMessage.user("Hello.")],
    )
    response = provider.complete(request)

    assert isinstance(response, ModelResponse)
    assert response.message.role == MessageRole.ASSISTANT
    assert response.message.content == "I am Claude."
    assert response.finish_reason == FinishReason.STOP
    assert response.provider == "anthropic"
    assert response.response_id == "msg_123"
    assert response.usage.input_tokens == 10


# ============================================================================
# 16 — complete() with tools round-trip
# ============================================================================

def test_complete_with_tool_calls() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.id = "msg_456"
    mock_response.stop_reason = "tool_use"

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "call_1"
    tool_block.name = "bash"
    tool_block.input = {"command": "pwd"}

    mock_response.content = [tool_block]
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 20
    mock_response.usage.output_tokens = 10

    mock_client.messages.create.return_value = mock_response

    provider = AnthropicProvider(client=mock_client)
    tools = [ToolDefinition(name="bash", description="Run command")]
    request = ModelRequest(
        model="claude-sonnet-5-20251001",
        messages=[ModelMessage.user("Run pwd.")],
        tools=tools,
    )
    response = provider.complete(request)

    assert response.finish_reason == FinishReason.TOOL_CALLS
    assert response.message.has_tool_calls
    assert len(response.message.tool_calls) == 1  # type: ignore[arg-type]
    assert response.message.tool_calls[0].name == "bash"  # type: ignore[index]
    assert response.message.tool_calls[0].arguments == {"command": "pwd"}  # type: ignore[index]


# ============================================================================
# _json_safe_boundary tests
# ============================================================================

class FakeAnthropicFieldInfo:
    """Simulates a Pydantic FieldInfo — must be dropped by _json_safe_boundary."""
    def __init__(self, default=None):
        self.default = default


def test_json_safe_boundary_allows_primitives() -> None:
    assert _json_safe_boundary(None) is None
    assert _json_safe_boundary(True) is True
    assert _json_safe_boundary(42) == 42
    assert _json_safe_boundary(3.14) == 3.14
    assert _json_safe_boundary("hello") == "hello"


def test_json_safe_boundary_drops_internals() -> None:
    assert _json_safe_boundary(FakeAnthropicFieldInfo()) is None
    assert _json_safe_boundary(lambda x: x) is None
    assert _json_safe_boundary(b"bytes") is None


def test_json_safe_boundary_recursive() -> None:
    result = _json_safe_boundary({"a": 1, "b": [2, None], "c": {"d": "keep"}})
    assert result == {"a": 1, "b": [2, None], "c": {"d": "keep"}}


def test_json_safe_boundary_mixed_drops_internals() -> None:
    result = _json_safe_boundary({
        "good": 1,
        "bad": FakeAnthropicFieldInfo(),
        "nested": [None, FakeAnthropicFieldInfo(), "keep"],
    })
    assert result == {"good": 1, "nested": [None, "keep"]}


# ============================================================================
# _decode_usage with Pydantic-like fake SDK object
# ============================================================================

def _make_anthropic_fake_usage(**overrides) -> MagicMock:
    """Build a MagicMock that mimics a real Anthropic SDK Usage object."""
    usage = MagicMock()
    usage.model_fields = {
        "input_tokens": FakeAnthropicFieldInfo(default=0),
        "output_tokens": FakeAnthropicFieldInfo(default=0),
    }
    usage.model_config = {"arbitrary_types_allowed": True}
    usage.model_computed_fields = {}

    usage.input_tokens = overrides.get("input_tokens", 100)
    usage.output_tokens = overrides.get("output_tokens", 50)
    usage.cache_read_input_tokens = overrides.get("cache_read_input_tokens", 30)
    usage.cache_creation_input_tokens = overrides.get("cache_creation_input_tokens", 20)
    return usage


def test_anthropic_decode_usage_drops_pydantic_internals() -> None:
    usage = _make_anthropic_fake_usage()
    result = _decode_usage(usage)

    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.cache_read_tokens == 30
    assert result.cache_creation_tokens == 20

    pd = result.provider_details
    assert "model_fields" not in pd
    assert "model_config" not in pd
    assert "model_computed_fields" not in pd

    import json as _json
    _json.dumps(pd)  # must not raise
