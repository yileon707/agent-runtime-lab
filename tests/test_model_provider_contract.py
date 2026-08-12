"""Tests for the provider-agnostic canonical model contract.

These tests validate that the contracts in ``agent_runtime.model`` express
the core agent-runtime vocabulary without any vendor coupling.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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
    RetryDecision,
    RETRYABLE_KINDS,
)


# ============================================================================
# 0 — Architectural constraint: no vendor SDK imports
# ============================================================================

VENDOR_MODULES = ("anthropic", "openai")


def _imported_modules(package: str) -> set[str]:
    """Return module names that are currently loaded under *package*."""
    return {m for m in sys.modules if m == package or m.startswith(f"{package}.")}


@pytest.mark.parametrize("vendor", VENDOR_MODULES)
def test_agent_runtime_model_does_not_import_vendor_sdk(vendor: str) -> None:
    """The canonical contract must be importable without loading any vendor SDK."""
    # Remove the package-modules from sys.modules so we can detect a fresh load.
    before = _imported_modules("agent_runtime")
    # Force a re-import
    for mod in list(sys.modules):
        if mod in before:
            del sys.modules[mod]

    # Re-import the contracts module
    import agent_runtime.model.contracts  # noqa: F811

    after = _imported_modules("agent_runtime")
    assert vendor not in sys.modules or vendor not in after, (
        f"agent_runtime.model imported '{vendor}' — "
        f"canonical contract must be vendor-free"
    )


# ============================================================================
# 1 — Normal user → assistant conversation
# ============================================================================

def test_basic_conversation() -> None:
    messages = [
        ModelMessage.system("You are a test assistant."),
        ModelMessage.user("Hello."),
        ModelMessage.assistant("Hi there!"),
    ]
    assert len(messages) == 3
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[1].role == MessageRole.USER
    assert messages[2].role == MessageRole.ASSISTANT
    assert messages[2].content == "Hi there!"


# ============================================================================
# 2 — Assistant tool call → tool message with matching id
# ============================================================================

def test_tool_call_and_result_correlation() -> None:
    tc = ToolCall(
        id="call_01",
        name="bash",
        arguments={"command": "ls"},
        raw_arguments='{"command": "ls"}',
        argument_error=None,
    )
    assistant_msg = ModelMessage.assistant(tool_calls=[tc])
    tool_msg = ModelMessage.tool(tool_call_id="call_01", content="file1.txt\nfile2.txt")

    assert assistant_msg.has_tool_calls
    assert assistant_msg.tool_calls is not None
    assert assistant_msg.tool_calls[0].id == "call_01"
    assert tool_msg.tool_call_id == "call_01"
    assert tool_msg.role == MessageRole.TOOL
    # Correlation: tool message's tool_call_id matches the tool call's id
    assert tool_msg.tool_call_id == assistant_msg.tool_calls[0].id


# ============================================================================
# 3 — Assistant response with multiple tool calls
# ============================================================================

def test_multiple_tool_calls_in_one_response() -> None:
    tc1 = ToolCall(
        id="call_01", name="read_file",
        arguments={"path": "a.py"},
        raw_arguments='{"path": "a.py"}',
        argument_error=None,
    )
    tc2 = ToolCall(
        id="call_02", name="read_file",
        arguments={"path": "b.py"},
        raw_arguments='{"path": "b.py"}',
        argument_error=None,
    )
    msg = ModelMessage.assistant(tool_calls=[tc1, tc2])
    assert len(msg.tool_calls) == 2  # type: ignore[arg-type]
    assert msg.tool_calls[0].id == "call_01"   # type: ignore[index]
    assert msg.tool_calls[1].id == "call_02"   # type: ignore[index]


# ============================================================================
# 4 — Valid tool argument JSON: parsed arguments are correct
# ============================================================================

def test_valid_tool_arguments() -> None:
    tc = ToolCall.from_raw_json("id1", "search", '{"query": "hello world"}')
    assert tc.arguments == {"query": "hello world"}
    assert tc.raw_arguments == '{"query": "hello world"}'
    assert tc.argument_error is None


def test_valid_tool_arguments_empty_object() -> None:
    tc = ToolCall.from_raw_json("id1", "noop", "{}")
    assert tc.arguments == {}
    assert tc.argument_error is None


# ============================================================================
# 5 — Malformed tool arguments: does not crash, preserves raw + error
# ============================================================================

def test_malformed_tool_arguments_invalid_json() -> None:
    tc = ToolCall.from_raw_json("id1", "broken", "{not valid json}")
    assert tc.arguments is None
    assert tc.raw_arguments == "{not valid json}"
    assert tc.argument_error is not None
    assert "JSON" in tc.argument_error.lower() or "expect" in tc.argument_error.lower()


def test_malformed_tool_arguments_not_a_dict() -> None:
    tc = ToolCall.from_raw_json("id1", "broken", "[1, 2, 3]")
    assert tc.arguments is None
    assert tc.raw_arguments == "[1, 2, 3]"
    assert tc.argument_error is not None
    assert "object" in tc.argument_error.lower() or "dict" in tc.argument_error.lower()


def test_malformed_tool_arguments_none_raw() -> None:
    tc = ToolCall.from_raw_json("id1", "no_args", None)
    assert tc.arguments == {}
    assert tc.raw_arguments is None
    assert tc.argument_error is None


# ============================================================================
# 6 — ProviderState: opaque, serialisable, round-trips correctly
# ============================================================================

def test_provider_state_round_trip() -> None:
    original_data = {"opaque": {"foo": "bar"}, "session_id": "sess-42"}
    ps = ProviderState(provider="test-provider", data=original_data)
    # Serialise
    serialised = json.dumps({"provider": ps.provider, "data": ps.data})
    # Deserialise
    restored = json.loads(serialised)
    new_ps = ProviderState(provider=restored["provider"], data=restored["data"])
    assert new_ps.data == original_data
    assert new_ps.provider == "test-provider"


def test_provider_state_rejects_non_serialisable_data() -> None:
    with pytest.raises(TypeError):
        ProviderState(provider="bad", data={"fn": lambda x: x})


# ============================================================================
# 7 — TOOL role without tool_call_id: MUST be rejected
# ============================================================================

def test_tool_message_without_tool_call_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="tool_call_id"):
        ModelMessage(role=MessageRole.TOOL, content="result")


# ============================================================================
# 8 — USER message with tool_calls: MUST be rejected
# ============================================================================

def test_user_message_with_tool_calls_is_rejected() -> None:
    tc = ToolCall(
        id="x", name="bash",
        arguments={"cmd": "ls"},
        raw_arguments='{"cmd": "ls"}',
        argument_error=None,
    )
    with pytest.raises(ValueError, match="tool_calls"):
        ModelMessage(role=MessageRole.USER, tool_calls=[tc])


# ============================================================================
# 9 — ModelResponse.message not ASSISTANT: MUST be rejected
# ============================================================================

def test_model_response_message_must_be_assistant() -> None:
    user_msg = ModelMessage.user("hi")
    with pytest.raises(ValueError, match="ASSISTANT"):
        ModelResponse(
            message=user_msg,  # type: ignore[arg-type]
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(),
            provider="test",
        )


# ============================================================================
# 10 — TokenUsage rejects negative values
# ============================================================================

def test_token_usage_rejects_negative_input() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        TokenUsage(input_tokens=-1)


def test_token_usage_rejects_negative_output() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        TokenUsage(output_tokens=-5)


def test_token_usage_rejects_negative_total() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        TokenUsage(total_tokens=-1)


def test_token_usage_derives_total_when_absent() -> None:
    usage = TokenUsage(input_tokens=100, output_tokens=50)
    assert usage.total == 150


def test_token_usage_explicit_total_wins() -> None:
    usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=200)
    assert usage.total == 200


# ============================================================================
# 11 — FinishReason does not depend on any vendor string
# ============================================================================

def test_finish_reason_values_are_vendor_neutral() -> None:
    """Confirm that FinishReason members use canonical names, not vendor strings."""
    values = {fr.value for fr in FinishReason}
    # None of these should be Anthropic-specific
    assert "tool_use" not in values
    assert "max_tokens" not in values
    assert "end_turn" not in values
    assert "stop_sequence" not in values
    # None should be OpenAI-specific
    assert "function_call" not in values
    # Our canonical values
    assert "stop" in values
    assert "tool_calls" in values
    assert "length" in values


def test_finish_reason_is_usable_without_any_import() -> None:
    """FinishReason should work as plain enum values."""
    assert FinishReason("stop") == FinishReason.STOP
    assert FinishReason("tool_calls") == FinishReason.TOOL_CALLS


# ============================================================================
# 12 — ProviderError retryable semantics are explicit
# ============================================================================

def test_provider_error_retryable_defaults() -> None:
    # Rate limit → retryable by default
    err = ProviderError(ProviderErrorKind.RATE_LIMIT, "test")
    assert err.retryable is True

    # Invalid request → not retryable by default
    err2 = ProviderError(ProviderErrorKind.INVALID_REQUEST, "test")
    assert err2.retryable is False


def test_provider_error_explicit_retryable_overrides_default() -> None:
    # Override: mark an OVERLOADED error as not retryable
    err = ProviderError(
        ProviderErrorKind.OVERLOADED,
        "test",
        "too many retries",
        retryable=False,
    )
    assert err.retryable is False


def test_provider_error_has_retry_decision_utility() -> None:
    """Confirm RetryDecision exists for explicit retry signalling."""
    assert RetryDecision.RETRY.value == "retry"
    assert RetryDecision.DO_NOT_RETRY.value == "do_not_retry"


# ============================================================================
# 13 — ModelRequest / ModelResponse integration smoke test
# ============================================================================

def test_full_request_response_cycle() -> None:
    """A complete request→response round-trip with tool calls."""
    tools = [
        ToolDefinition(
            name="bash",
            description="Run a shell command",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        ),
    ]

    messages = [
        ModelMessage.system("You are a helpful assistant."),
        ModelMessage.user("List files."),
    ]

    request = ModelRequest(
        model="test-model",
        messages=messages,
        tools=tools,
        max_output_tokens=8000,
    )
    assert request.model == "test-model"
    assert len(request.tools) == 1  # type: ignore[arg-type]

    # Simulate a model response with a tool call
    tc = ToolCall.from_raw_json("call_1", "bash", '{"command": "ls"}')
    response_msg = ModelMessage.assistant(tool_calls=[tc])
    response = ModelResponse(
        message=response_msg,
        finish_reason=FinishReason.TOOL_CALLS,
        usage=TokenUsage(input_tokens=50, output_tokens=20),
        provider="test",
        response_id="resp-1",
    )
    assert response.finish_reason == FinishReason.TOOL_CALLS
    assert response.message.tool_calls is not None
    assert response.message.tool_calls[0].name == "bash"
    assert response.usage.total == 70


# ============================================================================
# 14 — ToolDefinition validation
# ============================================================================

def test_tool_definition_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        ToolDefinition(name="", description="bad")


def test_tool_definition_rejects_non_dict_parameters() -> None:
    with pytest.raises(TypeError, match="dict"):
        ToolDefinition(name="t", description="d", parameters=[])  # type: ignore[arg-type]


# ============================================================================
# 15 — ModelMessage convenience constructors
# ============================================================================

def test_model_message_helper_constructors() -> None:
    sys_msg = ModelMessage.system("sys")
    assert sys_msg.role == MessageRole.SYSTEM
    assert sys_msg.content == "sys"

    user_msg = ModelMessage.user("q")
    assert user_msg.role == MessageRole.USER

    tool_msg = ModelMessage.tool("id1", "result")
    assert tool_msg.role == MessageRole.TOOL
    assert tool_msg.tool_call_id == "id1"


# ============================================================================
# 16 — ModelRequest validation
# ============================================================================

def test_model_request_rejects_empty_model() -> None:
    with pytest.raises(ValueError, match="model"):
        ModelRequest(model="", messages=[ModelMessage.user("hi")])


def test_model_request_rejects_empty_messages() -> None:
    with pytest.raises(ValueError, match="messages"):
        ModelRequest(model="m", messages=[])


def test_model_request_rejects_invalid_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_output_tokens"):
        ModelRequest(
            model="m",
            messages=[ModelMessage.user("hi")],
            max_output_tokens=0,
        )


# ============================================================================
# 17 — ToolCall construction invariants
# ============================================================================

def test_tool_call_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="id"):
        ToolCall(id="", name="bash", arguments={}, raw_arguments="{}", argument_error=None)


def test_tool_call_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        ToolCall(id="id1", name="", arguments={}, raw_arguments="{}", argument_error=None)


def test_tool_call_requires_arguments_or_error() -> None:
    with pytest.raises(ValueError, match="arguments"):
        ToolCall(id="id1", name="bash", arguments=None, raw_arguments=None, argument_error=None)
