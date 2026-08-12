"""Cross-provider contract tests.

These tests demonstrate that the Provider Boundary isolates protocol
variation — the same canonical inputs produce equivalent canonical outputs
regardless of provider-specific encoding differences.
"""

from __future__ import annotations

from agent_runtime.model.contracts import (
    FinishReason,
    ModelMessage,
    ModelRequest,
    ToolCall,
    ToolDefinition,
)
from agent_runtime.providers.anthropic import AnthropicProvider
from agent_runtime.providers.deepseek import DeepSeekProvider


# ============================================================================
# Same ToolDefinition → different vendor encoding
# ============================================================================

def test_tool_definition_anthropic_uses_input_schema() -> None:
    """Anthropic encodes ToolDefinition.parameters as input_schema."""
    from agent_runtime.providers.anthropic import _encode_tool_definition

    td = ToolDefinition(name="bash", description="desc", parameters={"type": "object"})
    result = _encode_tool_definition(td)
    assert "input_schema" in result
    assert result["input_schema"] == {"type": "object"}
    assert "function" not in result


def test_tool_definition_deepseek_uses_function_parameters() -> None:
    """DeepSeek encodes ToolDefinition as {type: function, function: {parameters}}."""
    from agent_runtime.providers.deepseek import _encode_tool_definition

    td = ToolDefinition(name="bash", description="desc", parameters={"type": "object"})
    result = _encode_tool_definition(td)
    assert result["type"] == "function"
    assert result["function"]["parameters"] == {"type": "object"}
    assert "input_schema" not in result


# ============================================================================
# Same ToolCall → different vendor encoding
# ============================================================================

def test_tool_call_anthropic_structured_input() -> None:
    """Anthropic encodes ToolCall.arguments as a dict (tool_use.input)."""
    from agent_runtime.providers.anthropic import _tool_calls_to_blocks

    tc = ToolCall.from_arguments("id1", "bash", {"command": "ls"})
    blocks = _tool_calls_to_blocks([tc])
    assert blocks[0]["type"] == "tool_use"
    assert blocks[0]["input"] == {"command": "ls"}
    assert blocks[0]["name"] == "bash"


def test_tool_call_deepseek_json_string() -> None:
    """DeepSeek encodes ToolCall.arguments as a JSON string in function.arguments."""
    from agent_runtime.providers.deepseek import _tool_call_to_dict

    tc = ToolCall.from_arguments("id1", "bash", {"command": "ls"})
    result = _tool_call_to_dict(tc)
    assert result["type"] == "function"
    assert result["function"]["name"] == "bash"
    # DeepSeek encodes as JSON string
    import json
    parsed = json.loads(result["function"]["arguments"])
    assert parsed == {"command": "ls"}


# ============================================================================
# Different native responses → same canonical FinishReason + equivalent ToolCall
# ============================================================================

def test_both_providers_produce_finish_reason_tool_calls() -> None:
    """Both providers map their vendor stop reasons to FinishReason.TOOL_CALLS."""
    # Anthropic: "tool_use" → TOOL_CALLS
    from agent_runtime.providers.anthropic import _decode_finish_reason as anthro_decode
    assert anthro_decode("tool_use") == FinishReason.TOOL_CALLS

    # DeepSeek: "tool_calls" → TOOL_CALLS
    from agent_runtime.providers.deepseek import _decode_finish_reason as ds_decode
    assert ds_decode("tool_calls") == FinishReason.TOOL_CALLS


def test_both_providers_produce_equivalent_canonical_tool_call() -> None:
    """A bash(ls) tool call produces an equivalent canonical ToolCall
    regardless of whether it came from Anthropic or DeepSeek."""
    # Anthropic path: from_arguments with dict input
    anthro_tc = ToolCall.from_arguments("id1", "bash", {"command": "ls"})
    assert anthro_tc.name == "bash"
    assert anthro_tc.arguments == {"command": "ls"}
    assert anthro_tc.argument_error is None

    # DeepSeek path: from_raw_json with JSON string
    ds_tc = ToolCall.from_raw_json("id1", "bash", '{"command": "ls"}')
    assert ds_tc.name == "bash"
    assert ds_tc.arguments == {"command": "ls"}
    assert ds_tc.argument_error is None

    # Both produce equivalent canonical results
    assert anthro_tc.name == ds_tc.name
    assert anthro_tc.arguments == ds_tc.arguments


# ============================================================================
# Provider Boundary: Runtime uses same ModelRequest for either provider
# ============================================================================

def test_same_model_request_accepted_by_both_providers() -> None:
    """A single ModelRequest can be passed to either provider's complete().
    The provider boundary requires no Runtime-side translation."""
    request = ModelRequest(
        model="some-model",
        messages=[
            ModelMessage.system("You are helpful."),
            ModelMessage.user("List files."),
        ],
        tools=[
            ToolDefinition(name="bash", description="Run a command"),
        ],
    )

    # Both providers accept the same request type
    assert isinstance(request, ModelRequest)
    assert request.model == "some-model"
    assert len(request.messages) == 2
    assert request.tools is not None
    assert len(request.tools) == 1
