"""Agent Runtime — Provider-agnostic Canonical Model Contract.

This module defines the vocabulary used by the Agent Runtime to describe
conversations, tool calls, and model responses WITHOUT coupling to any
specific vendor SDK (Anthropic, OpenAI, DeepSeek, etc.).

Design principles:
  - No Anthropic SDK imports     (no ``ContentBlock``, ``tool_use``, ``input_schema``)
  - No OpenAI SDK imports       (no ``tool_calls`` array on ``choice.message``)
  - No vendor-specific fields   (no ``reasoning_content``, no ``stop_reason``)
  - Provider state is opaque    (Runtime never interprets it)
  - Malformed tool arguments    (Runtime never crashes on invalid JSON)
"""

from __future__ import annotations

import copy
import enum
import json
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------

class MessageRole(str, enum.Enum):
    """Canonical conversation roles.

    These are intentionally the same string values as the OpenAI / Anthropic
    convention so that downstream code can use them without translation, but
    the *runtime* reasons about ``MessageRole``, not raw strings.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# ---------------------------------------------------------------------------
# Finish Reason
# ---------------------------------------------------------------------------

class FinishReason(str, enum.Enum):
    """Normalised stop condition.

    Every provider adapter is responsible for translating its vendor-specific
    ``stop_reason`` / ``finish_reason`` into one of these canonical values.

    **Error model**: ``FinishReason`` describes *successful* completions
    only.  Provider failures are raised as ``ProviderError`` — there is no
    ``ERROR`` finish reason.  This ensures a single error channel.
    """

    STOP = "stop"                 # model chose to end the turn (natural stop)
    TOOL_CALLS = "tool_calls"     # model emitted one or more tool calls
    LENGTH = "length"             # output hit max_tokens / max_output limit
    CONTENT_FILTER = "content_filter"  # provider safety filter blocked output
    UNKNOWN = "unknown"           # unrecognised / unhandled reason


# ---------------------------------------------------------------------------
# Tool Definition (provider-neutral)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolDefinition:
    """Provider-neutral tool schema.

    ``parameters`` is a JSON Schema dict describing the tool's input.
    We deliberately do NOT use the Anthropic name ``input_schema``.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ToolDefinition.name must be non-empty")
        if not isinstance(self.parameters, dict):
            raise TypeError("ToolDefinition.parameters must be a dict")


# ---------------------------------------------------------------------------
# Tool Call (with graceful parse-failure handling)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation requested by the model.

    **Malformed-arguments invariant**: if the provider returns arguments that
    cannot be parsed as valid JSON, ``arguments`` is ``None``,
    ``raw_arguments`` preserves the original string, and ``argument_error``
    contains the parse error message.  The runtime MUST NOT crash.
    """

    id: str
    name: str
    arguments: dict[str, Any] | None
    raw_arguments: str | None
    argument_error: str | None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("ToolCall.id must be non-empty")
        if not self.name.strip():
            raise ValueError("ToolCall.name must be non-empty")
        # At least one of arguments or argument_error must be present.
        if self.arguments is None and self.argument_error is None:
            raise ValueError(
                "ToolCall: arguments must be set when no argument_error exists"
            )
        if self.arguments is not None and not isinstance(self.arguments, dict):
            raise TypeError("ToolCall.arguments must be a dict or None")

    @classmethod
    def from_arguments(
        cls,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> "ToolCall":
        """Construct from an already-parsed dict (e.g. Anthropic-style input).

        The arguments dict is **defensively copied** so that provider-side
        mutation cannot affect Runtime state.

        This path is for providers that natively return structured tool
        input.  Adapters MUST NOT ``json.dumps() → json.loads()``
        round-trip an already-parsed dict — use ``from_arguments()``.
        """
        return cls(
            id=call_id,
            name=name,
            arguments=copy.deepcopy(arguments),
            raw_arguments=None,
            argument_error=None,
        )

    @classmethod
    def from_raw_json(
        cls,
        call_id: str,
        name: str,
        raw_arguments: str | None,
    ) -> "ToolCall":
        """Parse raw JSON arguments; return a ToolCall that never crashes.

        This is the canonical factory for providers that return tool
        arguments as JSON strings (e.g. OpenAI / DeepSeek).

        Every provider adapter should route through one of the two
        factories — ``from_arguments()`` or ``from_raw_json()`` — so that
        the Runtime sees a consistent ``ToolCall`` shape.
        """
        if raw_arguments is None:
            return cls(
                id=call_id,
                name=name,
                arguments={},
                raw_arguments=None,
                argument_error=None,
            )

        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            return cls(
                id=call_id,
                name=name,
                arguments=None,
                raw_arguments=raw_arguments,
                argument_error=str(exc),
            )

        if not isinstance(parsed, dict):
            return cls(
                id=call_id,
                name=name,
                arguments=None,
                raw_arguments=raw_arguments,
                argument_error=(
                    f"Expected a JSON object, got {type(parsed).__name__}"
                ),
            )

        return cls(
            id=call_id,
            name=name,
            arguments=parsed,
            raw_arguments=raw_arguments,
            argument_error=None,
        )


# ---------------------------------------------------------------------------
# Provider State (opaque replay-critical data)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderState:
    """Opaque, JSON-serialisable provider-owned state.

    **Invariant**: the Runtime MAY store, copy, serialise, and hand this
    object back to the provider unchanged, but it MUST NEVER interpret or
    depend on any key inside ``data``.

    This exists so that providers can stash replay-critical protocol state
    (e.g. multi-turn conversation IDs, cached-prompt identifiers, or
    reasoning tokens) without leaking vendor concepts into the Runtime.
    """

    provider: str
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("ProviderState.provider must be non-empty")
        if not isinstance(self.data, dict):
            raise TypeError("ProviderState.data must be a dict")
        # Guard: data must be JSON-serialisable
        json.dumps(self.data)  # raises TypeError on non-serialisable values


# ---------------------------------------------------------------------------
# Model Message
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelMessage:
    """A single turn in the conversation.

    Constraints enforced at construction time:

    * ``role=TOOL``  → ``tool_call_id`` must be set.
    * ``role!=ASSISTANT`` → ``tool_calls`` must be empty / ``None``.
    * ``content`` is ``None`` when the message has only tool calls.
    """

    role: MessageRole
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    provider_state: ProviderState | None = None

    def __post_init__(self) -> None:
        # -- TOOL role requires tool_call_id --
        if self.role == MessageRole.TOOL and not self.tool_call_id:
            raise ValueError(
                "ModelMessage with role=TOOL must have a non-empty tool_call_id"
            )

        # -- Only ASSISTANT may carry tool_calls --
        if self.role != MessageRole.ASSISTANT and self.tool_calls:
            raise ValueError(
                f"ModelMessage with role={self.role.value} cannot carry tool_calls"
            )

        # -- Only ASSISTANT may carry provider_state --
        if self.role != MessageRole.ASSISTANT and self.provider_state is not None:
            raise ValueError(
                f"ModelMessage with role={self.role.value} cannot carry provider_state "
                f"(provider_state is replay-critical assistant protocol state)"
            )

        # -- Guard content type --
        if self.content is not None and not isinstance(self.content, str):
            raise TypeError("ModelMessage.content must be str or None")

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @classmethod
    def system(cls, content: str) -> "ModelMessage":
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "ModelMessage":
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(
        cls,
        content: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        provider_state: ProviderState | None = None,
    ) -> "ModelMessage":
        return cls(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
            provider_state=provider_state,
        )

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> "ModelMessage":
        return cls(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=tool_call_id,
        )


# ---------------------------------------------------------------------------
# Token Usage
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TokenUsage:
    """Normalised token accounting.

    ``total_tokens`` is optional; when absent it is derived as
    ``input_tokens + output_tokens`` via the ``total`` property.

    No field may be negative.

    **Cache semantics**: ``cache_read_tokens``, ``cache_creation_tokens``,
    and ``cache_miss_tokens`` are independent dimensions — a cache miss is
    NOT automatically a cache write.  Adapters must map provider-specific
    cache fields without conflating these concepts.

    ``provider_details`` carries usage metadata that cannot be reliably
    canonicalised across providers.  The Runtime MAY store, trace, or
    serialise it, but MUST NOT depend on specific keys for core control.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    cache_miss_tokens: int | None = None
    provider_details: dict[str, Any] = field(default_factory=dict)

    _NON_NEGATIVE = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "reasoning_tokens",
        "cache_read_tokens",
        "cache_creation_tokens",
        "cache_miss_tokens",
    )

    def __post_init__(self) -> None:
        for attr in self._NON_NEGATIVE:
            value = getattr(self, attr)
            if value is not None and value < 0:
                raise ValueError(f"TokenUsage.{attr} must be non-negative, got {value}")
        if not isinstance(self.provider_details, dict):
            raise TypeError("TokenUsage.provider_details must be a dict")
        # Guard: provider_details must be JSON-serialisable
        json.dumps(self.provider_details)

    @property
    def total(self) -> int:
        if self.total_tokens is not None:
            return self.total_tokens
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# Model Request / Response
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelRequest:
    """A complete request to a model provider.

    ``api_key`` and ``base_url`` are intentionally NOT here — they belong
    to provider configuration, not the canonical request.
    """

    model: str
    messages: list[ModelMessage]
    tools: list[ToolDefinition] | None = None
    max_output_tokens: int = 8000

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("ModelRequest.model must be non-empty")
        if not self.messages:
            raise ValueError("ModelRequest.messages must be non-empty")
        if self.max_output_tokens < 1:
            raise ValueError("ModelRequest.max_output_tokens must be >= 1")


@dataclass(frozen=True)
class ModelResponse:
    """A complete model response.

    ``message`` MUST have ``role=ASSISTANT``.
    """

    message: ModelMessage
    finish_reason: FinishReason
    usage: TokenUsage
    provider: str
    response_id: str | None = None

    def __post_init__(self) -> None:
        if self.message.role != MessageRole.ASSISTANT:
            raise ValueError(
                f"ModelResponse.message must be ASSISTANT, got {self.message.role.value}"
            )
