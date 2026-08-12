"""Agent Runtime — Model Provider Protocol.

Defines the narrow contract that every provider adapter must fulfil.
"""

from __future__ import annotations

import enum
from typing import Protocol

from agent_runtime.model.contracts import ModelRequest, ModelResponse


# ---------------------------------------------------------------------------
# Provider Error
# ---------------------------------------------------------------------------

class ProviderErrorKind(str, enum.Enum):
    """Normalised provider error categories.

    Every provider adapter MUST translate its vendor-specific error codes
    (HTTP status, error.type strings, etc.) into one of these canonical
    kinds so that the Runtime can make decisions without vendor coupling.
    """

    INVALID_REQUEST = "invalid_request"    # bad parameters, missing fields
    AUTH = "auth"                          # key invalid / expired
    BILLING = "billing"                    # quota, payment required
    RATE_LIMIT = "rate_limit"              # too many requests
    CONTEXT_LIMIT = "context_limit"        # input too long
    SERVER = "server"                      # provider internal error (5xx)
    OVERLOADED = "overloaded"              # provider overloaded / shedding
    INVALID_RESPONSE = "invalid_response"  # response failed schema validation
    UNKNOWN = "unknown"                    # unclassified


@enum.unique
class RetryDecision(enum.Enum):
    """Whether a failed request should be retried."""

    RETRY = "retry"
    DO_NOT_RETRY = "do_not_retry"


# Canonical mapping: which error kinds are retryable by default.
RETRYABLE_KINDS: frozenset[ProviderErrorKind] = frozenset({
    ProviderErrorKind.RATE_LIMIT,
    ProviderErrorKind.SERVER,
    ProviderErrorKind.OVERLOADED,
})


class ProviderError(Exception):
    """A normalised error from a model provider.

    The Runtime SHOULD use ``kind`` (not vendor-specific fields) to decide
    recovery behaviour.  ``retryable`` is the adapter's recommendation;
    the Runtime may override it based on its own policy.
    """

    def __init__(
        self,
        kind: ProviderErrorKind,
        provider: str,
        message: str = "",
        *,
        status_code: int | None = None,
        provider_code: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.provider = provider
        self.message_text = message
        self.status_code = status_code
        self.provider_code = provider_code
        # Default: retryable if the kind is in the known set.
        self.retryable = (
            retryable
            if retryable is not None
            else kind in RETRYABLE_KINDS
        )

    def __str__(self) -> str:  # pragma: no cover
        parts = [f"[{self.kind.value}]"]
        if self.provider_code:
            parts.append(f"provider={self.provider}({self.provider_code})")
        else:
            parts.append(f"provider={self.provider}")
        if self.status_code is not None:
            parts.append(f"status={self.status_code}")
        if self.message_text:
            parts.append(self.message_text)
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Model Provider Protocol
# ---------------------------------------------------------------------------

class ModelProvider(Protocol):
    """The minimal synchronous model-provider interface.

    Every provider adapter (Anthropic, OpenAI, DeepSeek, …) must satisfy
    this protocol.  The Runtime only calls ``complete()`` — never touches
    vendor SDKs directly.
    """

    @property
    def name(self) -> str:
        """Return a short, human-readable provider name (e.g. 'deepseek')."""
        ...

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Execute a single synchronous model completion.

        Args:
            request: A fully-populated ``ModelRequest``.

        Returns:
            ``ModelResponse`` with a canonical ``ASSISTANT`` message.

        Raises:
            ProviderError: On any provider failure (auth, rate-limit,
                server error, invalid response, …).  The Runtime MUST NOT
                catch vendor-specific exception types directly.
        """
        ...
