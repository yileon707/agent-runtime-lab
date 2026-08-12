"""Agent Runtime — Provider Adapters.

Each adapter translates between the canonical :class:`ModelRequest` /
:class:`ModelResponse` types and a specific vendor protocol.

Adapters must satisfy the :class:`ModelProvider` protocol.
"""

from agent_runtime.model.provider import ModelProvider

__all__ = ["ModelProvider"]
