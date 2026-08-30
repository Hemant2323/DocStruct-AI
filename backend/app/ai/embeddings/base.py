"""The text embedding provider interface.

A separate seam from :class:`~app.ai.providers.base.LLMProvider`, deliberately not folded
into it: generation and embedding are different capabilities with different inputs and
outputs (text-in/text-out-plus-citations versus text-in/vector-out), and a provider that
offers one does not necessarily offer the other with the same request shape. Keeping them
apart means adding or swapping an embedding provider never touches generation code, and
vice versa.

Retrieval, vector storage, and chunk embedding orchestration are all out of scope here —
this interface only describes turning one piece of text into one vector.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.domain.embedding import EmbeddingResult


class Embedder(ABC):
    """Turns text into an embedding vector."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the embedding model in use."""

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Whether the provider has what it needs to make a call.

        Checked before use so a missing API key produces "AI is not configured" rather
        than an authentication failure surfaced from a third party.
        """

    @abstractmethod
    async def embed(self, text: str) -> EmbeddingResult:
        """Return the embedding for a piece of text.

        Args:
            text: The text to embed.

        Raises:
            AIConfigurationError: The provider is not configured.
            AIUnavailableError: The model could not be reached, timed out, or declined.
            InvalidModelResponseError: The reply did not contain a usable embedding.
        """
