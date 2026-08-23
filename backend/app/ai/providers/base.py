"""The language model provider interface.

The third of the project's extension seams, promised since Milestone 1. Swapping or
adding a model provider means writing one class; nothing in the service layer changes.

The interface is deliberately narrow and knows nothing about documents, databases or
provenance. It takes text blocks and returns text plus whatever citation spans the
provider reported. **Turning those spans into real source locations is our own
deterministic code's job** (``app/ai/citations.py``), never the provider's and never the
model's.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from app.models.domain.insight import ContextBlock, ModelResult


class LLMProvider(ABC):
    """Sends prompts to a language model and returns its reply."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the model in use, recorded on every result."""

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Whether the provider has what it needs to make a call.

        Checked before use so a missing API key produces "AI is not configured" rather
        than an authentication failure surfaced from a third party.
        """

    @abstractmethod
    async def generate(
        self,
        *,
        system: str,
        instruction: str,
        context_blocks: list[ContextBlock],
        document_title: str,
        enable_citations: bool = True,
    ) -> ModelResult:
        """Ask the model for a prose answer about the supplied content.

        Args:
            system: The system prompt — the rules the model must follow.
            instruction: What to do with the content.
            context_blocks: The document content, in order. **Position matters**: the
                index of a block in this list is what a citation refers back to.
            document_title: Shown to the model as the document's name.
            enable_citations: Ask the provider for verifiable citation spans.

        Returns:
            The reply text and any citation spans.

        Raises:
            AIConfigurationError: The provider is not configured.
            AIUnavailableError: The model could not be reached, timed out, or declined.
        """

    @abstractmethod
    async def generate_structured(
        self,
        *,
        system: str,
        instruction: str,
        context_blocks: list[ContextBlock],
        document_title: str,
        output_model: type[BaseModel],
    ) -> ModelResult:
        """Ask the model for a reply matching a schema.

        Used for extraction, where free prose would have to be parsed and could not be
        validated. The reply is validated against ``output_model`` before it is returned,
        so a malformed answer fails loudly instead of becoming half-correct data.

        Note that citations cannot be requested alongside a schema — the API rejects the
        combination. Extraction therefore establishes provenance by verifying the quotes
        the model returns against our own stored content; see ``app/ai/citations.py``.

        Raises:
            AIConfigurationError: The provider is not configured.
            AIUnavailableError: The model could not be reached, timed out, or declined.
            InvalidModelResponseError: The reply did not match the schema.
        """
