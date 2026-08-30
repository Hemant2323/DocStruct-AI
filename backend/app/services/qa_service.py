"""Orchestration for single-turn document Q&A (Milestone 4.4).

Mirrors ``AnalysisService``'s division of labour: retrieval, context assembly, and
citation resolution are all ordinary Python. The model is used only to answer the
question from content that retrieval has already selected as relevant — never to decide
what is relevant in the first place.

No conversation history, no session state: each call answers exactly one question against
one document, independent of any other call.
"""

from __future__ import annotations

from uuid import UUID

from app.ai.citations import resolve_source
from app.ai.context_builder import build_context
from app.ai.prompts import ask
from app.ai.providers.base import LLMProvider
from app.core.config import Settings
from app.core.exceptions import (
    AIConfigurationError,
    InsufficientContentError,
    InvalidModelResponseError,
    NotFoundError,
)
from app.core.logging import get_logger
from app.models.domain.document import Document, ProcessingStatus
from app.models.domain.qa import AnswerResult
from app.repositories.block_repo import ContentBlockRepository
from app.repositories.document_repo import DocumentRepository
from app.retrieval.base import Retriever

logger = get_logger(__name__)


class QAService:
    """Answers a single question about one document, grounded in retrieved content."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        provider: LLMProvider,
        settings: Settings,
        documents: DocumentRepository,
        blocks: ContentBlockRepository,
    ) -> None:
        self._retriever = retriever
        self._provider = provider
        self._settings = settings
        self._documents = documents
        self._blocks = blocks

    async def ask(
        self,
        document_id: UUID,
        question: str,
        *,
        owner_id: str,
        top_k: int = 5,
    ) -> AnswerResult:
        """Answer a question about a document, using only retrieved, relevant content.

        Raises:
            NotFoundError: The document does not exist, or is not visible to this owner.
            InsufficientContentError: The document is not ready, or has no content.
            EmbeddingsNotAvailableError: The document has not been prepared for search.
            AIConfigurationError: No model provider is configured.
            AIUnavailableError: The embedding or generation provider failed.
            InvalidModelResponseError: The model returned an empty answer, or an unusable
                query embedding.
        """
        document = await self._ready_document(document_id, owner_id=owner_id)
        self._require_provider()

        # Retrieval decides what is relevant; this service never re-implements that
        # decision. Its errors (InsufficientContentError, EmbeddingsNotAvailableError,
        # AIConfigurationError/AIUnavailableError/InvalidModelResponseError from the
        # embedder) are already the correct typed errors and propagate untouched.
        scored_chunks = await self._retriever.retrieve(
            document_id=document_id, query=question, top_k=top_k
        )

        # No optimisation pass: retrieval has already reduced the document to its most
        # relevant few chunks, so a second reduction would only lose detail for no
        # benefit — build_context is reused purely for its token-budget bookkeeping.
        context_blocks, _stats = build_context(
            [scored.chunk for scored in scored_chunks],
            max_context_tokens=self._settings.ai_max_context_tokens,
        )

        needed = [block_id for block in context_blocks for block_id in block.block_ids]
        blocks_by_id = await self._blocks.get_by_ids(list(dict.fromkeys(needed)))

        result = await self._provider.generate(
            system=ask.SYSTEM,
            instruction=ask.build_instruction(question),
            context_blocks=context_blocks,
            document_title=document.original_filename,
            enable_citations=True,
        )

        if not result.text.strip():
            raise InvalidModelResponseError("The AI returned an empty response.")

        # Every retrieved chunk is a source by construction — retrieval already decided
        # it was relevant enough to send, so there is no model-reported span to resolve
        # (and providers without native citation support never supply one anyway).
        citations = tuple(
            resolve_source(context_block, blocks_by_id) for context_block in context_blocks
        )

        logger.info(
            "Answered question document_id=%s chunks=%d citations=%d",
            document_id,
            len(context_blocks),
            len(citations),
        )

        return AnswerResult(
            document_id=document_id,
            question=question,
            answer=result.text,
            citations=citations,
            model=self._provider.model_name,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            chunk_count=len(context_blocks),
        )

    # ----------------------------------------------------------------- private
    async def _ready_document(self, document_id: UUID, *, owner_id: str) -> Document:
        """Fetch a document and confirm it can be searched.

        Mirrors ``AnalysisService._ready_document`` — kept as its own small copy rather
        than shared, since the check is a few lines and duplicating it avoids coupling
        this service to AnalysisService's internals for no real benefit.
        """
        document = await self._documents.get(document_id, owner_id=owner_id)
        if document is None:
            raise NotFoundError("Document not found", details={"document_id": str(document_id)})

        if document.status is not ProcessingStatus.READY:
            if document.status is ProcessingStatus.FAILED:
                raise InsufficientContentError(
                    "This document could not be processed, so there is nothing to ask "
                    "about. " + (document.error_message or "")
                )
            raise InsufficientContentError(
                "This document is still being processed. Try again once it is ready."
            )

        return document

    def _require_provider(self) -> None:
        if not self._provider.is_configured:
            raise AIConfigurationError(
                "AI features are not configured on this server. An API key is required."
            )
