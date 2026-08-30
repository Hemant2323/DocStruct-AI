"""The retrieval interface (Milestone 4).

A separate seam from ``LLMProvider`` and ``Embedder``: retrieval decides *which* chunks
are relevant to a query, generation decides what to say about them. Swapping the
retrieval strategy — cosine similarity today, a dedicated vector index or hybrid search
later — means writing one new class here; nothing above this layer changes.

Document Q&A, chat, and API routes are not part of this step. This interface only answers
"which chunks of this document match this query."
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from uuid import UUID

from app.repositories.chunk_repo import Chunk


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """One chunk returned by retrieval, with how well it matched the query.

    Attributes:
        chunk: The chunk itself — text, order, block_ids (the citation link), and
            location label, exactly as stored. Nothing about it is altered by retrieval.
        score: Similarity to the query. Higher is more relevant. The range and meaning
            are defined by the retriever implementation (cosine similarity here scores
            in [-1, 1]; a future retriever is not obliged to use the same scale).
    """

    chunk: Chunk
    score: float


class Retriever(ABC):
    """Finds the chunks of a document most relevant to a query."""

    @abstractmethod
    async def retrieve(self, *, document_id: UUID, query: str, top_k: int = 5) -> list[ScoredChunk]:
        """Return the ``top_k`` chunks of ``document_id`` most relevant to ``query``.

        Args:
            document_id: Which document to search. Retrieval never crosses documents —
                results for one document must never include another's chunks.
            query: The text to find relevant content for.
            top_k: Maximum number of chunks to return. Fewer are returned if the document
                has fewer usable chunks than that.

        Returns:
            Chunks ordered by descending relevance. Tied scores break deterministically,
            so the same query against unchanged data always returns the same order.

        Raises:
            ValidationError: ``top_k`` is not a positive integer.
            InsufficientContentError: The document has no extracted content at all.
            EmbeddingsNotAvailableError: The document has content, but no usable
                embedding exists for it yet (none computed, or all are stale/invalid).
            AIConfigurationError: The embedding provider is not configured.
            AIUnavailableError: The embedding provider could not be reached, timed out,
                or declined.
            InvalidModelResponseError: The embedding provider returned an embedding for
                the query that cannot be used (empty, non-finite, or zero-magnitude).
        """
