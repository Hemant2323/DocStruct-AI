"""Cosine-similarity retrieval over stored chunk embeddings (Milestone 4, Step 3).

Pure Python: no NumPy, no vector database. At the scale of one document's chunks (tens to
low hundreds), a brute-force pass in plain Python is fast enough and keeps this component
dependency-free — the same reasoning that already kept ``httpx`` over a dedicated SDK for
every provider in this codebase.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from uuid import UUID

from app.ai.embeddings.base import Embedder
from app.core.exceptions import (
    EmbeddingsNotAvailableError,
    InsufficientContentError,
    InvalidModelResponseError,
    ValidationError,
)
from app.core.logging import get_logger
from app.repositories.chunk_embedding_repo import ChunkEmbeddingRepository
from app.repositories.chunk_repo import ChunkRepository
from app.retrieval.base import Retriever, ScoredChunk
from app.retrieval.hashing import compute_content_hash

logger = get_logger(__name__)


class _InvalidEmbeddingError(Exception):
    """Internal marker: a vector cannot be scored. Never escapes this module —
    call sites either translate it into a typed error or skip the offending chunk."""


def _validate_vector(values: Sequence[float], *, label: str) -> None:
    """Reject an empty, non-finite, or zero-magnitude vector before it is ever compared
    to anything.

    NaN and infinity pass every Python arithmetic operation without raising, so an
    unchecked bad value would silently poison a similarity score rather than fail loudly.
    Zero magnitude gets the same treatment here — checking it only inside
    ``_cosine_similarity`` would let a zero-magnitude *query* vector slip past
    validation and instead fail once per stored chunk, which would look like every
    embedding in the document was individually broken rather than the query itself.
    """
    if not values:
        raise _InvalidEmbeddingError(f"{label} embedding is empty")
    if not all(math.isfinite(value) for value in values):
        raise _InvalidEmbeddingError(f"{label} embedding contains a non-finite value")
    if math.sqrt(sum(value * value for value in values)) == 0.0:
        raise _InvalidEmbeddingError(f"{label} embedding has zero magnitude")


def _cosine_similarity(query: Sequence[float], candidate: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1].

    Raises:
        _InvalidEmbeddingError: The vectors have different lengths, or either has zero
            magnitude (cosine similarity is undefined for a zero vector — the angle to
            anything is meaningless, not zero).
    """
    if len(query) != len(candidate):
        raise _InvalidEmbeddingError(
            f"dimension mismatch: query has {len(query)} values, candidate has {len(candidate)}"
        )

    query_magnitude = math.sqrt(sum(value * value for value in query))
    candidate_magnitude = math.sqrt(sum(value * value for value in candidate))
    if query_magnitude == 0.0 or candidate_magnitude == 0.0:
        raise _InvalidEmbeddingError("zero-magnitude vector")

    dot_product = sum(q * c for q, c in zip(query, candidate, strict=True))
    return dot_product / (query_magnitude * candidate_magnitude)


class CosineSimilarityRetriever(Retriever):
    """Finds relevant chunks by cosine similarity over stored embeddings.

    The first, simplest ``Retriever`` implementation: embed the query, compare it against
    every stored embedding for the document, and rank by similarity. No index structure
    and no external service — appropriate at the scale of one document's chunks, and the
    plainest thing that works rather than a dependency that isn't needed yet.

    Args:
        embedder: Used only to embed the query. Which embedding model's stored vectors
            are searched is determined by ``embedder.model_name`` — this retriever never
            mixes embeddings from two different models in one comparison.
        chunks: Source of a document's chunk text and ordering.
        embeddings: Source of a document's stored embeddings.
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        chunks: ChunkRepository,
        embeddings: ChunkEmbeddingRepository,
    ) -> None:
        self._embedder = embedder
        self._chunks = chunks
        self._embeddings = embeddings

    async def retrieve(self, *, document_id: UUID, query: str, top_k: int = 5) -> list[ScoredChunk]:
        if top_k < 1:
            raise ValidationError(f"top_k must be a positive integer, got {top_k}")

        chunks = await self._chunks.list_for_document(document_id)
        if not chunks:
            raise InsufficientContentError("This document has no extracted content to search.")

        # Let AIConfigurationError / AIUnavailableError / InvalidModelResponseError from
        # the embedder propagate untouched — they are already the correct typed errors
        # for "the embedding provider failed", and wrapping them here would only hide
        # which layer actually failed.
        embedding_result = await self._embedder.embed(query)

        try:
            _validate_vector(embedding_result.values, label="query")
        except _InvalidEmbeddingError as exc:
            raise InvalidModelResponseError(
                "The AI did not return a usable embedding for this query."
            ) from exc

        query_vector = embedding_result.values

        stored = await self._embeddings.list_for_document(
            document_id, model=self._embedder.model_name
        )
        if not stored:
            raise EmbeddingsNotAvailableError("This document has not been prepared for search yet.")

        chunks_by_id = {chunk.id: chunk for chunk in chunks}

        scored: list[ScoredChunk] = []
        skipped = 0

        for embedding in stored:
            chunk = chunks_by_id.get(embedding.chunk_id)
            if chunk is None:
                # An embedding whose chunk is no longer part of this document. The
                # chunk_id -> chunks.id foreign key cascades on delete/replace, so this
                # should not happen in practice; handled anyway rather than assumed away.
                skipped += 1
                logger.warning(
                    "Skipping embedding for a chunk no longer in this document "
                    "chunk_id=%s document_id=%s",
                    embedding.chunk_id,
                    document_id,
                )
                continue

            if embedding.content_hash != compute_content_hash(chunk.text):
                skipped += 1
                logger.warning(
                    "Skipping stale embedding chunk_id=%s document_id=%s",
                    chunk.id,
                    document_id,
                )
                continue

            try:
                _validate_vector(embedding.values, label="chunk")
                score = _cosine_similarity(query_vector, embedding.values)
            except _InvalidEmbeddingError as exc:
                skipped += 1
                logger.warning(
                    "Skipping unusable embedding chunk_id=%s document_id=%s reason=%s",
                    chunk.id,
                    document_id,
                    exc,
                )
                continue

            scored.append(ScoredChunk(chunk=chunk, score=score))

        if not scored:
            raise EmbeddingsNotAvailableError("This document has no usable embeddings to search.")

        if skipped:
            logger.info(
                "Retrieval skipped %d of %d stored embeddings for document_id=%s",
                skipped,
                len(stored),
                document_id,
            )

        # Highest similarity first; ties broken by the chunk's own document order so the
        # result is deterministic regardless of database or dict iteration order.
        scored.sort(key=lambda item: (-item.score, item.chunk.order))

        return scored[:top_k]
