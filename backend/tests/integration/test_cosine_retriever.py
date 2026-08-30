"""Tests for CosineSimilarityRetriever against a real database.

Chunks and their embeddings come from the real repositories (already proven correct in
``test_content_repositories.py`` / ``test_chunk_embedding_repository.py``) — only the
``Embedder`` is mocked, per the project's convention of mocking the model layer and
nothing else.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    AIUnavailableError,
    EmbeddingsNotAvailableError,
    InsufficientContentError,
    InvalidModelResponseError,
    ValidationError,
)
from app.models.domain.document import Document, SupportedFileType
from app.repositories.chunk_embedding_repo import ChunkEmbeddingRepository
from app.repositories.chunk_repo import Chunk, ChunkRepository
from app.repositories.document_repo import DocumentRepository
from app.retrieval.cosine import CosineSimilarityRetriever
from app.retrieval.hashing import compute_content_hash
from tests.fixtures.fake_embedder import FakeEmbedder

OWNER = "owner-a"


@pytest.fixture
async def document(db_session: AsyncSession) -> Document:
    repository = DocumentRepository(db_session)
    created = await repository.create(
        owner_id=OWNER,
        original_filename="report.pdf",
        file_type=SupportedFileType.PDF,
        size_bytes=1024,
    )
    await db_session.flush()
    return created


@pytest.fixture
def chunk_repository(db_session: AsyncSession) -> ChunkRepository:
    return ChunkRepository(db_session)


@pytest.fixture
def embedding_repository(db_session: AsyncSession) -> ChunkEmbeddingRepository:
    return ChunkEmbeddingRepository(db_session)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(values=[1.0, 0.0], model="fake-embedding-model")


@pytest.fixture
def retriever(
    embedder: FakeEmbedder,
    chunk_repository: ChunkRepository,
    embedding_repository: ChunkEmbeddingRepository,
) -> CosineSimilarityRetriever:
    return CosineSimilarityRetriever(
        embedder=embedder, chunks=chunk_repository, embeddings=embedding_repository
    )


async def _make_chunks(
    chunk_repository: ChunkRepository, document: Document, texts: list[str]
) -> list[Chunk]:
    await chunk_repository.replace_for_document(
        document.id, [(text, [], 1, "page 1") for text in texts]
    )
    return await chunk_repository.list_for_document(document.id)


async def _embed(
    embedding_repository: ChunkEmbeddingRepository,
    chunk: Chunk,
    document: Document,
    *,
    values: list[float],
    model: str = "fake-embedding-model",
    content_hash: str | None = None,
) -> None:
    """Save a valid (non-stale, unless overridden) embedding for one chunk."""
    await embedding_repository.save(
        chunk_id=chunk.id,
        document_id=document.id,
        model=model,
        values=values,
        content_hash=content_hash if content_hash is not None else compute_content_hash(chunk.text),
        token_count=1,
    )


class TestCosineSimilarityCorrectness:
    async def test_identical_direction_scores_close_to_one(
        self,
        retriever: CosineSimilarityRetriever,
        embedder: FakeEmbedder,
        chunk_repository: ChunkRepository,
        embedding_repository: ChunkEmbeddingRepository,
        document: Document,
    ) -> None:
        [chunk] = await _make_chunks(chunk_repository, document, ["match"])
        embedder.values = [1.0, 0.0]
        await _embed(embedding_repository, chunk, document, values=[2.0, 0.0])

        [result] = await retriever.retrieve(document_id=document.id, query="q")

        assert result.score == pytest.approx(1.0)

    async def test_orthogonal_vectors_score_zero(
        self,
        retriever: CosineSimilarityRetriever,
        embedder: FakeEmbedder,
        chunk_repository: ChunkRepository,
        embedding_repository: ChunkEmbeddingRepository,
        document: Document,
    ) -> None:
        [chunk] = await _make_chunks(chunk_repository, document, ["perpendicular"])
        embedder.values = [1.0, 0.0]
        await _embed(embedding_repository, chunk, document, values=[0.0, 1.0])

        [result] = await retriever.retrieve(document_id=document.id, query="q")

        assert result.score == pytest.approx(0.0)

    async def test_opposite_vectors_score_negative_one(
        self,
        retriever: CosineSimilarityRetriever,
        embedder: FakeEmbedder,
        chunk_repository: ChunkRepository,
        embedding_repository: ChunkEmbeddingRepository,
        document: Document,
    ) -> None:
        [chunk] = await _make_chunks(chunk_repository, document, ["opposite"])
        embedder.values = [1.0, 0.0]
        await _embed(embedding_repository, chunk, document, values=[-3.0, 0.0])

        [result] = await retriever.retrieve(document_id=document.id, query="q")

        assert result.score == pytest.approx(-1.0)


class TestRanking:
    async def test_more_similar_chunk_ranks_first(
        self,
        retriever: CosineSimilarityRetriever,
        embedder: FakeEmbedder,
        chunk_repository: ChunkRepository,
        embedding_repository: ChunkEmbeddingRepository,
        document: Document,
    ) -> None:
        close, far = await _make_chunks(chunk_repository, document, ["close", "far"])
        embedder.values = [1.0, 0.0]
        await _embed(embedding_repository, close, document, values=[1.0, 0.1])
        await _embed(embedding_repository, far, document, values=[0.1, 1.0])

        results = await retriever.retrieve(document_id=document.id, query="q")

        assert [r.chunk.text for r in results] == ["close", "far"]
        assert results[0].score > results[1].score


class TestTopK:
    async def test_top_k_limits_results(
        self,
        retriever: CosineSimilarityRetriever,
        embedder: FakeEmbedder,
        chunk_repository: ChunkRepository,
        embedding_repository: ChunkEmbeddingRepository,
        document: Document,
    ) -> None:
        chunks = await _make_chunks(chunk_repository, document, ["a", "b", "c", "d"])
        embedder.values = [1.0, 0.0]
        for chunk in chunks:
            await _embed(embedding_repository, chunk, document, values=[1.0, 0.0])

        results = await retriever.retrieve(document_id=document.id, query="q", top_k=2)

        assert len(results) == 2

    async def test_top_k_larger_than_available_returns_all(
        self,
        retriever: CosineSimilarityRetriever,
        embedder: FakeEmbedder,
        chunk_repository: ChunkRepository,
        embedding_repository: ChunkEmbeddingRepository,
        document: Document,
    ) -> None:
        [chunk] = await _make_chunks(chunk_repository, document, ["only one"])
        embedder.values = [1.0, 0.0]
        await _embed(embedding_repository, chunk, document, values=[1.0, 0.0])

        results = await retriever.retrieve(document_id=document.id, query="q", top_k=50)

        assert len(results) == 1

    async def test_top_k_below_one_is_rejected(
        self, retriever: CosineSimilarityRetriever, document: Document
    ) -> None:
        with pytest.raises(ValidationError):
            await retriever.retrieve(document_id=document.id, query="q", top_k=0)


class TestDocumentIsolation:
    async def test_retrieval_never_returns_another_documents_chunks(
        self,
        retriever: CosineSimilarityRetriever,
        embedder: FakeEmbedder,
        chunk_repository: ChunkRepository,
        embedding_repository: ChunkEmbeddingRepository,
        document: Document,
        db_session: AsyncSession,
    ) -> None:
        other_document = await DocumentRepository(db_session).create(
            owner_id=OWNER,
            original_filename="other.pdf",
            file_type=SupportedFileType.PDF,
            size_bytes=10,
        )
        await db_session.flush()

        [mine] = await _make_chunks(chunk_repository, document, ["mine"])
        await chunk_repository.replace_for_document(
            other_document.id, [("someone else's", [], 1, "page 1")]
        )
        [theirs] = await chunk_repository.list_for_document(other_document.id)

        embedder.values = [1.0, 0.0]
        await _embed(embedding_repository, mine, document, values=[1.0, 0.0])
        await _embed(embedding_repository, theirs, other_document, values=[1.0, 0.0])

        results = await retriever.retrieve(document_id=document.id, query="q", top_k=10)

        assert [r.chunk.text for r in results] == ["mine"]


class TestEmptyAndMissingEmbeddings:
    async def test_no_chunks_raises_insufficient_content(
        self, retriever: CosineSimilarityRetriever, document: Document
    ) -> None:
        with pytest.raises(InsufficientContentError):
            await retriever.retrieve(document_id=document.id, query="q")

    async def test_chunks_with_no_embeddings_raises_embeddings_not_available(
        self,
        retriever: CosineSimilarityRetriever,
        chunk_repository: ChunkRepository,
        document: Document,
    ) -> None:
        await _make_chunks(chunk_repository, document, ["unembedded"])

        with pytest.raises(EmbeddingsNotAvailableError):
            await retriever.retrieve(document_id=document.id, query="q")

    async def test_one_chunk_missing_an_embedding_does_not_block_the_others(
        self,
        retriever: CosineSimilarityRetriever,
        embedder: FakeEmbedder,
        chunk_repository: ChunkRepository,
        embedding_repository: ChunkEmbeddingRepository,
        document: Document,
    ) -> None:
        embedded, _unembedded = await _make_chunks(
            chunk_repository, document, ["has embedding", "missing embedding"]
        )
        embedder.values = [1.0, 0.0]
        await _embed(embedding_repository, embedded, document, values=[1.0, 0.0])

        results = await retriever.retrieve(document_id=document.id, query="q", top_k=10)

        assert [r.chunk.text for r in results] == ["has embedding"]


class TestDimensionMismatch:
    async def test_mismatched_dimension_chunk_is_skipped_not_fatal(
        self,
        retriever: CosineSimilarityRetriever,
        embedder: FakeEmbedder,
        chunk_repository: ChunkRepository,
        embedding_repository: ChunkEmbeddingRepository,
        document: Document,
    ) -> None:
        good, mismatched = await _make_chunks(chunk_repository, document, ["good", "mismatched"])
        embedder.values = [1.0, 0.0]
        await _embed(embedding_repository, good, document, values=[1.0, 0.0])
        await _embed(embedding_repository, mismatched, document, values=[1.0, 0.0, 0.0])

        results = await retriever.retrieve(document_id=document.id, query="q", top_k=10)

        assert [r.chunk.text for r in results] == ["good"]

    async def test_all_embeddings_mismatched_raises_embeddings_not_available(
        self,
        retriever: CosineSimilarityRetriever,
        embedder: FakeEmbedder,
        chunk_repository: ChunkRepository,
        embedding_repository: ChunkEmbeddingRepository,
        document: Document,
    ) -> None:
        [chunk] = await _make_chunks(chunk_repository, document, ["mismatched"])
        embedder.values = [1.0, 0.0]
        await _embed(embedding_repository, chunk, document, values=[1.0, 0.0, 0.0])

        with pytest.raises(EmbeddingsNotAvailableError):
            await retriever.retrieve(document_id=document.id, query="q")


class TestZeroVectors:
    async def test_zero_vector_query_raises_invalid_model_response(
        self,
        retriever: CosineSimilarityRetriever,
        embedder: FakeEmbedder,
        chunk_repository: ChunkRepository,
        document: Document,
    ) -> None:
        await _make_chunks(chunk_repository, document, ["some text"])
        embedder.values = [0.0, 0.0]

        with pytest.raises(InvalidModelResponseError):
            await retriever.retrieve(document_id=document.id, query="q")

    async def test_zero_vector_chunk_embedding_is_skipped(
        self,
        retriever: CosineSimilarityRetriever,
        embedder: FakeEmbedder,
        chunk_repository: ChunkRepository,
        embedding_repository: ChunkEmbeddingRepository,
        document: Document,
    ) -> None:
        good, zero = await _make_chunks(chunk_repository, document, ["good", "zero"])
        embedder.values = [1.0, 0.0]
        await _embed(embedding_repository, good, document, values=[1.0, 0.0])
        await _embed(embedding_repository, zero, document, values=[0.0, 0.0])

        results = await retriever.retrieve(document_id=document.id, query="q", top_k=10)

        assert [r.chunk.text for r in results] == ["good"]


class TestInvalidValues:
    async def test_non_finite_query_raises_invalid_model_response(
        self,
        retriever: CosineSimilarityRetriever,
        embedder: FakeEmbedder,
        chunk_repository: ChunkRepository,
        document: Document,
    ) -> None:
        await _make_chunks(chunk_repository, document, ["some text"])
        embedder.values = [float("nan"), 0.0]

        with pytest.raises(InvalidModelResponseError):
            await retriever.retrieve(document_id=document.id, query="q")

    async def test_infinite_query_raises_invalid_model_response(
        self,
        retriever: CosineSimilarityRetriever,
        embedder: FakeEmbedder,
        chunk_repository: ChunkRepository,
        document: Document,
    ) -> None:
        await _make_chunks(chunk_repository, document, ["some text"])
        embedder.values = [float("inf"), 0.0]

        with pytest.raises(InvalidModelResponseError):
            await retriever.retrieve(document_id=document.id, query="q")

    async def test_non_finite_chunk_embedding_is_skipped(
        self,
        retriever: CosineSimilarityRetriever,
        embedder: FakeEmbedder,
        chunk_repository: ChunkRepository,
        embedding_repository: ChunkEmbeddingRepository,
        document: Document,
    ) -> None:
        good, broken = await _make_chunks(chunk_repository, document, ["good", "broken"])
        embedder.values = [1.0, 0.0]
        await _embed(embedding_repository, good, document, values=[1.0, 0.0])
        await _embed(embedding_repository, broken, document, values=[float("nan"), 0.0])

        results = await retriever.retrieve(document_id=document.id, query="q", top_k=10)

        assert [r.chunk.text for r in results] == ["good"]

    async def test_empty_chunk_embedding_is_skipped(
        self,
        retriever: CosineSimilarityRetriever,
        embedder: FakeEmbedder,
        chunk_repository: ChunkRepository,
        embedding_repository: ChunkEmbeddingRepository,
        document: Document,
    ) -> None:
        good, empty = await _make_chunks(chunk_repository, document, ["good", "empty"])
        embedder.values = [1.0, 0.0]
        await _embed(embedding_repository, good, document, values=[1.0, 0.0])
        await _embed(embedding_repository, empty, document, values=[])

        results = await retriever.retrieve(document_id=document.id, query="q", top_k=10)

        assert [r.chunk.text for r in results] == ["good"]


class TestDeterministicTieBreaking:
    async def test_equal_scores_are_ordered_by_chunk_order(
        self,
        retriever: CosineSimilarityRetriever,
        embedder: FakeEmbedder,
        chunk_repository: ChunkRepository,
        embedding_repository: ChunkEmbeddingRepository,
        document: Document,
    ) -> None:
        first, second, third = await _make_chunks(
            chunk_repository, document, ["first", "second", "third"]
        )
        embedder.values = [1.0, 0.0]
        # Every chunk gets an identical embedding, so every score ties exactly.
        for chunk in (third, first, second):  # saved out of document order deliberately
            await _embed(embedding_repository, chunk, document, values=[5.0, 0.0])

        results = await retriever.retrieve(document_id=document.id, query="q", top_k=10)

        assert [r.chunk.text for r in results] == ["first", "second", "third"]

    async def test_tie_break_is_stable_across_repeated_calls(
        self,
        retriever: CosineSimilarityRetriever,
        embedder: FakeEmbedder,
        chunk_repository: ChunkRepository,
        embedding_repository: ChunkEmbeddingRepository,
        document: Document,
    ) -> None:
        chunks = await _make_chunks(chunk_repository, document, ["a", "b", "c"])
        embedder.values = [1.0, 0.0]
        for chunk in chunks:
            await _embed(embedding_repository, chunk, document, values=[2.0, 0.0])

        first_call = await retriever.retrieve(document_id=document.id, query="q", top_k=10)
        second_call = await retriever.retrieve(document_id=document.id, query="q", top_k=10)

        assert [r.chunk.id for r in first_call] == [r.chunk.id for r in second_call]


class TestQueryEmbeddingFailure:
    async def test_embedder_failure_propagates_untouched(
        self,
        retriever: CosineSimilarityRetriever,
        embedder: FakeEmbedder,
        chunk_repository: ChunkRepository,
        document: Document,
    ) -> None:
        await _make_chunks(chunk_repository, document, ["some text"])
        embedder.error = AIUnavailableError("The AI service is temporarily unavailable.")

        with pytest.raises(AIUnavailableError):
            await retriever.retrieve(document_id=document.id, query="q")


class TestStaleOrMissingEmbedding:
    async def test_stale_embedding_is_skipped_not_used(
        self,
        retriever: CosineSimilarityRetriever,
        embedder: FakeEmbedder,
        chunk_repository: ChunkRepository,
        embedding_repository: ChunkEmbeddingRepository,
        document: Document,
    ) -> None:
        current, stale = await _make_chunks(
            chunk_repository, document, ["current text", "stale text"]
        )
        embedder.values = [1.0, 0.0]
        await _embed(embedding_repository, current, document, values=[1.0, 0.0])
        # A hash that does not match compute_content_hash(stale.text) — simulating a
        # chunk whose text changed after its embedding was computed.
        await _embed(
            embedding_repository,
            stale,
            document,
            values=[1.0, 0.0],
            content_hash="stale-hash-does-not-match-current-text",
        )

        results = await retriever.retrieve(document_id=document.id, query="q", top_k=10)

        assert [r.chunk.text for r in results] == ["current text"]

    async def test_all_embeddings_stale_raises_embeddings_not_available(
        self,
        retriever: CosineSimilarityRetriever,
        embedder: FakeEmbedder,
        chunk_repository: ChunkRepository,
        embedding_repository: ChunkEmbeddingRepository,
        document: Document,
    ) -> None:
        [chunk] = await _make_chunks(chunk_repository, document, ["changed since embedding"])
        embedder.values = [1.0, 0.0]
        await _embed(
            embedding_repository,
            chunk,
            document,
            values=[1.0, 0.0],
            content_hash="does-not-match",
        )

        with pytest.raises(EmbeddingsNotAvailableError):
            await retriever.retrieve(document_id=document.id, query="q")


async def test_unknown_document_id_raises_insufficient_content(
    retriever: CosineSimilarityRetriever,
) -> None:
    with pytest.raises(InsufficientContentError):
        await retriever.retrieve(document_id=uuid4(), query="q")
