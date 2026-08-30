"""Tests for the chunk embedding repository against a real database.

Cascade deletion and the unique (chunk_id, model) constraint are both things a mock would
happily confirm while the real database did something else, so these use real SQL —
mirrors the structure of ``test_content_repositories.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.domain.document import Document, SupportedFileType
from app.repositories.chunk_embedding_repo import ChunkEmbedding, ChunkEmbeddingRepository
from app.repositories.chunk_repo import Chunk, ChunkRepository
from app.repositories.document_repo import DocumentRepository

OWNER = "owner-a"
MODEL = "gemini-embedding-2-preview"


@pytest.fixture
async def document(db_session: AsyncSession) -> Document:
    """A persisted document for a chunk to hang off."""
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
async def chunk(chunk_repository: ChunkRepository, document: Document) -> Chunk:
    """A persisted chunk for embeddings to attach to."""
    await chunk_repository.replace_for_document(document.id, [("Net 30 days.", [], 4, "page 1")])
    stored = await chunk_repository.list_for_document(document.id)
    return stored[0]


class TestSaveAndRetrieve:
    async def test_save_and_get_round_trip(
        self, embedding_repository: ChunkEmbeddingRepository, chunk: Chunk, document: Document
    ) -> None:
        await embedding_repository.save(
            chunk_id=chunk.id,
            document_id=document.id,
            model=MODEL,
            values=[0.1, 0.2, 0.3],
            content_hash="hash-abc",
            token_count=4,
        )

        stored = await embedding_repository.get_for_chunk(chunk.id, model=MODEL)

        assert stored is not None
        assert stored.values == [0.1, 0.2, 0.3]
        assert stored.model == MODEL
        assert stored.content_hash == "hash-abc"
        assert stored.token_count == 4
        assert stored.chunk_id == chunk.id
        assert stored.document_id == document.id

    async def test_dimensions_are_derived_from_values(
        self, embedding_repository: ChunkEmbeddingRepository, chunk: Chunk, document: Document
    ) -> None:
        await embedding_repository.save(
            chunk_id=chunk.id,
            document_id=document.id,
            model=MODEL,
            values=[0.1, 0.2, 0.3, 0.4],
            content_hash="hash-abc",
            token_count=4,
        )

        stored = await embedding_repository.get_for_chunk(chunk.id, model=MODEL)

        assert stored is not None
        assert stored.dimensions == 4

    async def test_get_returns_none_when_not_computed(
        self, embedding_repository: ChunkEmbeddingRepository, chunk: Chunk
    ) -> None:
        assert await embedding_repository.get_for_chunk(chunk.id, model=MODEL) is None

    async def test_get_is_scoped_to_the_requested_model(
        self, embedding_repository: ChunkEmbeddingRepository, chunk: Chunk, document: Document
    ) -> None:
        await embedding_repository.save(
            chunk_id=chunk.id,
            document_id=document.id,
            model=MODEL,
            values=[0.1],
            content_hash="hash-abc",
            token_count=1,
        )

        assert await embedding_repository.get_for_chunk(chunk.id, model="a-different-model") is None

    async def test_saving_the_same_chunk_and_model_twice_is_rejected(
        self,
        embedding_repository: ChunkEmbeddingRepository,
        chunk: Chunk,
        document: Document,
        db_session: AsyncSession,
    ) -> None:
        """The unique (chunk_id, model) constraint must be real, not just documented."""
        await embedding_repository.save(
            chunk_id=chunk.id,
            document_id=document.id,
            model=MODEL,
            values=[0.1],
            content_hash="hash-abc",
            token_count=1,
        )
        await db_session.flush()

        with pytest.raises(IntegrityError):
            await embedding_repository.save(
                chunk_id=chunk.id,
                document_id=document.id,
                model=MODEL,
                values=[0.2],
                content_hash="hash-def",
                token_count=1,
            )
            await db_session.flush()

    async def test_different_models_coexist_for_the_same_chunk(
        self, embedding_repository: ChunkEmbeddingRepository, chunk: Chunk, document: Document
    ) -> None:
        await embedding_repository.save(
            chunk_id=chunk.id,
            document_id=document.id,
            model="model-a",
            values=[0.1],
            content_hash="hash-abc",
            token_count=1,
        )
        await embedding_repository.save(
            chunk_id=chunk.id,
            document_id=document.id,
            model="model-b",
            values=[0.2],
            content_hash="hash-abc",
            token_count=1,
        )

        assert (await embedding_repository.get_for_chunk(chunk.id, model="model-a")).values == [0.1]  # type: ignore[union-attr]
        assert (await embedding_repository.get_for_chunk(chunk.id, model="model-b")).values == [0.2]  # type: ignore[union-attr]


class TestReplace:
    async def test_replace_recomputes_the_embedding(
        self, embedding_repository: ChunkEmbeddingRepository, chunk: Chunk, document: Document
    ) -> None:
        await embedding_repository.save(
            chunk_id=chunk.id,
            document_id=document.id,
            model=MODEL,
            values=[0.1, 0.1],
            content_hash="old-hash",
            token_count=4,
        )

        await embedding_repository.replace_for_chunk(
            chunk_id=chunk.id,
            document_id=document.id,
            model=MODEL,
            values=[0.9, 0.9],
            content_hash="new-hash",
            token_count=5,
        )

        stored = await embedding_repository.get_for_chunk(chunk.id, model=MODEL)
        assert stored is not None
        assert stored.values == [0.9, 0.9]
        assert stored.content_hash == "new-hash"

    async def test_replace_does_not_duplicate_rows(
        self, embedding_repository: ChunkEmbeddingRepository, chunk: Chunk, document: Document
    ) -> None:
        await embedding_repository.save(
            chunk_id=chunk.id,
            document_id=document.id,
            model=MODEL,
            values=[0.1],
            content_hash="a",
            token_count=1,
        )
        await embedding_repository.replace_for_chunk(
            chunk_id=chunk.id,
            document_id=document.id,
            model=MODEL,
            values=[0.2],
            content_hash="b",
            token_count=1,
        )

        assert await embedding_repository.count_for_document(document.id, model=MODEL) == 1

    async def test_replace_on_a_chunk_with_no_prior_embedding_still_saves(
        self, embedding_repository: ChunkEmbeddingRepository, chunk: Chunk, document: Document
    ) -> None:
        await embedding_repository.replace_for_chunk(
            chunk_id=chunk.id,
            document_id=document.id,
            model=MODEL,
            values=[0.5],
            content_hash="a",
            token_count=1,
        )

        stored = await embedding_repository.get_for_chunk(chunk.id, model=MODEL)
        assert stored is not None
        assert stored.values == [0.5]


class TestListAndCount:
    async def test_list_for_document_returns_every_chunks_embedding(
        self,
        embedding_repository: ChunkEmbeddingRepository,
        chunk_repository: ChunkRepository,
        document: Document,
    ) -> None:
        await chunk_repository.replace_for_document(
            document.id, [("first", [], 1, "page 1"), ("second", [], 1, "page 2")]
        )
        chunks = await chunk_repository.list_for_document(document.id)

        for one_chunk in chunks:
            await embedding_repository.save(
                chunk_id=one_chunk.id,
                document_id=document.id,
                model=MODEL,
                values=[0.1],
                content_hash="h",
                token_count=1,
            )

        listed = await embedding_repository.list_for_document(document.id)
        assert {e.chunk_id for e in listed} == {c.id for c in chunks}

    async def test_list_can_be_filtered_to_one_model(
        self, embedding_repository: ChunkEmbeddingRepository, chunk: Chunk, document: Document
    ) -> None:
        await embedding_repository.save(
            chunk_id=chunk.id,
            document_id=document.id,
            model="model-a",
            values=[0.1],
            content_hash="h",
            token_count=1,
        )
        await embedding_repository.save(
            chunk_id=chunk.id,
            document_id=document.id,
            model="model-b",
            values=[0.2],
            content_hash="h",
            token_count=1,
        )

        only_a = await embedding_repository.list_for_document(document.id, model="model-a")
        assert [e.model for e in only_a] == ["model-a"]

    async def test_count_for_document(
        self, embedding_repository: ChunkEmbeddingRepository, chunk: Chunk, document: Document
    ) -> None:
        await embedding_repository.save(
            chunk_id=chunk.id,
            document_id=document.id,
            model=MODEL,
            values=[0.1],
            content_hash="h",
            token_count=1,
        )
        assert await embedding_repository.count_for_document(document.id) == 1

    async def test_documents_are_isolated(
        self,
        embedding_repository: ChunkEmbeddingRepository,
        chunk: Chunk,
        document: Document,
        db_session: AsyncSession,
    ) -> None:
        other = await DocumentRepository(db_session).create(
            owner_id=OWNER,
            original_filename="other.txt",
            file_type=SupportedFileType.TXT,
            size_bytes=10,
        )
        await embedding_repository.save(
            chunk_id=chunk.id,
            document_id=document.id,
            model=MODEL,
            values=[0.1],
            content_hash="h",
            token_count=1,
        )

        assert await embedding_repository.count_for_document(other.id) == 0


class TestDelete:
    async def test_delete_for_chunk(
        self, embedding_repository: ChunkEmbeddingRepository, chunk: Chunk, document: Document
    ) -> None:
        await embedding_repository.save(
            chunk_id=chunk.id,
            document_id=document.id,
            model=MODEL,
            values=[0.1],
            content_hash="h",
            token_count=1,
        )

        removed = await embedding_repository.delete_for_chunk(chunk.id)

        assert removed == 1
        assert await embedding_repository.get_for_chunk(chunk.id, model=MODEL) is None

    async def test_delete_for_document(
        self, embedding_repository: ChunkEmbeddingRepository, chunk: Chunk, document: Document
    ) -> None:
        await embedding_repository.save(
            chunk_id=chunk.id,
            document_id=document.id,
            model=MODEL,
            values=[0.1],
            content_hash="h",
            token_count=1,
        )

        removed = await embedding_repository.delete_for_document(document.id)

        assert removed == 1
        assert await embedding_repository.count_for_document(document.id) == 0

    async def test_delete_for_unknown_chunk_removes_nothing(
        self, embedding_repository: ChunkEmbeddingRepository
    ) -> None:
        assert await embedding_repository.delete_for_chunk(uuid4()) == 0


class TestCascadeDelete:
    async def test_deleting_a_document_removes_its_embeddings(
        self,
        db_session: AsyncSession,
        embedding_repository: ChunkEmbeddingRepository,
        chunk: Chunk,
        document: Document,
    ) -> None:
        """SQLite only enforces this with PRAGMA foreign_keys=ON, which the engine sets."""
        await embedding_repository.save(
            chunk_id=chunk.id,
            document_id=document.id,
            model=MODEL,
            values=[0.1],
            content_hash="h",
            token_count=1,
        )
        await db_session.commit()

        deleted = await DocumentRepository(db_session).delete(document.id, owner_id=OWNER)
        await db_session.commit()

        assert deleted is True
        assert await embedding_repository.count_for_document(document.id) == 0

    async def test_replacing_a_documents_chunks_removes_the_old_embeddings(
        self,
        db_session: AsyncSession,
        embedding_repository: ChunkEmbeddingRepository,
        chunk_repository: ChunkRepository,
        chunk: Chunk,
        document: Document,
    ) -> None:
        """Re-chunking deletes and recreates chunk rows with new IDs — an embedding tied
        to the old chunk_id must not be left behind as an orphan."""
        await embedding_repository.save(
            chunk_id=chunk.id,
            document_id=document.id,
            model=MODEL,
            values=[0.1],
            content_hash="h",
            token_count=1,
        )
        await db_session.commit()

        await chunk_repository.replace_for_document(
            document.id, [("a whole new chunk", [], 1, "page 1")]
        )
        await db_session.commit()

        assert await embedding_repository.get_for_chunk(chunk.id, model=MODEL) is None
        assert await embedding_repository.count_for_document(document.id) == 0


def test_chunk_embedding_dataclass_is_read_only() -> None:
    """Guards against a mutable domain object silently drifting from the stored row."""
    embedding = ChunkEmbedding(
        id=uuid4(),
        chunk_id=uuid4(),
        document_id=uuid4(),
        model=MODEL,
        content_hash="h",
        values=[0.1],
        dimensions=1,
        token_count=1,
        created_at=datetime.now(UTC),
    )
    with pytest.raises(AttributeError):
        embedding.model = "different-model"  # type: ignore[misc]
