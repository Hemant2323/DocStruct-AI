"""Tests for the content block and chunk repositories against a real database.

Cascade deletion and provenance round-tripping are both things a mock would happily
confirm while the real database did something else, so these use real SQL.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.domain.content import BlockType, ContentBlock, Provenance
from app.models.domain.document import Document, SupportedFileType
from app.repositories.block_repo import ContentBlockRepository
from app.repositories.chunk_repo import ChunkRepository
from app.repositories.document_repo import DocumentRepository

OWNER = "owner-a"


@pytest.fixture
async def document(db_session: AsyncSession) -> Document:
    """A persisted document for content to hang off."""
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
def block_repository(db_session: AsyncSession) -> ContentBlockRepository:
    return ContentBlockRepository(db_session)


@pytest.fixture
def chunk_repository(db_session: AsyncSession) -> ChunkRepository:
    return ChunkRepository(db_session)


def _blocks() -> list[ContentBlock]:
    return [
        ContentBlock(BlockType.HEADING, "Payment Terms", 0, Provenance.for_page(1), level=1),
        ContentBlock(BlockType.PARAGRAPH, "Net 30 days.", 1, Provenance.for_page(1)),
        ContentBlock(
            BlockType.TABLE,
            "Fee | Amount\n---|---\nLate | 5%",
            2,
            Provenance.for_page(2),
            metadata={"rows": 2, "columns": 2},
        ),
    ]


class TestContentBlockRepository:
    async def test_replace_inserts_and_returns_ids_in_order(
        self, block_repository: ContentBlockRepository, document: Document
    ) -> None:
        ids = await block_repository.replace_for_document(document.id, _blocks())

        assert len(ids) == 3
        assert len(set(ids)) == 3

    async def test_blocks_round_trip_with_provenance(
        self, block_repository: ContentBlockRepository, document: Document
    ) -> None:
        """The whole point of the table: provenance must survive storage intact."""
        await block_repository.replace_for_document(document.id, _blocks())

        stored = await block_repository.list_for_document(document.id)

        assert [b.text for b in stored] == [b.text for b in _blocks()]
        assert [b.provenance for b in stored] == [b.provenance for b in _blocks()]
        assert stored[0].level == 1
        assert stored[2].metadata == {"rows": 2, "columns": 2}

    async def test_blocks_come_back_in_reading_order(
        self, block_repository: ContentBlockRepository, document: Document
    ) -> None:
        shuffled = [_blocks()[2], _blocks()[0], _blocks()[1]]

        await block_repository.replace_for_document(document.id, shuffled)
        stored = await block_repository.list_for_document(document.id)

        assert [b.order for b in stored] == [0, 1, 2]

    async def test_replace_removes_the_previous_generation(
        self, block_repository: ContentBlockRepository, document: Document
    ) -> None:
        """Re-parsing must not interleave two generations of blocks."""
        await block_repository.replace_for_document(document.id, _blocks())

        await block_repository.replace_for_document(
            document.id,
            [ContentBlock(BlockType.PARAGRAPH, "only block", 0, Provenance.for_page(1))],
        )

        stored = await block_repository.list_for_document(document.id)
        assert len(stored) == 1
        assert stored[0].text == "only block"

    async def test_pagination(
        self, block_repository: ContentBlockRepository, document: Document
    ) -> None:
        await block_repository.replace_for_document(document.id, _blocks())

        page = await block_repository.list_for_document(document.id, limit=2, offset=1)

        assert [b.order for b in page] == [1, 2]

    async def test_get_by_ids_resolves_blocks_for_citations(
        self, block_repository: ContentBlockRepository, document: Document
    ) -> None:
        """Chunk -> block IDs -> provenance is how a citation is built."""
        ids = await block_repository.replace_for_document(document.id, _blocks())

        resolved = await block_repository.get_by_ids([ids[0], ids[2]])

        assert set(resolved) == {ids[0], ids[2]}
        assert resolved[ids[2]].provenance.label == "page 2"

    async def test_get_by_ids_with_empty_list(
        self, block_repository: ContentBlockRepository
    ) -> None:
        assert await block_repository.get_by_ids([]) == {}

    async def test_count(
        self, block_repository: ContentBlockRepository, document: Document
    ) -> None:
        await block_repository.replace_for_document(document.id, _blocks())
        assert await block_repository.count_for_document(document.id) == 3

    async def test_documents_are_isolated(
        self,
        block_repository: ContentBlockRepository,
        document: Document,
        db_session: AsyncSession,
    ) -> None:
        other = await DocumentRepository(db_session).create(
            owner_id=OWNER,
            original_filename="other.txt",
            file_type=SupportedFileType.TXT,
            size_bytes=10,
        )
        await block_repository.replace_for_document(document.id, _blocks())

        assert await block_repository.count_for_document(other.id) == 0


class TestChunkRepository:
    async def test_chunks_round_trip_with_block_ids(
        self,
        block_repository: ContentBlockRepository,
        chunk_repository: ChunkRepository,
        document: Document,
    ) -> None:
        """block_ids is the citation link and must survive JSON storage as real UUIDs."""
        block_ids = await block_repository.replace_for_document(document.id, _blocks())

        await chunk_repository.replace_for_document(
            document.id,
            [
                ("Payment Terms\nNet 30 days.", block_ids[:2], 8, "page 1"),
                ("Fee | Amount", block_ids[2:], 4, "page 2"),
            ],
        )

        stored = await chunk_repository.list_for_document(document.id)

        assert len(stored) == 2
        assert stored[0].block_ids == block_ids[:2]
        assert stored[1].block_ids == block_ids[2:]
        assert stored[0].order == 0
        assert stored[1].location_label == "page 2"
        assert stored[0].char_count == len("Payment Terms\nNet 30 days.")

    async def test_replace_removes_previous_chunks(
        self, chunk_repository: ChunkRepository, document: Document
    ) -> None:
        await chunk_repository.replace_for_document(document.id, [("a", [], 1, "x")])
        await chunk_repository.replace_for_document(document.id, [("b", [], 1, "y")])

        stored = await chunk_repository.list_for_document(document.id)
        assert len(stored) == 1
        assert stored[0].text == "b"

    async def test_count(self, chunk_repository: ChunkRepository, document: Document) -> None:
        await chunk_repository.replace_for_document(
            document.id, [("a", [], 1, "x"), ("b", [], 1, "y")]
        )
        assert await chunk_repository.count_for_document(document.id) == 2

    async def test_empty_replace_clears(
        self, chunk_repository: ChunkRepository, document: Document
    ) -> None:
        await chunk_repository.replace_for_document(document.id, [("a", [], 1, "x")])
        await chunk_repository.replace_for_document(document.id, [])

        assert await chunk_repository.count_for_document(document.id) == 0


class TestCascadeDelete:
    async def test_deleting_a_document_removes_its_content(
        self,
        db_session: AsyncSession,
        block_repository: ContentBlockRepository,
        chunk_repository: ChunkRepository,
        document: Document,
    ) -> None:
        """SQLite only enforces this with PRAGMA foreign_keys=ON, which the engine sets.

        Without that pragma this test fails while PostgreSQL would pass — which is
        exactly the dev/production divergence it exists to catch.
        """
        block_ids = await block_repository.replace_for_document(document.id, _blocks())
        await chunk_repository.replace_for_document(document.id, [("text", block_ids, 5, "page 1")])
        await db_session.commit()

        deleted = await DocumentRepository(db_session).delete(document.id, owner_id=OWNER)
        await db_session.commit()

        assert deleted is True
        assert await block_repository.count_for_document(document.id) == 0
        assert await chunk_repository.count_for_document(document.id) == 0

    async def test_content_for_unknown_document_is_empty(
        self, block_repository: ContentBlockRepository
    ) -> None:
        assert await block_repository.list_for_document(uuid4()) == []
