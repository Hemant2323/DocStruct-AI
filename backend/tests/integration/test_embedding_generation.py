"""Tests for the embedding-generation step of ingestion.

Real ingestion runs end-to-end (parse -> chunk -> persist), exactly like
``test_background_processing.py`` — only the embedder is a fake, per the project's
convention of mocking the model layer and nothing else. The property that matters most
here is negative: embedding failure of any kind must never fail ingestion or change a
document's terminal status, since `/ask` is the only thing that depends on embeddings
existing — the core ingestion guarantee does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.database import get_session_factory
from app.core.exceptions import AIUnavailableError
from app.models.domain.document import Document, ProcessingStatus, SupportedFileType
from app.models.domain.embedding import EmbeddingResult
from app.repositories.chunk_embedding_repo import ChunkEmbeddingRepository
from app.repositories.chunk_repo import Chunk, ChunkRepository
from app.repositories.document_repo import DocumentRepository
from app.retrieval.hashing import compute_content_hash
from app.services.ingestion_service import IngestionService
from app.storage.local import LocalFileStorage
from tests.fixtures.builders import build_txt
from tests.fixtures.fake_embedder import FakeEmbedder

OWNER = "local-user"

DOCUMENT_TEXT = "First paragraph about invoices.\n\nSecond paragraph about payment terms."

#: Long enough (well past chunk_target_chars' default of 1500) to reliably split into
#: multiple chunks, for the one test that needs more than one to fail just one of them.
MULTI_CHUNK_DOCUMENT_TEXT = "\n\n".join(
    f"Paragraph {i}: this is filler content about clause {i} of the agreement, "
    "included only to push the document past the chunk size target so the chunker "
    "produces more than one chunk for this test."
    for i in range(1, 21)
)


@dataclass
class _FlakyEmbedder(FakeEmbedder):
    """Fails only for one specific chunk's text, succeeds for every other."""

    fail_on_text: str = ""

    async def embed(self, text: str) -> EmbeddingResult:
        self.calls.append(text)
        if text == self.fail_on_text:
            raise AIUnavailableError("simulated failure for one chunk")
        return EmbeddingResult(values=list(self.values), token_count=self.token_count)


async def _create_and_store_document(
    db_session: AsyncSession, file_storage: LocalFileStorage, *, text: str = DOCUMENT_TEXT
) -> Document:
    repository = DocumentRepository(db_session)
    document = await repository.create(
        owner_id=OWNER,
        original_filename="agreement.txt",
        file_type=SupportedFileType.TXT,
        size_bytes=100,
    )
    await db_session.commit()
    await file_storage.save(document.id, build_txt(text))
    return document


class TestEmbeddingsAreGeneratedAfterIngestion:
    async def test_an_embedding_is_persisted_for_each_chunk(
        self, settings: Settings, db_session: AsyncSession, file_storage: LocalFileStorage
    ) -> None:
        document = await _create_and_store_document(db_session, file_storage)
        embedder = FakeEmbedder(values=[0.1, 0.2, 0.3])

        service = IngestionService(get_session_factory(), file_storage, settings, embedder)
        result = await service.process_document(document.id, owner_id=OWNER)

        stored = await ChunkEmbeddingRepository(db_session).list_for_document(document.id)
        assert len(stored) == result.chunk_count
        assert embedder.call_count == result.chunk_count

    async def test_embedding_values_and_model_are_stored_correctly(
        self, settings: Settings, db_session: AsyncSession, file_storage: LocalFileStorage
    ) -> None:
        document = await _create_and_store_document(db_session, file_storage)
        embedder = FakeEmbedder(values=[0.5, -0.5], token_count=7, model="fake-embedding-model")

        service = IngestionService(get_session_factory(), file_storage, settings, embedder)
        await service.process_document(document.id, owner_id=OWNER)

        [stored] = await ChunkEmbeddingRepository(db_session).list_for_document(
            document.id, model="fake-embedding-model"
        )
        assert stored.values == [0.5, -0.5]
        assert stored.token_count == 7

    async def test_content_hash_matches_the_chunk_text_using_the_shared_function(
        self, settings: Settings, db_session: AsyncSession, file_storage: LocalFileStorage
    ) -> None:
        """Must use the exact same hashing function CosineSimilarityRetriever compares
        against later (M4.3's documented contract), not a re-derived one."""
        document = await _create_and_store_document(db_session, file_storage)
        embedder = FakeEmbedder()

        service = IngestionService(get_session_factory(), file_storage, settings, embedder)
        await service.process_document(document.id, owner_id=OWNER)

        chunks = await ChunkRepository(db_session).list_for_document(document.id)
        embeddings = await ChunkEmbeddingRepository(db_session).list_for_document(document.id)
        embeddings_by_chunk = {e.chunk_id: e for e in embeddings}

        for chunk in chunks:
            assert embeddings_by_chunk[chunk.id].content_hash == compute_content_hash(chunk.text)

    async def test_document_reaches_ready_regardless_of_embeddings(
        self, settings: Settings, db_session: AsyncSession, file_storage: LocalFileStorage
    ) -> None:
        document = await _create_and_store_document(db_session, file_storage)
        embedder = FakeEmbedder()

        service = IngestionService(get_session_factory(), file_storage, settings, embedder)
        await service.process_document(document.id, owner_id=OWNER)

        refreshed = await DocumentRepository(db_session).get(document.id, owner_id=OWNER)
        assert refreshed is not None
        assert refreshed.status is ProcessingStatus.READY


class TestEmbeddingIsBestEffort:
    async def test_no_embedder_configured_still_ingests_successfully(
        self, settings: Settings, db_session: AsyncSession, file_storage: LocalFileStorage
    ) -> None:
        """The existing (pre-M4) default: IngestionService built with no embedder at
        all, exactly as every M1/M2 test in test_background_processing.py does."""
        document = await _create_and_store_document(db_session, file_storage)

        service = IngestionService(get_session_factory(), file_storage, settings)
        result = await service.process_document(document.id, owner_id=OWNER)

        assert result.block_count > 0
        refreshed = await DocumentRepository(db_session).get(document.id, owner_id=OWNER)
        assert refreshed is not None
        assert refreshed.status is ProcessingStatus.READY

        stored = await ChunkEmbeddingRepository(db_session).list_for_document(document.id)
        assert stored == []

    async def test_an_unconfigured_embedder_skips_cleanly(
        self, settings: Settings, db_session: AsyncSession, file_storage: LocalFileStorage
    ) -> None:
        document = await _create_and_store_document(db_session, file_storage)
        embedder = FakeEmbedder(configured=False)

        service = IngestionService(get_session_factory(), file_storage, settings, embedder)
        await service.process_document(document.id, owner_id=OWNER)

        assert embedder.call_count == 0
        stored = await ChunkEmbeddingRepository(db_session).list_for_document(document.id)
        assert stored == []

    async def test_every_chunk_failing_to_embed_does_not_fail_ingestion(
        self, settings: Settings, db_session: AsyncSession, file_storage: LocalFileStorage
    ) -> None:
        document = await _create_and_store_document(db_session, file_storage)
        embedder = FakeEmbedder(error=AIUnavailableError("simulated total failure"))

        service = IngestionService(get_session_factory(), file_storage, settings, embedder)
        result = await service.process_document(document.id, owner_id=OWNER)

        assert result.block_count > 0
        refreshed = await DocumentRepository(db_session).get(document.id, owner_id=OWNER)
        assert refreshed is not None
        assert refreshed.status is ProcessingStatus.READY

        stored = await ChunkEmbeddingRepository(db_session).list_for_document(document.id)
        assert stored == []

    async def test_one_bad_chunk_does_not_stop_the_others(
        self, settings: Settings, db_session: AsyncSession, file_storage: LocalFileStorage
    ) -> None:
        document = await _create_and_store_document(
            db_session, file_storage, text=MULTI_CHUNK_DOCUMENT_TEXT
        )
        chunks_preview = await _persist_only(settings, db_session, file_storage, document.id)
        assert len(chunks_preview) >= 2, "fixture text must produce at least 2 chunks"

        embedder = _FlakyEmbedder(fail_on_text=chunks_preview[0].text)

        service = IngestionService(get_session_factory(), file_storage, settings, embedder)
        await service.process_document(document.id, owner_id=OWNER)

        stored = await ChunkEmbeddingRepository(db_session).list_for_document(document.id)
        # One chunk (re-created with a new id on this second ingestion pass) failed;
        # the rest must still have been embedded.
        assert 0 < len(stored) < len(chunks_preview)


async def _persist_only(
    settings: Settings, db_session: AsyncSession, file_storage: LocalFileStorage, document_id: UUID
) -> list[Chunk]:
    """Run ingestion once with no embedder, purely to inspect what chunks it produces."""
    service = IngestionService(get_session_factory(), file_storage, settings)
    await service.process_document(document_id, owner_id=OWNER)
    return await ChunkRepository(db_session).list_for_document(document_id)
