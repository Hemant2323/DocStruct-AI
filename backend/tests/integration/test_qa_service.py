"""Tests for QAService, with the Retriever and LLMProvider mocked.

Documents and content blocks are real (via ``db_session``), matching how
``AnalysisService`` is tested — only the model-calling layers (retrieval, generation) are
faked, per the project's convention of mocking the model layer and nothing else.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.prompts import ask
from app.core.config import Settings
from app.core.exceptions import (
    AIConfigurationError,
    AIUnavailableError,
    EmbeddingsNotAvailableError,
    InsufficientContentError,
    InvalidModelResponseError,
    NotFoundError,
)
from app.models.domain.content import BlockType, ContentBlock, Provenance
from app.models.domain.document import Document, ProcessingStatus, SupportedFileType
from app.repositories.block_repo import ContentBlockRepository
from app.repositories.chunk_repo import Chunk
from app.repositories.document_repo import DocumentRepository
from app.retrieval.base import ScoredChunk
from app.services.qa_service import QAService
from tests.fixtures.fake_provider import FakeLLMProvider
from tests.fixtures.fake_retriever import FakeRetriever

OWNER = "owner-a"


@pytest.fixture
def block_repository(db_session: AsyncSession) -> ContentBlockRepository:
    return ContentBlockRepository(db_session)


@pytest.fixture
def fake_retriever() -> FakeRetriever:
    return FakeRetriever()


@pytest.fixture
def fake_provider() -> FakeLLMProvider:
    return FakeLLMProvider(text="Invoices are due within 30 days.")


@pytest.fixture
def service(
    fake_retriever: FakeRetriever,
    fake_provider: FakeLLMProvider,
    settings: Settings,
    db_session: AsyncSession,
    block_repository: ContentBlockRepository,
) -> QAService:
    return QAService(
        retriever=fake_retriever,
        provider=fake_provider,
        settings=settings,
        documents=DocumentRepository(db_session),
        blocks=block_repository,
    )


async def _ready_document(db_session: AsyncSession, *, filename: str = "agreement.pdf") -> Document:
    repository = DocumentRepository(db_session)
    created = await repository.create(
        owner_id=OWNER,
        original_filename=filename,
        file_type=SupportedFileType.PDF,
        size_bytes=1024,
        status=ProcessingStatus.READY,
    )
    await db_session.flush()
    return created


async def _seeded_chunks(
    block_repository: ContentBlockRepository, document: Document, texts: list[str]
) -> list[Chunk]:
    """Persist real content blocks and return matching (unpersisted) Chunk objects.

    QAService never queries a real ``chunks`` table row by id — it only reads the text,
    order, and block_ids a retrieved ``Chunk`` carries — so these do not need to exist in
    the ``chunks`` table, only their referenced content blocks do (for provenance).
    """
    blocks = [
        ContentBlock(BlockType.PARAGRAPH, text, index, Provenance.for_page(index + 1))
        for index, text in enumerate(texts)
    ]
    block_ids = await block_repository.replace_for_document(document.id, blocks)
    return [
        Chunk(
            id=uuid4(),
            order=index,
            text=text,
            block_ids=[block_id],
            token_estimate=max(len(text) // 4, 1),
            char_count=len(text),
            location_label=f"page {index + 1}",
        )
        for index, (text, block_id) in enumerate(zip(texts, block_ids, strict=True))
    ]


class TestValidQuestion:
    async def test_returns_a_grounded_answer(
        self,
        service: QAService,
        fake_retriever: FakeRetriever,
        fake_provider: FakeLLMProvider,
        db_session: AsyncSession,
        block_repository: ContentBlockRepository,
    ) -> None:
        document = await _ready_document(db_session)
        [chunk] = await _seeded_chunks(
            block_repository, document, ["Invoices are due within 30 days."]
        )
        fake_retriever.results = [ScoredChunk(chunk=chunk, score=0.9)]

        result = await service.ask(document.id, "When is payment due?", owner_id=OWNER)

        assert result.answer == "Invoices are due within 30 days."
        assert result.question == "When is payment due?"
        assert result.document_id == document.id
        assert result.model == fake_provider.model


class TestMissingDocument:
    async def test_nonexistent_document_raises_not_found(self, service: QAService) -> None:
        with pytest.raises(NotFoundError):
            await service.ask(uuid4(), "a question", owner_id=OWNER)

    async def test_another_owners_document_is_not_found(
        self, service: QAService, db_session: AsyncSession
    ) -> None:
        document = await _ready_document(db_session)

        with pytest.raises(NotFoundError):
            await service.ask(document.id, "a question", owner_id="a-different-owner")

    async def test_not_yet_ready_document_raises_insufficient_content(
        self, service: QAService, db_session: AsyncSession
    ) -> None:
        repository = DocumentRepository(db_session)
        document = await repository.create(
            owner_id=OWNER,
            original_filename="still-processing.pdf",
            file_type=SupportedFileType.PDF,
            size_bytes=10,
            status=ProcessingStatus.PROCESSING,
        )
        await db_session.flush()

        with pytest.raises(InsufficientContentError):
            await service.ask(document.id, "a question", owner_id=OWNER)


class TestNoUsableEmbeddings:
    async def test_propagates_embeddings_not_available(
        self, service: QAService, fake_retriever: FakeRetriever, db_session: AsyncSession
    ) -> None:
        document = await _ready_document(db_session)
        fake_retriever.error = EmbeddingsNotAvailableError(
            "This document has not been prepared for search yet."
        )

        with pytest.raises(EmbeddingsNotAvailableError):
            await service.ask(document.id, "a question", owner_id=OWNER)


class TestRetrieverBehavior:
    async def test_retriever_is_called_with_the_document_and_question(
        self,
        service: QAService,
        fake_retriever: FakeRetriever,
        db_session: AsyncSession,
        block_repository: ContentBlockRepository,
    ) -> None:
        document = await _ready_document(db_session)
        [chunk] = await _seeded_chunks(block_repository, document, ["some content"])
        fake_retriever.results = [ScoredChunk(chunk=chunk, score=0.5)]

        await service.ask(document.id, "What is the deadline?", owner_id=OWNER, top_k=7)

        assert fake_retriever.last_call.document_id == document.id
        assert fake_retriever.last_call.query == "What is the deadline?"
        assert fake_retriever.last_call.top_k == 7

    async def test_no_relevant_chunks_raises_insufficient_content(
        self, service: QAService, fake_retriever: FakeRetriever, db_session: AsyncSession
    ) -> None:
        """Even if a Retriever implementation ever returned an empty list instead of
        raising, build_context's own guard must still refuse to call the model with
        nothing to ground an answer in."""
        document = await _ready_document(db_session)
        fake_retriever.results = []

        with pytest.raises(InsufficientContentError):
            await service.ask(document.id, "a question", owner_id=OWNER)


class TestProviderBehavior:
    async def test_provider_success_is_used_as_the_answer(
        self,
        service: QAService,
        fake_retriever: FakeRetriever,
        fake_provider: FakeLLMProvider,
        db_session: AsyncSession,
        block_repository: ContentBlockRepository,
    ) -> None:
        document = await _ready_document(db_session)
        [chunk] = await _seeded_chunks(block_repository, document, ["content"])
        fake_retriever.results = [ScoredChunk(chunk=chunk, score=0.5)]
        fake_provider.text = "The answer is 30 days."

        result = await service.ask(document.id, "q", owner_id=OWNER)

        assert result.answer == "The answer is 30 days."

    async def test_provider_failure_propagates(
        self,
        service: QAService,
        fake_retriever: FakeRetriever,
        fake_provider: FakeLLMProvider,
        db_session: AsyncSession,
        block_repository: ContentBlockRepository,
    ) -> None:
        document = await _ready_document(db_session)
        [chunk] = await _seeded_chunks(block_repository, document, ["content"])
        fake_retriever.results = [ScoredChunk(chunk=chunk, score=0.5)]
        fake_provider.error = AIUnavailableError("The AI service is temporarily unavailable.")

        with pytest.raises(AIUnavailableError):
            await service.ask(document.id, "q", owner_id=OWNER)

    async def test_provider_not_configured_raises_ai_configuration_error(
        self,
        service: QAService,
        fake_retriever: FakeRetriever,
        fake_provider: FakeLLMProvider,
        db_session: AsyncSession,
        block_repository: ContentBlockRepository,
    ) -> None:
        document = await _ready_document(db_session)
        [chunk] = await _seeded_chunks(block_repository, document, ["content"])
        fake_retriever.results = [ScoredChunk(chunk=chunk, score=0.5)]
        fake_provider.configured = False

        with pytest.raises(AIConfigurationError):
            await service.ask(document.id, "q", owner_id=OWNER)

    async def test_empty_answer_is_reported_not_returned(
        self,
        service: QAService,
        fake_retriever: FakeRetriever,
        fake_provider: FakeLLMProvider,
        db_session: AsyncSession,
        block_repository: ContentBlockRepository,
    ) -> None:
        document = await _ready_document(db_session)
        [chunk] = await _seeded_chunks(block_repository, document, ["content"])
        fake_retriever.results = [ScoredChunk(chunk=chunk, score=0.5)]
        fake_provider.text = "   "

        with pytest.raises(InvalidModelResponseError):
            await service.ask(document.id, "q", owner_id=OWNER)


class TestUsesRetrievedContextOnly:
    async def test_only_retrieved_chunks_reach_the_model(
        self,
        service: QAService,
        fake_retriever: FakeRetriever,
        fake_provider: FakeLLMProvider,
        db_session: AsyncSession,
        block_repository: ContentBlockRepository,
    ) -> None:
        """QAService has no ChunkRepository dependency at all -- it structurally cannot
        load the whole document, only what the retriever hands it."""
        document = await _ready_document(db_session)
        relevant, _irrelevant = await _seeded_chunks(
            block_repository, document, ["the relevant part", "unrelated other content"]
        )
        fake_retriever.results = [ScoredChunk(chunk=relevant, score=0.9)]

        await service.ask(document.id, "q", owner_id=OWNER)

        sent_texts = [block.text for block in fake_provider.last_call.context_blocks]
        assert sent_texts == ["the relevant part"]
        assert "unrelated other content" not in sent_texts


class TestDocumentIsolation:
    async def test_citations_never_reference_another_documents_blocks(
        self,
        service: QAService,
        fake_retriever: FakeRetriever,
        db_session: AsyncSession,
        block_repository: ContentBlockRepository,
    ) -> None:
        document_a = await _ready_document(db_session, filename="a.pdf")
        document_b = await _ready_document(db_session, filename="b.pdf")

        [chunk_a] = await _seeded_chunks(block_repository, document_a, ["document a content"])
        # Re-seeding blocks for document_b replaces document_a's block rows in the shared
        # repository call sequence, so document_a's chunk above must already have been
        # captured before this call -- its block_ids still point at real, resolvable rows.
        await _seeded_chunks(block_repository, document_b, ["document b content"])

        fake_retriever.results = [ScoredChunk(chunk=chunk_a, score=0.9)]

        result = await service.ask(document_a.id, "q", owner_id=OWNER)

        assert result.document_id == document_a.id
        for citation in result.citations:
            assert citation.quoted_text == "document a content"


class TestProvenancePreserved:
    async def test_source_provenance_matches_the_seeded_block(
        self,
        service: QAService,
        fake_retriever: FakeRetriever,
        db_session: AsyncSession,
        block_repository: ContentBlockRepository,
    ) -> None:
        document = await _ready_document(db_session)
        [chunk] = await _seeded_chunks(
            block_repository, document, ["Invoices are due within 30 days."]
        )
        fake_retriever.results = [ScoredChunk(chunk=chunk, score=0.9)]

        result = await service.ask(document.id, "q", owner_id=OWNER)

        assert len(result.citations) == 1
        citation = result.citations[0]
        assert citation.is_verified is True
        assert citation.provenance is not None
        assert citation.provenance.label == "page 1"
        assert citation.quoted_text == "Invoices are due within 30 days."

    async def test_multiple_retrieved_chunks_each_produce_a_citation(
        self,
        service: QAService,
        fake_retriever: FakeRetriever,
        db_session: AsyncSession,
        block_repository: ContentBlockRepository,
    ) -> None:
        document = await _ready_document(db_session)
        chunks = await _seeded_chunks(block_repository, document, ["first part", "second part"])
        fake_retriever.results = [
            ScoredChunk(chunk=chunks[0], score=0.9),
            ScoredChunk(chunk=chunks[1], score=0.8),
        ]

        result = await service.ask(document.id, "q", owner_id=OWNER)

        assert len(result.citations) == 2
        assert result.chunk_count == 2


class TestGroundingInstruction:
    def test_system_prompt_instructs_against_outside_knowledge(self) -> None:
        assert "ONLY the supplied document content" in ask.SYSTEM
        assert "does not contain enough information" in ask.SYSTEM

    def test_instruction_includes_the_question(self) -> None:
        instruction = ask.build_instruction("What is the deadline?")
        assert "What is the deadline?" in instruction


class TestErrorConventionsAreRespected:
    async def test_every_error_path_raises_a_typed_docstruct_error(
        self,
        service: QAService,
        fake_retriever: FakeRetriever,
        fake_provider: FakeLLMProvider,
        db_session: AsyncSession,
    ) -> None:
        from app.core.exceptions import DocStructError

        with pytest.raises(DocStructError) as missing_doc:
            await service.ask(uuid4(), "q", owner_id=OWNER)
        assert missing_doc.value.code == "not_found"

        document = await _ready_document(db_session)
        fake_retriever.error = EmbeddingsNotAvailableError("not ready")
        with pytest.raises(DocStructError) as no_embeddings:
            await service.ask(document.id, "q", owner_id=OWNER)
        assert no_embeddings.value.code == "embeddings_not_available"
