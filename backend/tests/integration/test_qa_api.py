"""End-to-end tests for the document Q&A endpoint.

Both the language model and the retriever are replaced by fakes — **no test here calls a
real API or computes a real embedding**. Everything else is real: a document is uploaded
and ingested through the actual HTTP endpoints, exactly like ``test_analysis_api.py``.

Real chunks (with their real ``block_ids``, computed by the actual Milestone 2 chunker
during ingestion) are read directly via ``ChunkRepository`` against the same test database
the HTTP client is using, and wrapped in a ``ScoredChunk`` to stand in for what a real
retriever would have found — a normal integration-test pattern, not a shortcut around the
API.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AIUnavailableError, EmbeddingsNotAvailableError
from app.repositories.chunk_repo import ChunkRepository
from app.retrieval.base import ScoredChunk
from app.services.job_service import JobService
from tests.fixtures.builders import build_txt
from tests.fixtures.fake_provider import FakeLLMProvider
from tests.fixtures.fake_retriever import FakeRetriever

DOCUMENT_TEXT = (
    "SERVICE AGREEMENT\n\n"
    "This agreement is between Acme Ltd and Globex Inc.\n\n"
    "Invoices are due within 30 days of receipt.\n\n"
    "The total contract value is $45,200.00 for the year."
)


async def _ingest(client: AsyncClient, api_prefix: str, jobs: JobService) -> str:
    response = await client.post(
        f"{api_prefix}/documents",
        files={"file": ("agreement.txt", build_txt(DOCUMENT_TEXT), "text/plain")},
    )
    assert response.status_code == 202, response.text
    document_id: str = response.json()["id"]

    await jobs.wait_for_all(timeout=30.0)

    status = await client.get(f"{api_prefix}/documents/{document_id}")
    assert status.json()["status"] == "ready", status.text
    return document_id


async def _first_scored_chunk(db_session: AsyncSession, document_id: str) -> ScoredChunk:
    """Wrap the ingested document's real, first chunk in a ScoredChunk.

    Uses the actual chunk the Milestone 2 pipeline produced during real ingestion — its
    ``block_ids`` are genuinely correct, unlike a hand-built ``Chunk`` would be.
    """
    chunks = await ChunkRepository(db_session).list_for_document(UUID(document_id))
    return ScoredChunk(chunk=chunks[0], score=0.9)


@pytest.fixture
def jobs(qa_app: FastAPI) -> JobService:
    return qa_app.state.job_service  # type: ignore[no-any-return]


class TestValidQuestion:
    async def test_returns_a_grounded_answer_with_sources(
        self,
        qa_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
        fake_retriever: FakeRetriever,
        db_session: AsyncSession,
    ) -> None:
        fake_provider.text = "Invoices are due within 30 days of receipt."
        document_id = await _ingest(qa_client, api_prefix, jobs)
        fake_retriever.results = [await _first_scored_chunk(db_session, document_id)]

        response = await qa_client.post(
            f"{api_prefix}/documents/{document_id}/ask",
            json={"question": "When is payment due?"},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["question"] == "When is payment due?"
        assert body["answer"] == "Invoices are due within 30 days of receipt."
        assert body["model"] == fake_provider.model
        assert body["chunk_count"] == 1
        assert len(body["citations"]) == 1
        assert body["citations"][0]["is_verified"] is True

    async def test_retriever_receives_the_question_and_top_k(
        self,
        qa_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_retriever: FakeRetriever,
        db_session: AsyncSession,
    ) -> None:
        document_id = await _ingest(qa_client, api_prefix, jobs)
        fake_retriever.results = [await _first_scored_chunk(db_session, document_id)]

        await qa_client.post(
            f"{api_prefix}/documents/{document_id}/ask",
            json={"question": "What is the total value?", "top_k": 3},
        )

        assert fake_retriever.last_call.query == "What is the total value?"
        assert fake_retriever.last_call.top_k == 3
        assert str(fake_retriever.last_call.document_id) == document_id


class TestQuestionValidation:
    async def test_empty_question_is_rejected(
        self, qa_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(qa_client, api_prefix, jobs)

        response = await qa_client.post(
            f"{api_prefix}/documents/{document_id}/ask", json={"question": ""}
        )

        assert response.status_code == 422, response.text

    async def test_whitespace_only_question_is_rejected(
        self, qa_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(qa_client, api_prefix, jobs)

        response = await qa_client.post(
            f"{api_prefix}/documents/{document_id}/ask", json={"question": "   \n\t  "}
        )

        assert response.status_code == 422, response.text

    async def test_question_over_the_length_limit_is_rejected(
        self, qa_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(qa_client, api_prefix, jobs)

        response = await qa_client.post(
            f"{api_prefix}/documents/{document_id}/ask",
            json={"question": "a" * 2001},
        )

        assert response.status_code == 422, response.text

    async def test_missing_question_field_is_rejected(
        self, qa_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(qa_client, api_prefix, jobs)

        response = await qa_client.post(f"{api_prefix}/documents/{document_id}/ask", json={})

        assert response.status_code == 422, response.text

    async def test_error_envelope_matches_existing_conventions(
        self, qa_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(qa_client, api_prefix, jobs)

        response = await qa_client.post(
            f"{api_prefix}/documents/{document_id}/ask", json={"question": ""}
        )

        body = response.json()
        assert "error" in body
        assert "code" in body["error"]
        assert "request_id" in body["error"]


class TestMissingDocument:
    async def test_nonexistent_document_returns_404(
        self, qa_client: AsyncClient, api_prefix: str
    ) -> None:
        response = await qa_client.post(
            f"{api_prefix}/documents/{uuid4()}/ask", json={"question": "a question"}
        )

        assert response.status_code == 404, response.text
        assert response.json()["error"]["code"] == "not_found"


class TestNoUsableEmbeddings:
    async def test_returns_422_when_retrieval_has_nothing_to_search(
        self,
        qa_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_retriever: FakeRetriever,
    ) -> None:
        document_id = await _ingest(qa_client, api_prefix, jobs)
        fake_retriever.error = EmbeddingsNotAvailableError(
            "This document has not been prepared for search yet."
        )

        response = await qa_client.post(
            f"{api_prefix}/documents/{document_id}/ask", json={"question": "a question"}
        )

        assert response.status_code == 422, response.text
        assert response.json()["error"]["code"] == "embeddings_not_available"


class TestProviderFailure:
    async def test_provider_failure_returns_a_5xx_with_the_generic_envelope(
        self,
        qa_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
        fake_retriever: FakeRetriever,
        db_session: AsyncSession,
    ) -> None:
        document_id = await _ingest(qa_client, api_prefix, jobs)
        fake_retriever.results = [await _first_scored_chunk(db_session, document_id)]
        fake_provider.error = AIUnavailableError("The AI service is temporarily unavailable.")

        response = await qa_client.post(
            f"{api_prefix}/documents/{document_id}/ask", json={"question": "a question"}
        )

        assert response.status_code == 500, response.text
        assert "request_id" in response.json()["error"]
