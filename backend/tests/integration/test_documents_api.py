"""Tests for the documents API, covering the request/response contract."""

from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import DEFAULT_OWNER_ID
from app.models.domain.document import Document, ProcessingStatus, SupportedFileType
from app.repositories.document_repo import DocumentRepository


async def _seed(
    session: AsyncSession,
    *,
    owner_id: str = DEFAULT_OWNER_ID,
    filename: str = "report.pdf",
    file_type: SupportedFileType = SupportedFileType.PDF,
    status: ProcessingStatus = ProcessingStatus.PENDING,
) -> Document:
    """Insert a document directly, bypassing the API (which has no upload route yet)."""
    repository = DocumentRepository(session)
    document = await repository.create(
        owner_id=owner_id,
        original_filename=filename,
        file_type=file_type,
        size_bytes=1234,
        status=status,
    )
    await session.commit()
    return document


class TestListDocuments:
    async def test_empty_list(self, client: AsyncClient, api_prefix: str) -> None:
        response = await client.get(f"{api_prefix}/documents")

        assert response.status_code == 200
        body = response.json()
        assert body == {"items": [], "total": 0, "limit": 50, "offset": 0}

    async def test_lists_seeded_documents(
        self, client: AsyncClient, api_prefix: str, db_session: AsyncSession
    ) -> None:
        await _seed(db_session, filename="a.pdf")
        await _seed(db_session, filename="b.csv", file_type=SupportedFileType.CSV)

        response = await client.get(f"{api_prefix}/documents")

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        assert {item["original_filename"] for item in body["items"]} == {"a.pdf", "b.csv"}

    async def test_owner_scoping_hides_other_users_documents(
        self, client: AsyncClient, api_prefix: str, db_session: AsyncSession
    ) -> None:
        await _seed(db_session, owner_id="someone-else", filename="private.pdf")

        response = await client.get(f"{api_prefix}/documents")

        assert response.status_code == 200
        assert response.json()["total"] == 0

    async def test_status_filter(
        self, client: AsyncClient, api_prefix: str, db_session: AsyncSession
    ) -> None:
        await _seed(db_session, filename="pending.pdf", status=ProcessingStatus.PENDING)
        await _seed(db_session, filename="ready.pdf", status=ProcessingStatus.READY)

        response = await client.get(f"{api_prefix}/documents", params={"status": "ready"})

        body = response.json()
        assert body["total"] == 1
        assert body["items"][0]["original_filename"] == "ready.pdf"

    async def test_pagination_parameters_are_echoed(
        self, client: AsyncClient, api_prefix: str, db_session: AsyncSession
    ) -> None:
        for index in range(3):
            await _seed(db_session, filename=f"doc{index}.pdf")

        response = await client.get(f"{api_prefix}/documents", params={"limit": 2, "offset": 1})

        body = response.json()
        assert body["limit"] == 2
        assert body["offset"] == 1
        assert body["total"] == 3
        assert len(body["items"]) == 2

    async def test_invalid_limit_is_rejected(self, client: AsyncClient, api_prefix: str) -> None:
        response = await client.get(f"{api_prefix}/documents", params={"limit": 500})

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    async def test_invalid_status_is_rejected(self, client: AsyncClient, api_prefix: str) -> None:
        response = await client.get(f"{api_prefix}/documents", params={"status": "banana"})

        assert response.status_code == 422


class TestGetDocument:
    async def test_returns_the_document(
        self, client: AsyncClient, api_prefix: str, db_session: AsyncSession
    ) -> None:
        document = await _seed(db_session, filename="contract.docx")

        response = await client.get(f"{api_prefix}/documents/{document.id}")

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == str(document.id)
        assert body["original_filename"] == "contract.docx"
        assert body["status"] == "pending"

    async def test_response_never_exposes_owner_id(
        self, client: AsyncClient, api_prefix: str, db_session: AsyncSession
    ) -> None:
        """owner_id is an internal access-control detail and must not be echoed back."""
        document = await _seed(db_session)

        response = await client.get(f"{api_prefix}/documents/{document.id}")

        assert "owner_id" not in response.json()

    async def test_unknown_id_returns_404_envelope(
        self, client: AsyncClient, api_prefix: str
    ) -> None:
        response = await client.get(f"{api_prefix}/documents/{uuid4()}")

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"

    async def test_another_owners_document_is_indistinguishable_from_missing(
        self, client: AsyncClient, api_prefix: str, db_session: AsyncSession
    ) -> None:
        """Returning 403 here would confirm the document exists, letting a caller probe
        for other users' document IDs. 404 leaks nothing."""
        document = await _seed(db_session, owner_id="someone-else")

        response = await client.get(f"{api_prefix}/documents/{document.id}")

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"

    async def test_malformed_uuid_is_rejected(self, client: AsyncClient, api_prefix: str) -> None:
        response = await client.get(f"{api_prefix}/documents/not-a-uuid")

        assert response.status_code == 422


class TestDeleteDocument:
    async def test_delete_returns_204_and_removes_it(
        self, client: AsyncClient, api_prefix: str, db_session: AsyncSession
    ) -> None:
        document = await _seed(db_session)

        response = await client.delete(f"{api_prefix}/documents/{document.id}")
        assert response.status_code == 204

        follow_up = await client.get(f"{api_prefix}/documents/{document.id}")
        assert follow_up.status_code == 404

    async def test_delete_unknown_id_returns_404(
        self, client: AsyncClient, api_prefix: str
    ) -> None:
        response = await client.delete(f"{api_prefix}/documents/{uuid4()}")

        assert response.status_code == 404

    async def test_delete_is_scoped_by_owner(
        self, client: AsyncClient, api_prefix: str, db_session: AsyncSession
    ) -> None:
        document = await _seed(db_session, owner_id="someone-else")

        response = await client.delete(f"{api_prefix}/documents/{document.id}")

        assert response.status_code == 404


class TestCors:
    async def test_allowed_origin_receives_cors_header(
        self, client: AsyncClient, api_prefix: str
    ) -> None:
        """The frontend dev server runs on a different port, so this must work."""
        response = await client.get(
            f"{api_prefix}/health", headers={"Origin": "http://localhost:5173"}
        )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://localhost:5173"

    async def test_unlisted_origin_is_not_granted_access(
        self, client: AsyncClient, api_prefix: str
    ) -> None:
        """Confirms the allowlist is real and CORS is not effectively '*'."""
        response = await client.get(
            f"{api_prefix}/health", headers={"Origin": "http://evil.example"}
        )

        assert "access-control-allow-origin" not in response.headers


class TestErrorEnvelope:
    async def test_unhandled_exception_does_not_leak_internals(
        self, app: FastAPI, api_prefix: str
    ) -> None:
        """A stack trace in an HTTP response is an information leak. The traceback goes
        to the log; the client gets a generic message."""
        from app.api.deps import get_document_service

        def _explode() -> None:
            raise RuntimeError("database password is hunter2")

        app.dependency_overrides[get_document_service] = _explode
        try:
            transport_client = AsyncClient(
                transport=__import__("httpx").ASGITransport(app=app, raise_app_exceptions=False),
                base_url="http://testserver",
            )
            async with transport_client as local_client:
                response = await local_client.get(f"{api_prefix}/documents")
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 500
        body = response.json()
        assert body["error"]["code"] == "internal_error"
        assert "hunter2" not in response.text
        assert "RuntimeError" not in response.text
