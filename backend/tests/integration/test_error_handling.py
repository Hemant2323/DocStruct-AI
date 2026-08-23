"""Tests for the error contract.

These pin down two promises that are easy to break by accident:

* a client never sees a stack trace, exception type, or internal detail;
* the server log still holds everything needed to debug, tied to the response by a
  correlation ID.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from app.api.deps import get_document_service
from app.api.middleware import REQUEST_ID_HEADER
from app.core.exceptions import NotFoundError, StorageError

SECRET = "postgresql://admin:hunter2@db.internal:5432/docstruct"


@pytest.fixture
async def failing_client(app: FastAPI) -> AsyncClient:
    """A client whose document service raises, to exercise the 500 path.

    ``raise_app_exceptions=False`` makes the transport return the error response the app
    produced instead of re-raising, which is what a real HTTP client would see.
    """
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    return AsyncClient(transport=transport, base_url="http://testserver")


class TestRequestId:
    async def test_every_response_carries_a_request_id_header(
        self, client: AsyncClient, api_prefix: str
    ) -> None:
        response = await client.get(f"{api_prefix}/health")

        assert response.status_code == 200
        assert response.headers.get(REQUEST_ID_HEADER)

    async def test_request_ids_are_unique_per_request(
        self, client: AsyncClient, api_prefix: str
    ) -> None:
        first = await client.get(f"{api_prefix}/health")
        second = await client.get(f"{api_prefix}/health")

        assert first.headers[REQUEST_ID_HEADER] != second.headers[REQUEST_ID_HEADER]

    async def test_client_supplied_request_id_is_not_trusted(
        self, client: AsyncClient, api_prefix: str
    ) -> None:
        """Echoing a caller's ID back would let anyone forge or collide log entries."""
        forged = "forged-request-id"

        response = await client.get(f"{api_prefix}/health", headers={REQUEST_ID_HEADER: forged})

        assert response.headers[REQUEST_ID_HEADER] != forged

    async def test_error_body_request_id_matches_the_header(
        self, client: AsyncClient, api_prefix: str
    ) -> None:
        """The whole point: the ID a user can quote is the ID in the log."""
        response = await client.get(f"{api_prefix}/documents/not-a-uuid")

        assert response.status_code == 422
        assert response.json()["error"]["request_id"] == response.headers[REQUEST_ID_HEADER]


class TestErrorEnvelopeShape:
    @pytest.mark.parametrize(
        ("path", "expected_status"),
        [
            ("/documents/not-a-uuid", 422),
            ("/documents/0b6c1f3e-6a4d-4c58-9a1f-2b7d8e5c4a90", 404),
            ("/does-not-exist", 404),
        ],
    )
    async def test_all_errors_share_one_envelope(
        self, client: AsyncClient, api_prefix: str, path: str, expected_status: int
    ) -> None:
        """One shape means the frontend needs one error path, not one per endpoint."""
        response = await client.get(f"{api_prefix}{path}")

        assert response.status_code == expected_status
        body = response.json()
        assert set(body) == {"error"}
        assert set(body["error"]) == {"code", "message", "details", "request_id"}
        assert isinstance(body["error"]["code"], str)
        assert isinstance(body["error"]["message"], str)

    async def test_validation_errors_name_the_field_but_not_the_value(
        self, client: AsyncClient, api_prefix: str
    ) -> None:
        """A rejected value can be a credential pasted into the wrong box."""
        response = await client.get(f"{api_prefix}/documents", params={"limit": 9999})

        assert response.status_code == 422
        fields = response.json()["error"]["details"]["fields"]
        assert fields
        assert all(set(field) == {"field", "message"} for field in fields)
        assert "9999" not in response.text


class TestUnexpectedErrorsAreSafe:
    async def test_internals_never_reach_the_client(
        self, app: FastAPI, failing_client: AsyncClient, api_prefix: str
    ) -> None:
        def _explode() -> None:
            raise RuntimeError(f"connection failed for {SECRET}")

        app.dependency_overrides[get_document_service] = _explode
        try:
            async with failing_client as http_client:
                response = await http_client.get(f"{api_prefix}/documents")
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 500
        body = response.json()

        assert body["error"]["code"] == "internal_error"
        # No secret, no exception type, no traceback, no source path.
        assert "hunter2" not in response.text
        assert SECRET not in response.text
        assert "RuntimeError" not in response.text
        assert "Traceback" not in response.text
        assert ".py" not in response.text

    async def test_unexpected_error_still_returns_a_usable_message_and_id(
        self, app: FastAPI, failing_client: AsyncClient, api_prefix: str
    ) -> None:
        """Safe must not mean useless: the user gets something to say and to quote."""

        def _explode() -> None:
            raise RuntimeError("internal detail")

        app.dependency_overrides[get_document_service] = _explode
        try:
            async with failing_client as http_client:
                response = await http_client.get(f"{api_prefix}/documents")
        finally:
            app.dependency_overrides.clear()

        error = response.json()["error"]
        assert error["message"]
        assert error["request_id"]
        assert response.headers[REQUEST_ID_HEADER] == error["request_id"]

    async def test_traceback_is_written_to_the_log(
        self,
        app: FastAPI,
        failing_client: AsyncClient,
        api_prefix: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The detail withheld from the client must be present in the log."""

        def _explode() -> None:
            raise RuntimeError("internal detail worth debugging")

        app.dependency_overrides[get_document_service] = _explode
        try:
            with caplog.at_level("ERROR"):
                async with failing_client as http_client:
                    response = await http_client.get(f"{api_prefix}/documents")
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 500
        assert "internal detail worth debugging" in caplog.text
        assert "RuntimeError" in caplog.text

    async def test_server_side_domain_errors_are_also_generic(
        self, app: FastAPI, failing_client: AsyncClient, api_prefix: str
    ) -> None:
        """A 5xx domain error is still an internal failure — no internal wording out."""

        def _explode() -> None:
            raise StorageError(
                "Failed to write document to /var/lib/docstruct/storage",
                details={"path": "/var/lib/docstruct/storage"},
            )

        app.dependency_overrides[get_document_service] = _explode
        try:
            async with failing_client as http_client:
                response = await http_client.get(f"{api_prefix}/documents")
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 500
        assert "/var/lib/docstruct" not in response.text
        # The code is still specific enough for a client to branch on.
        assert response.json()["error"]["code"] == "storage_error"


class TestActionableServerErrorsKeepTheirMessage:
    """A few 5xx errors carry a message that is safe AND useful.

    The generic-message rule exists to stop internal detail escaping. "AI features are
    not configured" leaks nothing and is something the operator can act on, so hiding it
    behind "something went wrong" would be unhelpful rather than safe.
    """

    async def test_a_safe_actionable_message_is_shown(
        self, app: FastAPI, failing_client: AsyncClient, api_prefix: str
    ) -> None:
        from app.core.exceptions import AIConfigurationError

        def _explode() -> None:
            raise AIConfigurationError(
                "AI features are not configured on this server. An Anthropic API key is required."
            )

        app.dependency_overrides[get_document_service] = _explode
        try:
            async with failing_client as http_client:
                response = await http_client.get(f"{api_prefix}/documents")
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 500
        body = response.json()
        assert body["error"]["code"] == "ai_not_configured"
        assert "not configured" in body["error"]["message"]

    async def test_ordinary_server_errors_stay_generic(
        self, app: FastAPI, failing_client: AsyncClient, api_prefix: str
    ) -> None:
        """The default is unchanged: opting in is per-error and deliberate."""

        def _explode() -> None:
            raise StorageError("Failed to write to /var/lib/docstruct/storage")

        app.dependency_overrides[get_document_service] = _explode
        try:
            async with failing_client as http_client:
                response = await http_client.get(f"{api_prefix}/documents")
        finally:
            app.dependency_overrides.clear()

        assert "/var/lib/docstruct" not in response.text


class TestClientErrorsStayInformative:
    async def test_4xx_messages_are_preserved(
        self, app: FastAPI, failing_client: AsyncClient, api_prefix: str
    ) -> None:
        """Client errors are the caller's business, so they keep their real message."""

        def _explode() -> None:
            raise NotFoundError("Document not found", details={"document_id": "abc"})

        app.dependency_overrides[get_document_service] = _explode
        try:
            async with failing_client as http_client:
                response = await http_client.get(f"{api_prefix}/documents")
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 404
        body = response.json()
        assert body["error"]["message"] == "Document not found"
        assert body["error"]["details"] == {"document_id": "abc"}
