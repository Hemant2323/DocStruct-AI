"""End-to-end tests for the complete ingestion pipeline.

Upload over real HTTP, let the background task run, then assert on what landed in the
database. Nothing is mocked: these exercise the parsers, normalizer, chunker, background
scheduling and persistence together, which is the only way to catch the wiring mistakes
that unit tests with fakes cannot see.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import DEFAULT_OWNER_ID
from app.models.domain.document import ProcessingStatus, SupportedFileType
from app.repositories.document_repo import DocumentRepository
from app.services.job_service import JobService
from tests.fixtures.builders import (
    build_csv,
    build_docx,
    build_txt,
    build_xlsx,
    build_zip_bomb,
    read_pdf_fixture,
)

MEDIA = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain",
    "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


async def _upload(
    client: AsyncClient, api_prefix: str, filename: str, content: bytes, media: str
) -> dict[str, Any]:
    """Upload a file and return the parsed response body."""
    response = await client.post(
        f"{api_prefix}/documents",
        files={"file": (filename, content, media)},
    )
    return {
        "status_code": response.status_code,
        "body": response.json() if response.content else None,
    }


async def _ingest(
    client: AsyncClient,
    api_prefix: str,
    jobs: JobService,
    filename: str,
    content: bytes,
    media: str,
) -> dict[str, Any]:
    """Upload, wait for background processing, and return the finished document."""
    result = await _upload(client, api_prefix, filename, content, media)
    assert result["status_code"] == 202, result["body"]

    # Deterministic: wait for the actual task rather than sleeping and hoping.
    await jobs.wait_for_all(timeout=30.0)

    document_id = result["body"]["id"]
    response = await client.get(f"{api_prefix}/documents/{document_id}")
    assert response.status_code == 200
    return response.json()


class TestUploadContract:
    async def test_upload_returns_202_and_a_pending_document(
        self, client: AsyncClient, api_prefix: str
    ) -> None:
        """202 says 'accepted, not finished' — the response cannot wait for parsing."""
        result = await _upload(
            client, api_prefix, "notes.txt", build_txt("Hello there."), MEDIA["txt"]
        )

        assert result["status_code"] == 202
        body = result["body"]
        assert body["status"] == ProcessingStatus.PENDING.value
        assert body["original_filename"] == "notes.txt"
        assert body["file_type"] == "txt"
        assert body["size_bytes"] > 0

    async def test_uploaded_document_appears_in_the_list(
        self, client: AsyncClient, api_prefix: str
    ) -> None:
        await _upload(client, api_prefix, "a.txt", build_txt("Content."), MEDIA["txt"])

        response = await client.get(f"{api_prefix}/documents")

        assert response.json()["total"] == 1

    async def test_missing_file_field_is_rejected(
        self, client: AsyncClient, api_prefix: str
    ) -> None:
        response = await client.post(f"{api_prefix}/documents")

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"


class TestEveryFormatIngestsEndToEnd:
    @pytest.mark.parametrize(
        ("filename", "media_key"),
        [
            ("report.pdf", "pdf"),
            ("contract.docx", "docx"),
            ("notes.txt", "txt"),
            ("data.csv", "csv"),
            ("budget.xlsx", "xlsx"),
        ],
    )
    async def test_document_reaches_ready_with_content(
        self,
        client: AsyncClient,
        api_prefix: str,
        job_service: JobService,
        filename: str,
        media_key: str,
    ) -> None:
        """The headline test: every supported format goes upload -> ready -> content."""
        contents = {
            "pdf": read_pdf_fixture("simple.pdf"),
            "docx": build_docx(
                [("Heading 1", "Payment Terms"), ("Normal", "Net 30 days applies.")]
            ),
            "txt": build_txt("First paragraph.\n\nSecond paragraph."),
            "csv": build_csv([["name", "age"], ["Alice", "30"], ["Bob", "25"]]),
            "xlsx": build_xlsx({"Sales": [["region", "amount"], ["North", 100]]}),
        }

        document = await _ingest(
            client, api_prefix, job_service, filename, contents[media_key], MEDIA[media_key]
        )

        assert document["status"] == ProcessingStatus.READY.value, document
        assert document["error_message"] is None
        assert document["failure_code"] is None
        assert document["processed_at"] is not None

        content = await client.get(f"{api_prefix}/documents/{document['id']}/content")
        blocks = content.json()["items"]

        assert blocks, f"{filename} produced no blocks"
        assert all(block["text"].strip() for block in blocks)
        assert all(block["location_label"].strip() for block in blocks)

    async def test_file_type_comes_from_content_not_the_name(
        self, client: AsyncClient, api_prefix: str, job_service: JobService
    ) -> None:
        """A Word document named .pdf is still processed as a Word document."""
        content = build_docx([("Normal", "Actually a Word document.")])

        document = await _ingest(
            client, api_prefix, job_service, "misnamed.pdf", content, MEDIA["pdf"]
        )

        assert document["file_type"] == SupportedFileType.DOCX.value
        assert document["status"] == ProcessingStatus.READY.value


class TestProvenanceSurvivesTheWholePipeline:
    async def test_pdf_page_numbers_reach_the_api(
        self, client: AsyncClient, api_prefix: str, job_service: JobService
    ) -> None:
        """The citation chain, verified end to end.

        Text placed on page 2 of the fixture must still report page 2 after parsing,
        normalization, chunking and storage. If any stage dropped provenance, this fails.
        """
        document = await _ingest(
            client,
            api_prefix,
            job_service,
            "report.pdf",
            read_pdf_fixture("simple.pdf"),
            MEDIA["pdf"],
        )

        content = await client.get(f"{api_prefix}/documents/{document['id']}/content")
        blocks = content.json()["items"]

        marker = next(b for b in blocks if "PAGE TWO MARKER" in b["text"])
        assert marker["location_label"] == "page 2"

        first = blocks[0]
        assert first["location_label"] == "page 1"

    async def test_csv_row_ranges_reach_the_api(
        self, client: AsyncClient, api_prefix: str, job_service: JobService
    ) -> None:
        rows = [["name", "age"], *[[f"person{i}", str(20 + i)] for i in range(60)]]

        document = await _ingest(
            client, api_prefix, job_service, "people.csv", build_csv(rows), MEDIA["csv"]
        )

        content = await client.get(f"{api_prefix}/documents/{document['id']}/content")
        blocks = content.json()["items"]

        row_blocks = [b for b in blocks if b["block_type"] == "sheet_row"]
        assert len(row_blocks) > 1
        assert all("row" in b["location_label"] for b in row_blocks)

    async def test_xlsx_sheet_and_cell_ranges_reach_the_api(
        self, client: AsyncClient, api_prefix: str, job_service: JobService
    ) -> None:
        content_bytes = build_xlsx({"Q3 Budget": [["item", "cost"], ["Rent", 500]]})

        document = await _ingest(
            client, api_prefix, job_service, "b.xlsx", content_bytes, MEDIA["xlsx"]
        )

        content = await client.get(f"{api_prefix}/documents/{document['id']}/content")
        labels = [b["location_label"] for b in content.json()["items"]]

        assert any(label.startswith("Q3 Budget!") for label in labels)

    async def test_inferred_locations_are_flagged_as_approximate(
        self, client: AsyncClient, api_prefix: str, job_service: JobService
    ) -> None:
        """A PDF heading is inferred from font size; the API must say so."""
        document = await _ingest(
            client,
            api_prefix,
            job_service,
            "report.pdf",
            read_pdf_fixture("simple.pdf"),
            MEDIA["pdf"],
        )

        content = await client.get(f"{api_prefix}/documents/{document['id']}/content")
        headings = [b for b in content.json()["items"] if b["block_type"] == "heading"]

        assert headings
        assert all(heading["is_exact"] is False for heading in headings)


class TestFailurePaths:
    async def test_scanned_pdf_fails_with_a_useful_reason(
        self, client: AsyncClient, api_prefix: str, job_service: JobService
    ) -> None:
        """The most important failure: never silently return an empty document."""
        document = await _ingest(
            client,
            api_prefix,
            job_service,
            "scan.pdf",
            read_pdf_fixture("scanned.pdf"),
            MEDIA["pdf"],
        )

        assert document["status"] == ProcessingStatus.FAILED.value
        assert document["failure_code"] == "requires_ocr"
        assert "OCR" in document["error_message"]
        assert document["processed_at"] is not None

    async def test_a_failed_document_always_explains_itself(
        self, client: AsyncClient, api_prefix: str, job_service: JobService
    ) -> None:
        document = await _ingest(
            client,
            api_prefix,
            job_service,
            "scan.pdf",
            read_pdf_fixture("scanned.pdf"),
            MEDIA["pdf"],
        )

        assert document["error_message"]
        assert document["failure_code"]

    async def test_failure_message_leaks_no_internals(
        self, client: AsyncClient, api_prefix: str, job_service: JobService
    ) -> None:
        """The message is shown to the user, so no paths, libraries, or tracebacks."""
        document = await _ingest(
            client,
            api_prefix,
            job_service,
            "scan.pdf",
            read_pdf_fixture("scanned.pdf"),
            MEDIA["pdf"],
        )

        message = document["error_message"]
        for leak in ("Traceback", ".py", "pdfplumber", "pdfminer", "site-packages"):
            assert leak not in message

    async def test_unsupported_type_is_rejected_at_upload(
        self, client: AsyncClient, api_prefix: str
    ) -> None:
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200

        result = await _upload(client, api_prefix, "image.png", png, "image/png")

        assert result["status_code"] == 415
        assert result["body"]["error"]["code"] == "unsupported_file_type"

    async def test_empty_file_is_rejected_at_upload(
        self, client: AsyncClient, api_prefix: str
    ) -> None:
        result = await _upload(client, api_prefix, "empty.txt", b"", MEDIA["txt"])

        assert result["status_code"] == 422

    async def test_oversized_file_is_rejected(
        self, client: AsyncClient, api_prefix: str, settings: Any
    ) -> None:
        oversized = b"a" * (settings.max_upload_size_bytes + 1024)

        result = await _upload(client, api_prefix, "big.txt", oversized, MEDIA["txt"])

        assert result["status_code"] == 413
        assert result["body"]["error"]["code"] == "file_too_large"

    async def test_zip_bomb_is_rejected(self, client: AsyncClient, api_prefix: str) -> None:
        bomb = build_zip_bomb(entries=4, uncompressed_mb=400)

        result = await _upload(client, api_prefix, "bomb.docx", bomb, MEDIA["docx"])

        assert result["status_code"] == 422

    async def test_rejected_upload_creates_no_document(
        self, client: AsyncClient, api_prefix: str
    ) -> None:
        """A failed validation must leave no trace behind."""
        await _upload(
            client, api_prefix, "image.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 200, "image/png"
        )

        response = await client.get(f"{api_prefix}/documents")
        assert response.json()["total"] == 0

    async def test_a_document_never_gets_stuck_in_processing(
        self, client: AsyncClient, api_prefix: str, job_service: JobService
    ) -> None:
        """Whatever goes wrong, the document must reach a terminal state."""
        document = await _ingest(
            client,
            api_prefix,
            job_service,
            "scan.pdf",
            read_pdf_fixture("scanned.pdf"),
            MEDIA["pdf"],
        )

        assert document["status"] in {
            ProcessingStatus.READY.value,
            ProcessingStatus.FAILED.value,
        }


class TestSecurityAtTheBoundary:
    async def test_hostile_filename_is_sanitised(
        self, client: AsyncClient, api_prefix: str
    ) -> None:
        result = await _upload(
            client, api_prefix, "../../etc/passwd.txt", build_txt("Content."), MEDIA["txt"]
        )

        assert result["status_code"] == 202
        stored_name = result["body"]["original_filename"]
        assert "/" not in stored_name
        assert ".." not in stored_name

    async def test_stored_file_is_named_by_uuid(
        self, client: AsyncClient, api_prefix: str, settings: Any
    ) -> None:
        """No user-controlled string ever reaches the filesystem."""
        result = await _upload(
            client, api_prefix, "payroll.txt", build_txt("Content."), MEDIA["txt"]
        )
        document_id = result["body"]["id"]

        files = list(settings.storage_dir.iterdir())

        assert len(files) == 1
        assert files[0].name == f"{document_id}.bin"

    async def test_content_is_scoped_by_owner(
        self, client: AsyncClient, api_prefix: str, job_service: JobService, app: FastAPI
    ) -> None:
        """Another user must not be able to read this document's extracted content."""
        from app.api.deps import get_current_owner_id

        document = await _ingest(
            client, api_prefix, job_service, "a.txt", build_txt("Secret text."), MEDIA["txt"]
        )

        app.dependency_overrides[get_current_owner_id] = lambda: "someone-else"
        try:
            response = await client.get(f"{api_prefix}/documents/{document['id']}/content")
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 404


class TestContentEndpoint:
    async def test_returns_blocks_in_reading_order(
        self, client: AsyncClient, api_prefix: str, job_service: JobService
    ) -> None:
        document = await _ingest(
            client,
            api_prefix,
            job_service,
            "a.txt",
            build_txt("One.\n\nTwo.\n\nThree."),
            MEDIA["txt"],
        )

        response = await client.get(f"{api_prefix}/documents/{document['id']}/content")
        orders = [block["order"] for block in response.json()["items"]]

        assert orders == sorted(orders)

    async def test_pagination(
        self, client: AsyncClient, api_prefix: str, job_service: JobService
    ) -> None:
        text = "\n\n".join(f"Paragraph number {i} with some text." for i in range(20))
        document = await _ingest(
            client, api_prefix, job_service, "a.txt", build_txt(text), MEDIA["txt"]
        )

        response = await client.get(
            f"{api_prefix}/documents/{document['id']}/content",
            params={"limit": 5, "offset": 2},
        )
        body = response.json()

        assert len(body["items"]) <= 5
        assert body["limit"] == 5
        assert body["offset"] == 2
        assert body["total"] >= len(body["items"])

    async def test_unprocessed_document_returns_empty_content(
        self, client: AsyncClient, api_prefix: str, db_session: AsyncSession
    ) -> None:
        """Not an error — the content simply does not exist yet.

        The document is seeded directly rather than uploaded: an upload schedules its
        background work immediately, so racing that task would make this test flaky.
        """
        document = await DocumentRepository(db_session).create(
            owner_id=DEFAULT_OWNER_ID,
            original_filename="pending.txt",
            file_type=SupportedFileType.TXT,
            size_bytes=10,
        )
        await db_session.commit()

        response = await client.get(f"{api_prefix}/documents/{document.id}/content")

        assert response.status_code == 200
        body = response.json()
        assert body["items"] == []
        assert body["total"] == 0

    async def test_unknown_document_returns_404(self, client: AsyncClient, api_prefix: str) -> None:
        from uuid import uuid4

        response = await client.get(f"{api_prefix}/documents/{uuid4()}/content")

        assert response.status_code == 404


class TestDeletionRemovesEverything:
    async def test_deleting_removes_content_and_file(
        self,
        client: AsyncClient,
        api_prefix: str,
        job_service: JobService,
        settings: Any,
    ) -> None:
        document = await _ingest(
            client, api_prefix, job_service, "a.txt", build_txt("Content here."), MEDIA["txt"]
        )

        response = await client.delete(f"{api_prefix}/documents/{document['id']}")
        assert response.status_code == 204

        # The record, its extracted content, and the stored bytes are all gone.
        assert (await client.get(f"{api_prefix}/documents/{document['id']}")).status_code == 404
        assert list(settings.storage_dir.iterdir()) == []
