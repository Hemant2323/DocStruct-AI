"""Tests for the document repository against a real database.

These run real SQL rather than mocks. Owner scoping is an access-control boundary, and a
mock would happily confirm behaviour the database does not actually have.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.models.domain.document import Document, ProcessingStatus, SupportedFileType
from app.repositories.document_repo import DocumentRepository

OWNER = "owner-a"
OTHER_OWNER = "owner-b"


async def _make_document(
    repository: DocumentRepository,
    *,
    owner_id: str = OWNER,
    filename: str = "report.pdf",
    file_type: SupportedFileType = SupportedFileType.PDF,
    size_bytes: int = 1024,
    status: ProcessingStatus = ProcessingStatus.PENDING,
) -> Document:
    return await repository.create(
        owner_id=owner_id,
        original_filename=filename,
        file_type=file_type,
        size_bytes=size_bytes,
        status=status,
    )


class TestCreate:
    async def test_create_populates_generated_fields(
        self, document_repository: DocumentRepository
    ) -> None:
        document = await document_repository.create(
            owner_id=OWNER,
            original_filename="contract.docx",
            file_type=SupportedFileType.DOCX,
            size_bytes=2048,
        )

        assert document.id is not None
        assert document.owner_id == OWNER
        assert document.original_filename == "contract.docx"
        assert document.file_type is SupportedFileType.DOCX
        assert document.size_bytes == 2048
        assert document.status is ProcessingStatus.PENDING
        assert document.error_message is None
        assert document.created_at is not None
        assert document.updated_at is not None

    async def test_timestamps_are_timezone_aware(
        self, document_repository: DocumentRepository
    ) -> None:
        """SQLite returns naive datetimes; the custom column type must re-attach UTC."""
        document = await _make_document(document_repository)

        assert document.created_at.tzinfo is not None
        assert document.updated_at.tzinfo is not None

    async def test_ids_are_unique(self, document_repository: DocumentRepository) -> None:
        first = await _make_document(document_repository)
        second = await _make_document(document_repository)

        assert first.id != second.id


class TestGet:
    async def test_get_returns_the_document(self, document_repository: DocumentRepository) -> None:
        created = await _make_document(document_repository)

        fetched = await document_repository.get(created.id, owner_id=OWNER)

        assert fetched is not None
        assert fetched.id == created.id

    async def test_get_unknown_id_returns_none(
        self, document_repository: DocumentRepository
    ) -> None:
        assert await document_repository.get(uuid4(), owner_id=OWNER) is None

    async def test_get_is_scoped_by_owner(self, document_repository: DocumentRepository) -> None:
        """The core multi-tenancy guarantee: another owner cannot read this document."""
        created = await _make_document(document_repository, owner_id=OWNER)

        assert await document_repository.get(created.id, owner_id=OTHER_OWNER) is None


class TestList:
    async def test_list_returns_only_the_owners_documents(
        self, document_repository: DocumentRepository
    ) -> None:
        await _make_document(document_repository, owner_id=OWNER)
        await _make_document(document_repository, owner_id=OWNER)
        await _make_document(document_repository, owner_id=OTHER_OWNER)

        mine = await document_repository.list_for_owner(owner_id=OWNER)
        theirs = await document_repository.list_for_owner(owner_id=OTHER_OWNER)

        assert len(mine) == 2
        assert len(theirs) == 1
        assert all(document.owner_id == OWNER for document in mine)

    async def test_list_is_newest_first(self, document_repository: DocumentRepository) -> None:
        for name in ("first.pdf", "second.pdf", "third.pdf"):
            await _make_document(document_repository, filename=name)

        documents = await document_repository.list_for_owner(owner_id=OWNER)

        timestamps = [document.created_at for document in documents]
        assert timestamps == sorted(timestamps, reverse=True)

    async def test_list_pagination(self, document_repository: DocumentRepository) -> None:
        for index in range(5):
            await _make_document(document_repository, filename=f"doc{index}.pdf")

        page_one = await document_repository.list_for_owner(owner_id=OWNER, limit=2, offset=0)
        page_two = await document_repository.list_for_owner(owner_id=OWNER, limit=2, offset=2)

        assert len(page_one) == 2
        assert len(page_two) == 2
        assert {d.id for d in page_one}.isdisjoint({d.id for d in page_two})

    async def test_list_filters_by_status(self, document_repository: DocumentRepository) -> None:
        await _make_document(document_repository, status=ProcessingStatus.PENDING)
        await _make_document(document_repository, status=ProcessingStatus.READY)
        await _make_document(document_repository, status=ProcessingStatus.READY)

        ready = await document_repository.list_for_owner(
            owner_id=OWNER, status=ProcessingStatus.READY
        )

        assert len(ready) == 2
        assert all(d.status is ProcessingStatus.READY for d in ready)

    async def test_list_empty_for_new_owner(self, document_repository: DocumentRepository) -> None:
        assert await document_repository.list_for_owner(owner_id="nobody") == []


class TestCount:
    async def test_count_is_scoped_by_owner(self, document_repository: DocumentRepository) -> None:
        await _make_document(document_repository, owner_id=OWNER)
        await _make_document(document_repository, owner_id=OTHER_OWNER)

        assert await document_repository.count_for_owner(owner_id=OWNER) == 1

    async def test_count_ignores_pagination(self, document_repository: DocumentRepository) -> None:
        """The total must reflect every match, not just the current page."""
        for index in range(7):
            await _make_document(document_repository, filename=f"doc{index}.pdf")

        assert await document_repository.count_for_owner(owner_id=OWNER) == 7

    async def test_count_filters_by_status(self, document_repository: DocumentRepository) -> None:
        await _make_document(document_repository, status=ProcessingStatus.PENDING)
        await _make_document(document_repository, status=ProcessingStatus.FAILED)

        count = await document_repository.count_for_owner(
            owner_id=OWNER, status=ProcessingStatus.FAILED
        )
        assert count == 1


class TestUpdateStatus:
    async def test_status_transition(self, document_repository: DocumentRepository) -> None:
        created = await _make_document(document_repository)

        updated = await document_repository.update_status(
            created.id,
            owner_id=OWNER,
            status=ProcessingStatus.PROCESSING,
        )

        assert updated is not None
        assert updated.status is ProcessingStatus.PROCESSING

    async def test_failure_records_the_reason(
        self, document_repository: DocumentRepository
    ) -> None:
        created = await _make_document(document_repository)

        updated = await document_repository.update_status(
            created.id,
            owner_id=OWNER,
            status=ProcessingStatus.FAILED,
            error_message="Encrypted PDF cannot be read",
        )

        assert updated is not None
        assert updated.status is ProcessingStatus.FAILED
        assert updated.error_message == "Encrypted PDF cannot be read"

    async def test_failure_without_a_reason_is_rejected(
        self, document_repository: DocumentRepository
    ) -> None:
        """A failed document must always be able to say why — no silent failures."""
        created = await _make_document(document_repository)

        with pytest.raises(ValueError, match="error_message is required"):
            await document_repository.update_status(
                created.id,
                owner_id=OWNER,
                status=ProcessingStatus.FAILED,
            )

    async def test_recovering_from_failure_clears_the_error(
        self, document_repository: DocumentRepository
    ) -> None:
        """Stale error text must not linger on a document that later succeeded."""
        created = await _make_document(document_repository)
        await document_repository.update_status(
            created.id,
            owner_id=OWNER,
            status=ProcessingStatus.FAILED,
            error_message="Temporary failure",
        )

        recovered = await document_repository.update_status(
            created.id,
            owner_id=OWNER,
            status=ProcessingStatus.READY,
        )

        assert recovered is not None
        assert recovered.error_message is None

    async def test_update_is_scoped_by_owner(self, document_repository: DocumentRepository) -> None:
        created = await _make_document(document_repository, owner_id=OWNER)

        result = await document_repository.update_status(
            created.id,
            owner_id=OTHER_OWNER,
            status=ProcessingStatus.READY,
        )

        assert result is None

    async def test_update_unknown_id_returns_none(
        self, document_repository: DocumentRepository
    ) -> None:
        result = await document_repository.update_status(
            uuid4(), owner_id=OWNER, status=ProcessingStatus.READY
        )
        assert result is None


class TestDelete:
    async def test_delete_removes_the_row(self, document_repository: DocumentRepository) -> None:
        created = await _make_document(document_repository)

        assert await document_repository.delete(created.id, owner_id=OWNER) is True
        assert await document_repository.get(created.id, owner_id=OWNER) is None

    async def test_delete_unknown_id_returns_false(
        self, document_repository: DocumentRepository
    ) -> None:
        assert await document_repository.delete(uuid4(), owner_id=OWNER) is False

    async def test_delete_is_scoped_by_owner(self, document_repository: DocumentRepository) -> None:
        created = await _make_document(document_repository, owner_id=OWNER)

        assert await document_repository.delete(created.id, owner_id=OTHER_OWNER) is False
        assert await document_repository.get(created.id, owner_id=OWNER) is not None
