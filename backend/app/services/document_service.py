"""Document orchestration.

The service layer decides *what happens in what order*; it does not know about HTTP and
does not write SQL. It raises typed domain errors, which ``app.api.errors`` turns into
HTTP responses.

Milestone 1 covers the read side of the document lifecycle (list, get, delete). Upload
and the parse/chunk pipeline arrive in Milestone 2 and will slot in here, calling the
storage seam that already exists.
"""

from __future__ import annotations

from uuid import UUID

from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.models.domain.document import Document, ProcessingStatus
from app.repositories.document_repo import DocumentRepository
from app.storage.base import FileStorage

logger = get_logger(__name__)


class DocumentService:
    """Coordinates document metadata and stored content.

    Args:
        repository: Database access for document records.
        storage: Where the raw file bytes live.
    """

    def __init__(self, repository: DocumentRepository, storage: FileStorage) -> None:
        self._repository = repository
        self._storage = storage

    async def get_document(self, document_id: UUID, *, owner_id: str) -> Document:
        """Fetch a single document.

        Raises:
            NotFoundError: If it does not exist or belongs to another owner.
        """
        document = await self._repository.get(document_id, owner_id=owner_id)
        if document is None:
            raise NotFoundError(
                "Document not found",
                details={"document_id": str(document_id)},
            )
        return document

    async def list_documents(
        self,
        *,
        owner_id: str,
        limit: int = 50,
        offset: int = 0,
        status: ProcessingStatus | None = None,
    ) -> tuple[list[Document], int]:
        """List an owner's documents.

        Returns:
            A ``(documents, total)`` pair. ``total`` counts every document matching the
            filter, not just the current page, so the client can paginate.
        """
        documents = await self._repository.list_for_owner(
            owner_id=owner_id, limit=limit, offset=offset, status=status
        )
        total = await self._repository.count_for_owner(owner_id=owner_id, status=status)
        return documents, total

    async def delete_document(self, document_id: UUID, *, owner_id: str) -> None:
        """Delete a document's record and its stored bytes.

        The stored file is removed first. If that fails the error propagates and the
        database row survives, leaving the document visible and retryable. Deleting the
        row first would risk an orphaned file that nothing references and nothing can
        clean up.

        Raises:
            NotFoundError: If the document does not exist or belongs to another owner.
            StorageError: If the stored file could not be removed.
        """
        # Confirms existence and ownership before touching storage.
        await self.get_document(document_id, owner_id=owner_id)

        await self._storage.delete(document_id)

        deleted = await self._repository.delete(document_id, owner_id=owner_id)
        if not deleted:
            raise NotFoundError(
                "Document not found",
                details={"document_id": str(document_id)},
            )

        logger.info("Deleted document document_id=%s owner_id=%s", document_id, owner_id)
