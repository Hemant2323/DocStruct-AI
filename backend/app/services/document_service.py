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

from app.core.config import Settings
from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.core.security import sanitize_filename
from app.ingestion.detection import ArchiveLimits, check_archive_safety, detect_file_type
from app.models.domain.content import ContentBlock
from app.models.domain.document import Document, ProcessingStatus, SupportedFileType
from app.repositories.block_repo import ContentBlockRepository
from app.repositories.chunk_repo import Chunk, ChunkRepository
from app.repositories.document_repo import DocumentRepository
from app.storage.base import FileStorage

logger = get_logger(__name__)


class DocumentService:
    """Coordinates document metadata and stored content.

    Args:
        repository: Database access for document records.
        storage: Where the raw file bytes live.
    """

    def __init__(
        self,
        repository: DocumentRepository,
        storage: FileStorage,
        settings: Settings,
        block_repository: ContentBlockRepository | None = None,
        chunk_repository: ChunkRepository | None = None,
    ) -> None:
        self._repository = repository
        self._storage = storage
        self._settings = settings
        self._block_repository = block_repository
        self._chunk_repository = chunk_repository

    async def create_from_upload(
        self,
        *,
        owner_id: str,
        filename: str | None,
        content: bytes,
        size_bytes: int,
        content_hash: str,
    ) -> Document:
        """Accept an uploaded file, store it, and record it as pending.

        Validation happens before anything is written, so a rejected file leaves no
        trace. The bytes are stored *before* the database row is created: a row pointing
        at a missing file is a broken state nothing can repair, whereas a stored file
        with no row is a harmless orphan.

        Raises:
            UnsupportedFileTypeError: The content is not a supported format.
            ValidationError: The file is empty, damaged, or unsafe to expand.
        """
        display_name = sanitize_filename(filename)

        # Content decides the type; the filename is only a hint for the CSV/TXT tie.
        file_type = detect_file_type(content, filename=display_name)

        if file_type in (SupportedFileType.DOCX, SupportedFileType.XLSX):
            check_archive_safety(
                content,
                ArchiveLimits(
                    max_uncompressed_bytes=self._settings.max_archive_uncompressed_mb * 1024 * 1024,
                    max_entries=self._settings.max_archive_entries,
                ),
            )

        document = await self._repository.create(
            owner_id=owner_id,
            original_filename=display_name,
            file_type=file_type,
            size_bytes=size_bytes,
            status=ProcessingStatus.PENDING,
            content_hash=content_hash,
        )

        try:
            await self._storage.save(document.id, content)
        except Exception:
            # The row is only useful if its bytes exist. The caller's transaction is
            # rolled back by the session dependency when this propagates.
            logger.error("Failed to store uploaded file document_id=%s", document.id)
            raise

        logger.info(
            "Accepted upload document_id=%s type=%s size=%d",
            document.id,
            file_type.value,
            size_bytes,
        )
        return document

    async def get_content_blocks(
        self, document_id: UUID, *, owner_id: str, limit: int = 200, offset: int = 0
    ) -> tuple[list[ContentBlock], int]:
        """Return a page of a document's extracted blocks, newest-first by position.

        Raises:
            NotFoundError: If the document does not exist or belongs to another owner.
        """
        await self.get_document(document_id, owner_id=owner_id)

        if self._block_repository is None:  # pragma: no cover - wiring guard
            raise RuntimeError("DocumentService was constructed without a block repository")

        blocks = await self._block_repository.list_for_document(
            document_id, limit=limit, offset=offset
        )
        total = await self._block_repository.count_for_document(document_id)
        return blocks, total

    async def get_chunks(self, document_id: UUID, *, owner_id: str) -> list[Chunk]:
        """Return a document's retrieval chunks.

        Raises:
            NotFoundError: If the document does not exist or belongs to another owner.
        """
        await self.get_document(document_id, owner_id=owner_id)

        if self._chunk_repository is None:  # pragma: no cover - wiring guard
            raise RuntimeError("DocumentService was constructed without a chunk repository")

        return await self._chunk_repository.list_for_document(document_id)

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
