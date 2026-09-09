"""The file storage interface.

One of the project's extension seams. Milestone 1 ships a local-disk implementation;
swapping in S3 or Azure Blob later means writing one class, not editing service code.

Files are addressed by the document's UUID, never by a user-supplied filename. That
single rule removes path traversal and filename collisions at the same time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID


class FileStorage(ABC):
    """Stores and retrieves the raw bytes of uploaded documents."""

    @abstractmethod
    async def save(self, document_id: UUID, content: bytes) -> int:
        """Persist a document's bytes.

        Args:
            document_id: Identifier the content is stored under.
            content: Raw file bytes.

        Returns:
            The number of bytes written.

        Raises:
            StorageError: If the write fails.
        """

    @abstractmethod
    async def load(self, document_id: UUID) -> bytes:
        """Read a document's bytes back.

        Raises:
            NotFoundError: If nothing is stored under this identifier.
            StorageError: If the read fails.
        """

    @abstractmethod
    async def delete(self, document_id: UUID) -> None:
        """Remove a document's bytes.

        Deleting something that is already absent succeeds — the desired end state is
        "not present", and it has been reached.

        Raises:
            StorageError: If the deletion fails for any other reason.
        """

    @abstractmethod
    async def exists(self, document_id: UUID) -> bool:
        """Return whether content is stored under this identifier."""
