"""Local filesystem implementation of :class:`FileStorage`.

Files are written to a configured directory under their document UUID. Disk I/O is run in
a worker thread so the event loop is never blocked by a large file.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID

from app.core.exceptions import NotFoundError, StorageError
from app.core.logging import get_logger
from app.storage.base import FileStorage

logger = get_logger(__name__)


class LocalFileStorage(FileStorage):
    """Stores document bytes on the local filesystem.

    Args:
        root: Directory to store files in. Created if it does not exist.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """The resolved storage directory."""
        return self._root

    def _path_for(self, document_id: UUID) -> Path:
        """Return the on-disk path for a document.

        The filename is derived solely from the UUID, so no user-controlled string ever
        reaches the filesystem. ``UUID`` is validated by its own constructor, which makes
        traversal sequences such as ``../`` unrepresentable rather than merely filtered.
        """
        return self._root / f"{document_id}.bin"

    async def save(self, document_id: UUID, content: bytes) -> int:
        path = self._path_for(document_id)
        try:
            await asyncio.to_thread(path.write_bytes, content)
        except OSError as exc:
            raise StorageError(
                "Failed to write document to storage",
                details={"document_id": str(document_id)},
            ) from exc

        logger.info("Stored document bytes=%d document_id=%s", len(content), document_id)
        return len(content)

    async def load(self, document_id: UUID) -> bytes:
        path = self._path_for(document_id)
        try:
            return await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError as exc:
            raise NotFoundError(
                "Document content not found in storage",
                details={"document_id": str(document_id)},
            ) from exc
        except OSError as exc:
            raise StorageError(
                "Failed to read document from storage",
                details={"document_id": str(document_id)},
            ) from exc

    async def delete(self, document_id: UUID) -> None:
        path = self._path_for(document_id)
        try:
            # missing_ok: the goal is "not present", which is already true.
            await asyncio.to_thread(path.unlink, True)
        except OSError as exc:
            raise StorageError(
                "Failed to delete document from storage",
                details={"document_id": str(document_id)},
            ) from exc

        logger.info("Deleted document from storage document_id=%s", document_id)

    async def exists(self, document_id: UUID) -> bool:
        return await asyncio.to_thread(self._path_for(document_id).is_file)
