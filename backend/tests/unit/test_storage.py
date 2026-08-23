"""Tests for local file storage, including its failure paths."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from app.core.exceptions import NotFoundError
from app.storage.local import LocalFileStorage


class TestLocalFileStorage:
    async def test_save_and_load_round_trip(self, file_storage: LocalFileStorage) -> None:
        document_id = uuid4()
        content = b"DocStruct AI test content"

        written = await file_storage.save(document_id, content)

        assert written == len(content)
        assert await file_storage.load(document_id) == content

    async def test_save_returns_byte_count(self, file_storage: LocalFileStorage) -> None:
        assert await file_storage.save(uuid4(), b"12345") == 5

    async def test_empty_content_round_trips(self, file_storage: LocalFileStorage) -> None:
        """An empty file is a real case; it must not be confused with a missing file."""
        document_id = uuid4()
        await file_storage.save(document_id, b"")

        assert await file_storage.exists(document_id) is True
        assert await file_storage.load(document_id) == b""

    async def test_binary_content_is_not_corrupted(self, file_storage: LocalFileStorage) -> None:
        """Files are written as bytes, so no encoding/newline translation may occur."""
        document_id = uuid4()
        content = bytes(range(256)) + b"\r\n\x00binary\xff"

        await file_storage.save(document_id, content)

        assert await file_storage.load(document_id) == content

    async def test_load_missing_document_raises_not_found(
        self, file_storage: LocalFileStorage
    ) -> None:
        with pytest.raises(NotFoundError) as exc_info:
            await file_storage.load(uuid4())

        assert exc_info.value.code == "not_found"

    async def test_exists_reflects_state(self, file_storage: LocalFileStorage) -> None:
        document_id = uuid4()

        assert await file_storage.exists(document_id) is False
        await file_storage.save(document_id, b"content")
        assert await file_storage.exists(document_id) is True

    async def test_delete_removes_content(self, file_storage: LocalFileStorage) -> None:
        document_id = uuid4()
        await file_storage.save(document_id, b"content")

        await file_storage.delete(document_id)

        assert await file_storage.exists(document_id) is False

    async def test_delete_is_idempotent(self, file_storage: LocalFileStorage) -> None:
        """Deleting something already absent has reached the desired end state."""
        document_id = uuid4()

        await file_storage.delete(document_id)
        await file_storage.delete(document_id)

        assert await file_storage.exists(document_id) is False

    async def test_overwrite_replaces_content(self, file_storage: LocalFileStorage) -> None:
        document_id = uuid4()
        await file_storage.save(document_id, b"original content here")
        await file_storage.save(document_id, b"short")

        assert await file_storage.load(document_id) == b"short"

    async def test_documents_are_isolated_from_each_other(
        self, file_storage: LocalFileStorage
    ) -> None:
        first, second = uuid4(), uuid4()
        await file_storage.save(first, b"first")
        await file_storage.save(second, b"second")

        assert await file_storage.load(first) == b"first"
        assert await file_storage.load(second) == b"second"

    def test_root_directory_is_created_if_absent(self, tmp_path: Path) -> None:
        target = tmp_path / "does" / "not" / "exist"

        storage = LocalFileStorage(target)

        assert storage.root.is_dir()

    async def test_files_are_named_by_uuid_not_by_user_input(
        self, file_storage: LocalFileStorage
    ) -> None:
        """The security property that makes path traversal unrepresentable.

        The on-disk name is derived only from the UUID, so no user-supplied string can
        influence the path. Nothing is escaped or filtered because nothing user-controlled
        ever reaches the filesystem in the first place.
        """
        document_id = uuid4()
        await file_storage.save(document_id, b"content")

        files = list(file_storage.root.iterdir())

        assert len(files) == 1
        assert files[0].name == f"{document_id}.bin"

    async def test_written_file_stays_inside_the_storage_root(
        self, file_storage: LocalFileStorage
    ) -> None:
        document_id = uuid4()
        await file_storage.save(document_id, b"content")

        written = next(file_storage.root.iterdir()).resolve()

        assert written.parent == file_storage.root
