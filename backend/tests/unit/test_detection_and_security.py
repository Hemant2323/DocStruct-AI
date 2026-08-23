"""Tests for upload security and file type detection.

These cover the boundary where untrusted input first reaches the system, so the emphasis
is on what happens with hostile or malformed input rather than the happy path.
"""

from __future__ import annotations

import pytest

from app.core.exceptions import FileTooLargeError, UnsupportedFileTypeError, ValidationError
from app.core.security import (
    MAX_FILENAME_LENGTH,
    read_upload_within_limit,
    sanitize_filename,
)
from app.ingestion.detection import ArchiveLimits, check_archive_safety, detect_file_type
from app.models.domain.document import SupportedFileType
from tests.fixtures.builders import (
    build_csv,
    build_docx,
    build_txt,
    build_xlsx,
    build_zip_bomb,
    read_pdf_fixture,
)


class _FakeUpload:
    """Minimal stand-in for Starlette's UploadFile."""

    def __init__(self, content: bytes) -> None:
        self._content = content
        self._position = 0

    async def read(self, size: int = -1) -> bytes:
        if size < 0:
            chunk = self._content[self._position :]
            self._position = len(self._content)
            return chunk
        chunk = self._content[self._position : self._position + size]
        self._position += len(chunk)
        return chunk


class TestReadUploadWithinLimit:
    async def test_reads_content_and_reports_size(self) -> None:
        result = await read_upload_within_limit(_FakeUpload(b"hello world"), 1024)

        assert result.content == b"hello world"
        assert result.size_bytes == 11

    async def test_computes_sha256_during_the_read(self) -> None:
        """Hashing while reading means the bytes are only walked once."""
        import hashlib

        content = b"DocStruct AI"
        result = await read_upload_within_limit(_FakeUpload(content), 1024)

        assert result.sha256 == hashlib.sha256(content).hexdigest()

    async def test_content_exactly_at_the_limit_is_accepted(self) -> None:
        content = b"x" * 100
        result = await read_upload_within_limit(_FakeUpload(content), 100)
        assert result.size_bytes == 100

    async def test_one_byte_over_the_limit_is_rejected(self) -> None:
        with pytest.raises(FileTooLargeError):
            await read_upload_within_limit(_FakeUpload(b"x" * 101), 100)

    async def test_large_content_is_rejected_without_buffering_it_all(self) -> None:
        """The point of streaming: a huge upload costs a chunk, not its full size."""
        with pytest.raises(FileTooLargeError):
            await read_upload_within_limit(_FakeUpload(b"x" * (5 * 1024 * 1024)), 1024)

    async def test_empty_upload_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            await read_upload_within_limit(_FakeUpload(b""), 1024)

    async def test_multi_chunk_content_is_reassembled_correctly(self) -> None:
        content = bytes(range(256)) * 1000  # larger than one read chunk

        result = await read_upload_within_limit(_FakeUpload(content), 10 * 1024 * 1024)

        assert result.content == content


class TestSanitizeFilename:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("report.pdf", "report.pdf"),
            ("My Document.docx", "My Document.docx"),
            # Runs of whitespace collapse to one space; leading/trailing are stripped.
            ("  spaced  .txt  ", "spaced .txt"),
        ],
    )
    def test_ordinary_names_survive(self, raw: str, expected: str) -> None:
        assert sanitize_filename(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "   ", "...", "///"])
    def test_unusable_names_become_a_placeholder(self, raw: str | None) -> None:
        assert sanitize_filename(raw) == "untitled"

    def test_path_separators_are_stripped(self) -> None:
        """Files are stored by UUID, so this protects displays and logs, not the disk."""
        assert "/" not in sanitize_filename("../../etc/passwd")
        assert "\\" not in sanitize_filename(r"..\..\windows\system32")

    def test_control_characters_are_stripped(self) -> None:
        """A newline in a filename could forge a second line in the log."""
        cleaned = sanitize_filename("report\n2026-01-01 ERROR fake log line.pdf")

        assert "\n" not in cleaned
        assert "\r" not in cleaned

    def test_null_byte_is_stripped(self) -> None:
        assert "\x00" not in sanitize_filename("report\x00.pdf")

    def test_long_names_are_truncated_but_keep_the_extension(self) -> None:
        cleaned = sanitize_filename("a" * 400 + ".pdf")

        assert len(cleaned) <= MAX_FILENAME_LENGTH
        assert cleaned.endswith(".pdf")

    def test_unicode_is_preserved(self) -> None:
        assert "café" in sanitize_filename("café résumé.pdf")


class TestDetectFileType:
    def test_detects_each_supported_format_from_content(self) -> None:
        samples = {
            SupportedFileType.PDF: read_pdf_fixture("one_page.pdf"),
            SupportedFileType.DOCX: build_docx([("Normal", "Body text.")]),
            SupportedFileType.XLSX: build_xlsx({"S": [["a", "b"], ["1", "2"]]}),
        }
        for expected, content in samples.items():
            assert detect_file_type(content) is expected, expected

    def test_csv_is_detected_from_structure_and_extension(self) -> None:
        content = build_csv([["a", "b"], ["1", "2"], ["3", "4"]])
        assert detect_file_type(content, filename="data.csv") is SupportedFileType.CSV

    def test_plain_text_is_detected_as_txt(self) -> None:
        content = build_txt("Just some ordinary prose, written in sentences.")
        assert detect_file_type(content, filename="notes.txt") is SupportedFileType.TXT

    def test_a_lying_extension_does_not_win(self) -> None:
        """The core security property: content decides, the extension only hints."""
        docx_bytes = build_docx([("Normal", "Actually a Word document.")])

        assert detect_file_type(docx_bytes, filename="totally_a.pdf") is (SupportedFileType.DOCX)

    def test_pdf_bytes_named_txt_are_still_a_pdf(self) -> None:
        pdf_bytes = read_pdf_fixture("one_page.pdf")
        assert detect_file_type(pdf_bytes, filename="notes.txt") is SupportedFileType.PDF

    def test_prose_named_csv_falls_back_to_text(self) -> None:
        """An extension cannot make undelimited prose into a table."""
        content = build_txt("This is a paragraph of prose, with commas, but no columns.")

        assert detect_file_type(content, filename="data.csv") is SupportedFileType.TXT

    def test_unsupported_binary_is_rejected(self) -> None:
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        with pytest.raises(UnsupportedFileTypeError):
            detect_file_type(png, filename="image.png")

    def test_zip_that_is_not_an_office_document_is_rejected(self) -> None:
        import io
        import zipfile

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("readme.txt", "hello")

        with pytest.raises(UnsupportedFileTypeError):
            detect_file_type(buffer.getvalue(), filename="archive.zip")

    def test_empty_content_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            detect_file_type(b"")

    def test_damaged_office_document_gives_a_clear_message(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            detect_file_type(b"PK\x03\x04" + b"\xff" * 200, filename="report.docx")

        assert "damaged" in exc_info.value.message.lower()

    def test_cp1252_text_is_detected_as_text_not_binary(self) -> None:
        content = "Café naïve, written on Windows.".encode("cp1252")

        assert detect_file_type(content, filename="notes.txt") is SupportedFileType.TXT


class TestArchiveSafety:
    LIMITS = ArchiveLimits(max_uncompressed_bytes=50 * 1024 * 1024, max_entries=100)

    def test_normal_documents_pass(self) -> None:
        check_archive_safety(build_docx([("Normal", "Body.")]), self.LIMITS)
        check_archive_safety(build_xlsx({"S": [["a"], ["1"]]}), self.LIMITS)

    def test_zip_bomb_is_rejected(self) -> None:
        """A few KB on disk declaring hundreds of MB of content."""
        bomb = build_zip_bomb(entries=4, uncompressed_mb=400)

        # Confirms the fixture really is a bomb, so the test is meaningful.
        assert len(bomb) < 1024 * 1024

        with pytest.raises(ValidationError) as exc_info:
            check_archive_safety(bomb, self.LIMITS)

        assert "expands to too much" in exc_info.value.message

    def test_too_many_entries_is_rejected(self) -> None:
        import io
        import zipfile

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("word/document.xml", "<xml/>")
            for index in range(50):
                archive.writestr(f"f{index}.txt", "x")

        with pytest.raises(ValidationError):
            check_archive_safety(
                buffer.getvalue(),
                ArchiveLimits(max_uncompressed_bytes=10 * 1024 * 1024, max_entries=10),
            )

    def test_damaged_archive_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            check_archive_safety(b"PK\x03\x04 broken", self.LIMITS)
