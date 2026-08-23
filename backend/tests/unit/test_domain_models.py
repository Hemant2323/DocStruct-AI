"""Tests for the document domain models."""

from __future__ import annotations

import pytest

from app.models.domain.document import ProcessingStatus, SupportedFileType


class TestProcessingStatus:
    def test_values_are_stable_strings(self) -> None:
        """These strings are persisted and sent over the API; changing them is a breaking
        change, so they are pinned by a test."""
        assert ProcessingStatus.PENDING.value == "pending"
        assert ProcessingStatus.PROCESSING.value == "processing"
        assert ProcessingStatus.READY.value == "ready"
        assert ProcessingStatus.FAILED.value == "failed"

    @pytest.mark.parametrize(
        ("status", "terminal"),
        [
            (ProcessingStatus.PENDING, False),
            (ProcessingStatus.PROCESSING, False),
            (ProcessingStatus.READY, True),
            (ProcessingStatus.FAILED, True),
        ],
    )
    def test_is_terminal(self, status: ProcessingStatus, terminal: bool) -> None:
        assert status.is_terminal is terminal


class TestSupportedFileType:
    def test_mvp_supports_exactly_five_types(self) -> None:
        assert {t.value for t in SupportedFileType} == {"pdf", "docx", "txt", "csv", "xlsx"}

    def test_every_type_declares_a_media_type(self) -> None:
        for file_type in SupportedFileType:
            assert file_type.media_type

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("report.pdf", SupportedFileType.PDF),
            ("contract.docx", SupportedFileType.DOCX),
            ("notes.txt", SupportedFileType.TXT),
            ("data.csv", SupportedFileType.CSV),
            ("budget.xlsx", SupportedFileType.XLSX),
            ("REPORT.PDF", SupportedFileType.PDF),
            ("archive.tar.gz", None),
            ("image.png", None),
            ("no_extension", None),
            ("", None),
            (".hidden", None),
        ],
    )
    def test_from_filename(self, filename: str, expected: SupportedFileType | None) -> None:
        assert SupportedFileType.from_filename(filename) == expected

    def test_from_filename_uses_the_last_extension(self) -> None:
        """A double extension must resolve to the final one, not the first."""
        assert SupportedFileType.from_filename("report.pdf.txt") is SupportedFileType.TXT

    def test_from_filename_is_only_a_hint(self) -> None:
        """Documents the security boundary: the extension is untrusted metadata.

        `from_filename` happily reports PDF for a file whose bytes are anything at all.
        Acceptance must be decided by content sniffing (Milestone 2), which is exactly
        why this returns an optional hint rather than raising.
        """
        assert SupportedFileType.from_filename("definitely_an_executable.pdf") is (
            SupportedFileType.PDF
        )
