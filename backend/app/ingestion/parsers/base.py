"""The document parser interface.

This is the project's main extension seam. Every supported format has exactly one parser,
and adding a format means writing one new file and adding one registry line — nothing
else in the system changes.

Two rules make that work:

* A parser takes **bytes in and a** :class:`~app.models.domain.content.ParsedDocument`
  **out**. It never touches the database, storage, configuration, or logging setup. That
  is what makes parsers trivially testable and keeps format knowledge from leaking.
* Every block a parser emits carries accurate provenance. A parser that drops it is a
  broken parser (CLAUDE.md §4.3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.core.config import Settings
from app.models.domain.content import ParsedDocument
from app.models.domain.document import SupportedFileType


@dataclass(frozen=True, slots=True)
class ParserLimits:
    """Resource ceilings a parser must enforce.

    Passed in rather than read from global settings so that parsers stay pure functions
    of their inputs, and so a test can impose a tiny limit without touching the
    environment.
    """

    max_pdf_pages: int = 500
    max_sheets: int = 50
    max_rows: int = 100_000
    max_columns: int = 512
    max_blocks: int = 50_000
    rows_per_block: int = 25

    @classmethod
    def from_settings(cls, settings: Settings) -> ParserLimits:
        """Build limits from application configuration."""
        return cls(
            max_pdf_pages=settings.max_pdf_pages,
            max_sheets=settings.max_sheets,
            max_rows=settings.max_rows,
            max_columns=settings.max_columns,
            max_blocks=settings.max_blocks,
            rows_per_block=settings.tabular_rows_per_block,
        )


class DocumentParser(ABC):
    """Turns the bytes of one file format into canonical content blocks."""

    #: The format this parser handles. Used by the registry to route files.
    file_type: SupportedFileType

    @abstractmethod
    def parse(self, content: bytes, limits: ParserLimits) -> ParsedDocument:
        """Extract content from a file.

        Implementations are **synchronous**. Parsing is CPU-bound work, so marking it
        ``async`` would be a lie that blocks the event loop; the caller runs it in a
        worker thread instead.

        Args:
            content: The raw file bytes.
            limits: Ceilings to enforce.

        Returns:
            The extracted blocks, in reading order, each with provenance.

        Raises:
            CorruptFileError: The file is damaged and cannot be read.
            EncryptedFileError: The file is password-protected.
            EmptyDocumentError: The file held no extractable content.
            ScannedDocumentError: The file needs OCR, which is not available.
            DocumentTooComplexError: A limit was exceeded.
            ParserError: Parsing failed for a reason that is not the file's fault.
        """
