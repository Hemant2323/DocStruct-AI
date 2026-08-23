"""Maps a file type to the parser that handles it.

This is the single lookup point that keeps format knowledge out of the rest of the
system. Supporting a new format means writing one parser and adding one line here —
nothing else changes, which is the whole point of the ``DocumentParser`` seam.
"""

from __future__ import annotations

from app.core.exceptions import UnsupportedFileTypeError
from app.ingestion.parsers.base import DocumentParser
from app.ingestion.parsers.csv_parser import CsvParser
from app.ingestion.parsers.docx_parser import DocxParser
from app.ingestion.parsers.pdf_parser import PdfParser
from app.ingestion.parsers.txt_parser import TxtParser
from app.ingestion.parsers.xlsx_parser import XlsxParser
from app.models.domain.document import SupportedFileType

#: One parser per supported format. Built once at import time; parsers hold no state.
_PARSERS: dict[SupportedFileType, DocumentParser] = {
    parser.file_type: parser
    for parser in (
        PdfParser(),
        DocxParser(),
        TxtParser(),
        CsvParser(),
        XlsxParser(),
    )
}


def get_parser(file_type: SupportedFileType) -> DocumentParser:
    """Return the parser for a file type.

    Raises:
        UnsupportedFileTypeError: If no parser is registered for the type. This should be
            unreachable while every ``SupportedFileType`` has a parser — a condition a
            test enforces — but failing loudly beats returning ``None``.
    """
    parser = _PARSERS.get(file_type)
    if parser is None:
        raise UnsupportedFileTypeError(
            f"No parser is available for {file_type.value.upper()} files",
            details={"file_type": file_type.value},
        )
    return parser


def supported_file_types() -> frozenset[SupportedFileType]:
    """Every file type that currently has a parser."""
    return frozenset(_PARSERS)
