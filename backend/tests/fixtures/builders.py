"""Builders for synthetic test documents.

DOCX, XLSX, CSV and TXT fixtures are generated at test time using libraries the project
already depends on, so nothing binary or of unknown origin is committed. PDFs cannot be
generated that way and are committed instead — see ``make_pdf_fixtures.py``.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import docx
import openpyxl

FIXTURES_DIR = Path(__file__).parent


def read_pdf_fixture(name: str) -> bytes:
    """Read one of the committed synthetic PDFs."""
    return (FIXTURES_DIR / name).read_bytes()


def build_docx(
    items: list[tuple[str, str]], *, tables: list[list[list[str]]] | None = None
) -> bytes:
    """Build a Word document.

    Args:
        items: ``(style, text)`` pairs, e.g. ``("Heading 1", "Payment Terms")`` or
            ``("Normal", "Body text.")``.
        tables: Optional tables, each a list of rows of cell strings. Appended after the
            paragraphs.

    Returns:
        The .docx file's bytes.
    """
    document = docx.Document()
    for style, text in items:
        document.add_paragraph(text, style=style)

    for table_rows in tables or []:
        table = document.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        for row_index, row in enumerate(table_rows):
            for column_index, value in enumerate(row):
                table.cell(row_index, column_index).text = value

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def build_xlsx(sheets: dict[str, list[list[Any]]]) -> bytes:
    """Build a workbook.

    Args:
        sheets: Sheet name to rows (the first row being the header).

    Returns:
        The .xlsx file's bytes.
    """
    workbook = openpyxl.Workbook()
    # A new workbook starts with one sheet; reuse it for the first, then add the rest.
    default_sheet = workbook.active
    assert default_sheet is not None

    for index, (name, rows) in enumerate(sheets.items()):
        sheet = default_sheet if index == 0 else workbook.create_sheet()
        sheet.title = name
        for row in rows:
            sheet.append(row)

    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def build_csv(rows: list[list[str]], *, delimiter: str = ",") -> bytes:
    """Build a CSV file from rows (the first being the header)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=delimiter, lineterminator="\n")
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def build_txt(text: str, *, encoding: str = "utf-8") -> bytes:
    """Build a text file in a given encoding."""
    return text.encode(encoding)


def build_zip_bomb(*, entries: int = 5, uncompressed_mb: int = 400) -> bytes:
    """Build an archive that declares far more content than it stores.

    Highly compressible zeroes, so a few hundred megabytes of declared content costs a
    few kilobytes on disk — the shape of a real zip bomb.
    """
    import zipfile

    payload = b"\x00" * (uncompressed_mb * 1024 * 1024 // entries)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        # Present as a Word document so it passes type detection and reaches the check.
        archive.writestr("word/document.xml", "<xml/>")
        for index in range(entries):
            archive.writestr(f"payload_{index}.bin", payload)
    return buffer.getvalue()
