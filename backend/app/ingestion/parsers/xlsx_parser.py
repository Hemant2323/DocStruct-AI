"""Excel workbook parser.

Applies the same tabular treatment as CSV, but **per worksheet**: one workbook holds
several independent tables, so each sheet gets its own schema block and row blocks.

``openpyxl`` is opened ``read_only`` (rows stream instead of loading the whole workbook
into memory) and ``data_only`` (cells yield their last computed value, and formulas are
never evaluated — evaluating attacker-supplied formulas would be a serious hole).
"""

from __future__ import annotations

import io
import zipfile
from datetime import date, datetime
from typing import Any

import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.utils.exceptions import InvalidFileException

from app.core.exceptions import (
    CorruptFileError,
    DocumentTooComplexError,
    EmptyDocumentError,
)
from app.ingestion.parsers.base import DocumentParser, ParserLimits
from app.ingestion.parsers.tabular import normalise_headers, profile_table, render_rows
from app.models.domain.content import (
    BlockType,
    ContentBlock,
    ParsedDocument,
    ParseStats,
    Provenance,
)
from app.models.domain.document import SupportedFileType

#: Leading blank rows tolerated while looking for a header. Exported spreadsheets often
#: start with a title and spacing before the real table begins.
_MAX_HEADER_SEARCH_ROWS = 20


class XlsxParser(DocumentParser):
    """Extracts schema and row blocks from every worksheet in a workbook."""

    file_type = SupportedFileType.XLSX

    def parse(self, content: bytes, limits: ParserLimits) -> ParsedDocument:
        workbook = self._open(content)
        warnings: list[str] = []
        blocks: list[ContentBlock] = []
        total_rows = 0
        sheets_read = 0

        try:
            sheet_names = list(workbook.sheetnames)

            if len(sheet_names) > limits.max_sheets:
                warnings.append(
                    f"Only the first {limits.max_sheets} of {len(sheet_names)} "
                    "worksheets were read."
                )
                sheet_names = sheet_names[: limits.max_sheets]

            for sheet_name in sheet_names:
                sheet_blocks, rows_read, sheet_warnings = self._parse_sheet(
                    workbook[sheet_name], sheet_name, len(blocks), limits
                )
                warnings.extend(sheet_warnings)

                if sheet_blocks:
                    blocks.extend(sheet_blocks)
                    total_rows += rows_read
                    sheets_read += 1
        finally:
            # read_only workbooks hold an open archive handle until closed.
            workbook.close()

        if not blocks:
            raise EmptyDocumentError("This spreadsheet contains no data")

        return ParsedDocument(
            blocks=blocks,
            file_type=self.file_type,
            stats=ParseStats(
                sheets=sheets_read,
                rows=total_rows,
                tables=sheets_read,
                characters=sum(block.char_count for block in blocks),
            ),
            warnings=warnings,
        )

    def _open(self, content: bytes) -> openpyxl.Workbook:
        """Open the workbook in streaming, values-only mode."""
        try:
            return openpyxl.load_workbook(
                io.BytesIO(content),
                read_only=True,
                data_only=True,
            )
        except (
            InvalidFileException,
            zipfile.BadZipFile,
            KeyError,
            ValueError,
            OSError,
        ) as exc:
            raise CorruptFileError(
                "This spreadsheet could not be opened. It may be damaged or password protected."
            ) from exc

    def _parse_sheet(
        self,
        sheet: Any,
        sheet_name: str,
        order_offset: int,
        limits: ParserLimits,
    ) -> tuple[list[ContentBlock], int, list[str]]:
        """Parse a single worksheet into a schema block plus row blocks."""
        warnings: list[str] = []

        raw_rows: list[tuple[Any, ...]] = []
        for index, row in enumerate(sheet.iter_rows(values_only=True)):
            if index >= limits.max_rows + _MAX_HEADER_SEARCH_ROWS:
                warnings.append(
                    f"Worksheet '{sheet_name}' was truncated at {limits.max_rows:,} rows."
                )
                break
            raw_rows.append(row)

        header_index = _find_header_row(raw_rows)
        if header_index is None:
            return [], 0, warnings

        raw_headers = [_stringify(value) for value in raw_rows[header_index]]
        headers = normalise_headers(raw_headers, limits.max_columns)
        if len(raw_headers) > limits.max_columns:
            warnings.append(
                f"Worksheet '{sheet_name}': only the first {limits.max_columns} columns were read."
            )

        data_rows = [
            [_stringify(value) for value in row[: len(headers)]]
            for row in raw_rows[header_index + 1 :]
        ]
        data_rows = [row for row in data_rows if any(cell.strip() for cell in row)]

        if not data_rows:
            return [], 0, warnings

        if len(data_rows) > limits.max_rows:
            raise DocumentTooComplexError(
                f"Worksheet '{sheet_name}' has more than {limits.max_rows:,} rows, "
                "which is too many to process",
                details={"max_rows": limits.max_rows},
            )

        # Spreadsheet rows are 1-indexed, so the first data row is the header's row + 1.
        first_data_row = header_index + 2
        last_data_row = first_data_row + len(data_rows) - 1
        last_column = get_column_letter(max(len(headers), 1))

        profile = profile_table(headers, data_rows)
        blocks: list[ContentBlock] = [
            ContentBlock(
                block_type=BlockType.SCHEMA,
                text=profile.describe(title=f"Worksheet '{sheet_name}' structure"),
                order=order_offset,
                provenance=Provenance.for_cells(
                    sheet_name,
                    f"A{first_data_row}:{last_column}{last_data_row}",
                    row_start=first_data_row,
                    row_end=last_data_row,
                ),
                metadata={"sheet": sheet_name, **profile.to_metadata()},
            )
        ]

        for start in range(0, len(data_rows), limits.rows_per_block):
            batch = data_rows[start : start + limits.rows_per_block]

            if order_offset + len(blocks) >= limits.max_blocks:
                raise DocumentTooComplexError(
                    "This workbook produces too many sections to process",
                    details={"max_blocks": limits.max_blocks},
                )

            row_start = first_data_row + start
            row_end = row_start + len(batch) - 1

            blocks.append(
                ContentBlock(
                    block_type=BlockType.SHEET_ROW,
                    text=render_rows(headers, batch),
                    order=order_offset + len(blocks),
                    provenance=Provenance.for_cells(
                        sheet_name,
                        f"A{row_start}:{last_column}{row_end}",
                        row_start=row_start,
                        row_end=row_end,
                    ),
                    metadata={"sheet": sheet_name, "row_count": len(batch)},
                )
            )

        return blocks, len(data_rows), warnings


def _find_header_row(rows: list[tuple[Any, ...]]) -> int | None:
    """Find the row that holds the column headers.

    Real spreadsheets rarely start their table at A1 — a title, a blank line and then the
    headers is the norm. The header is taken to be the first row with at least two
    non-empty cells, searched only within the first few rows so a sparse data row deep in
    the sheet is never mistaken for one.
    """
    for index, row in enumerate(rows[:_MAX_HEADER_SEARCH_ROWS]):
        filled = sum(1 for value in row if _stringify(value).strip())
        if filled >= 2:
            return index
    return None


def _stringify(value: Any) -> str:
    """Render a cell value as text.

    Dates become ISO strings so they are both readable and parseable by the type
    inference in ``tabular.py``. Floats that are whole numbers lose their ``.0`` so an
    integer column is not mistaken for a decimal one.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, datetime):
        # A midnight timestamp is almost always a plain date in a spreadsheet.
        if value.time() == datetime.min.time():
            return value.date().isoformat()
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
