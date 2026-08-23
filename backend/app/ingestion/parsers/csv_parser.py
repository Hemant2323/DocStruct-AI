"""CSV parser.

Reads a delimited text file as **data, not prose**: one schema block describing the
columns, then batches of rows. See ``tabular.py`` for why that split matters.
"""

from __future__ import annotations

import csv
import io

from charset_normalizer import from_bytes

from app.core.exceptions import (
    CorruptFileError,
    DocumentTooComplexError,
    EmptyDocumentError,
)
from app.ingestion.parsers.base import DocumentParser, ParserLimits
from app.ingestion.parsers.tabular import (
    normalise_headers,
    profile_table,
    render_rows,
)
from app.models.domain.content import (
    BlockType,
    ContentBlock,
    ParsedDocument,
    ParseStats,
    Provenance,
)
from app.models.domain.document import SupportedFileType

#: Sample used for dialect sniffing. Large enough to see the shape, small enough to be fast.
_SNIFF_BYTES = 16 * 1024

_SUPPORTED_DELIMITERS = ",;\t|"


class CsvParser(DocumentParser):
    """Extracts a schema block and row blocks from a CSV file."""

    file_type = SupportedFileType.CSV

    def parse(self, content: bytes, limits: ParserLimits) -> ParsedDocument:
        text = _decode(content)
        warnings: list[str] = []

        dialect, has_header = _detect_dialect(text)

        reader = csv.reader(io.StringIO(text, newline=""), dialect)
        try:
            raw_rows = list(reader)
        except csv.Error as exc:
            # A malformed CSV is the user's file being wrong, not our code failing, so
            # this is a 4xx-class error carrying a message they can act on.
            raise CorruptFileError(
                "This CSV file could not be read. It may be malformed or contain an "
                "unterminated quoted value."
            ) from exc

        raw_rows = [row for row in raw_rows if any(cell.strip() for cell in row)]
        if not raw_rows:
            raise EmptyDocumentError("This CSV file contains no data")

        if has_header:
            headers = normalise_headers(raw_rows[0], limits.max_columns)
            data_rows = raw_rows[1:]
            first_data_line = 2
        else:
            # No header line: synthesise names so rendered rows are still labelled.
            column_count = min(max(len(row) for row in raw_rows), limits.max_columns)
            headers = normalise_headers([""] * column_count, limits.max_columns)
            data_rows = raw_rows
            first_data_line = 1
            warnings.append("No header row was detected; columns were numbered instead.")

        if not data_rows:
            raise EmptyDocumentError("This CSV file contains headers but no data rows")

        if len(data_rows) > limits.max_rows:
            raise DocumentTooComplexError(
                f"This file has more than {limits.max_rows:,} rows, which is too many to process",
                details={"max_rows": limits.max_rows},
            )

        widest = max(len(row) for row in data_rows)
        if widest > limits.max_columns:
            warnings.append(
                f"Only the first {limits.max_columns} columns were read; the file has {widest}."
            )
        data_rows = [row[: limits.max_columns] for row in data_rows]

        blocks = self._build_blocks(headers, data_rows, first_data_line, limits)

        return ParsedDocument(
            blocks=blocks,
            file_type=self.file_type,
            stats=ParseStats(
                rows=len(data_rows),
                tables=1,
                characters=sum(block.char_count for block in blocks),
            ),
            warnings=warnings,
        )

    def _build_blocks(
        self,
        headers: list[str],
        data_rows: list[list[str]],
        first_data_line: int,
        limits: ParserLimits,
    ) -> list[ContentBlock]:
        """Build the schema block followed by batched row blocks."""
        profile = profile_table(headers, data_rows)

        last_line = first_data_line + len(data_rows) - 1
        blocks: list[ContentBlock] = [
            ContentBlock(
                block_type=BlockType.SCHEMA,
                text=profile.describe(title="Table structure"),
                order=0,
                # The schema describes the whole file, so its provenance is the whole
                # data range rather than a single row.
                provenance=Provenance.for_rows(first_data_line, last_line),
                metadata=profile.to_metadata(),
            )
        ]

        for start in range(0, len(data_rows), limits.rows_per_block):
            batch = data_rows[start : start + limits.rows_per_block]

            if len(blocks) >= limits.max_blocks:
                raise DocumentTooComplexError(
                    "This file produces too many sections to process",
                    details={"max_blocks": limits.max_blocks},
                )

            line_start = first_data_line + start
            line_end = line_start + len(batch) - 1

            blocks.append(
                ContentBlock(
                    block_type=BlockType.SHEET_ROW,
                    text=render_rows(headers, batch),
                    order=len(blocks),
                    provenance=Provenance.for_rows(line_start, line_end),
                    metadata={"row_count": len(batch)},
                )
            )

        return blocks


def _decode(content: bytes) -> str:
    """Decode CSV bytes, detecting the encoding rather than assuming UTF-8."""
    best_match = from_bytes(content).best()
    encoding = best_match.encoding if best_match is not None else "utf-8"
    try:
        return content.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        return content.decode("utf-8", errors="replace")


def _detect_dialect(text: str) -> tuple[type[csv.Dialect] | csv.Dialect, bool]:
    """Work out the delimiter and whether the first row is a header.

    Falls back to a comma-delimited default when sniffing fails: a file that reached this
    parser has already been detected as CSV, so refusing to guess would fail a file we
    can still read usefully.
    """
    sample = text[:_SNIFF_BYTES]
    sniffer = csv.Sniffer()

    try:
        dialect: type[csv.Dialect] | csv.Dialect = sniffer.sniff(
            sample, delimiters=_SUPPORTED_DELIMITERS
        )
    except csv.Error:
        dialect = csv.excel

    try:
        has_header = sniffer.has_header(sample)
    except csv.Error:
        # Assume a header: exported data almost always has one, and mislabelling the
        # first data row as a header is less damaging than losing the column names.
        has_header = True

    return dialect, has_header
