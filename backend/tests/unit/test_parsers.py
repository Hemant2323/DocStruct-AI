"""Tests for all five document parsers.

Every parser gets the same treatment: a happy path, an assertion that **provenance is
correct** (not merely present), and its failure modes. A parser that drops or invents
provenance is a broken parser, so those assertions are the point of this file.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from app.core.exceptions import (
    CorruptFileError,
    DocStructError,
    DocumentTooComplexError,
    EmptyDocumentError,
    ScannedDocumentError,
)
from app.ingestion.parsers.base import ParserLimits
from app.ingestion.parsers.csv_parser import CsvParser
from app.ingestion.parsers.docx_parser import DocxParser
from app.ingestion.parsers.pdf_parser import PdfParser
from app.ingestion.parsers.txt_parser import TxtParser
from app.ingestion.parsers.xlsx_parser import XlsxParser
from app.ingestion.registry import get_parser, supported_file_types
from app.models.domain.content import BlockType, ProvenanceKind
from app.models.domain.document import SupportedFileType
from tests.fixtures.builders import (
    build_csv,
    build_docx,
    build_txt,
    build_xlsx,
    read_pdf_fixture,
)

LIMITS = ParserLimits()


# ---------------------------------------------------------------------------
# TXT
# ---------------------------------------------------------------------------
class TestTxtParser:
    def test_splits_on_blank_lines(self) -> None:
        content = build_txt("First paragraph.\n\nSecond paragraph.\n\nThird.")

        result = TxtParser().parse(content, LIMITS)

        assert len(result.blocks) == 3
        assert result.blocks[0].text == "First paragraph."
        assert result.blocks[2].text == "Third."

    def test_line_provenance_is_correct(self) -> None:
        """The second paragraph really does start on line 3 of the file."""
        content = build_txt("Line one.\n\nLine three.\nLine four.")

        result = TxtParser().parse(content, LIMITS)

        first, second = result.blocks
        assert first.provenance.line_start == 1
        assert first.provenance.line_end == 1
        assert second.provenance.line_start == 3
        assert second.provenance.line_end == 4
        assert second.provenance.label == "lines 3-4"

    def test_blocks_are_ordered_from_zero(self) -> None:
        result = TxtParser().parse(build_txt("A\n\nB\n\nC"), LIMITS)
        assert [block.order for block in result.blocks] == [0, 1, 2]

    @pytest.mark.parametrize(
        "heading",
        ["INTRODUCTION", "1. Scope", "2.3 Payment Terms"],
    )
    def test_detects_headings(self, heading: str) -> None:
        result = TxtParser().parse(build_txt(f"{heading}\n\nBody text here."), LIMITS)

        assert result.blocks[0].block_type is BlockType.HEADING

    def test_heading_detection_is_marked_inexact(self) -> None:
        """Plain text declares no headings, so the classification is inference."""
        result = TxtParser().parse(build_txt("INTRODUCTION\n\nBody."), LIMITS)

        assert result.blocks[0].provenance.is_exact is False

    def test_sentence_is_not_a_heading(self) -> None:
        result = TxtParser().parse(build_txt("This is a sentence.\n\nAnother."), LIMITS)
        assert result.blocks[0].block_type is BlockType.PARAGRAPH

    def test_cp1252_is_decoded_without_mojibake(self) -> None:
        """A Windows-authored file is often cp1252; assuming UTF-8 would corrupt it."""
        content = "Café — naïve".encode("cp1252")

        result = TxtParser().parse(content, LIMITS)

        assert "Caf" in result.blocks[0].text
        assert "�" not in result.blocks[0].text

    def test_utf8_with_bom(self) -> None:
        result = TxtParser().parse("Héllo wörld".encode("utf-8-sig"), LIMITS)
        assert "Héllo wörld" in result.blocks[0].text

    def test_empty_file_raises(self) -> None:
        with pytest.raises(EmptyDocumentError):
            TxtParser().parse(b"", LIMITS)

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(EmptyDocumentError):
            TxtParser().parse(build_txt("   \n\n  \t \n"), LIMITS)

    def test_block_limit_enforced(self) -> None:
        content = build_txt("\n\n".join(f"Paragraph {i}." for i in range(50)))

        with pytest.raises(DocumentTooComplexError):
            TxtParser().parse(content, ParserLimits(max_blocks=5))


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
class TestCsvParser:
    ROWS: ClassVar[list[list[str]]] = [
        ["name", "age", "joined"],
        ["Alice", "30", "2024-01-15"],
        ["Bob", "25", "2024-02-20"],
        ["Carol", "35", "2024-03-10"],
    ]

    def test_produces_schema_block_first(self) -> None:
        """The schema answers structural questions without reading any data rows."""
        result = CsvParser().parse(build_csv(self.ROWS), LIMITS)

        assert result.blocks[0].block_type is BlockType.SCHEMA
        assert "name" in result.blocks[0].text
        assert "3 data rows" in result.blocks[0].text

    def test_infers_column_types(self) -> None:
        result = CsvParser().parse(build_csv(self.ROWS), LIMITS)

        columns = {c["name"]: c["type"] for c in result.blocks[0].metadata["columns"]}
        assert columns == {"name": "text", "age": "integer", "joined": "date"}

    def test_numeric_range_is_profiled(self) -> None:
        result = CsvParser().parse(build_csv(self.ROWS), LIMITS)

        age = next(c for c in result.blocks[0].metadata["columns"] if c["name"] == "age")
        assert age["min"] == "25"
        assert age["max"] == "35"

    def test_rows_are_batched_into_blocks(self) -> None:
        result = CsvParser().parse(build_csv(self.ROWS), ParserLimits(rows_per_block=2))

        row_blocks = [b for b in result.blocks if b.block_type is BlockType.SHEET_ROW]
        assert len(row_blocks) == 2  # 3 rows in batches of 2

    def test_row_provenance_points_at_real_file_lines(self) -> None:
        """Data starts on line 2 because line 1 is the header."""
        result = CsvParser().parse(build_csv(self.ROWS), ParserLimits(rows_per_block=2))

        row_blocks = [b for b in result.blocks if b.block_type is BlockType.SHEET_ROW]
        assert row_blocks[0].provenance.row_start == 2
        assert row_blocks[0].provenance.row_end == 3
        assert row_blocks[1].provenance.row_start == 4
        assert row_blocks[0].provenance.kind is ProvenanceKind.ROW_RANGE

    def test_rows_are_rendered_with_column_names(self) -> None:
        """A retrieved row block must be readable on its own, without its header row."""
        result = CsvParser().parse(build_csv(self.ROWS), LIMITS)

        row_block = next(b for b in result.blocks if b.block_type is BlockType.SHEET_ROW)
        assert "name: Alice" in row_block.text
        assert "age: 30" in row_block.text

    def test_semicolon_delimiter_is_detected(self) -> None:
        result = CsvParser().parse(build_csv(self.ROWS, delimiter=";"), LIMITS)

        columns = [c["name"] for c in result.blocks[0].metadata["columns"]]
        assert columns == ["name", "age", "joined"]

    def test_duplicate_headers_are_made_unique(self) -> None:
        rows = [["a", "a", "b"], ["1", "2", "3"], ["4", "5", "6"]]

        result = CsvParser().parse(build_csv(rows), LIMITS)

        names = [c["name"] for c in result.blocks[0].metadata["columns"]]
        assert len(set(names)) == 3

    def test_ragged_rows_do_not_crash(self) -> None:
        raw = b"a,b,c\n1,2,3\n4,5\n6,7,8,9\n"

        result = CsvParser().parse(raw, LIMITS)

        assert any(b.block_type is BlockType.SHEET_ROW for b in result.blocks)

    def test_mixed_type_column_is_text(self) -> None:
        """One stray word makes the whole column text; a wrong type would make the
        profile's stated range a lie."""
        rows = [["value"], ["1"], ["2"], ["unknown"]]

        result = CsvParser().parse(build_csv(rows), LIMITS)

        column = result.blocks[0].metadata["columns"][0]
        assert column["type"] == "text"

    def test_null_tokens_are_counted(self) -> None:
        """Both "N/A" and a genuinely blank cell count as missing.

        The blank cell sits in a two-column row on purpose: a row that is blank in
        *every* column is noise and is dropped before profiling.
        """
        rows = [["value", "label"], ["1", "a"], ["N/A", "b"], ["", "c"], ["3", "d"]]

        result = CsvParser().parse(build_csv(rows), LIMITS)

        value_column = result.blocks[0].metadata["columns"][0]
        assert value_column["name"] == "value"
        assert value_column["null_count"] == 2

    def test_header_only_raises(self) -> None:
        with pytest.raises(EmptyDocumentError):
            CsvParser().parse(build_csv([["a", "b"]]), LIMITS)

    def test_empty_raises(self) -> None:
        with pytest.raises(EmptyDocumentError):
            CsvParser().parse(b"\n\n", LIMITS)

    def test_row_limit_enforced(self) -> None:
        rows = [["n"], *[[str(i)] for i in range(50)]]

        with pytest.raises(DocumentTooComplexError):
            CsvParser().parse(build_csv(rows), ParserLimits(max_rows=10))

    def test_unterminated_quote_is_a_clear_error(self) -> None:
        with pytest.raises(CorruptFileError):
            CsvParser().parse(b'a,b\n"unterminated,2\n' + b"x" * 200000, LIMITS)


# ---------------------------------------------------------------------------
# XLSX
# ---------------------------------------------------------------------------
class TestXlsxParser:
    SHEETS: ClassVar[dict[str, list[list[Any]]]] = {
        "Sales": [
            ["region", "amount"],
            ["North", 100],
            ["South", 250],
        ],
        "Costs": [
            ["item", "cost"],
            ["Rent", 500],
        ],
    }

    def test_each_sheet_gets_a_schema_block(self) -> None:
        result = XlsxParser().parse(build_xlsx(self.SHEETS), LIMITS)

        schemas = [b for b in result.blocks if b.block_type is BlockType.SCHEMA]
        assert len(schemas) == 2
        assert {s.metadata["sheet"] for s in schemas} == {"Sales", "Costs"}

    def test_cell_provenance_names_sheet_and_range(self) -> None:
        result = XlsxParser().parse(build_xlsx(self.SHEETS), LIMITS)

        row_block = next(b for b in result.blocks if b.block_type is BlockType.SHEET_ROW)
        assert row_block.provenance.kind is ProvenanceKind.CELL_RANGE
        assert row_block.provenance.sheet_name == "Sales"
        # Data begins at row 2 because row 1 holds the headers.
        assert row_block.provenance.row_start == 2
        assert row_block.provenance.label.startswith("Sales!A2:")

    def test_orders_are_unique_across_sheets(self) -> None:
        """`order` is document-wide, so sheets must not restart numbering."""
        result = XlsxParser().parse(build_xlsx(self.SHEETS), LIMITS)

        orders = [b.order for b in result.blocks]
        assert orders == sorted(orders)
        assert len(set(orders)) == len(orders)

    def test_header_not_in_first_row_is_found(self) -> None:
        """Real spreadsheets often start with a title and a blank line."""
        sheets = {
            "S": [
                ["Quarterly Report"],
                [],
                ["region", "amount"],
                ["North", 100],
            ]
        }

        result = XlsxParser().parse(build_xlsx(sheets), LIMITS)

        columns = [c["name"] for c in result.blocks[0].metadata["columns"]]
        assert columns == ["region", "amount"]

    def test_dates_are_rendered_iso(self) -> None:
        from datetime import datetime

        sheets = {"S": [["when", "what"], [datetime(2024, 5, 17), "launch"]]}

        result = XlsxParser().parse(build_xlsx(sheets), LIMITS)

        row_block = next(b for b in result.blocks if b.block_type is BlockType.SHEET_ROW)
        assert "2024-05-17" in row_block.text

    def test_whole_number_floats_lose_their_decimal(self) -> None:
        sheets = {"S": [["n", "label"], [3.0, "three"]]}

        result = XlsxParser().parse(build_xlsx(sheets), LIMITS)

        row_block = next(b for b in result.blocks if b.block_type is BlockType.SHEET_ROW)
        assert "n: 3" in row_block.text
        assert "3.0" not in row_block.text

    def test_empty_sheets_are_skipped_not_fatal(self) -> None:
        sheets = {"Empty": [], "Real": [["a", "b"], ["1", "2"]]}

        result = XlsxParser().parse(build_xlsx(sheets), LIMITS)

        assert result.stats.sheets == 1

    def test_workbook_with_no_data_raises(self) -> None:
        with pytest.raises(EmptyDocumentError):
            XlsxParser().parse(build_xlsx({"Empty": []}), LIMITS)

    def test_corrupt_workbook_raises(self) -> None:
        with pytest.raises(CorruptFileError):
            XlsxParser().parse(b"PK\x03\x04 not really a workbook", LIMITS)

    def test_sheet_limit_produces_a_warning_not_a_failure(self) -> None:
        sheets = {f"S{i}": [["a", "b"], ["1", "2"]] for i in range(5)}

        result = XlsxParser().parse(build_xlsx(sheets), ParserLimits(max_sheets=2))

        assert result.stats.sheets == 2
        assert any("worksheets were read" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------
class TestDocxParser:
    ITEMS: ClassVar[list[tuple[str, str]]] = [
        ("Heading 1", "Payment Terms"),
        ("Normal", "Invoices are due within 30 days."),
        ("Heading 2", "Late Fees"),
        ("Normal", "A 5% fee applies after the due date."),
    ]

    def test_extracts_paragraphs_and_headings(self) -> None:
        result = DocxParser().parse(build_docx(self.ITEMS), LIMITS)

        types = [b.block_type for b in result.blocks]
        assert types == [
            BlockType.HEADING,
            BlockType.PARAGRAPH,
            BlockType.HEADING,
            BlockType.PARAGRAPH,
        ]

    def test_heading_levels_come_from_styles(self) -> None:
        """DOCX declares its headings, so this is fact rather than inference."""
        result = DocxParser().parse(build_docx(self.ITEMS), LIMITS)

        assert result.blocks[0].level == 1
        assert result.blocks[2].level == 2
        assert result.blocks[0].provenance.is_exact is True

    def test_heading_path_gives_section_context(self) -> None:
        """Turns a citation into 'Payment Terms > Late Fees' instead of 'paragraph 4'."""
        result = DocxParser().parse(build_docx(self.ITEMS), LIMITS)

        last_paragraph = result.blocks[3]
        assert last_paragraph.provenance.heading_path == ("Payment Terms", "Late Fees")

    def test_paragraph_index_provenance(self) -> None:
        result = DocxParser().parse(build_docx(self.ITEMS), LIMITS)

        assert result.blocks[0].provenance.paragraph_index == 0
        assert result.blocks[1].provenance.paragraph_index == 1
        assert result.blocks[1].provenance.kind is ProvenanceKind.PARAGRAPH

    def test_tables_are_extracted_as_single_blocks(self) -> None:
        content = build_docx(
            [("Normal", "Before the table.")],
            tables=[[["Fee", "Amount"], ["Late", "5%"]]],
        )

        result = DocxParser().parse(content, LIMITS)

        table = next(b for b in result.blocks if b.block_type is BlockType.TABLE)
        assert "Fee" in table.text
        assert "5%" in table.text
        assert table.metadata["rows"] == 2

    def test_tables_and_paragraphs_keep_document_order(self) -> None:
        """python-docx exposes paragraphs and tables separately; order must be restored."""
        content = build_docx(
            [("Normal", "First.")],
            tables=[[["a", "b"], ["1", "2"]]],
        )

        result = DocxParser().parse(content, LIMITS)

        assert result.blocks[0].text == "First."
        assert result.blocks[1].block_type is BlockType.TABLE

    def test_empty_paragraphs_are_skipped(self) -> None:
        content = build_docx(
            [("Normal", "Real text."), ("Normal", "   "), ("Normal", "More text.")]
        )

        result = DocxParser().parse(content, LIMITS)

        assert len(result.blocks) == 2

    def test_document_with_no_text_raises(self) -> None:
        with pytest.raises(EmptyDocumentError):
            DocxParser().parse(build_docx([("Normal", "")]), LIMITS)

    def test_corrupt_document_raises(self) -> None:
        with pytest.raises(CorruptFileError):
            DocxParser().parse(b"PK\x03\x04 not a real docx", LIMITS)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
class TestPdfParser:
    def test_extracts_text(self) -> None:
        result = PdfParser().parse(read_pdf_fixture("simple.pdf"), LIMITS)

        text = " ".join(block.text for block in result.blocks)
        assert "Quarterly Report" in text
        assert "Revenue increased" in text

    def test_page_provenance_is_correct(self) -> None:
        """Content placed on page 2 must report page 2 — the citation depends on it."""
        result = PdfParser().parse(read_pdf_fixture("simple.pdf"), LIMITS)

        marker_block = next(b for b in result.blocks if "PAGE TWO MARKER" in b.text)
        assert marker_block.provenance.page == 2
        assert marker_block.provenance.label == "page 2"
        assert marker_block.provenance.kind is ProvenanceKind.PAGE

        first_block = result.blocks[0]
        assert first_block.provenance.page == 1

    def test_reports_page_count(self) -> None:
        result = PdfParser().parse(read_pdf_fixture("simple.pdf"), LIMITS)
        assert result.stats.pages == 2

    def test_large_text_is_detected_as_a_heading(self) -> None:
        result = PdfParser().parse(read_pdf_fixture("simple.pdf"), LIMITS)

        headings = [b for b in result.blocks if b.block_type is BlockType.HEADING]
        assert any("Quarterly Report" in b.text for b in headings)

    def test_inferred_headings_are_marked_inexact(self) -> None:
        """A PDF declares no headings, so the classification is a guess and says so."""
        result = PdfParser().parse(read_pdf_fixture("simple.pdf"), LIMITS)

        headings = [b for b in result.blocks if b.block_type is BlockType.HEADING]
        assert headings
        assert all(b.provenance.is_exact is False for b in headings)

    def test_single_page_document(self) -> None:
        result = PdfParser().parse(read_pdf_fixture("one_page.pdf"), LIMITS)

        assert result.stats.pages == 1
        assert "only sentence" in result.blocks[0].text

    def test_scanned_pdf_is_reported_not_silently_empty(self) -> None:
        """The single most important PDF failure mode.

        A scan yields no text. Returning an empty result would be indistinguishable from
        a successfully processed blank document, so it must raise instead.
        """
        with pytest.raises(ScannedDocumentError) as exc_info:
            PdfParser().parse(read_pdf_fixture("scanned.pdf"), LIMITS)

        assert exc_info.value.code == "requires_ocr"
        assert "OCR" in exc_info.value.message

    def test_corrupt_pdf_raises(self) -> None:
        with pytest.raises(CorruptFileError):
            PdfParser().parse(b"%PDF-1.4\nbut then garbage", LIMITS)

    def test_page_limit_enforced(self) -> None:
        with pytest.raises(DocumentTooComplexError):
            PdfParser().parse(read_pdf_fixture("simple.pdf"), ParserLimits(max_pdf_pages=1))


# ---------------------------------------------------------------------------
# Cross-parser conformance
# ---------------------------------------------------------------------------
class TestAllParsersConform:
    """Proves the abstraction actually holds across every format.

    If these pass, everything downstream can treat all five formats identically — which
    is the entire justification for the ContentBlock design.
    """

    @pytest.fixture
    def samples(self) -> dict[SupportedFileType, bytes]:
        return {
            SupportedFileType.TXT: build_txt("A heading\n\nSome body text here."),
            SupportedFileType.CSV: build_csv([["a", "b"], ["1", "2"], ["3", "4"]]),
            SupportedFileType.XLSX: build_xlsx({"S": [["a", "b"], [1, 2]]}),
            SupportedFileType.DOCX: build_docx([("Heading 1", "Title"), ("Normal", "Body.")]),
            SupportedFileType.PDF: read_pdf_fixture("simple.pdf"),
        }

    def test_every_supported_type_has_a_parser(self) -> None:
        assert supported_file_types() == set(SupportedFileType)

    def test_every_parser_produces_blocks_with_the_right_file_type(
        self, samples: dict[SupportedFileType, bytes]
    ) -> None:
        for file_type, content in samples.items():
            result = get_parser(file_type).parse(content, LIMITS)
            assert result.file_type is file_type, file_type
            assert result.blocks, file_type

    def test_every_block_from_every_parser_has_provenance(
        self, samples: dict[SupportedFileType, bytes]
    ) -> None:
        """The guarantee the whole citation feature rests on."""
        for file_type, content in samples.items():
            result = get_parser(file_type).parse(content, LIMITS)
            for block in result.blocks:
                assert block.provenance is not None, f"{file_type} block {block.order}"
                assert block.provenance.label.strip(), f"{file_type} block {block.order}"
                assert block.provenance.kind is not ProvenanceKind.UNKNOWN

    def test_every_parser_numbers_blocks_contiguously_from_zero(
        self, samples: dict[SupportedFileType, bytes]
    ) -> None:
        """Downstream code relies on `order` being a dense sequence."""
        for file_type, content in samples.items():
            result = get_parser(file_type).parse(content, LIMITS)
            orders = [block.order for block in result.blocks]
            assert orders == list(range(len(orders))), file_type

    def test_no_parser_emits_blank_text(self, samples: dict[SupportedFileType, bytes]) -> None:
        for file_type, content in samples.items():
            result = get_parser(file_type).parse(content, LIMITS)
            assert all(block.text.strip() for block in result.blocks), file_type

    def test_every_parser_rejects_empty_input(self) -> None:
        for file_type in SupportedFileType:
            with pytest.raises((EmptyDocumentError, CorruptFileError)):
                get_parser(file_type).parse(b"", LIMITS)

    def test_every_parser_survives_garbage_input(self) -> None:
        """Garbage must produce a typed domain error or a valid result — never a crash.

        Binary content does not reach a parser in practice: ``detection.py`` rejects it
        first. So the invariant worth pinning here is not "always rejects" but "never
        fails in an unhandled way" — an unhandled failure is what would turn a bad
        upload into an opaque 500 instead of a clear message.
        """
        garbage = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4

        for file_type in SupportedFileType:
            try:
                result = get_parser(file_type).parse(garbage, LIMITS)
            except DocStructError:
                continue  # A typed, user-facing failure is a correct outcome.

            # If it did parse, every invariant must still hold.
            assert all(block.provenance.label.strip() for block in result.blocks)
            assert [b.order for b in result.blocks] == list(range(len(result.blocks)))
