"""Tests for the canonical content models and provenance."""

from __future__ import annotations

import pytest

from app.models.domain.content import (
    BlockType,
    ContentBlock,
    ParsedDocument,
    ParseStats,
    Provenance,
    ProvenanceKind,
)
from app.models.domain.document import SupportedFileType


class TestBlockType:
    def test_values_are_stable(self) -> None:
        """Persisted and sent over the API; changing one is a breaking change."""
        assert {t.value for t in BlockType} == {
            "heading",
            "paragraph",
            "list_item",
            "table",
            "schema",
            "sheet_row",
            "caption",
        }

    @pytest.mark.parametrize(
        ("block_type", "tabular"),
        [
            (BlockType.TABLE, True),
            (BlockType.SCHEMA, True),
            (BlockType.SHEET_ROW, True),
            (BlockType.PARAGRAPH, False),
            (BlockType.HEADING, False),
        ],
    )
    def test_is_tabular(self, block_type: BlockType, tabular: bool) -> None:
        assert block_type.is_tabular is tabular

    def test_atomic_blocks_are_never_split(self) -> None:
        """Half a table is worse than no table, so the chunker must not split these."""
        assert BlockType.TABLE.is_atomic is True
        assert BlockType.SCHEMA.is_atomic is True
        assert BlockType.PARAGRAPH.is_atomic is False


class TestProvenanceConstructors:
    def test_for_page(self) -> None:
        provenance = Provenance.for_page(4)

        assert provenance.kind is ProvenanceKind.PAGE
        assert provenance.page == 4
        assert provenance.label == "page 4"
        assert provenance.is_exact is True

    def test_for_paragraph_label_is_one_indexed(self) -> None:
        """Index 0 is the first paragraph; humans count from one."""
        provenance = Provenance.for_paragraph(0)

        assert provenance.paragraph_index == 0
        assert provenance.label == "paragraph 1"

    def test_for_paragraph_carries_heading_path(self) -> None:
        provenance = Provenance.for_paragraph(12, heading_path=("Payment", "Late fees"))

        assert provenance.heading_path == ("Payment", "Late fees")

    @pytest.mark.parametrize(
        ("start", "end", "label"),
        [(5, 5, "line 5"), (40, 48, "lines 40-48")],
    )
    def test_for_lines(self, start: int, end: int, label: str) -> None:
        assert Provenance.for_lines(start, end).label == label

    @pytest.mark.parametrize(
        ("start", "end", "label"),
        [(7, 7, "row 7"), (120, 140, "rows 120-140")],
    )
    def test_for_rows(self, start: int, end: int, label: str) -> None:
        assert Provenance.for_rows(start, end).label == label

    def test_for_cells(self) -> None:
        provenance = Provenance.for_cells("Q3 Budget", "A12:F30", row_start=12, row_end=30)

        assert provenance.kind is ProvenanceKind.CELL_RANGE
        assert provenance.label == "Q3 Budget!A12:F30"
        assert provenance.sheet_name == "Q3 Budget"
        assert provenance.row_start == 12

    def test_unknown_is_explicit_and_inexact(self) -> None:
        """CLAUDE.md forbids inventing a location; an honest 'unknown' is the alternative."""
        provenance = Provenance.unknown("no page marker")

        assert provenance.kind is ProvenanceKind.UNKNOWN
        assert provenance.is_exact is False
        assert "unknown" in provenance.label

    def test_every_constructor_produces_a_non_empty_label(self) -> None:
        """A citation with a blank location is useless."""
        provenances = [
            Provenance.for_page(1),
            Provenance.for_paragraph(0),
            Provenance.for_lines(1, 2),
            Provenance.for_rows(1, 2),
            Provenance.for_cells("S", "A1:B2", row_start=1, row_end=2),
            Provenance.unknown("reason"),
        ]
        assert all(p.label.strip() for p in provenances)


class TestProvenanceSerialisation:
    @pytest.mark.parametrize(
        "provenance",
        [
            Provenance.for_page(4),
            Provenance.for_paragraph(17, heading_path=("A", "B")),
            Provenance.for_lines(40, 48),
            Provenance.for_rows(120, 140),
            Provenance.for_cells("Q3", "A12:F30", row_start=12, row_end=30),
            Provenance.unknown("scanned"),
        ],
    )
    def test_round_trip_through_json_shape(self, provenance: Provenance) -> None:
        """Provenance survives the database, or citations break silently."""
        assert Provenance.from_dict(provenance.to_dict()) == provenance

    def test_unset_fields_are_omitted(self) -> None:
        """Keeps the stored JSON small and readable."""
        data = Provenance.for_page(4).to_dict()

        assert "page" in data
        assert "sheet_name" not in data
        assert "row_start" not in data

    def test_serialised_form_is_json_compatible(self) -> None:
        import json

        provenance = Provenance.for_cells("Q3", "A1:B2", row_start=1, row_end=2)
        assert json.loads(json.dumps(provenance.to_dict()))


class TestContentBlock:
    def test_char_count(self) -> None:
        block = ContentBlock(BlockType.PARAGRAPH, "hello", 0, Provenance.for_page(1))
        assert block.char_count == 5

    def test_is_frozen(self) -> None:
        """A parser's output is a record of what it saw; nothing may mutate it."""
        block = ContentBlock(BlockType.PARAGRAPH, "hello", 0, Provenance.for_page(1))

        with pytest.raises(AttributeError):
            block.text = "changed"  # type: ignore[misc]

    def test_with_order_preserves_everything_else(self) -> None:
        original = ContentBlock(
            BlockType.HEADING, "Title", 0, Provenance.for_page(2), level=1, metadata={"a": 1}
        )

        moved = original.with_order(5)

        assert moved.order == 5
        assert moved.text == original.text
        assert moved.provenance == original.provenance
        assert moved.level == original.level
        assert moved.metadata == original.metadata

    def test_with_text_preserves_provenance(self) -> None:
        """The normalizer rewrites text; the link to the source must survive."""
        original = ContentBlock(BlockType.PARAGRAPH, "  messy  ", 3, Provenance.for_page(2))

        cleaned = original.with_text("messy")

        assert cleaned.text == "messy"
        assert cleaned.order == 3
        assert cleaned.provenance == original.provenance


class TestParsedDocument:
    def test_is_empty_for_whitespace_only_content(self) -> None:
        document = ParsedDocument(
            blocks=[ContentBlock(BlockType.PARAGRAPH, "   \n ", 0, Provenance.for_page(1))],
            file_type=SupportedFileType.PDF,
        )
        assert document.is_empty is True

    def test_is_not_empty_with_real_content(self) -> None:
        document = ParsedDocument(
            blocks=[ContentBlock(BlockType.PARAGRAPH, "real", 0, Provenance.for_page(1))],
            file_type=SupportedFileType.PDF,
        )
        assert document.is_empty is False

    def test_no_blocks_is_empty(self) -> None:
        assert ParsedDocument(blocks=[], file_type=SupportedFileType.TXT).is_empty is True

    def test_warnings_default_to_empty_but_are_carried(self) -> None:
        """Partial success must not look like clean success."""
        document = ParsedDocument(
            blocks=[],
            file_type=SupportedFileType.PDF,
            warnings=["3 pages had no extractable text"],
        )
        assert document.warnings == ["3 pages had no extractable text"]

    def test_total_characters(self) -> None:
        document = ParsedDocument(
            blocks=[
                ContentBlock(BlockType.PARAGRAPH, "12345", 0, Provenance.for_page(1)),
                ContentBlock(BlockType.PARAGRAPH, "123", 1, Provenance.for_page(1)),
            ],
            file_type=SupportedFileType.PDF,
        )
        assert document.total_characters == 8


class TestParseStats:
    def test_page_count_prefers_pages_then_sheets(self) -> None:
        assert ParseStats(pages=10).page_count == 10
        assert ParseStats(sheets=3).page_count == 3
        assert ParseStats().page_count is None
