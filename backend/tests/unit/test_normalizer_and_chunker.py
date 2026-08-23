"""Tests for text normalization and structure-aware chunking."""

from __future__ import annotations

import pytest

from app.ingestion.chunker import chunk_blocks, estimate_tokens
from app.ingestion.normalizer import merge_short_blocks, normalize_document, normalize_text
from app.models.domain.content import (
    BlockType,
    ContentBlock,
    ParsedDocument,
    Provenance,
)
from app.models.domain.document import SupportedFileType


def _block(
    text: str,
    order: int = 0,
    block_type: BlockType = BlockType.PARAGRAPH,
    page: int = 1,
) -> ContentBlock:
    return ContentBlock(
        block_type=block_type,
        text=text,
        order=order,
        provenance=Provenance.for_page(page),
    )


class TestNormalizeText:
    def test_collapses_runs_of_spaces(self) -> None:
        assert normalize_text("too     many    spaces") == "too many spaces"

    def test_collapses_excess_blank_lines(self) -> None:
        assert normalize_text("a\n\n\n\n\nb") == "a\n\nb"

    def test_normalises_line_endings(self) -> None:
        assert normalize_text("a\r\nb\rc") == "a\nb\nc"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("“quoted”", '"quoted"'),
            ("it’s", "it's"),
            ("dash — here", "dash - here"),
            # The "fi" ligature, as PDFs and Word commonly emit it.
            ("ofﬁce", "office"),
        ],
    )
    def test_normalises_typography(self, raw: str, expected: str) -> None:
        """A search for a quoted phrase must match text copied out of Word or a PDF."""
        assert normalize_text(raw) == expected

    def test_strips_invisible_characters(self) -> None:
        """Zero-width characters look like nothing but break search and word counts."""
        assert normalize_text("hel​lo­") == "hello"

    def test_strips_byte_order_mark(self) -> None:
        assert normalize_text("﻿hello") == "hello"

    def test_preserve_lines_keeps_row_structure(self) -> None:
        """Table rows are records; collapsing their newlines would merge them."""
        text = "a: 1 | b: 2\na: 3 | b: 4"

        assert normalize_text(text, preserve_lines=True) == text

    def test_preserve_lines_drops_blank_lines(self) -> None:
        assert normalize_text("row one\n\n\nrow two", preserve_lines=True) == ("row one\nrow two")

    def test_is_idempotent(self) -> None:
        """Normalizing twice must equal normalizing once."""
        raw = "  “Messy”   text — with​ junk  \n\n\n\n  and more.  "

        once = normalize_text(raw)
        assert normalize_text(once) == once

    def test_empty_input(self) -> None:
        assert normalize_text("") == ""
        assert normalize_text("   \n  ") == ""


class TestNormalizeDocument:
    def test_preserves_provenance(self) -> None:
        """The single rule normalization must never break."""
        document = ParsedDocument(
            blocks=[_block("  messy   text  ", 0, page=7)],
            file_type=SupportedFileType.PDF,
        )

        result = normalize_document(document)

        assert result.blocks[0].text == "messy text"
        assert result.blocks[0].provenance.page == 7
        assert result.blocks[0].provenance.label == "page 7"

    def test_renumbers_after_dropping_empty_blocks(self) -> None:
        """`order` must stay a dense sequence for downstream code."""
        document = ParsedDocument(
            blocks=[_block("real", 0), _block("​  ­", 1), _block("also real", 2)],
            file_type=SupportedFileType.TXT,
        )

        result = normalize_document(document)

        assert [b.order for b in result.blocks] == [0, 1]
        assert [b.text for b in result.blocks] == ["real", "also real"]

    def test_dropped_blocks_are_reported_not_silently_lost(self) -> None:
        document = ParsedDocument(
            blocks=[_block("real", 0), _block("​", 1)],
            file_type=SupportedFileType.TXT,
        )

        result = normalize_document(document)

        assert any("empty sections were removed" in w for w in result.warnings)

    def test_removes_repeated_page_headers(self) -> None:
        """A running head extracted from every page would otherwise pollute summaries."""
        blocks = [
            _block(f"ACME CONFIDENTIAL\nBody text for page {page}.", page - 1, page=page)
            for page in range(1, 7)
        ]
        document = ParsedDocument(blocks=blocks, file_type=SupportedFileType.PDF)

        result = normalize_document(document)

        assert all("ACME CONFIDENTIAL" not in b.text for b in result.blocks)
        assert all("Body text for page" in b.text for b in result.blocks)
        assert any("page header/footer" in w for w in result.warnings)

    def test_does_not_strip_repeats_from_short_documents(self) -> None:
        """Two pages is a coincidence, not a pattern — removing content would be worse."""
        blocks = [_block(f"Shared line\nUnique {page}.", page - 1, page=page) for page in (1, 2)]
        document = ParsedDocument(blocks=blocks, file_type=SupportedFileType.PDF)

        result = normalize_document(document)

        assert all("Shared line" in b.text for b in result.blocks)

    def test_tabular_blocks_keep_their_lines(self) -> None:
        document = ParsedDocument(
            blocks=[_block("a: 1\na: 2", 0, block_type=BlockType.SHEET_ROW)],
            file_type=SupportedFileType.CSV,
        )

        result = normalize_document(document)

        assert result.blocks[0].text == "a: 1\na: 2"


class TestMergeShortBlocks:
    def test_merges_tiny_consecutive_paragraphs(self) -> None:
        blocks = [_block("Short.", 0), _block("Also short.", 1)]

        merged = merge_short_blocks(blocks, minimum=40)

        assert len(merged) == 1
        assert merged[0].text == "Short. Also short."

    def test_never_merges_headings(self) -> None:
        blocks = [_block("Title", 0, BlockType.HEADING), _block("Body.", 1)]

        merged = merge_short_blocks(blocks, minimum=40)

        assert len(merged) == 2

    def test_never_merges_tables(self) -> None:
        blocks = [_block("a|b", 0, BlockType.TABLE), _block("x|y", 1, BlockType.TABLE)]

        assert len(merge_short_blocks(blocks, minimum=40)) == 2

    def test_does_not_merge_across_pages(self) -> None:
        blocks = [_block("Short.", 0, page=1), _block("Other.", 1, page=2)]

        assert len(merge_short_blocks(blocks, minimum=40)) == 2

    def test_does_not_merge_line_based_blocks(self) -> None:
        """Merging would attribute the second block's text to the first block's lines.

        The merged block keeps the first block's provenance, which is truthful only when
        both came from the same place. Two paragraphs on different lines of a text file
        do not qualify, so they are left alone.
        """
        blocks = [
            ContentBlock(BlockType.PARAGRAPH, "Short.", 0, Provenance.for_lines(1, 1)),
            ContentBlock(BlockType.PARAGRAPH, "Other.", 1, Provenance.for_lines(3, 3)),
        ]

        merged = merge_short_blocks(blocks, minimum=40)

        assert len(merged) == 2
        assert merged[0].provenance.line_start == 1
        assert merged[1].provenance.line_start == 3

    def test_orders_stay_contiguous(self) -> None:
        blocks = [_block("a", 0), _block("b", 1), _block("c" * 100, 2)]

        merged = merge_short_blocks(blocks, minimum=40)

        assert [b.order for b in merged] == list(range(len(merged)))


class TestChunker:
    def test_empty_input(self) -> None:
        assert chunk_blocks([]) == []

    def test_small_document_becomes_one_chunk(self) -> None:
        blocks = [_block("Short paragraph.", 0), _block("Another one.", 1)]

        chunks = chunk_blocks(blocks, target_chars=1500, overlap_chars=0)

        assert len(chunks) == 1
        assert chunks[0].block_indices == [0, 1]

    def test_respects_the_target_size(self) -> None:
        blocks = [_block("x" * 400, index) for index in range(10)]

        chunks = chunk_blocks(blocks, target_chars=1000, overlap_chars=0)

        assert all(chunk.char_count <= 1000 for chunk in chunks)

    def test_headings_start_a_new_chunk(self) -> None:
        """Keeps a heading with the text it introduces, not stranded at the end."""
        blocks = [
            _block("Intro paragraph.", 0),
            _block("Section Two", 1, BlockType.HEADING),
            _block("Section two body.", 2),
        ]

        chunks = chunk_blocks(blocks, target_chars=1500, overlap_chars=0)

        assert len(chunks) == 2
        assert chunks[1].block_indices == [1, 2]
        assert chunks[1].text.startswith("Section Two")

    def test_tables_are_never_split(self) -> None:
        """Half a table is worse than no table."""
        table_text = "\n".join(f"row {i} | value {i}" for i in range(200))
        blocks = [_block(table_text, 0, BlockType.TABLE)]

        chunks = chunk_blocks(blocks, target_chars=200, overlap_chars=0)

        assert len(chunks) == 1
        assert chunks[0].text == table_text

    def test_schema_blocks_are_never_split(self) -> None:
        blocks = [_block("x" * 5000, 0, BlockType.SCHEMA)]

        chunks = chunk_blocks(blocks, target_chars=100, overlap_chars=0)

        assert len(chunks) == 1

    def test_tables_do_not_share_a_chunk(self) -> None:
        """A retrieved table should arrive whole and uncluttered."""
        blocks = [
            _block("Before.", 0),
            _block("a | b", 1, BlockType.TABLE),
            _block("After.", 2),
        ]

        chunks = chunk_blocks(blocks, target_chars=1500, overlap_chars=0)

        table_chunk = next(c for c in chunks if 1 in c.block_indices)
        assert table_chunk.block_indices == [1]

    def test_oversized_block_splits_on_sentences(self) -> None:
        sentence = "This is a complete sentence with some length to it. "
        blocks = [_block(sentence * 40, 0)]

        chunks = chunk_blocks(blocks, target_chars=500, overlap_chars=0)

        assert len(chunks) > 1
        # Every piece still points back to the block it came from.
        assert all(chunk.block_indices == [0] for chunk in chunks)

    def test_a_single_enormous_sentence_still_splits(self) -> None:
        """No sentence boundary exists, so a hard split is the only option left."""
        blocks = [_block("word " * 2000, 0)]

        chunks = chunk_blocks(blocks, target_chars=500, overlap_chars=0)

        assert len(chunks) > 1
        assert all(chunk.char_count <= 600 for chunk in chunks)

    def test_every_block_appears_in_some_chunk(self) -> None:
        """The invariant that proves no content is silently dropped."""
        blocks = [
            _block("Heading", 0, BlockType.HEADING),
            _block("x" * 900, 1),
            _block("a | b", 2, BlockType.TABLE),
            _block("y" * 900, 3),
            _block("Short.", 4),
        ]

        chunks = chunk_blocks(blocks, target_chars=1000, overlap_chars=50)

        covered = {index for chunk in chunks for index in chunk.block_indices}
        assert covered == {0, 1, 2, 3, 4}

    def test_chunks_are_in_document_order(self) -> None:
        blocks = [_block("x" * 400, index) for index in range(8)]

        chunks = chunk_blocks(blocks, target_chars=800, overlap_chars=0)

        firsts = [chunk.block_indices[0] for chunk in chunks]
        assert firsts == sorted(firsts)

    def test_overlap_repeats_text_from_the_previous_chunk(self) -> None:
        """A fact spanning a boundary must not be lost to both chunks."""
        blocks = [_block("alpha " * 100, 0), _block("beta " * 100, 1)]

        with_overlap = chunk_blocks(blocks, target_chars=700, overlap_chars=100)
        without_overlap = chunk_blocks(blocks, target_chars=700, overlap_chars=0)

        assert len(with_overlap) == len(without_overlap)
        assert with_overlap[1].char_count > without_overlap[1].char_count

    def test_overlap_must_be_smaller_than_the_target(self) -> None:
        """Otherwise chunking could never advance."""
        with pytest.raises(ValueError, match="smaller than"):
            chunk_blocks([_block("text", 0)], target_chars=100, overlap_chars=100)

    def test_location_label_for_a_single_page(self) -> None:
        blocks = [_block("a", 0, page=3), _block("b", 1, page=3)]

        chunks = chunk_blocks(blocks, target_chars=1500, overlap_chars=0)

        assert chunks[0].location_label == "page 3"

    def test_location_label_spans_multiple_pages(self) -> None:
        blocks = [_block("a", 0, page=3), _block("b", 1, page=4)]

        chunks = chunk_blocks(blocks, target_chars=1500, overlap_chars=0)

        assert chunks[0].location_label == "pages 3-4"

    def test_token_estimate_is_proportional(self) -> None:
        assert estimate_tokens("x" * 400) == 100
        assert estimate_tokens("") == 1  # never zero, so sizing maths cannot divide by it
