"""Tests for context building, citation resolution and optional optimization.

All three are deterministic, so they are tested with no model involved at all. That is
the point of keeping this work in ordinary code.

The most important tests here are the ones asserting that a location is **never
invented** — an unresolvable citation must come back unverified, not plausible.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from app.ai.citations import resolve_citations, verify_quote
from app.ai.context_builder import build_context, describe_context
from app.ai.optimization import score_chunk, select_chunks, should_offer_optimization
from app.core.exceptions import InsufficientContentError
from app.models.domain.content import BlockType, ContentBlock, Provenance
from app.models.domain.insight import ContextBlock, ContextMode, RawCitation
from app.repositories.chunk_repo import Chunk


def _chunk(
    text: str, *, order: int = 0, label: str = "page 1", block_id: UUID | None = None
) -> Chunk:
    return Chunk(
        id=uuid4(),
        order=order,
        text=text,
        block_ids=[block_id or uuid4()],
        token_estimate=max(1, len(text) // 4),
        char_count=len(text),
        location_label=label,
    )


def _content_block(page: int = 1, text: str = "block text") -> ContentBlock:
    return ContentBlock(
        block_type=BlockType.PARAGRAPH,
        text=text,
        order=0,
        provenance=Provenance.for_page(page),
    )


class TestBuildContext:
    def test_sends_everything_when_it_fits(self) -> None:
        """The default and the best answer: the model sees the whole document."""
        chunks = [_chunk("Some content here.", order=i) for i in range(5)]

        blocks, stats = build_context(chunks, max_context_tokens=600_000)

        assert len(blocks) == 5
        assert stats.mode is ContextMode.FULL
        assert stats.is_complete
        assert stats.omitted_chunk_count == 0

    def test_block_order_matches_chunk_order(self) -> None:
        """Position in this list *is* the citation coordinate system."""
        chunks = [_chunk(f"chunk {i}", order=i) for i in range(6)]

        blocks, _ = build_context(chunks, max_context_tokens=600_000)

        assert [b.text for b in blocks] == [c.text for c in chunks]
        assert [b.chunk_id for b in blocks] == [c.id for c in chunks]

    def test_blocks_carry_identity_for_later_resolution(self) -> None:
        block_id = uuid4()
        chunks = [_chunk("text", label="page 7", block_id=block_id)]

        blocks, _ = build_context(chunks, max_context_tokens=600_000)

        assert blocks[0].block_ids == (block_id,)
        assert blocks[0].location_label == "page 7"

    def test_empty_document_is_rejected(self) -> None:
        """Better than asking the model to summarise nothing, which invites invention."""
        with pytest.raises(InsufficientContentError):
            build_context([], max_context_tokens=600_000)

    def test_optimization_is_skipped_for_small_documents(self) -> None:
        """Trimming a short document loses detail for no benefit."""
        chunks = [_chunk("Short content.", order=i) for i in range(3)]

        blocks, stats = build_context(
            chunks, max_context_tokens=600_000, optimize=True, optimization_min_tokens=20_000
        )

        assert stats.mode is ContextMode.FULL
        assert len(blocks) == 3

    def test_optimization_reduces_a_large_document_when_requested(self) -> None:
        chunks = [_chunk("word " * 500, order=i) for i in range(40)]

        blocks, stats = build_context(
            chunks,
            max_context_tokens=600_000,
            optimize=True,
            optimization_min_tokens=1_000,
            optimization_target_ratio=0.2,
        )

        assert stats.mode is ContextMode.OPTIMIZED
        assert len(blocks) < len(chunks)
        assert stats.omitted_chunk_count == len(chunks) - len(blocks)

    def test_oversized_document_is_reduced_even_without_opting_in(self) -> None:
        """Refusing outright would be worse; the omission is reported instead."""
        chunks = [_chunk("word " * 400, order=i) for i in range(50)]

        blocks, stats = build_context(chunks, max_context_tokens=3_000)

        assert stats.mode is ContextMode.OPTIMIZED
        assert len(blocks) < len(chunks)

    def test_omissions_are_reported_to_the_user(self) -> None:
        """CLAUDE.md forbids silently truncating user content."""
        chunks = [_chunk("word " * 400, order=i) for i in range(50)]

        _, stats = build_context(chunks, max_context_tokens=3_000)
        warnings = describe_context(stats)

        assert warnings
        assert "left out" in warnings[0]

    def test_optimized_context_says_the_original_is_untouched(self) -> None:
        chunks = [_chunk("word " * 500, order=i) for i in range(40)]

        _, stats = build_context(
            chunks,
            max_context_tokens=600_000,
            optimize=True,
            optimization_min_tokens=1_000,
            optimization_target_ratio=0.5,
        )
        warnings = describe_context(stats)

        assert any("unchanged" in warning for warning in warnings)

    def test_full_context_produces_no_warnings(self) -> None:
        chunks = [_chunk("Content.", order=i) for i in range(3)]

        _, stats = build_context(chunks, max_context_tokens=600_000)

        assert describe_context(stats) == []


class TestOptimizationPreservesWhatMatters:
    def test_originals_are_never_modified(self) -> None:
        """The whole safety property: selection reads, it does not write."""
        chunks = [_chunk(f"Original text {i}. " * 50, order=i) for i in range(30)]
        snapshot = [(c.id, c.text, tuple(c.block_ids), c.location_label) for c in chunks]

        select_chunks(chunks, target_tokens=200)

        after = [(c.id, c.text, tuple(c.block_ids), c.location_label) for c in chunks]
        assert after == snapshot

    def test_selected_chunks_are_verbatim(self) -> None:
        """Kept text is byte-identical, which is what keeps citations exact."""
        chunks = [_chunk(f"Exact wording {i} must survive. " * 20, order=i) for i in range(20)]

        selected = select_chunks(chunks, target_tokens=300)

        originals = {c.id: c.text for c in chunks}
        assert all(chunk.text == originals[chunk.id] for chunk in selected)

    def test_document_order_is_preserved(self) -> None:
        """Sections read out of sequence would mislead the model about what follows what."""
        chunks = [_chunk(f"Section {i} content. " * 30, order=i) for i in range(30)]

        selected = select_chunks(chunks, target_tokens=400)

        orders = [chunk.order for chunk in selected]
        assert orders == sorted(orders)

    def test_the_opening_of_the_document_is_always_kept(self) -> None:
        """Titles, parties and dates live at the front."""
        chunks = [_chunk(f"Opening or body text {i}. " * 20, order=i) for i in range(40)]

        selected = select_chunks(chunks, target_tokens=300)

        assert selected[0].order == 0

    def test_content_with_amounts_and_dates_outranks_filler(self) -> None:
        """The facts users actually ask about must survive the cut."""
        facts = _chunk("The total is $45,200.00 and payment is due 2025-03-15.")
        filler = _chunk("This section contains general introductory narrative prose.")

        assert score_chunk(facts) > score_chunk(filler)

    def test_table_structure_blocks_are_prioritised(self) -> None:
        """A schema block describes a whole sheet in a few lines."""
        schema = _chunk("Worksheet structure: 3 columns", label="Q3 Budget structure")
        prose = _chunk("Some ordinary paragraph text here.", label="page 4")

        assert score_chunk(schema) > score_chunk(prose)

    def test_obligations_and_deadlines_are_prioritised(self) -> None:
        obligation = _chunk("The tenant must give 60 days notice before termination.")
        neutral = _chunk("The building has a reception area on the ground floor.")

        assert score_chunk(obligation) > score_chunk(neutral)

    def test_empty_chunk_scores_zero(self) -> None:
        assert score_chunk(_chunk("   ")) == 0.0

    def test_selection_respects_the_budget(self) -> None:
        chunks = [_chunk("word " * 200, order=i) for i in range(50)]

        selected = select_chunks(chunks, target_tokens=500)

        total = sum(max(1, len(c.text) // 4) for c in selected)
        assert total <= 500

    def test_empty_input(self) -> None:
        assert select_chunks([], target_tokens=100) == []

    @pytest.mark.parametrize(
        ("tokens", "minimum", "expected"),
        [(5_000, 20_000, False), (20_000, 20_000, True), (50_000, 20_000, True)],
    )
    def test_offer_threshold(self, tokens: int, minimum: int, expected: bool) -> None:
        assert should_offer_optimization(tokens, minimum) is expected


class TestResolveCitations:
    def _setup(self) -> tuple[list[ContextBlock], dict[object, ContentBlock]]:
        block_a, block_b = uuid4(), uuid4()
        context_blocks = [
            ContextBlock("First chunk text.", uuid4(), (block_a,), "page 1"),
            ContextBlock("Second chunk text.", uuid4(), (block_b,), "page 2"),
        ]
        blocks_by_id = {
            block_a: _content_block(page=1),
            block_b: _content_block(page=2),
        }
        return context_blocks, blocks_by_id

    def test_index_resolves_to_real_provenance(self) -> None:
        """The core mechanism: an index we assigned maps back to a location we recorded."""
        context_blocks, blocks_by_id = self._setup()
        raw = [RawCitation("Second chunk text.", start_block_index=1, end_block_index=2)]

        citations = resolve_citations(raw, context_blocks, blocks_by_id)

        assert len(citations) == 1
        assert citations[0].is_verified
        assert citations[0].location_label == "page 2"
        assert citations[0].provenance is not None
        assert citations[0].provenance.page == 2

    def test_citation_carries_chunk_and_block_identity(self) -> None:
        context_blocks, blocks_by_id = self._setup()
        raw = [RawCitation("First chunk text.", 0, 1)]

        citation = resolve_citations(raw, context_blocks, blocks_by_id)[0]

        assert citation.chunk_id == context_blocks[0].chunk_id
        assert citation.block_ids == context_blocks[0].block_ids

    def test_out_of_range_index_is_reported_not_guessed(self) -> None:
        """Pointing at the wrong place is worse than admitting we cannot place it."""
        context_blocks, blocks_by_id = self._setup()
        raw = [RawCitation("Something", start_block_index=99, end_block_index=100)]

        citations = resolve_citations(raw, context_blocks, blocks_by_id)

        assert len(citations) == 1
        assert citations[0].is_verified is False
        assert citations[0].provenance is None
        assert citations[0].location_label == "source not verified"

    def test_negative_index_is_rejected(self) -> None:
        context_blocks, blocks_by_id = self._setup()
        raw = [RawCitation("Something", start_block_index=-1, end_block_index=0)]

        assert resolve_citations(raw, context_blocks, blocks_by_id)[0].is_verified is False

    def test_unknown_block_id_yields_unverified_citation(self) -> None:
        """The index is valid but we have no provenance for it — say so."""
        context_blocks = [ContextBlock("text", uuid4(), (uuid4(),), "page 1")]

        citations = resolve_citations([RawCitation("text", 0, 1)], context_blocks, {})

        assert citations[0].is_verified is False

    def test_duplicate_citations_are_collapsed(self) -> None:
        context_blocks, blocks_by_id = self._setup()
        raw = [RawCitation("First chunk text.", 0, 1), RawCitation("First chunk text.", 0, 1)]

        assert len(resolve_citations(raw, context_blocks, blocks_by_id)) == 1

    def test_no_citations_yields_no_results(self) -> None:
        context_blocks, blocks_by_id = self._setup()
        assert resolve_citations([], context_blocks, blocks_by_id) == []


class TestVerifyQuote:
    def _setup(self) -> tuple[list[ContextBlock], dict[object, ContentBlock]]:
        block_id = uuid4()
        context_blocks = [
            ContextBlock(
                "The total amount payable is $45,200.00 under this agreement.",
                uuid4(),
                (block_id,),
                "page 3",
            )
        ]
        return context_blocks, {block_id: _content_block(page=3)}

    def test_a_real_quote_is_located(self) -> None:
        """Extraction provenance comes from finding the quote, not from trusting a claim."""
        context_blocks, blocks_by_id = self._setup()

        citation = verify_quote(
            "The total amount payable is $45,200.00", context_blocks, blocks_by_id
        )

        assert citation is not None
        assert citation.is_verified
        assert citation.location_label == "page 3"

    def test_an_invented_quote_is_rejected(self) -> None:
        """The single most important test in this file.

        A model that fabricates a quote must get no provenance at all — not a nearby
        location, not a best guess.
        """
        context_blocks, blocks_by_id = self._setup()

        citation = verify_quote(
            "The penalty for late payment is 15% per month", context_blocks, blocks_by_id
        )

        assert citation is None

    def test_whitespace_differences_do_not_break_a_genuine_quote(self) -> None:
        context_blocks, blocks_by_id = self._setup()

        citation = verify_quote(
            "The  total   amount\npayable is $45,200.00", context_blocks, blocks_by_id
        )

        assert citation is not None

    def test_smart_punctuation_differences_are_tolerated(self) -> None:
        block_id = uuid4()
        context_blocks = [
            ContextBlock(
                "The tenant's obligations - as described - apply.", uuid4(), (block_id,), "page 1"
            )
        ]
        blocks_by_id = {block_id: _content_block(page=1)}

        citation = verify_quote(
            "The tenant’s obligations — as described — apply", context_blocks, blocks_by_id
        )

        assert citation is not None

    def test_a_too_short_quote_is_not_accepted(self) -> None:
        """A three-word fragment matches almost anything, so a match proves nothing."""
        context_blocks, blocks_by_id = self._setup()

        assert verify_quote("the", context_blocks, blocks_by_id) is None

    def test_empty_quote(self) -> None:
        context_blocks, blocks_by_id = self._setup()
        assert verify_quote("", context_blocks, blocks_by_id) is None
