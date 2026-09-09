"""Tests for `app.ai.windows` — how a document is divided for windowed extraction.

Pure function, pure tests: no database, no provider, no event loop. Everything this
module decides is decidable from a list of chunks and a budget, which is the reason the
decision lives in its own file rather than inside the service that calls it.

The property under test throughout is that **windowing divides, it does not edit**. Every
chunk goes into exactly one window, whole, in order. A chunk is the unit the ingestion
chunker already established as safe to split on — it never cuts a table and always starts
fresh at a heading — so a window built from whole chunks inherits that guarantee. These
tests exist to make sure nobody later "optimises" that away by slicing text to balance
window sizes.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from app.ai.windows import (
    OUTPUT_SAFETY_RATIO,
    ContextWindow,
    adaptive_window_tokens,
    output_tokens_per_record,
    plan_windows,
)
from app.repositories.chunk_repo import Chunk

#: The provider ceiling these tests size against, matching `ai_max_output_tokens`.
CEILING = 8_000


def _chunk(
    order: int, tokens: int, *, label: str = "", block_ids: list[UUID] | None = None
) -> Chunk:
    """A chunk of a stated size. Text is padded so the stored estimate is honest."""
    return Chunk(
        id=uuid4(),
        order=order,
        text="x" * (tokens * 4),
        block_ids=block_ids if block_ids is not None else [uuid4()],
        token_estimate=tokens,
        char_count=tokens * 4,
        location_label=label or f"page {order + 1}",
    )


class TestCoverage:
    """Every chunk lands in exactly one window, unchanged."""

    def test_no_chunks_gives_no_windows(self) -> None:
        # What an empty document *means* is the caller's decision — it has a typed error
        # for it — so this returns nothing rather than raising.
        assert plan_windows([], window_tokens=1_000) == []

    def test_every_chunk_appears_exactly_once(self) -> None:
        chunks = [_chunk(index, 300) for index in range(20)]

        windows = plan_windows(chunks, window_tokens=1_000)

        seen = [chunk.id for window in windows for chunk in window.chunks]
        assert seen == [chunk.id for chunk in chunks]

    def test_windows_do_not_overlap(self) -> None:
        """Overlap would extract the same record twice and leave the merge deciding
        whether two records are the same — a judgement about the user's data."""
        chunks = [_chunk(index, 400) for index in range(12)]

        windows = plan_windows(chunks, window_tokens=1_000)

        seen = [chunk.id for window in windows for chunk in window.chunks]
        assert len(seen) == len(set(seen))

    def test_chunk_text_is_never_altered(self) -> None:
        chunks = [_chunk(index, 250) for index in range(8)]

        windows = plan_windows(chunks, window_tokens=600)

        rebuilt = [chunk.text for window in windows for chunk in window.chunks]
        assert rebuilt == [chunk.text for chunk in chunks]


class TestBoundaries:
    """A chunk is the smallest unit. It is never divided."""

    def test_a_chunk_larger_than_the_budget_becomes_its_own_window(self) -> None:
        """The case that would tempt a naive implementation into slicing text.

        A single oversized chunk is one indivisible structure — the chunker only produces
        one when a table or a schema block could not be broken up — so it is sent whole
        and slightly over budget. Half a table is worse than a large request.
        """
        chunks = [_chunk(0, 100), _chunk(1, 5_000), _chunk(2, 100)]

        windows = plan_windows(chunks, window_tokens=1_000)

        oversized = [w for w in windows if any(c.token_estimate == 5_000 for c in w.chunks)]
        assert len(oversized) == 1
        assert len(oversized[0].chunks) == 1

    def test_a_window_closes_before_it_overflows_never_after(self) -> None:
        """Sized on the way in. A window allowed to overflow and split afterwards has
        already built the oversized request the budget exists to prevent."""
        chunks = [_chunk(index, 400) for index in range(9)]

        windows = plan_windows(chunks, window_tokens=1_000)

        # 400 + 400 = 800 fits; a third would make 1200. So two per window.
        assert [len(window.chunks) for window in windows] == [2, 2, 2, 2, 1]
        for window in windows[:-1]:
            assert window.token_estimate <= 1_000

    def test_a_document_that_fits_produces_exactly_one_window(self) -> None:
        """The small-document case. One window is the signal the service uses to send a
        document down the original single-call path untouched."""
        chunks = [_chunk(index, 100) for index in range(5)]

        assert len(plan_windows(chunks, window_tokens=40_000)) == 1

    def test_a_budget_of_zero_still_makes_progress(self) -> None:
        """Degenerate configuration must not loop forever or drop content."""
        chunks = [_chunk(index, 10) for index in range(3)]

        windows = plan_windows(chunks, window_tokens=0)

        assert [len(window.chunks) for window in windows] == [1, 1, 1]


class TestOrdering:
    """Window order is document order, which is what makes merging a concatenation."""

    def test_windows_are_indexed_in_order_from_zero(self) -> None:
        chunks = [_chunk(index, 500) for index in range(10)]

        windows = plan_windows(chunks, window_tokens=1_000)

        assert [window.index for window in windows] == list(range(len(windows)))

    def test_chunks_keep_their_document_order_inside_a_window(self) -> None:
        chunks = [_chunk(index, 100) for index in range(6)]

        windows = plan_windows(chunks, window_tokens=10_000)

        assert [chunk.order for chunk in windows[0].chunks] == [0, 1, 2, 3, 4, 5]

    def test_the_short_window_falls_at_the_end(self) -> None:
        """Greedy filling, not balancing. Balancing would choose boundaries by
        arithmetic, which is how a run of related rows gets divided at a midpoint."""
        chunks = [_chunk(index, 300) for index in range(7)]

        windows = plan_windows(chunks, window_tokens=900)

        sizes = [len(window.chunks) for window in windows]
        assert sizes == [3, 3, 1]


class TestLabels:
    """A window can say where it is, in the same words a citation would use."""

    def test_a_single_chunk_window_reports_that_chunk_s_label(self) -> None:
        window = ContextWindow(index=0, chunks=(_chunk(0, 10, label="page 4"),))

        assert window.location_label == "page 4"

    def test_a_multi_chunk_window_reports_a_span(self) -> None:
        window = ContextWindow(
            index=0,
            chunks=(_chunk(0, 10, label="page 4"), _chunk(1, 10, label="page 9")),
        )

        assert window.location_label == "page 4 to page 9"

    def test_the_token_estimate_is_the_sum_of_its_chunks(self) -> None:
        window = ContextWindow(index=0, chunks=(_chunk(0, 120), _chunk(1, 80)))

        assert window.token_estimate == 200


class TestReplanningTheTail:
    """Re-planning the rest of a document at a new size keeps window numbering honest."""

    def test_windows_can_be_numbered_from_an_offset(self) -> None:
        chunks = [_chunk(index, 100) for index in range(6)]

        windows = plan_windows(chunks, window_tokens=200, start_index=1)

        assert [window.index for window in windows] == [1, 2, 3]

    def test_without_an_offset_numbering_starts_at_zero(self) -> None:
        chunks = [_chunk(index, 100) for index in range(4)]

        assert plan_windows(chunks, window_tokens=200)[0].index == 0


class TestRecordCost:
    """What one record costs in the reply, from the schema alone."""

    def test_cost_scales_with_the_number_of_fields(self) -> None:
        """The half of the sizing problem that needs no measurement. A two-field and a
        twenty-field extraction differ tenfold over the same text."""
        assert output_tokens_per_record(20) > output_tokens_per_record(10)
        assert output_tokens_per_record(10) == 10 * 25

    def test_a_tiny_schema_still_costs_something(self) -> None:
        """A record carries JSON structure even with one field. Treating it as nearly
        free would size windows far too generously for very short records."""
        assert output_tokens_per_record(1) == 40
        assert output_tokens_per_record(0) == 40


class TestAdaptiveSizing:
    """Choosing a window size from measured density and the schema.

    The property that matters is stated as an assertion in
    `test_the_chosen_window_never_plans_more_records_than_a_reply_holds`: whatever the
    inputs, the records a window is expected to hold must cost less than the ceiling.
    Everything else here is about how the bounds and the degenerate cases behave.
    """

    def test_a_sparse_document_gets_a_large_window(self) -> None:
        """Prose: a few records per thousand tokens, so a long section fits one reply.

        8,000 x 0.6 = 4,800 output tokens to spend, at 5 x 25 = 125 per record, is 38
        records; at 3 records per 1,000 tokens that is a 12,800-token window. Asserted
        exactly, because a formula whose output nobody has worked out by hand is a
        formula nobody is checking.
        """
        chosen = adaptive_window_tokens(
            observed_records=6,  # 3 records per 1,000 tokens
            observed_tokens=2_000,
            field_count=5,
            max_output_tokens=CEILING,
            min_window_tokens=2_000,
            max_window_tokens=40_000,
        )

        assert chosen == 12_800

    def test_a_very_sparse_document_is_capped_by_the_maximum(self) -> None:
        """The ceiling is what stops a document of mostly prose from being read in one
        enormous window, which would give up the point of windowing."""
        chosen = adaptive_window_tokens(
            observed_records=1,  # 0.5 records per 1,000 tokens
            observed_tokens=2_000,
            field_count=5,
            max_output_tokens=CEILING,
            min_window_tokens=2_000,
            max_window_tokens=40_000,
        )

        assert chosen == 40_000

    def test_a_dense_document_gets_a_small_window(self) -> None:
        """A register: many records per thousand tokens, so the same size would overrun
        the reply many times over. This is the case a fixed 40,000 got wrong."""
        chosen = adaptive_window_tokens(
            observed_records=50,  # 25 records per 1,000 tokens
            observed_tokens=2_000,
            field_count=5,
            max_output_tokens=CEILING,
            min_window_tokens=2_000,
            max_window_tokens=40_000,
        )

        assert chosen < 10_000
        assert chosen >= 2_000

    def test_density_is_what_separates_the_two(self) -> None:
        """Same schema, same ceiling, same observed window size — only the record count
        differs, and the window size moves with it."""
        sparse = adaptive_window_tokens(
            observed_records=5,
            observed_tokens=2_000,
            field_count=5,
            max_output_tokens=CEILING,
            min_window_tokens=500,
            max_window_tokens=40_000,
        )
        dense = adaptive_window_tokens(
            observed_records=100,
            observed_tokens=2_000,
            field_count=5,
            max_output_tokens=CEILING,
            min_window_tokens=500,
            max_window_tokens=40_000,
        )

        assert sparse > dense

    def test_more_fields_shrink_the_window_at_the_same_density(self) -> None:
        """The other half of the cost. A twenty-field schema fills a reply with a
        fraction of the records a two-field one would."""
        few_fields = adaptive_window_tokens(
            observed_records=20,
            observed_tokens=2_000,
            field_count=2,
            max_output_tokens=CEILING,
            min_window_tokens=500,
            max_window_tokens=40_000,
        )
        many_fields = adaptive_window_tokens(
            observed_records=20,
            observed_tokens=2_000,
            field_count=20,
            max_output_tokens=CEILING,
            min_window_tokens=500,
            max_window_tokens=40_000,
        )

        assert many_fields < few_fields

    def test_the_chosen_window_never_plans_more_records_than_a_reply_holds(self) -> None:
        """The safety property, checked across the whole space rather than at a point.

        For every density and schema below, the records the chosen window is expected to
        contain must cost less than the ceiling. Where the floor forces a window larger
        than the formula wanted, that is exempt — no window size can rescue a document
        denser than the floor allows, which is why an overrun there is reported as a
        failed window rather than hidden.
        """
        for records in (1, 5, 20, 100, 500):
            for fields in (1, 3, 8, 20):
                chosen = adaptive_window_tokens(
                    observed_records=records,
                    observed_tokens=2_000,
                    field_count=fields,
                    max_output_tokens=CEILING,
                    min_window_tokens=500,
                    max_window_tokens=40_000,
                )
                if chosen == 500:
                    continue  # clamped to the floor; see the docstring

                density = records / 2_000
                expected_records = chosen * density
                expected_output = expected_records * output_tokens_per_record(fields)
                assert expected_output <= CEILING, (
                    f"{records} records / {fields} fields -> window {chosen} "
                    f"plans {expected_output:.0f} output tokens"
                )

    def test_the_safety_ratio_leaves_real_headroom(self) -> None:
        """Every term feeding the formula is an estimate, so the target is deliberately
        below the ceiling rather than at it."""
        chosen = adaptive_window_tokens(
            observed_records=20,
            observed_tokens=2_000,
            field_count=5,
            max_output_tokens=CEILING,
            min_window_tokens=100,
            max_window_tokens=1_000_000,
        )

        planned_output = (chosen * 20 / 2_000) * output_tokens_per_record(5)
        assert planned_output <= CEILING * OUTPUT_SAFETY_RATIO + 1

    def test_no_records_observed_falls_back_to_the_maximum(self) -> None:
        """A cover page or a preamble states no records. That is no evidence of density,
        and forcing the whole document to minimum windows on the strength of its first
        page would multiply the cost of every long report."""
        chosen = adaptive_window_tokens(
            observed_records=0,
            observed_tokens=2_000,
            field_count=5,
            max_output_tokens=CEILING,
            min_window_tokens=2_000,
            max_window_tokens=40_000,
        )

        assert chosen == 40_000

    def test_a_zero_token_observation_falls_back_rather_than_dividing_by_zero(self) -> None:
        chosen = adaptive_window_tokens(
            observed_records=5,
            observed_tokens=0,
            field_count=5,
            max_output_tokens=CEILING,
            min_window_tokens=2_000,
            max_window_tokens=40_000,
        )

        assert chosen == 40_000

    def test_the_result_is_always_within_the_bounds(self) -> None:
        for records in (0, 1, 10, 1_000, 100_000):
            chosen = adaptive_window_tokens(
                observed_records=records,
                observed_tokens=1_000,
                field_count=6,
                max_output_tokens=CEILING,
                min_window_tokens=1_500,
                max_window_tokens=30_000,
            )
            assert 1_500 <= chosen <= 30_000
