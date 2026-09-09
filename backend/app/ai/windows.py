"""Grouping a document's stored chunks into windows, one per extraction call.

Windowing exists because of an output ceiling, not an input one. A conversion's
extraction step must emit *every* record it finds in a single structured reply, and that
reply is capped by ``ai_max_output_tokens`` however much context was sent in. Past some
size a document simply holds more records than one reply can carry, and the surplus is
lost — silently, before the provider learned to say so. Reading the document in windows
means each reply only has to carry the records of its own window.

**Nothing here re-chunks anything.** The ingestion chunker already did the structural
work: a heading starts a new chunk, and a table or schema block is never split across
chunks. Every chunk boundary is therefore already a safe place to divide a document, and
a window built out of whole chunks inherits that property without re-deriving it. This
module only decides *how many* consecutive chunks travel together.

Two consequences worth stating, because they are the reason this file is short:

* **A chunk is never split.** If one chunk alone exceeds the budget it becomes a window
  of its own — sending it whole and slightly over is better than cutting a table in half,
  which is the exact failure the chunker was written to avoid.
* **Windows never overlap.** Overlap would put the same content in two calls, produce the
  same record twice, and leave the merge step deciding whether two records are "the
  same" — a judgement about the user's data that no part of this system is entitled to
  make.

The size of a window is not a constant, and cannot be. What fills a reply is
``records x tokens per record``, and both vary per document: how densely a source packs
records is a property of the source, and what one record costs depends on how many fields
the schema asks for. A single number can only be right at one point in that space — a
40,000-token window is about right for a contract and eight times too large for a register
of the same length. :func:`adaptive_window_tokens` derives the size from those two terms
instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.repositories.chunk_repo import Chunk

#: Output tokens one extracted field costs in the reply.
#:
#: A field emits its name, its value and a source quote, and the quote — which repeats
#: text from the document — is most of it. This is why a reply can be larger than the
#: passage that produced it, and why the *output* ceiling binds long before the input
#: context window does.
OUTPUT_TOKENS_PER_FIELD = 25

#: Floor on a record's output cost, for a schema of one or two fields. Even a tiny record
#: carries JSON structure, and treating it as nearly free would size windows far too
#: generously for a document made of very short records.
MIN_OUTPUT_TOKENS_PER_RECORD = 40

#: Share of the output ceiling a window is allowed to aim for.
#:
#: The headroom is not timidity: every term feeding this is an estimate — the token
#: figures above, and a density measured from one window of the document. Aiming at the
#: whole ceiling would mean a document only slightly denser than its opening section
#: overruns it, and an overrun costs a whole window's work.
OUTPUT_SAFETY_RATIO = 0.6


@dataclass(frozen=True, slots=True)
class ContextWindow:
    """One extraction call's worth of a document.

    Attributes:
        index: Position in the window sequence, from zero. This is document order —
            windows are built from chunks in order and never reordered — which is what
            lets merged records be concatenated window by window and still read in the
            order the document wrote them.
        chunks: The chunks it covers, in order. Whole chunks, always.
    """

    index: int
    chunks: tuple[Chunk, ...]

    @property
    def token_estimate(self) -> int:
        """Estimated size, from the counts ingestion already stored per chunk."""
        return sum(chunk.token_estimate for chunk in self.chunks)

    @property
    def location_label(self) -> str:
        """Where this window sits in the source, for logs and warnings.

        Built from the chunks' own labels rather than from a page count of our own: the
        label is the same string the citation system shows, so a warning about window 3
        and a citation inside it describe the same place in the same words.
        """
        if not self.chunks:
            return "empty"
        first = self.chunks[0].location_label
        last = self.chunks[-1].location_label
        return first if first == last else f"{first} to {last}"


def plan_windows(
    chunks: list[Chunk], *, window_tokens: int, start_index: int = 0
) -> list[ContextWindow]:
    """Group consecutive chunks into windows of at most ``window_tokens`` each.

    Greedy and single-pass: chunks accumulate in document order until the next one would
    push the window over budget, at which point the window closes and a new one opens.

    Greedy rather than balanced on purpose. A balanced split would even out the window
    sizes, but only by choosing boundaries on the basis of arithmetic — which is exactly
    how a run of related rows ends up divided because it happened to fall near a
    midpoint. Filling each window until it is full keeps related content together for as
    long as the budget allows, and leaves the short window at the end rather than in the
    middle of the document.

    Args:
        chunks: The document's chunks, in order, as ingestion stored them.
        window_tokens: Budget per window, in estimated tokens.
        start_index: Number the first window from here. Used when re-planning the tail of
            a document at a new size, so window indices stay document positions rather
            than restarting at zero halfway through.

    Returns:
        Windows covering every chunk exactly once, in document order. An empty input
        gives an empty list — deciding what a document with no content means belongs to
        the caller, which has a typed error for it.
    """
    if not chunks:
        return []

    budget = max(window_tokens, 1)

    windows: list[ContextWindow] = []
    current: list[Chunk] = []
    current_tokens = 0

    for chunk in chunks:
        size = chunk.token_estimate

        # Close the window *before* adding, so the check is "would this overflow?" and
        # never "did this overflow?". A window that is allowed to go over and then split
        # afterwards has already built the oversized request.
        if current and current_tokens + size > budget:
            windows.append(ContextWindow(index=len(windows), chunks=tuple(current)))
            current = []
            current_tokens = 0

        current.append(chunk)
        current_tokens += size

    if current:
        windows.append(ContextWindow(index=len(windows), chunks=tuple(current)))

    if start_index:
        # Re-numbered rather than restarted, so a window's index stays its position in
        # the document even when this call only planned the tail of one. The merge relies
        # on that: index order is document order.
        windows = [
            ContextWindow(index=start_index + offset, chunks=window.chunks)
            for offset, window in enumerate(windows)
        ]

    return windows


def output_tokens_per_record(field_count: int) -> int:
    """What one extracted record is expected to cost in the reply.

    Known before any extraction runs, because the schema is resolved first — which is the
    half of the sizing problem that needs no measurement. A two-field extraction and a
    twenty-field one differ tenfold for the same passage of text, and a window size that
    ignores that is wrong for one of them by an order of magnitude.
    """
    return max(MIN_OUTPUT_TOKENS_PER_RECORD, field_count * OUTPUT_TOKENS_PER_FIELD)


def adaptive_window_tokens(
    *,
    observed_records: int,
    observed_tokens: int,
    field_count: int,
    max_output_tokens: int,
    min_window_tokens: int,
    max_window_tokens: int,
) -> int:
    """Size a window so the records inside it fit in one reply.

    Derived *from* the output ceiling rather than checked against it. A window of ``W``
    tokens holds ``W x density`` records, each costing ``output_tokens_per_record``, and
    that product has to stay under the ceiling. Rearranged, ``W`` is whatever leaves the
    reply room to finish — which is the largest window that is safe, not a guess that
    happens to work.

    Every error lands on the safe side. Over-estimating density gives windows that are too
    small: more calls, same records, nothing lost. Under-estimating is the only dangerous
    direction, which is what :data:`OUTPUT_SAFETY_RATIO` and the bounds are for.

    Args:
        observed_records: Records returned by the window already extracted.
        observed_tokens: Tokens that window was sent.
        field_count: Fields in the resolved schema.
        max_output_tokens: The provider's reply ceiling, from settings.
        min_window_tokens: Floor. A denser document than this allows cannot be made safe
            by shrinking further — see the note below.
        max_window_tokens: Ceiling, so a sparse document does not produce one enormous
            window and lose the point of the exercise.

    Returns:
        A window size in estimated tokens, within the bounds.

    Note:
        With no records observed, there is no evidence of density and the maximum is
        returned — the size this document would have used before adaptive sizing existed.
        A section that states no records (a cover page, a preamble) is common, and letting
        it force the whole document down to minimum windows would multiply the cost of
        every long report on the strength of its first page.
    """
    if observed_records <= 0 or observed_tokens <= 0:
        return max_window_tokens

    per_record = output_tokens_per_record(field_count)
    budget_records = (max_output_tokens * OUTPUT_SAFETY_RATIO) / per_record

    # records per token, from the one window whose answer we have actually seen.
    density = observed_records / observed_tokens
    fitting = int(budget_records / density)

    return max(min_window_tokens, min(fitting, max_window_tokens))
