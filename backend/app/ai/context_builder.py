"""Assembling the context sent to the model.

Entirely deterministic: no model is involved in deciding what to send. Choosing chunks,
measuring size and keeping track of what came from where is ordinary bookkeeping, and
doing it in code makes it fast, free, and testable.

The output is an ordered list of :class:`ContextBlock`. **Its order is the citation
coordinate system** — position *n* in this list is what the API means by block index *n* —
so the ordering here and the resolution in ``citations.py`` are two halves of one
mechanism.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.core.exceptions import ContextTooLargeError, InsufficientContentError
from app.ingestion.chunker import estimate_tokens
from app.models.domain.insight import ContextBlock, ContextMode, ContextStats
from app.repositories.chunk_repo import Chunk

#: Rough allowance for the system prompt and instruction, so a context sized exactly to
#: the limit does not push the whole request over it.
_PROMPT_OVERHEAD_TOKENS = 2_000


def build_context(
    chunks: list[Chunk],
    *,
    max_context_tokens: int,
    optimize: bool = False,
    optimization_min_tokens: int = 20_000,
    optimization_target_ratio: float = 0.5,
) -> tuple[list[ContextBlock], ContextStats]:
    """Turn a document's chunks into the context for one model request.

    The default is to send **everything**. A complete document gives the best answer, and
    Claude Opus 5's context window is large enough that most documents fit comfortably.
    Content is left out only when the user asked for that, or when it genuinely will not
    fit — and either way the omission is reported rather than hidden.

    Args:
        chunks: The document's chunks, in order, from Milestone 2.
        max_context_tokens: Ceiling on the assembled context.
        optimize: The user opted in to reducing the context.
        optimization_min_tokens: Below this size, optimization is skipped even when
            requested — trimming a short document only loses detail for no benefit.
        optimization_target_ratio: Share of the budget an optimized context may use.

    Returns:
        The context blocks and a description of what was included.

    Raises:
        InsufficientContentError: There is nothing to analyse.
        ContextTooLargeError: The document does not fit even after selection.
    """
    if not chunks:
        raise InsufficientContentError("This document has no extracted content to analyse.")

    budget = max(max_context_tokens - _PROMPT_OVERHEAD_TOKENS, 1)
    full_tokens = sum(estimate_tokens(chunk.text) for chunk in chunks)

    # Deliberately narrow: optimization is opt-in, and even then only for documents big
    # enough for it to matter.
    should_optimize = optimize and full_tokens >= optimization_min_tokens

    if not should_optimize and full_tokens <= budget:
        blocks = [_to_block(chunk) for chunk in chunks]
        return blocks, ContextStats(
            chunk_count=len(blocks),
            total_chunks=len(chunks),
            estimated_tokens=full_tokens,
            mode=ContextMode.FULL,
        )

    if should_optimize:
        # The ratio applies to how big this document actually is, not to the whole
        # budget. Measuring against the budget would mean a document that already fits
        # comfortably gets no reduction at all — so a user who explicitly asked to
        # optimize would see nothing happen.
        target = max(int(min(budget, full_tokens) * optimization_target_ratio), 1)
        mode = ContextMode.OPTIMIZED
    else:
        # Too large to send whole and the user did not opt in. Selecting is still better
        # than refusing outright, and the response says plainly that it happened.
        target = budget
        mode = ContextMode.OPTIMIZED

    from app.ai.optimization import select_chunks

    selected = select_chunks(chunks, target_tokens=target)

    if not selected:
        raise ContextTooLargeError(
            "This document is too large to analyse, even after reducing the context.",
            details={"chunks": len(chunks)},
        )

    blocks = [_to_block(chunk) for chunk in selected]
    return blocks, ContextStats(
        chunk_count=len(blocks),
        total_chunks=len(chunks),
        estimated_tokens=sum(estimate_tokens(chunk.text) for chunk in selected),
        mode=mode,
        omitted_chunk_count=len(chunks) - len(selected),
        was_requested=should_optimize,
    )


def blocks_from_chunks(chunks: Sequence[Chunk]) -> list[ContextBlock]:
    """Turn chunks into context blocks, in order, sending all of them.

    The path for a caller that has already decided what to send — windowed extraction
    sizes each window to fit before it gets here, so there is nothing left to select and
    no ``ContextStats`` to report. It exists so that windowing does not grow a second
    way of building a context block: position *n* in the returned list is block index
    *n* to the model, and that convention has to be identical everywhere or citations
    resolve to the wrong place.
    """
    return [_to_block(chunk) for chunk in chunks]


def _to_block(chunk: Chunk) -> ContextBlock:
    """Convert a stored chunk into a context block, carrying its identity along.

    The chunk id and block ids travel with the text so that a citation pointing at this
    position can be resolved back to a real location without another lookup — and without
    anything being inferred.
    """
    return ContextBlock(
        text=chunk.text,
        chunk_id=chunk.id,
        block_ids=tuple(chunk.block_ids),
        location_label=chunk.location_label,
    )


def describe_context(stats: ContextStats) -> list[str]:
    """Turn context statistics into warnings the user should see.

    CLAUDE.md forbids silently truncating user content. When something was left out, that
    fact is stated in plain language and attached to the result.
    """
    warnings: list[str] = []

    if not stats.mode.is_reduced:
        return warnings

    if stats.was_requested:
        # The user chose this, so lead with reassurance about what was *not* touched.
        warnings.append(
            "Context was optimized for the AI. Your original file and its extracted "
            "content are unchanged."
        )
        if stats.omitted_chunk_count:
            warnings.append(
                f"{stats.omitted_chunk_count} of {stats.total_chunks} sections were left "
                "out of the AI context to reduce its size."
            )
    else:
        # Not their choice — say plainly that the result is partial.
        warnings.append(
            f"This document was too large to send in full, so {stats.omitted_chunk_count} "
            f"of {stats.total_chunks} sections were left out. The result is based on the "
            "remaining content."
        )

    return warnings
