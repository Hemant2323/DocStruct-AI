"""Turning model citations into real source locations.

This module is the last link in the chain that has been built since Milestone 2::

    file -> ContentBlock(provenance) -> Chunk(block_ids) -> ContextBlock -> citation

Everything here is a **lookup in our own records**. Two rules hold without exception:

* A location is only ever produced by resolving an index or a quote against data we
  stored ourselves. Nothing the model writes is treated as a location.
* When a citation cannot be resolved, it is returned with ``provenance=None`` and shown
  as "source not verified". An unresolvable citation is never quietly dropped, and never
  given a plausible-looking page number (CLAUDE.md §4.3).
"""

from __future__ import annotations

import re
import unicodedata
from uuid import UUID

from app.models.domain.content import ContentBlock
from app.models.domain.insight import Citation, ContextBlock, RawCitation

#: Shortest quote worth trying to verify. A three-word fragment matches almost anywhere,
#: so a "match" would be meaningless.
_MIN_VERIFIABLE_QUOTE_CHARS = 12

_WHITESPACE = re.compile(r"\s+")


def resolve_citations(
    raw_citations: tuple[RawCitation, ...] | list[RawCitation],
    context_blocks: list[ContextBlock],
    blocks_by_id: dict[UUID, ContentBlock],
) -> list[Citation]:
    """Resolve API citation spans into source locations.

    Each span names a range of positions in the list we sent, so resolution is an index
    lookup. The block's provenance is then read from what the parser recorded — the model
    never supplies it.

    Args:
        raw_citations: Spans as the API reported them.
        context_blocks: Exactly the list that was sent, in the same order.
        blocks_by_id: Content blocks keyed by id, for reading provenance.

    Returns:
        One citation per span, in order, with duplicates collapsed.
    """
    resolved: list[Citation] = []
    seen: set[tuple[str, int]] = set()

    for raw in raw_citations:
        # Out-of-range indices should not happen, but accepting one would mean pointing
        # at the wrong part of the document, which is worse than reporting nothing.
        if not (0 <= raw.start_block_index < len(context_blocks)):
            resolved.append(_unverified(raw.quoted_text))
            continue

        key = (raw.quoted_text.strip(), raw.start_block_index)
        if key in seen:
            continue
        seen.add(key)

        context_block = context_blocks[raw.start_block_index]
        provenance_source = _first_block_with_provenance(context_block, blocks_by_id)

        resolved.append(
            Citation(
                quoted_text=raw.quoted_text.strip(),
                provenance=provenance_source.provenance if provenance_source else None,
                chunk_id=context_block.chunk_id,
                block_ids=context_block.block_ids,
            )
        )

    return resolved


def verify_quote(
    quote: str,
    context_blocks: list[ContextBlock],
    blocks_by_id: dict[UUID, ContentBlock],
) -> Citation | None:
    """Check a model-supplied quote against the document, and locate it if it is real.

    Used for structured extraction, where citations cannot be requested from the API
    because they are incompatible with schema-constrained output. Instead the model is
    asked for the exact wording it relied on, and that wording is **searched for in our
    own stored text**.

    A match is evidence; the model's opinion about where something appears is not. If the
    quote is not found, ``None`` is returned and the field is reported without a source
    rather than with a guess.

    Args:
        quote: The wording the model claims to have used.
        context_blocks: The content that was sent, in order.
        blocks_by_id: Content blocks keyed by id, for reading provenance.

    Returns:
        A citation with real provenance, or ``None`` if the quote could not be found.
    """
    cleaned = quote.strip()
    if len(cleaned) < _MIN_VERIFIABLE_QUOTE_CHARS:
        return None

    needle = _normalise(cleaned)
    if not needle:
        return None

    for context_block in context_blocks:
        if needle not in _normalise(context_block.text):
            continue

        provenance_source = _first_block_with_provenance(context_block, blocks_by_id)
        return Citation(
            quoted_text=cleaned,
            provenance=provenance_source.provenance if provenance_source else None,
            chunk_id=context_block.chunk_id,
            block_ids=context_block.block_ids,
        )

    return None


def resolve_source(context_block: ContextBlock, blocks_by_id: dict[UUID, ContentBlock]) -> Citation:
    """Build a citation directly from a context block already known to be a source.

    Used by retrieval-grounded answers (Milestone 4.4): unlike ``resolve_citations`` and
    ``verify_quote``, no model output is involved here. Retrieval already decided this
    chunk was relevant enough to send, so its provenance is resolved unconditionally
    rather than depending on the model reporting a citation span — which providers
    without native citation support (see ``GeminiProvider``, ``NVIDIAProvider``) never do.

    Args:
        context_block: One block that was sent to the model.
        blocks_by_id: Content blocks keyed by id, for reading provenance.

    Returns:
        A citation for this block. ``provenance`` is ``None`` only if the block's own
        content blocks could not be found — the same honest fallback ``resolve_citations``
        and ``verify_quote`` use.
    """
    provenance_source = _first_block_with_provenance(context_block, blocks_by_id)
    return Citation(
        quoted_text=context_block.text,
        provenance=provenance_source.provenance if provenance_source else None,
        chunk_id=context_block.chunk_id,
        block_ids=context_block.block_ids,
    )


def _first_block_with_provenance(
    context_block: ContextBlock, blocks_by_id: dict[UUID, ContentBlock]
) -> ContentBlock | None:
    """Return the first content block behind a chunk, for its provenance.

    A chunk may span several blocks. The first is used because that is where its text
    begins, which matches how the chunker built it — the same rule already applied when
    merging blocks during ingestion.
    """
    for block_id in context_block.block_ids:
        block = blocks_by_id.get(block_id)
        if block is not None:
            return block
    return None


def _unverified(quoted_text: str) -> Citation:
    """A citation we could not place, stated honestly."""
    return Citation(quoted_text=quoted_text.strip(), provenance=None)


def _normalise(text: str) -> str:
    """Fold text for comparison.

    Quotes come back with different whitespace and, occasionally, different punctuation
    forms than the stored text. Comparing a folded form avoids rejecting a genuine quote
    over a line break — while still requiring the actual words to be present.
    """
    folded = unicodedata.normalize("NFKC", text)
    folded = folded.replace("’", "'").replace("“", '"').replace("”", '"')
    folded = folded.replace("–", "-").replace("—", "-")
    return _WHITESPACE.sub(" ", folded).strip().lower()
