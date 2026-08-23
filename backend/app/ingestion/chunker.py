"""Structure-aware chunking.

Groups content blocks into retrieval-sized pieces. The naive approach — slice the
document every N characters — destroys exactly the structure the parsers worked to
recover, splitting tables in half and severing sentences.

Two rules shape this implementation:

* **Structure is a boundary.** A heading starts a new chunk; a table or schema block is
  never split, because half a table is worse than no table.
* **Every chunk records which blocks it came from.** That list is the middle link in the
  citation chain (``chunk -> blocks -> provenance -> "report.pdf, page 7"``). Without it
  a chunk could be retrieved but never attributed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.models.domain.content import BlockType, ContentBlock

#: Rough characters-per-token ratio for English prose.
#:
#: Deliberately an estimate. Anthropic ships no local tokenizer, and `count_tokens` is a
#: network call, which would be wrong inside an ingest loop — it would make ingestion
#: depend on an API being reachable. The value only needs to be good enough to keep
#: chunks in a sensible size band; retrieval quality work in a later milestone can refine
#: it against real measurements.
CHARS_PER_TOKEN = 4

#: Sentence boundary: terminal punctuation followed by whitespace and a capital or digit.
#: Used only when a single block is too large to fit a chunk on its own.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def estimate_tokens(text: str) -> int:
    """Estimate how many tokens a piece of text is worth."""
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrieval-sized piece of a document.

    Attributes:
        text: The chunk's text.
        block_indices: Positions (``ContentBlock.order``) of the blocks it was built
            from. The caller maps these to database IDs.
        location_label: Span covered, e.g. ``"pages 3-4"``.
    """

    text: str
    block_indices: list[int]
    location_label: str

    @property
    def token_estimate(self) -> int:
        return estimate_tokens(self.text)

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class _Accumulator:
    """Blocks being gathered into the chunk currently under construction."""

    blocks: list[ContentBlock] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        # +1 per join for the separator between blocks.
        return sum(block.char_count + 1 for block in self.blocks)

    def is_empty(self) -> bool:
        return not self.blocks


def chunk_blocks(
    blocks: list[ContentBlock],
    *,
    target_chars: int = 1500,
    overlap_chars: int = 150,
) -> list[Chunk]:
    """Group blocks into chunks.

    Args:
        blocks: Normalized content blocks in reading order.
        target_chars: Preferred maximum size of a chunk.
        overlap_chars: How much text to repeat from the end of the previous chunk, so a
            fact spanning a boundary is not lost to both chunks.

    Returns:
        The chunks, in document order. Every block is represented in at least one chunk.

    Raises:
        ValueError: If ``overlap_chars`` is not smaller than ``target_chars``, which
            would prevent chunking from advancing.
    """
    if overlap_chars >= target_chars:
        raise ValueError("overlap_chars must be smaller than target_chars")

    if not blocks:
        return []

    chunks: list[Chunk] = []
    current = _Accumulator()

    def flush() -> None:
        if current.is_empty():
            return
        chunks.append(_build_chunk(current.blocks, chunks, overlap_chars))
        current.blocks = []

    for block in blocks:
        # A heading opens a new section, so it starts a new chunk — keeping the heading
        # with the text it introduces rather than orphaned at the end of the previous one.
        if block.block_type is BlockType.HEADING and not current.is_empty():
            flush()

        # Atomic blocks (tables, schemas) are never split and never share space, so that
        # a retrieved table arrives whole.
        if block.block_type.is_atomic:
            flush()
            chunks.append(_build_chunk([block], chunks, overlap_chars=0))
            continue

        # A block too large for any chunk is split on sentence boundaries.
        if block.char_count > target_chars:
            flush()
            for piece in _split_oversized_block(block, target_chars, overlap_chars):
                chunks.append(piece)
            continue

        if current.char_count + block.char_count > target_chars:
            flush()

        current.blocks.append(block)

    flush()
    return chunks


def _build_chunk(blocks: list[ContentBlock], previous: list[Chunk], overlap_chars: int) -> Chunk:
    """Assemble a chunk from blocks, prefixing overlap from the preceding chunk."""
    body = "\n\n".join(block.text for block in blocks)

    if overlap_chars and previous:
        tail = _tail(previous[-1].text, overlap_chars)
        if tail:
            body = f"{tail}\n\n{body}"

    return Chunk(
        text=body,
        block_indices=[block.order for block in blocks],
        location_label=_span_label(blocks),
    )


def _split_oversized_block(
    block: ContentBlock, target_chars: int, overlap_chars: int
) -> list[Chunk]:
    """Split one very large block into several chunks on sentence boundaries.

    All the pieces share the parent block's provenance and index: they came from the same
    place in the source, so a citation to any of them points to the same location.
    """
    sentences = _SENTENCE_BOUNDARY.split(block.text)
    chunks: list[Chunk] = []
    buffer: list[str] = []
    length = 0

    def flush() -> None:
        nonlocal buffer, length
        if not buffer:
            return
        text = " ".join(buffer).strip()
        if text:
            if overlap_chars and chunks:
                tail = _tail(chunks[-1].text, overlap_chars)
                if tail:
                    text = f"{tail} {text}"
            chunks.append(
                Chunk(
                    text=text,
                    block_indices=[block.order],
                    location_label=block.provenance.label,
                )
            )
        buffer = []
        length = 0

    for sentence in sentences:
        # A single sentence longer than the target is split by hard character count;
        # there is no smaller safe boundary left to use.
        if len(sentence) > target_chars:
            flush()
            for start in range(0, len(sentence), target_chars):
                chunks.append(
                    Chunk(
                        text=sentence[start : start + target_chars].strip(),
                        block_indices=[block.order],
                        location_label=block.provenance.label,
                    )
                )
            continue

        if length + len(sentence) > target_chars:
            flush()

        buffer.append(sentence)
        length += len(sentence) + 1

    flush()
    return chunks


def _tail(text: str, overlap_chars: int) -> str:
    """Return the last ``overlap_chars`` of text, trimmed to a word boundary."""
    if len(text) <= overlap_chars:
        return text

    tail = text[-overlap_chars:]
    # Drop a leading partial word so the overlap starts cleanly.
    space = tail.find(" ")
    return tail[space + 1 :].strip() if space != -1 else tail.strip()


def _span_label(blocks: list[ContentBlock]) -> str:
    """Describe the source span a chunk covers.

    Lets a retrieved chunk be attributed without loading every block it came from.
    """
    if not blocks:
        return ""

    labels = [block.provenance.label for block in blocks]
    if len(set(labels)) == 1:
        return labels[0]

    pages = sorted({block.provenance.page for block in blocks if block.provenance.page})
    if len(pages) > 1:
        return f"pages {pages[0]}-{pages[-1]}"

    return f"{labels[0]} to {labels[-1]}"
