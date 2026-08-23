"""Text normalization.

Runs once over the blocks *after* parsing, so the cleanup rules exist in one place
instead of being reimplemented (slightly differently) in five parsers.

Two invariants hold throughout:

* **Provenance is never changed.** Text is cleaned; the link back to the source file is
  untouched, because that link is what citations depend on.
* **Dropped content is counted, never silently discarded.** If normalizing empties a
  block, that is reported as a warning.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.models.domain.content import BlockType, ContentBlock, ParsedDocument

#: Characters that look like text but are not: zero-width spaces, soft hyphens, and the
#: byte-order mark. They break search and word counting while being invisible.
_INVISIBLE = dict.fromkeys(
    [0x00AD, 0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF],
    None,
)

#: Typographic characters replaced with ASCII equivalents so that a search for a quoted
#: phrase matches text copied out of Word or a PDF.
# NOTE: RUF001 (ambiguous unicode) is disabled for this file in pyproject.toml. The
# look-alike characters below are precisely what this table exists to replace, so
# flagging them would be flagging the feature.
_PUNCTUATION = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "–": "-",
    "—": "-",
    "−": "-",
    " ": " ",
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
}

_PUNCTUATION_TABLE = str.maketrans(_PUNCTUATION)

#: Three or more blank lines collapse to two — paragraph separation without the gaps.
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
#: Runs of spaces and tabs, but not newlines, which carry structure in tables.
_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]{2,}")

#: A line must repeat on at least this share of pages to count as a header or footer.
#: Set high on purpose: eating real content is far worse than leaving a page number in.
_REPEAT_THRESHOLD = 0.6
#: Repetition is only meaningful with enough pages to establish a pattern.
_MIN_PAGES_FOR_REPEAT_DETECTION = 4
#: Long lines are body text that happens to recur, not running heads.
_MAX_REPEATED_LINE_LENGTH = 100


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """Normalized blocks plus anything worth telling the user about."""

    blocks: list[ContentBlock]
    warnings: list[str]


def normalize_document(document: ParsedDocument) -> NormalizationResult:
    """Normalize every block in a parsed document.

    Args:
        document: The parser's output.

    Returns:
        The cleaned blocks, renumbered contiguously, plus any warnings.
    """
    warnings: list[str] = []
    blocks = document.blocks

    if _should_strip_running_heads(blocks):
        blocks, removed = _strip_running_heads(blocks)
        if removed:
            warnings.append(f"Removed {removed} repeated page header/footer lines.")

    cleaned: list[ContentBlock] = []
    dropped = 0

    for block in blocks:
        text = normalize_text(block.text, preserve_lines=block.block_type.is_tabular)
        if not text:
            dropped += 1
            continue
        # Renumbered so `order` stays a dense sequence after drops.
        cleaned.append(block.with_text(text).with_order(len(cleaned)))

    if dropped:
        warnings.append(f"{dropped} empty sections were removed during cleanup.")

    return NormalizationResult(blocks=cleaned, warnings=warnings)


def normalize_text(text: str, *, preserve_lines: bool = False) -> str:
    """Clean a single piece of extracted text.

    Args:
        text: The raw extracted text.
        preserve_lines: Keep line breaks. True for tables and row batches, where each
            line is a record; collapsing them would merge separate rows into one.

    Returns:
        The cleaned text, stripped of surrounding whitespace.
    """
    if not text:
        return ""

    # NFKC folds compatibility characters (ligatures, full-width forms) into their
    # ordinary equivalents, so the same word is spelled the same way everywhere.
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_INVISIBLE)
    text = text.translate(_PUNCTUATION_TABLE)

    # Normalize line endings before any line-based work.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = _HORIZONTAL_WHITESPACE.sub(" ", text)

    if preserve_lines:
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(line for line in lines if line)
    else:
        text = _EXCESS_BLANK_LINES.sub("\n\n", text)
        text = "\n".join(line.strip() for line in text.split("\n"))

    return text.strip()


def _should_strip_running_heads(blocks: list[ContentBlock]) -> bool:
    """Only PDFs repeat headers and footers on every page."""
    pages = {block.provenance.page for block in blocks if block.provenance.page}
    return len(pages) >= _MIN_PAGES_FOR_REPEAT_DETECTION


def _strip_running_heads(blocks: list[ContentBlock]) -> tuple[list[ContentBlock], int]:
    """Remove lines that repeat across most pages of a PDF.

    Page headers and footers are extracted as ordinary text, so without this every page's
    running title ends up in the content — and then in summaries and search results.

    The threshold is deliberately conservative. Removing a line that only *looks*
    repetitive would delete real content, which is much worse than leaving a page number
    in place.
    """
    pages = {block.provenance.page for block in blocks if block.provenance.page}
    page_count = len(pages)

    # Count how many distinct pages each short line appears on.
    line_pages: dict[str, set[int]] = {}
    for block in blocks:
        page = block.provenance.page
        if page is None or block.block_type.is_tabular:
            continue
        for line in block.text.split("\n"):
            stripped = line.strip()
            if stripped and len(stripped) <= _MAX_REPEATED_LINE_LENGTH:
                line_pages.setdefault(stripped, set()).add(page)

    repeated = {
        line
        for line, seen_pages in line_pages.items()
        if len(seen_pages) / page_count >= _REPEAT_THRESHOLD
    }

    if not repeated:
        return blocks, 0

    result: list[ContentBlock] = []
    removed = 0

    for block in blocks:
        if block.block_type.is_tabular:
            result.append(block)
            continue

        kept_lines = []
        for line in block.text.split("\n"):
            if line.strip() in repeated:
                removed += 1
            else:
                kept_lines.append(line)

        result.append(block.with_text("\n".join(kept_lines)))

    return result, removed


def merge_short_blocks(blocks: list[ContentBlock], *, minimum: int = 40) -> list[ContentBlock]:
    """Merge tiny consecutive prose blocks into their neighbour.

    PDF extraction often breaks a paragraph across several fragments. Left alone, each
    fragment becomes its own block and later its own chunk, which makes retrieval noisy.

    Headings, tables and tabular blocks are never merged: their separateness is what
    gives them meaning.

    **Only page-based blocks are merged.** The merged block keeps the *first* block's
    provenance, which is only truthful when both came from the same place. That holds for
    two fragments of one PDF page, but not for two paragraphs on different lines of a text
    file — merging those would silently attribute the second one's text to the first one's
    line numbers, which is exactly the invented provenance CLAUDE.md §4.3 forbids.
    """
    if not blocks:
        return []

    merged: list[ContentBlock] = []

    for block in blocks:
        previous = merged[-1] if merged else None
        mergeable = (
            block.block_type is BlockType.PARAGRAPH
            and len(block.text) < minimum
            and previous is not None
            and previous.block_type is BlockType.PARAGRAPH
            # Both must carry a real page number, and it must be the same one.
            and block.provenance.page is not None
            and previous.provenance.page == block.provenance.page
        )
        if mergeable and previous is not None:
            # Safe here: both blocks are on the same page, so the retained provenance
            # remains accurate for the combined text.
            merged[-1] = previous.with_text(f"{previous.text} {block.text}".strip())
        else:
            merged.append(block.with_order(len(merged)))

    return [block.with_order(index) for index, block in enumerate(merged)]
