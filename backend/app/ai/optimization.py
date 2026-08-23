"""Optional context optimization.

Reduces how much of a large document is sent to the model. **It is off by default, and it
never touches anything the user owns**: the uploaded file, the content blocks, the chunks
and their provenance are all read-only inputs here. The result is a shorter *list of
chunks to send*, computed fresh for one request and never stored in their place.

The reduction is **selection, not rewriting** — chosen deliberately for two reasons:

* Every kept chunk stays byte-for-byte identical to what the parser produced, so
  citations remain exact and no fact can be altered on the way through.
* Deciding what to keep is scoring and arithmetic, which is ordinary code. Asking a model
  to shorten text before asking it to read the text would be paying twice and introducing
  a place where information could silently change.
"""

from __future__ import annotations

import re

from app.ingestion.chunker import estimate_tokens
from app.repositories.chunk_repo import Chunk

#: Chunks whose location label mentions these are structural summaries of tabular data —
#: the highest-value content per token in a spreadsheet or CSV.
_SCHEMA_MARKERS = ("structure",)

#: Signals that a chunk carries hard facts worth keeping: money, dates, percentages,
#: quantities. These are exactly what a user asks about and what must not be lost.
_MONEY = re.compile(r"[$£€]\s?\d|(?<!\w)\d[\d,]*\.?\d*\s?(?:usd|eur|gbp|dollars|euros)", re.I)
_DATE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b",
    re.I,
)
_PERCENT = re.compile(r"\d+(?:\.\d+)?\s?%")
_NUMBER = re.compile(r"(?<!\w)\d[\d,]*(?:\.\d+)?(?!\w)")

#: Words that tend to introduce an obligation, a deadline or an exception — the parts of
#: a document a reader most needs and most often misses.
_SIGNIFICANT_TERMS = re.compile(
    r"\b(must|shall|require[ds]?|deadline|due|terminat\w+|penalt\w+|liabilit\w+|"
    r"notice|renew\w+|cancel\w+|except\w*|unless|provided that|warrant\w+|"
    r"indemnif\w+|confidential\w*|total|balance|amount)\b",
    re.I,
)


def score_chunk(chunk: Chunk) -> float:
    """Rate how much a chunk is worth keeping, per token spent on it.

    Higher is more valuable. The score rewards density of real information rather than
    length, so a short paragraph stating an amount outranks a long passage of boilerplate.
    """
    text = chunk.text
    if not text.strip():
        return 0.0

    score = 1.0

    label = chunk.location_label.lower()
    if any(marker in label for marker in _SCHEMA_MARKERS):
        # A table's structure block describes an entire sheet in a few lines.
        score += 6.0

    # Hard facts. Counted with a cap so one number-heavy table cannot crowd out
    # everything else in the document.
    score += min(len(_MONEY.findall(text)), 5) * 1.5
    score += min(len(_DATE.findall(text)), 5) * 1.5
    score += min(len(_PERCENT.findall(text)), 5) * 1.0
    score += min(len(_NUMBER.findall(text)), 10) * 0.2
    score += min(len(_SIGNIFICANT_TERMS.findall(text)), 8) * 1.0

    # Prefer information per token: two short informative chunks beat one long vague one.
    tokens = max(estimate_tokens(text), 1)
    return score / (tokens**0.5)


def select_chunks(chunks: list[Chunk], *, target_tokens: int) -> list[Chunk]:
    """Choose which chunks to send, staying within a token target.

    The opening of a document is always kept: titles, parties and dates live there, and a
    summary that omits them is close to useless. The rest are taken by score until the
    budget runs out, then **restored to document order** — a model reading sections out of
    sequence would draw the wrong conclusions about what follows what.

    Args:
        chunks: Every chunk of the document, in order.
        target_tokens: Approximate ceiling for the selection.

    Returns:
        The chosen chunks, in their original order. Never reordered, never rewritten.
    """
    if not chunks:
        return []

    remaining = target_tokens
    chosen: dict[int, Chunk] = {}

    # The beginning of a document is disproportionately informative.
    for index, chunk in enumerate(chunks[: _opening_size(len(chunks))]):
        cost = estimate_tokens(chunk.text)
        if cost > remaining:
            break
        chosen[index] = chunk
        remaining -= cost

    ranked = sorted(
        ((index, chunk) for index, chunk in enumerate(chunks) if index not in chosen),
        key=lambda pair: score_chunk(pair[1]),
        reverse=True,
    )

    for index, chunk in ranked:
        cost = estimate_tokens(chunk.text)
        if cost > remaining:
            continue  # A later, smaller chunk may still fit.
        chosen[index] = chunk
        remaining -= cost

    return [chunk for _, chunk in sorted(chosen.items())]


def _opening_size(total_chunks: int) -> int:
    """How many leading chunks to keep unconditionally."""
    if total_chunks <= 3:
        return total_chunks
    return max(2, min(5, total_chunks // 10))


def should_offer_optimization(estimated_tokens: int, minimum_tokens: int) -> bool:
    """Whether optimization is worth offering for a document of this size.

    Used by the API so the interface can hide the option for documents that fit
    comfortably. Offering it there would invite users to lose detail for no gain.
    """
    return estimated_tokens >= minimum_tokens
