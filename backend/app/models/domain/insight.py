"""Domain models for AI-produced understanding of a document.

An *insight* is one AI result about one document — a summary, a plain-language
explanation, or a set of extracted facts.

The type that matters most here is :class:`Citation`. It is what connects a sentence the
model wrote back to the place in the user's file it came from, and it is deliberately
built so that a location can only ever be **resolved from our own records** — never taken
from anything the model said.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from app.models.domain.content import Provenance


class InsightKind(StrEnum):
    """What kind of AI result this is.

    Persisted and sent over the API, so these values are part of the contract.
    """

    SUMMARY = "summary"
    EXPLANATION = "explanation"
    EXTRACTION = "extraction"


class ContextMode(StrEnum):
    """How the document was presented to the model.

    Recorded on every result so a user can tell whether they are reading something
    derived from the whole document or from a reduced version of it.
    """

    #: Every chunk was sent. The default, and the best quality.
    FULL = "full"
    #: A deterministically selected subset was sent (user opted in, or the document
    #: was too large to send whole).
    OPTIMIZED = "optimized"

    @property
    def is_reduced(self) -> bool:
        return self is ContextMode.OPTIMIZED


@dataclass(frozen=True, slots=True)
class Citation:
    """A link from something the model wrote back to the source it used.

    Attributes:
        quoted_text: The exact text the model cited. For prose results this comes from
            the API's own citation data; for extraction it is the quote the model
            returned, and it is only kept if it was **found in our stored content**.
        provenance: Where that text lives in the original file. ``None`` when the
            location could not be established — which is reported honestly rather than
            filled in with a guess.
        chunk_id: The chunk the citation resolved to, when known.
        block_ids: The content blocks behind that chunk.

    The class deliberately has no way to construct a location from model output. A
    :class:`Citation` either carries provenance that came out of our own database, or it
    carries ``None``.
    """

    quoted_text: str
    provenance: Provenance | None = None
    chunk_id: UUID | None = None
    block_ids: tuple[UUID, ...] = ()

    @property
    def is_verified(self) -> bool:
        """True when this citation resolved to a real location in the document."""
        return self.provenance is not None

    @property
    def location_label(self) -> str:
        """How to display the source, or an honest statement that it is unknown."""
        if self.provenance is None:
            return "source not verified"
        return self.provenance.label


@dataclass(frozen=True, slots=True)
class ExtractedField:
    """One fact pulled out of a document.

    Attributes:
        name: What was extracted, e.g. ``"total_amount"``.
        value: The value, as text. Kept as text because a document's "amount" may be
            "$1,200" or "1200 USD" and reshaping it would be interpretation.
        citation: Where it came from, if the model's quote could be verified.
    """

    name: str
    value: str
    citation: Citation | None = None


@dataclass(frozen=True, slots=True)
class Insight:
    """One AI result about one document.

    Attributes:
        document_id: The document analysed.
        kind: Summary, explanation, or extraction.
        content: The prose result. Empty for extractions, which use ``fields``.
        fields: Extracted facts. Empty for prose results.
        citations: Sources supporting the result.
        context_mode: Whether the whole document or a reduced context was used.
        model: Which model produced it, recorded so a result can be traced to it.
        input_tokens / output_tokens: What the call cost, for transparency.
        warnings: Anything the user should know — sections omitted, sources
            unverified. Surfaced, never swallowed.
        created_at: When it was produced.
        from_cache: True when served from a previous identical run rather than
            regenerated. Set at read time; not persisted.
    """

    document_id: UUID
    kind: InsightKind
    content: str = ""
    fields: tuple[ExtractedField, ...] = ()
    citations: tuple[Citation, ...] = ()
    context_mode: ContextMode = ContextMode.FULL
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    warnings: tuple[str, ...] = ()
    created_at: datetime | None = None
    from_cache: bool = False

    @property
    def verified_citation_count(self) -> int:
        """How many citations resolved to a real location."""
        return sum(1 for citation in self.citations if citation.is_verified)


@dataclass(frozen=True, slots=True)
class ContextStats:
    """What was actually sent to the model, for display and for tests.

    Attributes:
        chunk_count: Chunks included.
        total_chunks: Chunks the document has in total. A difference between the two is
            what makes an omission visible rather than silent.
        estimated_tokens: Size of the assembled context.
        mode: Whether this was the full document or a reduced selection.
        omitted_chunk_count: Chunks left out.
        was_requested: True when the user asked for optimization; False when the
            document simply did not fit. The distinction matters for wording — a chosen
            reduction is reassurance, a forced one is a caveat.
    """

    chunk_count: int
    total_chunks: int
    estimated_tokens: int
    mode: ContextMode
    omitted_chunk_count: int = 0
    was_requested: bool = False

    @property
    def is_complete(self) -> bool:
        """True when every chunk of the document was included."""
        return self.omitted_chunk_count == 0


@dataclass(frozen=True, slots=True)
class ModelResult:
    """A raw reply from the language model, before any interpretation.

    Kept separate from :class:`Insight` so the provider layer never needs to know what a
    document or a citation is — it returns text plus whatever citation spans the API gave
    it, and the mapping to provenance happens in our own deterministic code.

    Attributes:
        text: The reply.
        raw_citations: Citation spans exactly as the API returned them, each holding the
            index of the context block it refers to.
        parsed: Validated structured output, when the request asked for it.
        input_tokens / output_tokens: Usage reported by the API.
        stop_reason: Why generation ended. ``"refusal"`` is handled explicitly.
    """

    text: str = ""
    raw_citations: tuple[RawCitation, ...] = ()
    parsed: Any = None
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RawCitation:
    """A citation span as the API reported it.

    ``start_block_index`` and ``end_block_index`` are positions in the list of content
    blocks *we* supplied, 0-indexed with an exclusive end. Because we built that list,
    the index maps back to a known chunk with no guesswork — which is exactly why this
    approach was chosen over asking the model to write page numbers in its prose.
    """

    quoted_text: str
    start_block_index: int
    end_block_index: int
    document_index: int = 0


@dataclass(frozen=True, slots=True)
class ContextBlock:
    """One piece of context sent to the model.

    Attributes:
        text: What the model sees.
        chunk_id: The chunk it came from.
        block_ids: The content blocks behind that chunk.
        location_label: Where it lives in the original file.

    The position of this object in the assembled list *is* the index the API cites, which
    is what makes citation resolution a lookup rather than an inference.
    """

    text: str
    chunk_id: UUID
    block_ids: tuple[UUID, ...]
    location_label: str
    metadata: dict[str, Any] = field(default_factory=dict)
