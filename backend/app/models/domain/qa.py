"""Domain model for a single-turn question-and-answer result (Milestone 4.4).

Kept separate from ``app/models/domain/insight.py``'s ``Insight``/``InsightKind``: a Q&A
answer is not cached the way a summary, explanation or extraction is — every question is
different, so there is nothing to fingerprint against. Forcing it into ``Insight`` would
mean stretching a type built for three specific, cacheable analysis kinds to cover
something with a genuinely different shape: a question-answer pair, not a kind-plus-content
record.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from app.models.domain.insight import Citation


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """The answer to one question about one document.

    Attributes:
        document_id: The document the question was about.
        question: The question that was asked, as sent (already validated and trimmed).
        answer: The model's answer, grounded in the retrieved content only.
        citations: Sources the answer is grounded in — one per chunk that retrieval
            selected and sent to the model. Reuses the same ``Citation`` type summaries
            and explanations use, resolved the same way: from our own stored provenance,
            never from anything the model said.
        model: Which model produced the answer.
        input_tokens / output_tokens: What the call cost.
        chunk_count: How many chunks were retrieved and used to answer.
    """

    document_id: UUID
    question: str
    answer: str
    citations: tuple[Citation, ...] = ()
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    chunk_count: int = 0
