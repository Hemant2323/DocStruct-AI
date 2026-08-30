"""Request and response schemas for document Q&A (Milestone 4.4)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.api.schemas.analysis import CitationResponse
from app.models.domain.qa import AnswerResult

#: Generous enough for a real question, small enough to keep a request bounded.
MAX_QUESTION_LENGTH = 2000


class AskRequest(BaseModel):
    """A question to ask about a document."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"question": "What is the payment deadline?", "top_k": 5}}
    )

    question: str = Field(
        min_length=1,
        max_length=MAX_QUESTION_LENGTH,
        description="The question to ask about this document.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of document sections to retrieve and consider.",
    )

    @field_validator("question")
    @classmethod
    def _validate_question(cls, value: str) -> str:
        """Reject a question that is empty once whitespace is stripped.

        ``min_length=1`` alone would accept a string of only spaces — its length is
        checked before stripping.
        """
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be empty or whitespace-only")
        return cleaned


class AskResponse(BaseModel):
    """An answer to a question about a document, grounded in its content."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What is the payment deadline?",
                "answer": "Invoices are due within 30 days of the invoice date.",
                "citations": [
                    {
                        "quoted_text": "Invoices are due within 30 days.",
                        "location_label": "page 2",
                        "is_verified": True,
                        "block_ids": [],
                    }
                ],
                "model": "claude-opus-5",
                "input_tokens": 1200,
                "output_tokens": 40,
                "chunk_count": 3,
            }
        }
    )

    question: str = Field(description="The question that was asked.")
    answer: str = Field(
        description=(
            "The answer, grounded in the document. States plainly when the document "
            "does not contain enough information to answer, rather than guessing."
        )
    )
    citations: list[CitationResponse] = Field(
        default_factory=list,
        description="The document sections the answer is grounded in.",
    )
    model: str = Field(description="Which model produced this answer.")
    input_tokens: int = 0
    output_tokens: int = 0
    chunk_count: int = Field(description="How many document sections were used to answer.")

    @classmethod
    def from_domain(cls, result: AnswerResult) -> AskResponse:
        return cls(
            question=result.question,
            answer=result.answer,
            citations=[CitationResponse.from_domain(item) for item in result.citations],
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            chunk_count=result.chunk_count,
        )
