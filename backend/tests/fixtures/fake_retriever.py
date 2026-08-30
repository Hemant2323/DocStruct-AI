"""A stand-in retriever for tests.

No automated test that exercises the QA service or its API route needs real embeddings
or real cosine similarity — that mechanism is already covered directly by
``test_cosine_retriever.py``. This fake lets a test control exactly which chunks
"retrieval" returns, without a real document, real chunks, or real embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from app.retrieval.base import Retriever, ScoredChunk


@dataclass
class RecordedRetrieval:
    """What the retriever was asked to find."""

    document_id: UUID
    query: str
    top_k: int


@dataclass
class FakeRetriever(Retriever):
    """A retriever that returns canned results and remembers its calls.

    Args:
        results: What :meth:`retrieve` should return.
        error: If set, raised instead of returning — for testing failure paths (e.g. an
            ``EmbeddingsNotAvailableError`` from a document with no usable embeddings).
    """

    results: list[ScoredChunk] = field(default_factory=list)
    error: Exception | None = None

    calls: list[RecordedRetrieval] = field(default_factory=list)

    async def retrieve(self, *, document_id: UUID, query: str, top_k: int = 5) -> list[ScoredChunk]:
        self.calls.append(RecordedRetrieval(document_id=document_id, query=query, top_k=top_k))
        if self.error is not None:
            raise self.error
        return list(self.results)

    @property
    def last_call(self) -> RecordedRetrieval:
        """The most recent call, for assertions."""
        if not self.calls:
            raise AssertionError("The retriever was never called")
        return self.calls[-1]

    @property
    def call_count(self) -> int:
        return len(self.calls)
