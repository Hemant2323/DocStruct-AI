"""A stand-in embedding provider for tests.

**No automated test ever calls a real embedding API**, for the same reason
``fake_provider.py`` exists for ``LLMProvider``: it costs money, takes time, and returns
something different every run. This fake implements ``Embedder`` and returns whatever a
test tells it to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.ai.embeddings.base import Embedder
from app.models.domain.embedding import EmbeddingResult


@dataclass
class FakeEmbedder(Embedder):
    """An embedder that returns a canned vector and remembers what it was asked.

    Args:
        values: What :meth:`embed` should return as the embedding vector.
        token_count: Reported alongside it.
        error: If set, raised instead of returning — for testing failure paths.
        configured: Whether the embedder reports itself as usable.
        model: Reported as ``model_name``.
    """

    values: list[float] = field(default_factory=lambda: [1.0, 0.0])
    token_count: int = 1
    error: Exception | None = None
    configured: bool = True
    model: str = "fake-embedding-model"

    calls: list[str] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def is_configured(self) -> bool:
        return self.configured

    async def embed(self, text: str) -> EmbeddingResult:
        self.calls.append(text)
        if self.error is not None:
            raise self.error
        return EmbeddingResult(values=list(self.values), token_count=self.token_count)

    @property
    def call_count(self) -> int:
        return len(self.calls)
