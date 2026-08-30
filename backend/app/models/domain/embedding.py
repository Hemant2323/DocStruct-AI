"""Domain model for a text embedding.

Kept separate from ``app/models/domain/insight.py``: an embedding is a different kind of
result from a generation call (a vector, not text or citations), produced by a different
kind of provider (``Embedder``, not ``LLMProvider``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """A single text embedding.

    Attributes:
        values: The embedding vector.
        token_count: Tokens billed for the input text, as reported by the provider.
    """

    values: list[float]
    token_count: int
