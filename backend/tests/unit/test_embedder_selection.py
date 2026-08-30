"""Tests for embedder dependency-injection wiring.

Confirms `get_embedder()` uses the embedding-specific model setting, not the
generation/chat model setting. The two are easy to conflate since both currently come
from the same provider (Gemini), but a chat model name is not valid for the embeddings
endpoint — this is the exact regression the M4 audit found and this file guards against.
"""

from __future__ import annotations

import pytest

from app.ai.embeddings.gemini import GeminiEmbedder
from app.api.deps import get_embedder, reset_embedder
from app.core.config import Settings


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {"gemini_api_key": "gemini-key"}
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.fixture(autouse=True)
def _reset_embedder_cache() -> None:
    """`get_embedder()` caches a process-wide singleton; without resetting it, one
    test's embedder would leak into the next regardless of its own settings."""
    reset_embedder()
    yield
    reset_embedder()


class TestEmbedderSelection:
    def test_uses_the_embedding_model_by_default(self) -> None:
        embedder = get_embedder(_settings())

        assert isinstance(embedder, GeminiEmbedder)
        assert embedder.model_name == "gemini-embedding-2-preview"

    def test_does_not_use_the_generation_model(self) -> None:
        """The exact regression this fixes: wiring must never silently reuse
        gemini_model (a chat model) as the embedding model."""
        settings = _settings(gemini_model="gemini-3.6-flash")

        embedder = get_embedder(settings)

        assert embedder.model_name != "gemini-3.6-flash"
        assert embedder.model_name == "gemini-embedding-2-preview"

    def test_embedding_model_is_independently_configurable(self) -> None:
        settings = _settings(
            gemini_model="gemini-3.6-flash",
            gemini_embedding_model="gemini-embedding-3-preview",
        )

        embedder = get_embedder(settings)

        assert embedder.model_name == "gemini-embedding-3-preview"
        assert embedder.model_name != settings.gemini_model
