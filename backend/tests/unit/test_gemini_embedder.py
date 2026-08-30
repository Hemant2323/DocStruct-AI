"""Tests for the Gemini embedding component, with the HTTP client mocked.

Never touches the network — every test patches ``httpx.AsyncClient.post``. Mirrors the
structure of ``test_gemini_provider.py`` so the two Gemini components are easy to compare.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.ai.embeddings.gemini import DEFAULT_BASE_URL, DEFAULT_MODEL, GeminiEmbedder
from app.core.exceptions import (
    AIConfigurationError,
    AIUnavailableError,
    InvalidModelResponseError,
)


def _http_response(
    *,
    status_code: int = 200,
    values: list[float] | None = None,
    token_count: int = 5,
    include_embedding: bool = True,
    request: httpx.Request | None = None,
) -> httpx.Response:
    """Build a real ``httpx.Response`` so ``.raise_for_status()``/``.json()`` behave
    exactly as they do against a live server."""
    body: dict[str, Any] = {"usageMetadata": {"promptTokenCount": token_count}}
    if include_embedding:
        body["embedding"] = {"values": values if values is not None else [0.1, 0.2, 0.3]}
    return httpx.Response(
        status_code=status_code,
        json=body,
        request=request
        or httpx.Request("POST", "https://example.test/models/gemini-embedding:embedContent"),
    )


class TestConfiguration:
    def test_reports_unconfigured_without_a_key(self) -> None:
        embedder = GeminiEmbedder(api_key=None)
        assert embedder.is_configured is False

    def test_reports_configured_with_a_key(self) -> None:
        assert GeminiEmbedder(api_key="test-key-not-real").is_configured is True

    def test_blank_key_counts_as_unconfigured(self) -> None:
        """An empty string in .env is a common mistake and must not look configured."""
        assert GeminiEmbedder(api_key="   ").is_configured is False

    def test_model_name_comes_from_the_constructor(self) -> None:
        embedder = GeminiEmbedder(api_key="k", model="gemini-embedding-2-preview")
        assert embedder.model_name == "gemini-embedding-2-preview"

    def test_defaults_match_the_requested_configuration(self) -> None:
        assert DEFAULT_BASE_URL == "https://generativelanguage.googleapis.com/v1beta"
        assert DEFAULT_MODEL == "gemini-embedding-2-preview"

    async def test_embedding_without_a_key_is_a_clear_error(self) -> None:
        embedder = GeminiEmbedder(api_key=None)

        with pytest.raises(AIConfigurationError) as exc_info:
            await embedder.embed("some text")

        assert exc_info.value.code == "ai_not_configured"


class TestRequestConstruction:
    async def test_requests_go_to_the_embed_content_endpoint(self) -> None:
        embedder = GeminiEmbedder(
            api_key="k",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            model="gemini-embedding-2-preview",
        )
        captured: dict[str, Any] = {}

        async def capture(url: str, *, json: dict[str, Any]) -> httpx.Response:
            captured["url"] = url
            captured["json"] = json
            return _http_response()

        with patch.object(embedder._client, "post", side_effect=capture):
            await embedder.embed("hello world")

        assert captured["url"] == (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-embedding-2-preview:embedContent"
        )
        assert captured["json"] == {"content": {"parts": [{"text": "hello world"}]}}

    async def test_a_trailing_slash_on_the_base_url_does_not_double_up(self) -> None:
        embedder = GeminiEmbedder(
            api_key="k",
            base_url="https://generativelanguage.googleapis.com/v1beta/",
            model="gemini-embedding-2-preview",
        )
        captured: dict[str, Any] = {}

        async def capture(url: str, *, json: dict[str, Any]) -> httpx.Response:
            captured["url"] = url
            return _http_response()

        with patch.object(embedder._client, "post", side_effect=capture):
            await embedder.embed("text")

        assert captured["url"] == (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-embedding-2-preview:embedContent"
        )

    async def test_the_key_is_sent_as_a_header_never_a_query_parameter(self) -> None:
        """A key in a URL can end up in access logs, proxy logs, or browser history —
        the header form must be what is actually used."""
        embedder = GeminiEmbedder(api_key="secret-key-value")

        assert embedder._client is not None
        assert embedder._client.headers["x-goog-api-key"] == "secret-key-value"
        assert "key" not in embedder._embed_url


class TestResponseHandling:
    async def test_embedding_values_are_returned(self) -> None:
        embedder = GeminiEmbedder(api_key="k")
        response = _http_response(values=[0.5, -0.25, 1.0])

        with patch.object(embedder._client, "post", AsyncMock(return_value=response)):
            result = await embedder.embed("some text")

        assert result.values == [0.5, -0.25, 1.0]

    async def test_token_count_is_reported(self) -> None:
        embedder = GeminiEmbedder(api_key="k")
        response = _http_response(token_count=42)

        with patch.object(embedder._client, "post", AsyncMock(return_value=response)):
            result = await embedder.embed("some text")

        assert result.token_count == 42

    async def test_missing_embedding_is_reported_not_silently_empty(self) -> None:
        embedder = GeminiEmbedder(api_key="k")
        response = _http_response(include_embedding=False)

        with (
            patch.object(embedder._client, "post", AsyncMock(return_value=response)),
            pytest.raises(InvalidModelResponseError),
        ):
            await embedder.embed("some text")

    async def test_empty_values_list_is_reported_not_silently_empty(self) -> None:
        embedder = GeminiEmbedder(api_key="k")
        response = _http_response(values=[])

        with (
            patch.object(embedder._client, "post", AsyncMock(return_value=response)),
            pytest.raises(InvalidModelResponseError),
        ):
            await embedder.embed("some text")


class TestErrorsAreSafeAndTyped:
    """Every upstream failure must become a typed error with a message safe to display."""

    async def test_401_becomes_a_configuration_error(self) -> None:
        embedder = GeminiEmbedder(api_key="k")
        response = _http_response(status_code=401)

        with (
            patch.object(embedder._client, "post", AsyncMock(return_value=response)),
            pytest.raises(AIConfigurationError) as exc_info,
        ):
            await embedder.embed("text")

        assert exc_info.value.code == "ai_not_configured"

    async def test_429_becomes_unavailable_after_exhausting_retries(self) -> None:
        embedder = GeminiEmbedder(api_key="k", max_retries=1)
        response = _http_response(status_code=429)

        with (
            patch.object(embedder._client, "post", AsyncMock(return_value=response)),
            pytest.raises(AIUnavailableError) as exc_info,
        ):
            await embedder.embed("text")

        assert exc_info.value.code == "ai_unavailable"

    async def test_429_is_retried_and_can_succeed(self) -> None:
        embedder = GeminiEmbedder(api_key="k", max_retries=2)
        attempts = [
            _http_response(status_code=429),
            _http_response(values=[1.0, 2.0]),
        ]

        with patch.object(embedder._client, "post", AsyncMock(side_effect=attempts)):
            result = await embedder.embed("text")

        assert result.values == [1.0, 2.0]

    async def test_timeout_becomes_a_clear_error(self) -> None:
        embedder = GeminiEmbedder(api_key="k", max_retries=0)

        with (
            patch.object(
                embedder._client, "post", AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
            ),
            pytest.raises(AIUnavailableError) as exc_info,
        ):
            await embedder.embed("text")

        assert "took too long" in exc_info.value.message

    async def test_connection_failure_becomes_a_clear_error(self) -> None:
        embedder = GeminiEmbedder(api_key="k", max_retries=0)

        with (
            patch.object(
                embedder._client, "post", AsyncMock(side_effect=httpx.ConnectError("refused"))
            ),
            pytest.raises(AIUnavailableError) as exc_info,
        ):
            await embedder.embed("text")

        assert "reach the AI service" in exc_info.value.message

    async def test_an_api_key_never_appears_in_an_error_message(self) -> None:
        """The single most important safety test in this file."""
        embedder = GeminiEmbedder(api_key="AIzaSySUPERSECRET123")
        response = _http_response(status_code=401)

        with (
            patch.object(embedder._client, "post", AsyncMock(return_value=response)),
            pytest.raises(AIConfigurationError) as exc_info,
        ):
            await embedder.embed("text")

        assert "SUPERSECRET" not in exc_info.value.message
        assert "AIzaSy" not in exc_info.value.message

    async def test_no_bare_exception_types_are_ever_raised(self) -> None:
        embedder = GeminiEmbedder(api_key="k", max_retries=0)

        with (
            patch.object(
                embedder._client, "post", AsyncMock(side_effect=httpx.ConnectTimeout("x"))
            ),
            pytest.raises((AIConfigurationError, AIUnavailableError, InvalidModelResponseError)),
        ):
            await embedder.embed("text")
