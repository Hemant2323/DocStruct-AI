"""Google Gemini implementation of :class:`Embedder`.

Talks to the Gemini REST API directly over ``httpx``, mirroring
:class:`~app.ai.providers.gemini.GeminiProvider`'s conventions: an absolute URL built once
in the constructor, the API key sent as an ``x-goog-api-key`` header (never a ``?key=``
query parameter, which could end up in a logged URL), and upstream failures translated into
the same typed errors used everywhere else in the AI layer. The retry loop and error
translation are intentionally self-contained here rather than imported from
``GeminiProvider`` — the two files stay independent so a change to one can never alter the
other's behaviour, matching how ``NVIDIAProvider`` and ``GeminiProvider`` are already kept
separate from each other.

This is deliberately narrow: it embeds one piece of text and returns one vector. Batching,
retrieval, and vector storage are not part of this component.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from app.ai.embeddings.base import Embedder
from app.core.exceptions import (
    AIConfigurationError,
    AIUnavailableError,
    InvalidModelResponseError,
)
from app.core.logging import get_logger
from app.models.domain.embedding import EmbeddingResult

logger = get_logger(__name__)

#: Google's Generative Language API root — the same default as GeminiProvider.
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

#: The model this component was built for.
DEFAULT_MODEL = "gemini-embedding-2-preview"

#: HTTP statuses worth retrying: rate limiting and transient server-side failure.
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

#: Backoff between retries, scaled by attempt number.
_RETRY_BACKOFF_SECONDS = 0.5


class GeminiEmbedder(Embedder):
    """Embeds text using a Gemini embedding model through Google's REST API.

    Args:
        api_key: Gemini API key. The provider reports itself as unconfigured when this
            is missing or blank — never raised as an authentication failure from a
            partially-built client.
        base_url: The API root, e.g. ``https://generativelanguage.googleapis.com/v1beta``.
        model: Embedding model identifier to request.
        timeout_seconds: Per-request timeout.
        max_retries: Retries for rate limiting and transient server errors.

    The key is read once, used only to build the client's auth header, and never stored
    on its own attribute — so there is nothing on this object to accidentally log or
    serialise later.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self._model = model
        self._max_retries = max_retries
        # Built once, up front, so every call sends the same, already-validated absolute
        # URL rather than assembling it per request.
        self._embed_url = f"{base_url.rstrip('/')}/models/{model}:embedContent"
        self._client: httpx.AsyncClient | None = None

        key = (api_key or "").strip()
        if key:
            self._client = httpx.AsyncClient(
                headers={
                    # Header, not the documented `?key=` query parameter: a key in a URL
                    # can end up in access logs, proxy logs, or browser history.
                    "x-goog-api-key": key,
                    "Content-Type": "application/json",
                },
                timeout=timeout_seconds,
            )
            logger.info("Gemini embedder configured model=%s", model)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    async def embed(self, text: str) -> EmbeddingResult:
        client = self._require_client()
        payload = {"content": {"parts": [{"text": text}]}}

        try:
            response = await self._post_with_retries(client, payload)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise self._translate(exc) from exc

        values = ((data.get("embedding") or {}).get("values")) or []
        if not values:
            raise InvalidModelResponseError("The AI did not return an embedding for this text.")

        usage = data.get("usageMetadata") or {}
        return EmbeddingResult(
            values=[float(value) for value in values],
            token_count=int(usage.get("promptTokenCount", 0) or 0),
        )

    # ----------------------------------------------------------------- helpers
    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise AIConfigurationError(
                "AI features are not configured on this server. An API key is required."
            )
        return self._client

    async def _post_with_retries(
        self, client: httpx.AsyncClient, payload: dict[str, Any]
    ) -> httpx.Response:
        """POST with a small retry budget for rate limiting and transient failures."""
        attempts = self._max_retries + 1

        for attempt in range(attempts):
            is_last_attempt = attempt == attempts - 1

            try:
                response = await client.post(self._embed_url, json=payload)
            except (httpx.TimeoutException, httpx.TransportError):
                if is_last_attempt:
                    raise
                await asyncio.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue

            if is_last_attempt or response.status_code not in _RETRYABLE_STATUS_CODES:
                return response

            await asyncio.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))

        # Unreachable: every iteration above returns or raises.
        raise AssertionError("retry loop exited without returning or raising")  # pragma: no cover

    def _translate(self, exc: Exception) -> Exception:
        """Convert an httpx/transport failure into one of our typed errors.

        Upstream response bodies and headers are never included in what reaches the
        client — only the exception type and, where safe, an HTTP status code. The
        original is logged in full (server-side only) and chained for the log.
        """
        if isinstance(exc, AIConfigurationError | AIUnavailableError | InvalidModelResponseError):
            return exc

        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code

            if status in (401, 403):
                logger.error("Gemini rejected our credentials: HTTP %s", status)
                return AIConfigurationError(
                    "AI features are not configured correctly on this server."
                )

            if status == 429:
                logger.warning("Gemini rate limit reached")
                return AIUnavailableError(
                    "The AI service is busy right now. Please try again in a moment."
                )

            if status == 400:
                logger.error("Gemini rejected the request: HTTP 400 body=%s", exc.response.text)
                return AIUnavailableError(
                    "The AI could not process this text. The problem has been logged."
                )

            logger.error("Gemini returned HTTP %s", status, exc_info=True)
            return AIUnavailableError(
                "The AI service is temporarily unavailable. Please try again shortly."
            )

        if isinstance(exc, httpx.TimeoutException):
            logger.error("Gemini embedding request timed out")
            return AIUnavailableError("The AI took too long to respond. Please try again.")

        if isinstance(exc, httpx.TransportError):
            logger.error("Could not reach Gemini: %s", type(exc).__name__)
            return AIUnavailableError(
                "Could not reach the AI service. Please check the connection and try again."
            )

        logger.error("Unexpected AI provider failure: %s", type(exc).__name__, exc_info=True)
        return AIUnavailableError(
            "Something went wrong while contacting the AI service. The problem has been logged."
        )
