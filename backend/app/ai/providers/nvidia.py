"""NVIDIA NIM implementation of :class:`LLMProvider`.

Talks to NVIDIA's OpenAI-compatible chat completions endpoint directly over ``httpx``
rather than the ``openai`` package — no new dependency is needed since ``httpx`` is
already installed (it underlies the ``anthropic`` SDK), and a single JSON POST is all an
OpenAI-compatible chat completion requires.

**Not yet wired into the application.** Nothing in ``app/core/config.py`` or
``app/api/deps.py`` references this class; it exists as a second, independently testable
:class:`LLMProvider` implementation, proving the interface — not this file — is what the
rest of the system depends on.

Two differences from :class:`~app.ai.providers.claude.ClaudeProvider` follow directly from
what an OpenAI-compatible chat completion can and cannot do:

* **No native citations.** Anthropic's custom-content document + citation span mechanism
  is Anthropic-specific; there is no equivalent here. ``generate()`` therefore always
  returns an empty ``raw_citations`` tuple rather than inventing one. Downstream code
  already treats "no citations" as an honest, valid outcome — a result with no confirmed
  sources, never a fabricated one.
* **No native structured output.** Instead of schema-constrained generation, the schema
  is described to the model in the prompt, JSON mode is requested, and the reply is
  parsed and validated against ``output_model`` before it is returned — a malformed or
  merely plausible-looking reply fails loudly rather than becoming half-correct data,
  exactly as it does for Claude.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from app.ai.providers.base import LLMProvider
from app.core.exceptions import (
    AIConfigurationError,
    AIUnavailableError,
    InvalidModelResponseError,
)
from app.core.logging import get_logger
from app.models.domain.insight import ContextBlock, ModelResult

logger = get_logger(__name__)

#: NVIDIA NIM's OpenAI-compatible endpoint.
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

#: The model requested for this integration.
DEFAULT_MODEL = "deepseek-ai/deepseek-v4-flash-0731"

#: HTTP statuses worth retrying: rate limiting and transient server-side failure.
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

#: Backoff between retries, scaled by attempt number. Kept short: the SDK-based
#: ClaudeProvider delegates retry timing to the Anthropic client, which does not expose
#: its backoff schedule for reuse here, so a small fixed step is used instead.
_RETRY_BACKOFF_SECONDS = 0.5

_CONTENT_FILTER_MESSAGE = (
    "The AI declined to analyse this document. This can happen with unusual or "
    "sensitive content. Try a different document."
)


class NVIDIAProvider(LLMProvider):
    """Talks to a DeepSeek model through NVIDIA NIM's OpenAI-compatible API.

    Args:
        api_key: NVIDIA NIM API key. The provider reports itself as unconfigured when
            this is missing or blank — never raised as an authentication failure from a
            partially-built client.
        base_url: The OpenAI-compatible API root, e.g. ``https://integrate.api.nvidia.com/v1``.
        model: Model identifier to request.
        timeout_seconds: Per-request timeout, mirroring ``ai_timeout_seconds``.
        max_retries: Retries for rate limiting and transient server errors, mirroring
            ``ai_max_retries``.
        max_output_tokens: Ceiling on the reply, mirroring ``ai_max_output_tokens``.

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
        timeout_seconds: float = 300.0,
        max_retries: int = 2,
        max_output_tokens: int = 8_000,
    ) -> None:
        self._model = model
        self._max_retries = max_retries
        self._max_output_tokens = max_output_tokens
        # Built once, up front, so every call site sends the same, already-validated
        # absolute URL rather than relying on httpx's base_url + relative-path merge
        # rules, which silently drop a base path segment unless the two are formatted
        # exactly right. An explicit absolute URL has no such ambiguity.
        self._chat_completions_url = f"{base_url.rstrip('/')}/chat/completions"
        self._client: httpx.AsyncClient | None = None

        key = (api_key or "").strip()
        if key:
            self._client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                timeout=timeout_seconds,
            )
            # Host only. The Authorization header is never touched again after this
            # constructor, so there is no later code path that could log it.
            logger.info(
                "NVIDIA provider configured host=%s model=%s",
                urlsplit(base_url).netloc,
                model,
            )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    # ------------------------------------------------------------------ prose
    async def generate(
        self,
        *,
        system: str,
        instruction: str,
        context_blocks: list[ContextBlock],
        document_title: str,
        enable_citations: bool = True,
    ) -> ModelResult:
        # `enable_citations` is accepted to satisfy the interface but has no effect:
        # this API has no citation mechanism to enable. See the module docstring.
        client = self._require_client()

        prompt = _build_prompt(context_blocks, document_title, instruction)
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self._max_output_tokens,
        }

        try:
            response = await self._post_with_retries(client, payload)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise self._translate(exc) from exc

        choice = _first_choice(data)
        finish_reason = choice.get("finish_reason")
        self._check_content_filter(finish_reason)

        text = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}

        return ModelResult(
            text=text.strip(),
            raw_citations=(),
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            stop_reason=finish_reason,
        )

    # ------------------------------------------------------------- structured
    async def generate_structured(
        self,
        *,
        system: str,
        instruction: str,
        context_blocks: list[ContextBlock],
        document_title: str,
        output_model: type[BaseModel],
    ) -> ModelResult:
        client = self._require_client()

        prompt = _build_structured_prompt(context_blocks, document_title, instruction, output_model)
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self._max_output_tokens,
            # JSON mode: the closest widely-supported OpenAI-compatible equivalent to
            # Anthropic's schema-constrained output. It guarantees syntactically valid
            # JSON, not conformance to our schema — model_validate() below does that.
            "response_format": {"type": "json_object"},
        }

        try:
            response = await self._post_with_retries(client, payload)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise self._translate(exc) from exc

        choice = _first_choice(data)
        finish_reason = choice.get("finish_reason")
        self._check_content_filter(finish_reason)

        raw_content = (choice.get("message") or {}).get("content") or ""

        try:
            parsed_json = json.loads(raw_content)
            parsed = output_model.model_validate(parsed_json)
        except (json.JSONDecodeError, PydanticValidationError) as exc:
            # The model replied but the shape was wrong. Reported rather than repaired:
            # guessing at a fix is how invented data gets into results.
            raise InvalidModelResponseError(
                "The AI returned a response that did not match the expected format."
            ) from exc

        usage = data.get("usage") or {}
        return ModelResult(
            parsed=parsed,
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            stop_reason=finish_reason,
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
        """POST with a small retry budget for rate limiting and transient failures.

        The SDK-based ClaudeProvider gets this from the Anthropic client's own
        ``max_retries``; there is no SDK here, so the same behaviour — retry 429s and
        5xxs, give up on anything else — is written out directly.
        """
        attempts = self._max_retries + 1

        for attempt in range(attempts):
            is_last_attempt = attempt == attempts - 1

            try:
                response = await client.post(self._chat_completions_url, json=payload)
            except (httpx.TimeoutException, httpx.TransportError):
                if is_last_attempt:
                    raise
                await asyncio.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue

            if is_last_attempt or response.status_code not in _RETRYABLE_STATUS_CODES:
                return response

            await asyncio.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))

        # Unreachable: every iteration above returns or raises. Kept explicit so the
        # function's return type is provably complete rather than relying on that fact.
        raise AssertionError("retry loop exited without returning or raising")  # pragma: no cover

    def _check_content_filter(self, finish_reason: str | None) -> None:
        """Turn a content-filtered reply into a clear, typed failure.

        The OpenAI-compatible equivalent of Anthropic's ``stop_reason: "refusal"``: the
        request succeeds at the transport level but the model produced nothing usable,
        so without this check the user would see a blank result and no explanation.
        """
        if finish_reason != "content_filter":
            return

        logger.warning("Model declined the request finish_reason=content_filter")
        raise AIUnavailableError(_CONTENT_FILTER_MESSAGE)

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
                logger.error("NVIDIA NIM rejected our credentials: HTTP %s", status)
                return AIConfigurationError(
                    "AI features are not configured correctly on this server."
                )

            if status == 429:
                logger.warning("NVIDIA NIM rate limit reached")
                return AIUnavailableError(
                    "The AI service is busy right now. Please try again in a moment."
                )

            if status == 400:
                # Ours to fix, not the user's: a malformed request or an oversized
                # context. The response body is logged, never returned to the client.
                logger.error("NVIDIA NIM rejected the request: HTTP 400 body=%s", exc.response.text)
                return AIUnavailableError(
                    "The AI could not process this document. The problem has been logged."
                )

            logger.error("NVIDIA NIM returned HTTP %s", status, exc_info=True)
            return AIUnavailableError(
                "The AI service is temporarily unavailable. Please try again shortly."
            )

        if isinstance(exc, httpx.TimeoutException):
            logger.error("NVIDIA NIM request timed out")
            return AIUnavailableError(
                "The AI took too long to respond. Large documents can take a while — "
                "please try again."
            )

        if isinstance(exc, httpx.TransportError):
            logger.error("Could not reach NVIDIA NIM: %s", type(exc).__name__)
            return AIUnavailableError(
                "Could not reach the AI service. Please check the connection and try again."
            )

        logger.error("Unexpected AI provider failure: %s", type(exc).__name__, exc_info=True)
        return AIUnavailableError(
            "Something went wrong while contacting the AI service. The problem has been logged."
        )


def _first_choice(data: dict[str, Any]) -> dict[str, Any]:
    """Return the first completion choice, or an empty one if the reply is malformed.

    An empty choice yields empty text and a ``None`` finish reason rather than an
    ``IndexError`` — the caller's existing "empty response" handling covers the rest.
    """
    choices = data.get("choices") or []
    first: dict[str, Any] = choices[0] if choices else {}
    return first


def _build_prompt(context_blocks: list[ContextBlock], document_title: str, instruction: str) -> str:
    """Render the document's content and the instruction as a single user message.

    There is no document-with-citable-sections concept in a plain chat completion, so
    sections are numbered for the model's own organisation only — never presented as
    something it could cite, since nothing downstream could resolve such a citation.
    """
    sections = [
        f"--- Section {index} ---\n{block.text}"
        for index, block in enumerate(context_blocks, start=1)
    ]
    body = "\n\n".join(sections)
    return f"Document: {document_title}\n\n{body}\n\n{instruction}"


def _build_structured_prompt(
    context_blocks: list[ContextBlock],
    document_title: str,
    instruction: str,
    output_model: type[BaseModel],
) -> str:
    """Extend the base prompt with the target schema, since JSON mode only guarantees
    syntactically valid JSON — not conformance to any particular shape."""
    schema_json = json.dumps(output_model.model_json_schema(), indent=2)
    base = _build_prompt(context_blocks, document_title, instruction)
    return (
        f"{base}\n\n"
        "Respond with a single JSON object that matches exactly this JSON Schema. "
        "Do not include any text before or after the JSON object.\n\n"
        f"{schema_json}"
    )
