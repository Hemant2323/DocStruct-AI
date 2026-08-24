"""Google Gemini implementation of :class:`LLMProvider`.

Talks to the Gemini REST API directly over ``httpx`` — the same choice made for
:class:`~app.ai.providers.nvidia.NVIDIAProvider` and for the same reason: ``httpx`` is
already installed, and a single JSON POST is all ``generateContent`` requires, so a new
SDK dependency (``google-genai``) is not needed.

**Not yet wired into the application.** Nothing in ``app/core/config.py`` or
``app/api/deps.py`` references this class. It exists as a third, independently testable
:class:`LLMProvider` implementation — configuration/dependency wiring is a deliberately
separate step, exactly as it was for NVIDIA.

Design decisions, and why they follow the same shape as ``NVIDIAProvider``:

* **No native citations.** Gemini's default ``generateContent`` response carries no
  span mapping back to the supplied input, the way Anthropic's custom-content document
  does. Citation-style ``groundingMetadata`` only appears when a grounding tool (such as
  Search) is explicitly enabled, which is out of scope here. ``generate()`` therefore
  always returns an empty ``raw_citations`` tuple rather than inventing one — the same
  honest "no confirmed sources" outcome the rest of the system already handles.
* **Schema described in the prompt, not passed as `responseSchema`.** Gemini does accept
  a native ``responseSchema``, but in its own OpenAPI-3.0-flavoured dialect — not
  standard JSON Schema, and not guaranteed to accept everything
  ``BaseModel.model_json_schema()`` produces (``$defs``, certain ``anyOf`` shapes).
  Rather than depend on that dialect matching, ``responseMimeType: "application/json"``
  is requested for guaranteed-valid JSON, the target shape is described in the prompt
  text, and the reply is validated against ``output_model`` afterward — identical to how
  NVIDIA's JSON-mode path works, and provably correct regardless of schema dialect.
* **The key travels in a header, never a URL.** Gemini's docs describe two auth methods:
  a ``?key=`` query parameter, or an ``x-goog-api-key`` header. The header is used
  deliberately — a key living in a URL can end up in access logs, proxy logs, or browser
  history; a header does not.
* **Two distinct "nothing usable" signals, both handled.** Gemini can block a prompt
  outright (``promptFeedback.blockReason``, no candidates at all) or cut a reply short
  partway through (``candidates[0].finishReason == "SAFETY"`` or ``"RECITATION"``). Both
  are treated the same way NVIDIA's ``content_filter`` is: a clear, typed failure instead
  of a silent empty result.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

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

#: Google's Generative Language API root.
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

#: Best-effort default. Unlike the NVIDIA integration, no exact model string was
#: specified for this one — override via the constructor (and, once configuration
#: wiring is added, via GEMINI_MODEL) rather than relying on this being current.
DEFAULT_MODEL = "gemini-2.5-flash"

#: HTTP statuses worth retrying: rate limiting and transient server-side failure.
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

#: Backoff between retries, scaled by attempt number.
_RETRY_BACKOFF_SECONDS = 0.5

#: finishReason values meaning generation was cut short by safety filtering rather than
#: completing normally.
_BLOCKED_FINISH_REASONS = frozenset({"SAFETY", "RECITATION"})

_BLOCKED_MESSAGE = (
    "The AI declined to analyse this document. This can happen with unusual or "
    "sensitive content. Try a different document."
)


class GeminiProvider(LLMProvider):
    """Talks to a Gemini model through Google's Generative Language REST API.

    Args:
        api_key: Gemini API key. The provider reports itself as unconfigured when this
            is missing or blank — never raised as an authentication failure from a
            partially-built client.
        base_url: The API root, e.g. ``https://generativelanguage.googleapis.com/v1beta``.
        model: Model identifier to request.
        timeout_seconds: Per-request timeout.
        max_retries: Retries for rate limiting and transient server errors.
        max_output_tokens: Ceiling on the reply.

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
        # absolute URL rather than assembling it per request.
        self._generate_url = f"{base_url.rstrip('/')}/models/{model}:generateContent"
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
            logger.info("Gemini provider configured model=%s", model)

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
        payload = _base_payload(system, prompt, self._max_output_tokens)

        try:
            response = await self._post_with_retries(client, payload)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise self._translate(exc) from exc

        self._check_blocked(data)

        candidate = _first_candidate(data)
        finish_reason = candidate.get("finishReason")
        text = _extract_text(candidate)
        usage = data.get("usageMetadata") or {}

        return ModelResult(
            text=text.strip(),
            raw_citations=(),
            input_tokens=int(usage.get("promptTokenCount", 0) or 0),
            output_tokens=int(usage.get("candidatesTokenCount", 0) or 0),
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
        payload = _base_payload(system, prompt, self._max_output_tokens)
        # JSON mode: guarantees syntactically valid JSON, not conformance to our schema
        # — model_validate() below does that. See the module docstring for why this is
        # preferred over Gemini's native `responseSchema`.
        payload["generationConfig"]["responseMimeType"] = "application/json"

        try:
            response = await self._post_with_retries(client, payload)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise self._translate(exc) from exc

        self._check_blocked(data)

        candidate = _first_candidate(data)
        raw_content = _extract_text(candidate)

        try:
            parsed_json = json.loads(raw_content)
            parsed = output_model.model_validate(parsed_json)
        except (json.JSONDecodeError, PydanticValidationError) as exc:
            # The model replied but the shape was wrong. Reported rather than repaired:
            # guessing at a fix is how invented data gets into results.
            raise InvalidModelResponseError(
                "The AI returned a response that did not match the expected format."
            ) from exc

        usage = data.get("usageMetadata") or {}
        return ModelResult(
            parsed=parsed,
            input_tokens=int(usage.get("promptTokenCount", 0) or 0),
            output_tokens=int(usage.get("candidatesTokenCount", 0) or 0),
            stop_reason=candidate.get("finishReason"),
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

        There is no SDK here to supply retry behaviour, so it is written out directly —
        identical in shape to NVIDIAProvider's version of the same helper.
        """
        attempts = self._max_retries + 1

        for attempt in range(attempts):
            is_last_attempt = attempt == attempts - 1

            try:
                response = await client.post(self._generate_url, json=payload)
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

    def _check_blocked(self, data: dict[str, Any]) -> None:
        """Turn either of Gemini's two block signals into a clear, typed failure.

        A blocked prompt or a safety-truncated reply both succeed at the transport
        level, so without this check the user would see a blank or missing result and
        no explanation.
        """
        prompt_feedback = data.get("promptFeedback") or {}
        if prompt_feedback.get("blockReason"):
            logger.warning(
                "Gemini blocked the prompt reason=%s", prompt_feedback.get("blockReason")
            )
            raise AIUnavailableError(_BLOCKED_MESSAGE)

        candidates = data.get("candidates") or []
        if candidates and candidates[0].get("finishReason") in _BLOCKED_FINISH_REASONS:
            logger.warning(
                "Gemini reply was blocked finish_reason=%s", candidates[0].get("finishReason")
            )
            raise AIUnavailableError(_BLOCKED_MESSAGE)

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
                # Ours to fix, not the user's: a malformed request or an oversized
                # context. The response body is logged, never returned to the client.
                logger.error("Gemini rejected the request: HTTP 400 body=%s", exc.response.text)
                return AIUnavailableError(
                    "The AI could not process this document. The problem has been logged."
                )

            logger.error("Gemini returned HTTP %s", status, exc_info=True)
            return AIUnavailableError(
                "The AI service is temporarily unavailable. Please try again shortly."
            )

        if isinstance(exc, httpx.TimeoutException):
            logger.error("Gemini request timed out")
            return AIUnavailableError(
                "The AI took too long to respond. Large documents can take a while — "
                "please try again."
            )

        if isinstance(exc, httpx.TransportError):
            logger.error("Could not reach Gemini: %s", type(exc).__name__)
            return AIUnavailableError(
                "Could not reach the AI service. Please check the connection and try again."
            )

        logger.error("Unexpected AI provider failure: %s", type(exc).__name__, exc_info=True)
        return AIUnavailableError(
            "Something went wrong while contacting the AI service. The problem has been logged."
        )


def _base_payload(system: str, prompt: str, max_output_tokens: int) -> dict[str, Any]:
    """Build the shared request body for both prose and structured requests."""
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {"maxOutputTokens": max_output_tokens},
    }


def _first_candidate(data: dict[str, Any]) -> dict[str, Any]:
    """Return the first candidate, or an empty one if the reply is malformed.

    An empty candidate yields empty text and a ``None`` finish reason rather than an
    ``IndexError`` — a genuinely blocked prompt is already handled separately by
    ``_check_blocked`` before this is called.
    """
    candidates = data.get("candidates") or []
    first: dict[str, Any] = candidates[0] if candidates else {}
    return first


def _extract_text(candidate: dict[str, Any]) -> str:
    """Join every text part of one candidate's content, in order."""
    parts = (candidate.get("content") or {}).get("parts") or []
    return "".join(part.get("text", "") for part in parts if "text" in part)


def _build_prompt(context_blocks: list[ContextBlock], document_title: str, instruction: str) -> str:
    """Render the document's content and the instruction as a single user message.

    There is no document-with-citable-sections concept in a plain generateContent
    request, so sections are numbered for the model's own organisation only — never
    presented as something it could cite, since nothing downstream could resolve such a
    citation.
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
