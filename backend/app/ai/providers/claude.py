"""Anthropic Claude implementation of :class:`LLMProvider`.

Uses the official ``anthropic`` SDK — never an OpenAI-compatible shim or raw HTTP
(CLAUDE.md §5).

Two API details drive the shape of this file:

* Content is sent as a **custom-content document**: a list of text blocks that we build.
  Citations then come back as indices into *that list*, so a citation resolves to a known
  chunk by lookup rather than by interpreting anything the model wrote. This is what lets
  the product show a real page number without ever inventing one.
* **Citations and structured outputs cannot be combined** — the API returns 400. Prose
  results therefore use citations; extraction uses a schema and verifies its quotes
  separately.
"""

from __future__ import annotations

from typing import Any, cast
from urllib.parse import urlsplit

import anthropic
from anthropic.types import MessageParam, OutputConfigParam
from anthropic.types.beta import BetaMessageParam, BetaOutputConfigParam
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from app.ai.providers.base import LLMProvider
from app.core.config import Settings
from app.core.exceptions import (
    AIConfigurationError,
    AIUnavailableError,
    InvalidModelResponseError,
)
from app.core.logging import get_logger
from app.models.domain.insight import ContextBlock, ModelResult, RawCitation

logger = get_logger(__name__)

#: Beta flag enabling server-side fallback routing. If a safety classifier declines the
#: request, the API retries on another model instead of returning nothing useful.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

#: Minimum blocks before it is worth asking the API to cache the document prefix. Below
#: roughly 1024 tokens the API will not cache anyway, so the marker would be noise.
_MIN_BLOCKS_FOR_CACHING = 4

_REFUSAL_MESSAGE = (
    "The AI declined to analyse this document. This can happen with unusual or "
    "sensitive content. Try a different document."
)


class ClaudeProvider(LLMProvider):
    """Talks to Claude through the Anthropic API.

    Args:
        settings: Supplies the API key, model, limits and timeouts. The key is read here
            and never leaves the server — it is not returned by any endpoint and never
            appears in a log line.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: anthropic.AsyncAnthropic | None = None

        if settings.ai_configured:
            # `base_url` is omitted entirely when unset so the SDK uses its own default;
            # passing None explicitly would also work, but leaving it out keeps the
            # default path identical to before this option existed.
            optional: dict[str, Any] = {}
            if settings.anthropic_base_url:
                optional["base_url"] = settings.anthropic_base_url

            self._client = anthropic.AsyncAnthropic(
                api_key=settings.anthropic_api_key,
                timeout=settings.ai_timeout_seconds,
                # The SDK retries 429s, 5xx and connection errors on its own, which is
                # the right layer for it — no hand-rolled retry loop.
                max_retries=settings.ai_max_retries,
                **optional,
            )

            if settings.anthropic_base_url:
                # Host only: a base URL can legitimately carry a path, and it must never
                # be logged in a form that could include credentials.
                logger.info(
                    "Using a custom API endpoint host=%s model=%s",
                    urlsplit(settings.anthropic_base_url).netloc,
                    settings.anthropic_model,
                )

    @property
    def model_name(self) -> str:
        return self._settings.anthropic_model

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
        client = self._require_client()

        document = self._build_document(context_blocks, document_title, enable_citations)
        # The request bodies are assembled as plain dicts because their shape depends on
        # runtime choices (citations on or off, caching on or off). Cast to the SDK's
        # TypedDicts at the boundary rather than weakening type checking anywhere else.
        messages = [
            cast(
                "BetaMessageParam",
                {"role": "user", "content": [document, {"type": "text", "text": instruction}]},
            )
        ]
        output_config = cast("BetaOutputConfigParam", {"effort": self._settings.ai_effort})

        try:
            # Streaming even though the reply is short: the *input* can be very large,
            # and streaming is what stops a long request from hitting an HTTP timeout.
            async with client.beta.messages.stream(
                model=self.model_name,
                max_tokens=self._settings.ai_max_output_tokens,
                system=system,
                messages=messages,
                thinking={"type": "adaptive"},
                output_config=output_config,
                betas=[_FALLBACK_BETA],
                fallbacks="default",
            ) as stream:
                response = await stream.get_final_message()
        except Exception as exc:
            raise self._translate(exc) from exc

        self._check_refusal(response)
        text, citations = _extract_text_and_citations(response)

        return ModelResult(
            text=text,
            raw_citations=tuple(citations),
            input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
            stop_reason=getattr(response, "stop_reason", None),
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

        # Citations are deliberately disabled: combining them with a schema is rejected
        # by the API. Provenance for extraction comes from verifying quotes instead.
        document = self._build_document(context_blocks, document_title, enable_citations=False)
        messages = [
            cast(
                "MessageParam",
                {"role": "user", "content": [document, {"type": "text", "text": instruction}]},
            )
        ]
        output_config = cast("OutputConfigParam", {"effort": self._settings.ai_effort})

        try:
            response = await client.messages.parse(
                model=self.model_name,
                max_tokens=self._settings.ai_max_output_tokens,
                system=system,
                messages=messages,
                thinking={"type": "adaptive"},
                output_config=output_config,
                output_format=output_model,
            )
        except PydanticValidationError as exc:
            # The model replied but the shape was wrong. Reported rather than repaired:
            # guessing at a fix is how invented data gets into results.
            raise InvalidModelResponseError(
                "The AI returned a response that did not match the expected format.",
            ) from exc
        except Exception as exc:
            raise self._translate(exc) from exc

        self._check_refusal(response)

        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            raise InvalidModelResponseError(
                "The AI did not return the structured information that was requested."
            )

        return ModelResult(
            parsed=parsed,
            input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
            stop_reason=getattr(response, "stop_reason", None),
        )

    # ----------------------------------------------------------------- helpers
    def _require_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            raise AIConfigurationError(
                "AI features are not configured on this server. An Anthropic API key is required."
            )
        return self._client

    def _build_document(
        self, context_blocks: list[ContextBlock], title: str, enable_citations: bool
    ) -> dict[str, Any]:
        """Assemble the document content block sent to the API.

        The ``content`` list is the citation coordinate system: a citation names a range
        of positions in it. Order therefore must match ``context_blocks`` exactly, which
        is what makes ``citations.py`` a lookup rather than a guess.
        """
        content: list[dict[str, Any]] = [
            {"type": "text", "text": block.text} for block in context_blocks
        ]

        document: dict[str, Any] = {
            "type": "document",
            "source": {"type": "content", "content": content},
            "title": title,
            # `context` is shown to the model but is not itself citable, which makes it
            # the right place for a caveat that should never be quoted back as evidence.
            "context": ("This is the user's uploaded document, extracted into ordered sections."),
        }

        if enable_citations:
            document["citations"] = {"enabled": True}

        if self._settings.ai_prompt_caching and len(content) >= _MIN_BLOCKS_FOR_CACHING:
            # The same document is summarised, explained and extracted from. Caching the
            # prefix makes the second and third call much cheaper.
            document["cache_control"] = {"type": "ephemeral"}

        return document

    def _check_refusal(self, response: Any) -> None:
        """Turn a declined request into a clear, typed failure.

        A refusal arrives as HTTP 200 with an empty-looking body, so without this check
        the user would see a blank result and no explanation.
        """
        if getattr(response, "stop_reason", None) != "refusal":
            return

        details = getattr(response, "stop_details", None)
        logger.warning(
            "Model declined the request category=%s",
            getattr(details, "category", None),
        )
        raise AIUnavailableError(_REFUSAL_MESSAGE)

    def _translate(self, exc: Exception) -> Exception:
        """Convert an SDK exception into one of our typed errors.

        Upstream messages can name models, internal endpoints and request details, so
        none of them are passed through. The original is logged and chained; the user
        gets a sentence they can act on.
        """
        if isinstance(exc, AIConfigurationError | AIUnavailableError | InvalidModelResponseError):
            return exc

        if isinstance(exc, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
            logger.error("Anthropic rejected our credentials: %s", type(exc).__name__)
            return AIConfigurationError("AI features are not configured correctly on this server.")

        if isinstance(exc, anthropic.RateLimitError):
            logger.warning("Anthropic rate limit reached")
            return AIUnavailableError(
                "The AI service is busy right now. Please try again in a moment."
            )

        if isinstance(exc, anthropic.APITimeoutError):
            logger.error("Anthropic request timed out after %ss", self._settings.ai_timeout_seconds)
            return AIUnavailableError(
                "The AI took too long to respond. Large documents can take a while — "
                "please try again."
            )

        if isinstance(exc, anthropic.APIConnectionError):
            logger.error("Could not reach Anthropic: %s", type(exc).__name__)
            return AIUnavailableError(
                "Could not reach the AI service. Please check the connection and try again."
            )

        if isinstance(exc, anthropic.BadRequestError):
            # Ours to fix, not the user's: a malformed request or an oversized context.
            logger.error("Anthropic rejected the request: %s", exc, exc_info=True)
            return AIUnavailableError(
                "The AI could not process this document. The problem has been logged."
            )

        if isinstance(exc, anthropic.APIStatusError):
            logger.error("Anthropic returned %s", getattr(exc, "status_code", "?"), exc_info=True)
            return AIUnavailableError(
                "The AI service is temporarily unavailable. Please try again shortly."
            )

        logger.error("Unexpected AI provider failure: %s", type(exc).__name__, exc_info=True)
        return AIUnavailableError(
            "Something went wrong while contacting the AI service. The problem has been logged."
        )


def _extract_text_and_citations(response: Any) -> tuple[str, list[RawCitation]]:
    """Pull the reply text and citation spans out of an API response.

    With citations enabled the reply arrives as several text blocks, each optionally
    carrying the spans that support it. The text is rejoined in order, and every span is
    kept with the block index it names.

    Only ``content_block_location`` spans are collected — the kind produced by
    custom-content documents. Other location types cannot occur here, and silently
    accepting one would mean recording a location we cannot map.
    """
    parts: list[str] = []
    citations: list[RawCitation] = []

    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) != "text":
            continue

        parts.append(getattr(block, "text", "") or "")

        for citation in getattr(block, "citations", None) or []:
            if getattr(citation, "type", None) != "content_block_location":
                continue
            start = getattr(citation, "start_block_index", None)
            end = getattr(citation, "end_block_index", None)
            if start is None or end is None:
                continue
            citations.append(
                RawCitation(
                    quoted_text=getattr(citation, "cited_text", "") or "",
                    start_block_index=int(start),
                    end_block_index=int(end),
                    document_index=int(getattr(citation, "document_index", 0) or 0),
                )
            )

    return "".join(parts).strip(), citations
