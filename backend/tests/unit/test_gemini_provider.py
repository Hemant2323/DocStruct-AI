"""Tests for the Gemini provider, with the HTTP client mocked.

Never touches the network — every test patches ``httpx.AsyncClient.post``. What is being
tested is *our* code: how a request is assembled, how a reply is read, and that every
kind of failure becomes a typed error carrying a message that is safe to show a user.

Mirrors the structure of ``test_nvidia_provider.py`` so the three providers are easy to
compare and keep in step.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
import pytest
from pydantic import BaseModel

from app.ai.providers.gemini import DEFAULT_BASE_URL, DEFAULT_MODEL, GeminiProvider
from app.core.exceptions import (
    AIConfigurationError,
    AIUnavailableError,
    InvalidModelResponseError,
    OutputTruncatedError,
)
from app.models.domain.insight import ContextBlock


def _blocks(count: int = 2) -> list[ContextBlock]:
    return [
        ContextBlock(f"Chunk {i} text.", uuid4(), (uuid4(),), f"page {i + 1}") for i in range(count)
    ]


def _http_response(
    *,
    status_code: int = 200,
    content: str = "Result.",
    finish_reason: str = "STOP",
    usage: dict[str, int] | None = None,
    request: httpx.Request | None = None,
) -> httpx.Response:
    """Build a real ``httpx.Response`` so ``.raise_for_status()``/``.json()`` behave
    exactly as they do against a live server — nothing about the response is faked
    beyond the transport itself."""
    body = {
        "candidates": [
            {
                "content": {"parts": [{"text": content}], "role": "model"},
                "finishReason": finish_reason,
            }
        ],
        "usageMetadata": usage or {"promptTokenCount": 100, "candidatesTokenCount": 50},
    }
    return httpx.Response(
        status_code=status_code,
        json=body,
        request=request
        or httpx.Request("POST", "https://example.test/models/gemini:generateContent"),
    )


class TestConfiguration:
    def test_reports_unconfigured_without_a_key(self) -> None:
        provider = GeminiProvider(api_key=None)
        assert provider.is_configured is False

    def test_reports_configured_with_a_key(self) -> None:
        assert GeminiProvider(api_key="test-key-not-real").is_configured is True

    def test_blank_key_counts_as_unconfigured(self) -> None:
        """An empty string in .env is a common mistake and must not look configured."""
        assert GeminiProvider(api_key="   ").is_configured is False

    def test_model_name_comes_from_the_constructor(self) -> None:
        provider = GeminiProvider(api_key="k", model="gemini-2.5-pro")
        assert provider.model_name == "gemini-2.5-pro"

    def test_defaults_point_at_the_generative_language_api(self) -> None:
        assert DEFAULT_BASE_URL == "https://generativelanguage.googleapis.com/v1beta"
        assert DEFAULT_MODEL

    async def test_generating_without_a_key_is_a_clear_error(self) -> None:
        provider = GeminiProvider(api_key=None)

        with pytest.raises(AIConfigurationError) as exc_info:
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert exc_info.value.code == "ai_not_configured"

    async def test_structured_generation_without_a_key_is_a_clear_error(self) -> None:
        class Out(BaseModel):
            value: str

        provider = GeminiProvider(api_key=None)

        with pytest.raises(AIConfigurationError):
            await provider.generate_structured(
                system="s",
                instruction="i",
                context_blocks=_blocks(),
                document_title="d.pdf",
                output_model=Out,
            )


class TestRequestConstruction:
    async def test_content_is_sent_in_document_order(self) -> None:
        """Order must be preserved even though there is no citation mechanism to key
        off it — an out-of-order prompt would still mislead the model about structure."""
        provider = GeminiProvider(api_key="k")
        blocks = _blocks(3)
        captured: dict[str, Any] = {}

        async def capture(url: str, *, json: dict[str, Any]) -> httpx.Response:
            captured["url"] = url
            captured["json"] = json
            return _http_response()

        with patch.object(provider._client, "post", side_effect=capture):
            await provider.generate(
                system="s", instruction="i", context_blocks=blocks, document_title="report.pdf"
            )

        prompt = captured["json"]["contents"][0]["parts"][0]["text"]
        assert (
            prompt.index(blocks[0].text)
            < prompt.index(blocks[1].text)
            < prompt.index(blocks[2].text)
        )
        assert "report.pdf" in prompt

    async def test_system_prompt_is_sent_as_the_system_instruction(self) -> None:
        provider = GeminiProvider(api_key="k")
        captured: dict[str, Any] = {}

        async def capture(url: str, *, json: dict[str, Any]) -> httpx.Response:
            captured["json"] = json
            return _http_response()

        with patch.object(provider._client, "post", side_effect=capture):
            await provider.generate(
                system="Follow these rules exactly.",
                instruction="i",
                context_blocks=_blocks(),
                document_title="d.pdf",
            )

        assert captured["json"]["systemInstruction"] == {
            "parts": [{"text": "Follow these rules exactly."}]
        }

    async def test_requests_go_to_the_generate_content_endpoint(self) -> None:
        provider = GeminiProvider(
            api_key="k",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            model="gemini-3.6-flash",
        )
        captured: dict[str, Any] = {}

        async def capture(url: str, *, json: dict[str, Any]) -> httpx.Response:
            captured["url"] = url
            return _http_response()

        with patch.object(provider._client, "post", side_effect=capture):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert captured["url"] == (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-3.6-flash:generateContent"
        )

    async def test_a_trailing_slash_on_the_base_url_does_not_double_up(self) -> None:
        provider = GeminiProvider(
            api_key="k",
            base_url="https://generativelanguage.googleapis.com/v1beta/",
            model="gemini-3.6-flash",
        )
        captured: dict[str, Any] = {}

        async def capture(url: str, *, json: dict[str, Any]) -> httpx.Response:
            captured["url"] = url
            return _http_response()

        with patch.object(provider._client, "post", side_effect=capture):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert captured["url"] == (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-3.6-flash:generateContent"
        )

    async def test_the_key_is_sent_as_a_header_never_a_query_parameter(self) -> None:
        """A key in a URL can end up in access logs, proxy logs, or browser history —
        the header form must be what is actually used."""
        provider = GeminiProvider(api_key="secret-key-value")

        assert provider._client is not None
        assert provider._client.headers["x-goog-api-key"] == "secret-key-value"
        assert "key" not in provider._generate_url

    async def test_max_tokens_comes_from_the_constructor(self) -> None:
        provider = GeminiProvider(api_key="k", max_output_tokens=1234)
        captured: dict[str, Any] = {}

        async def capture(url: str, *, json: dict[str, Any]) -> httpx.Response:
            captured["json"] = json
            return _http_response()

        with patch.object(provider._client, "post", side_effect=capture):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert captured["json"]["generationConfig"]["maxOutputTokens"] == 1234

    async def test_structured_requests_ask_for_json_mime_type(self) -> None:
        class Out(BaseModel):
            value: str

        provider = GeminiProvider(api_key="k")
        captured: dict[str, Any] = {}

        async def capture(url: str, *, json: dict[str, Any]) -> httpx.Response:
            captured["json"] = json
            return _http_response(content=json_dumps({"value": "ok"}))

        with patch.object(provider._client, "post", side_effect=capture):
            await provider.generate_structured(
                system="s",
                instruction="i",
                context_blocks=_blocks(),
                document_title="d.pdf",
                output_model=Out,
            )

        assert captured["json"]["generationConfig"]["responseMimeType"] == "application/json"

    async def test_structured_prompt_describes_the_target_schema(self) -> None:
        """JSON mode only guarantees valid JSON, not our shape — the schema must be in
        the prompt or the model has no way to know what fields are expected."""

        class Out(BaseModel):
            total_amount: str

        provider = GeminiProvider(api_key="k")
        captured: dict[str, Any] = {}

        async def capture(url: str, *, json: dict[str, Any]) -> httpx.Response:
            captured["json"] = json
            return _http_response(content=json_dumps({"total_amount": "$1"}))

        with patch.object(provider._client, "post", side_effect=capture):
            await provider.generate_structured(
                system="s",
                instruction="i",
                context_blocks=_blocks(),
                document_title="d.pdf",
                output_model=Out,
            )

        prompt = captured["json"]["contents"][0]["parts"][0]["text"]
        assert "total_amount" in prompt


class TestResponseHandling:
    async def test_text_is_returned_and_stripped(self) -> None:
        provider = GeminiProvider(api_key="k")

        with patch.object(
            provider._client,
            "post",
            AsyncMock(return_value=_http_response(content="  The answer.  ")),
        ):
            result = await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert result.text == "The answer."

    async def test_citations_are_always_empty(self) -> None:
        """The single most important behavioural fact about this provider: it has no
        citation mechanism, so it must never invent one."""
        provider = GeminiProvider(api_key="k")

        with patch.object(provider._client, "post", AsyncMock(return_value=_http_response())):
            result = await provider.generate(
                system="s",
                instruction="i",
                context_blocks=_blocks(),
                document_title="d.pdf",
                enable_citations=True,
            )

        assert result.raw_citations == ()

    async def test_usage_is_reported(self) -> None:
        provider = GeminiProvider(api_key="k")
        response = _http_response(usage={"promptTokenCount": 321, "candidatesTokenCount": 45})

        with patch.object(provider._client, "post", AsyncMock(return_value=response)):
            result = await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert result.input_tokens == 321
        assert result.output_tokens == 45

    async def test_missing_usage_defaults_to_zero(self) -> None:
        provider = GeminiProvider(api_key="k")
        response = httpx.Response(
            status_code=200,
            json={"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]},
            request=httpx.Request("POST", "https://example.test/models/gemini:generateContent"),
        )

        with patch.object(provider._client, "post", AsyncMock(return_value=response)):
            result = await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert result.input_tokens == 0
        assert result.output_tokens == 0

    async def test_empty_candidates_does_not_crash(self) -> None:
        """A malformed reply must degrade to empty text, not an unhandled IndexError."""
        provider = GeminiProvider(api_key="k")
        response = httpx.Response(
            status_code=200,
            json={"candidates": []},
            request=httpx.Request("POST", "https://example.test/models/gemini:generateContent"),
        )

        with patch.object(provider._client, "post", AsyncMock(return_value=response)):
            result = await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert result.text == ""

    async def test_a_blocked_prompt_becomes_a_clear_error(self) -> None:
        """promptFeedback.blockReason means generation never started at all: HTTP 200
        with no candidates. Without this check the user would see a blank result and no
        explanation."""
        provider = GeminiProvider(api_key="k")
        response = httpx.Response(
            status_code=200,
            json={"promptFeedback": {"blockReason": "SAFETY"}},
            request=httpx.Request("POST", "https://example.test/models/gemini:generateContent"),
        )

        with (
            patch.object(provider._client, "post", AsyncMock(return_value=response)),
            pytest.raises(AIUnavailableError) as exc_info,
        ):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert "declined" in exc_info.value.message

    async def test_a_safety_finish_reason_becomes_a_clear_error(self) -> None:
        """Generation can also be cut short partway through, distinct from an outright
        blocked prompt — both must be handled."""
        provider = GeminiProvider(api_key="k")
        response = _http_response(content="", finish_reason="SAFETY")

        with (
            patch.object(provider._client, "post", AsyncMock(return_value=response)),
            pytest.raises(AIUnavailableError) as exc_info,
        ):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert "declined" in exc_info.value.message

    async def test_a_recitation_finish_reason_becomes_a_clear_error(self) -> None:
        provider = GeminiProvider(api_key="k")
        response = _http_response(content="", finish_reason="RECITATION")

        with (
            patch.object(provider._client, "post", AsyncMock(return_value=response)),
            pytest.raises(AIUnavailableError),
        ):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

    async def test_structured_output_is_parsed_and_validated(self) -> None:
        class Out(BaseModel):
            name: str
            value: int

        provider = GeminiProvider(api_key="k")
        response = _http_response(content=json.dumps({"name": "total", "value": 42}))

        with patch.object(provider._client, "post", AsyncMock(return_value=response)):
            result = await provider.generate_structured(
                system="s",
                instruction="i",
                context_blocks=_blocks(),
                document_title="d.pdf",
                output_model=Out,
            )

        assert isinstance(result.parsed, Out)
        assert result.parsed.name == "total"
        assert result.parsed.value == 42

    async def test_malformed_json_is_reported_not_repaired(self) -> None:
        class Out(BaseModel):
            value: str

        provider = GeminiProvider(api_key="k")
        response = _http_response(content="not valid json {{{")

        with (
            patch.object(provider._client, "post", AsyncMock(return_value=response)),
            pytest.raises(InvalidModelResponseError),
        ):
            await provider.generate_structured(
                system="s",
                instruction="i",
                context_blocks=_blocks(),
                document_title="d.pdf",
                output_model=Out,
            )

    async def test_json_missing_a_required_field_is_reported(self) -> None:
        """Syntactically valid JSON that does not match the schema must still fail —
        JSON mode alone cannot guarantee shape."""

        class Out(BaseModel):
            required_field: str

        provider = GeminiProvider(api_key="k")
        response = _http_response(content=json.dumps({"something_else": "x"}))

        with (
            patch.object(provider._client, "post", AsyncMock(return_value=response)),
            pytest.raises(InvalidModelResponseError),
        ):
            await provider.generate_structured(
                system="s",
                instruction="i",
                context_blocks=_blocks(),
                document_title="d.pdf",
                output_model=Out,
            )


class TestErrorsAreSafeAndTyped:
    """Every upstream failure must become a typed error with a message safe to display."""

    async def test_401_becomes_a_configuration_error(self) -> None:
        provider = GeminiProvider(api_key="k")
        response = _http_response(status_code=401, content="")

        with (
            patch.object(provider._client, "post", AsyncMock(return_value=response)),
            pytest.raises(AIConfigurationError) as exc_info,
        ):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert exc_info.value.code == "ai_not_configured"

    async def test_403_becomes_a_configuration_error(self) -> None:
        provider = GeminiProvider(api_key="k")
        response = _http_response(status_code=403, content="")

        with (
            patch.object(provider._client, "post", AsyncMock(return_value=response)),
            pytest.raises(AIConfigurationError),
        ):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

    async def test_429_becomes_unavailable_after_exhausting_retries(self) -> None:
        provider = GeminiProvider(api_key="k", max_retries=1)
        response = _http_response(status_code=429, content="")

        with (
            patch.object(provider._client, "post", AsyncMock(return_value=response)),
            pytest.raises(AIUnavailableError) as exc_info,
        ):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert exc_info.value.code == "ai_unavailable"

    async def test_429_is_retried_and_can_succeed(self) -> None:
        """The retry budget must actually be used, not merely accepted as a parameter."""
        provider = GeminiProvider(api_key="k", max_retries=2)
        attempts = [
            _http_response(status_code=429, content=""),
            _http_response(status_code=200, content="Recovered."),
        ]

        with patch.object(provider._client, "post", AsyncMock(side_effect=attempts)):
            result = await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert result.text == "Recovered."

    async def test_500_is_retried_then_reported(self) -> None:
        provider = GeminiProvider(api_key="k", max_retries=1)
        response = _http_response(status_code=500, content="")

        with (
            patch.object(provider._client, "post", AsyncMock(return_value=response)) as mock_post,
            pytest.raises(AIUnavailableError),
        ):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        # max_retries=1 means the initial attempt plus one retry: two calls total.
        assert mock_post.call_count == 2

    async def test_400_becomes_unavailable_not_configuration(self) -> None:
        """A bad request is our bug, not the user's credentials — it must not be
        reported as a configuration problem."""
        provider = GeminiProvider(api_key="k")
        response = _http_response(status_code=400, content="")

        with (
            patch.object(provider._client, "post", AsyncMock(return_value=response)),
            pytest.raises(AIUnavailableError) as exc_info,
        ):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert exc_info.value.code == "ai_unavailable"

    async def test_timeout_becomes_a_clear_error(self) -> None:
        provider = GeminiProvider(api_key="k", max_retries=0)

        with (
            patch.object(
                provider._client, "post", AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
            ),
            pytest.raises(AIUnavailableError) as exc_info,
        ):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert "took too long" in exc_info.value.message

    async def test_connection_failure_becomes_a_clear_error(self) -> None:
        provider = GeminiProvider(api_key="k", max_retries=0)

        with (
            patch.object(
                provider._client, "post", AsyncMock(side_effect=httpx.ConnectError("refused"))
            ),
            pytest.raises(AIUnavailableError) as exc_info,
        ):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert "reach the AI service" in exc_info.value.message

    async def test_an_unexpected_exception_still_becomes_a_typed_error(self) -> None:
        provider = GeminiProvider(api_key="k", max_retries=0)

        with (
            patch.object(provider._client, "post", AsyncMock(side_effect=RuntimeError("boom"))),
            pytest.raises(AIUnavailableError),
        ):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

    async def test_an_api_key_never_appears_in_an_error_message(self) -> None:
        """The single most important safety test in this file."""
        provider = GeminiProvider(api_key="AIzaSySUPERSECRET123")
        response = _http_response(status_code=401, content="invalid key AIzaSySUPERSECRET123")

        with (
            patch.object(provider._client, "post", AsyncMock(return_value=response)),
            pytest.raises(AIConfigurationError) as exc_info,
        ):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert "SUPERSECRET" not in exc_info.value.message
        assert "AIzaSy" not in exc_info.value.message

    async def test_upstream_response_body_is_not_passed_through(self) -> None:
        provider = GeminiProvider(api_key="k")
        response = _http_response(
            status_code=400,
            content="internal routing failure on node tpu-cluster-7 model gemini-internal-x",
        )

        with (
            patch.object(provider._client, "post", AsyncMock(return_value=response)),
            pytest.raises(AIUnavailableError) as exc_info,
        ):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert "tpu-cluster-7" not in exc_info.value.message
        assert "gemini-internal-x" not in exc_info.value.message

    async def test_no_bare_exception_types_are_ever_raised(self) -> None:
        """Every failure path must produce one of our own exception types, never let an
        httpx or json exception escape to the caller."""
        provider = GeminiProvider(api_key="k", max_retries=0)

        with (
            patch.object(
                provider._client, "post", AsyncMock(side_effect=httpx.ConnectTimeout("x"))
            ),
            pytest.raises((AIConfigurationError, AIUnavailableError, InvalidModelResponseError)),
        ):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )


class TestOutputTokenCeiling:
    """A reply cut short by `ai_max_output_tokens`, which is not a safety stop.

    Gemini reports it as ``finishReason: "MAX_TOKENS"`` on an otherwise successful
    response, so without an explicit check it looks like a normal completion whose JSON
    happens to be malformed. That distinction matters: it is the failure a long document
    actually hits, and reporting it as a schema mismatch sends whoever reads the log
    hunting for a bug that is not there.
    """

    async def test_a_truncated_structured_reply_is_reported_as_running_out_of_room(
        self,
    ) -> None:
        class Out(BaseModel):
            value: str

        provider = GeminiProvider(api_key="k")
        # Valid JSON, so this cannot pass merely because parsing failed — the check has
        # to be looking at the finish reason.
        response = _http_response(content=json_dumps({"value": "ok"}), finish_reason="MAX_TOKENS")

        with (
            patch.object(provider._client, "post", AsyncMock(return_value=response)),
            pytest.raises(InvalidModelResponseError) as exc_info,
        ):
            await provider.generate_structured(
                system="s",
                instruction="i",
                context_blocks=_blocks(),
                document_title="d.pdf",
                output_model=Out,
            )

        assert "ran out of room" in exc_info.value.message

    async def test_the_error_is_distinguishable_from_a_schema_mismatch(self) -> None:
        """The distinction a caller acts on. A schema mismatch is a dead end; running
        out of room means the request asked for more than one reply can hold, which is
        why conversion can respond by asking for less at a time. Still an
        `InvalidModelResponseError`, so every caller that does not care is unaffected —
        including the HTTP status mapping, which matches on the parent class.
        """

        class Out(BaseModel):
            value: str

        provider = GeminiProvider(api_key="k")
        response = _http_response(content=json_dumps({"value": "ok"}), finish_reason="MAX_TOKENS")

        with (
            patch.object(provider._client, "post", AsyncMock(return_value=response)),
            pytest.raises(OutputTruncatedError) as exc_info,
        ):
            await provider.generate_structured(
                system="s",
                instruction="i",
                context_blocks=_blocks(),
                document_title="d.pdf",
                output_model=Out,
            )

        assert isinstance(exc_info.value, InvalidModelResponseError)
        assert exc_info.value.code == "output_truncated"

    async def test_a_genuine_schema_mismatch_is_not_reported_as_truncation(self) -> None:
        """The other side of the same distinction: a well-formed reply that does not fit
        the schema must stay the plain error, or conversion would re-run the whole
        document through windows to reach the identical mismatch."""

        class Out(BaseModel):
            value: str

        provider = GeminiProvider(api_key="k")
        response = _http_response(content=json_dumps({"wrong": "shape"}), finish_reason="STOP")

        with (
            patch.object(provider._client, "post", AsyncMock(return_value=response)),
            pytest.raises(InvalidModelResponseError) as exc_info,
        ):
            await provider.generate_structured(
                system="s",
                instruction="i",
                context_blocks=_blocks(),
                document_title="d.pdf",
                output_model=Out,
            )

        assert not isinstance(exc_info.value, OutputTruncatedError)

    async def test_the_message_does_not_blame_the_response_format(self) -> None:
        """The wording is the point of the fix, so it is asserted rather than assumed."""

        class Out(BaseModel):
            value: str

        provider = GeminiProvider(api_key="k")
        response = _http_response(content='{"value": "trunca', finish_reason="MAX_TOKENS")

        with (
            patch.object(provider._client, "post", AsyncMock(return_value=response)),
            pytest.raises(InvalidModelResponseError) as exc_info,
        ):
            await provider.generate_structured(
                system="s",
                instruction="i",
                context_blocks=_blocks(),
                document_title="d.pdf",
                output_model=Out,
            )

        assert "did not match the expected format" not in exc_info.value.message

    async def test_a_normal_finish_reason_still_parses(self) -> None:
        """Guards the check itself: it must not reject healthy replies."""

        class Out(BaseModel):
            value: str

        provider = GeminiProvider(api_key="k")
        response = _http_response(content=json_dumps({"value": "ok"}), finish_reason="STOP")

        with patch.object(provider._client, "post", AsyncMock(return_value=response)):
            result = await provider.generate_structured(
                system="s",
                instruction="i",
                context_blocks=_blocks(),
                document_title="d.pdf",
                output_model=Out,
            )

        assert isinstance(result.parsed, Out)
        assert result.parsed.value == "ok"

    async def test_prose_generation_is_unaffected_by_a_length_stop(self) -> None:
        """Truncated prose is still readable text, so it is returned with the finish
        reason attached rather than thrown away. Only structured output is fatal."""
        provider = GeminiProvider(api_key="k")
        response = _http_response(content="A partial answer", finish_reason="MAX_TOKENS")

        with patch.object(provider._client, "post", AsyncMock(return_value=response)):
            result = await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert result.text == "A partial answer"
        assert result.stop_reason == "MAX_TOKENS"


def json_dumps(value: dict[str, Any]) -> str:
    return json.dumps(value)
