"""Tests for the NVIDIA NIM provider, with the HTTP client mocked.

Never touches the network — every test patches ``httpx.AsyncClient.post``. What is being
tested is *our* code: how a request is assembled, how a reply is read, and that every
kind of failure becomes a typed error carrying a message that is safe to show a user.

Mirrors the structure of ``test_claude_provider.py`` so the two providers are easy to
compare and keep in step.
"""

from __future__ import annotations

import json
import json as json_lib
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
import pytest
from pydantic import BaseModel

from app.ai.providers.nvidia import DEFAULT_BASE_URL, DEFAULT_MODEL, NVIDIAProvider
from app.core.exceptions import (
    AIConfigurationError,
    AIUnavailableError,
    InvalidModelResponseError,
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
    finish_reason: str = "stop",
    usage: dict[str, int] | None = None,
    request: httpx.Request | None = None,
) -> httpx.Response:
    """Build a real ``httpx.Response`` so ``.raise_for_status()``/``.json()`` behave
    exactly as they do against a live server — nothing about the response is faked
    beyond the transport itself."""
    body = {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": usage or {"prompt_tokens": 100, "completion_tokens": 50},
    }
    return httpx.Response(
        status_code=status_code,
        json=body,
        request=request or httpx.Request("POST", "https://example.test/chat/completions"),
    )


class TestConfiguration:
    def test_reports_unconfigured_without_a_key(self) -> None:
        provider = NVIDIAProvider(api_key=None)
        assert provider.is_configured is False

    def test_reports_configured_with_a_key(self) -> None:
        assert NVIDIAProvider(api_key="test-key-not-real").is_configured is True

    def test_blank_key_counts_as_unconfigured(self) -> None:
        """An empty string in .env is a common mistake and must not look configured."""
        assert NVIDIAProvider(api_key="   ").is_configured is False

    def test_model_name_comes_from_the_constructor(self) -> None:
        provider = NVIDIAProvider(api_key="k", model="deepseek-ai/deepseek-v4-flash-0731")
        assert provider.model_name == "deepseek-ai/deepseek-v4-flash-0731"

    def test_defaults_match_the_requested_nvidia_nim_configuration(self) -> None:
        """Pins the two constants this integration exists for."""
        assert DEFAULT_BASE_URL == "https://integrate.api.nvidia.com/v1"
        assert DEFAULT_MODEL == "deepseek-ai/deepseek-v4-flash-0731"

    async def test_generating_without_a_key_is_a_clear_error(self) -> None:
        provider = NVIDIAProvider(api_key=None)

        with pytest.raises(AIConfigurationError) as exc_info:
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert exc_info.value.code == "ai_not_configured"

    async def test_structured_generation_without_a_key_is_a_clear_error(self) -> None:
        class Out(BaseModel):
            value: str

        provider = NVIDIAProvider(api_key=None)

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
        provider = NVIDIAProvider(api_key="k")
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

        prompt = captured["json"]["messages"][1]["content"]
        assert (
            prompt.index(blocks[0].text)
            < prompt.index(blocks[1].text)
            < prompt.index(blocks[2].text)
        )
        assert "report.pdf" in prompt

    async def test_system_prompt_is_sent_as_the_system_message(self) -> None:
        provider = NVIDIAProvider(api_key="k")
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

        assert captured["json"]["messages"][0] == {
            "role": "system",
            "content": "Follow these rules exactly.",
        }

    async def test_requests_go_to_the_chat_completions_endpoint(self) -> None:
        provider = NVIDIAProvider(api_key="k", base_url="https://integrate.api.nvidia.com/v1")
        captured: dict[str, Any] = {}

        async def capture(url: str, *, json: dict[str, Any]) -> httpx.Response:
            captured["url"] = url
            return _http_response()

        with patch.object(provider._client, "post", side_effect=capture):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert captured["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"

    async def test_a_trailing_slash_on_the_base_url_does_not_double_up(self) -> None:
        provider = NVIDIAProvider(api_key="k", base_url="https://integrate.api.nvidia.com/v1/")
        captured: dict[str, Any] = {}

        async def capture(url: str, *, json: dict[str, Any]) -> httpx.Response:
            captured["url"] = url
            return _http_response()

        with patch.object(provider._client, "post", side_effect=capture):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert captured["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"

    async def test_max_tokens_comes_from_the_constructor(self) -> None:
        provider = NVIDIAProvider(api_key="k", max_output_tokens=1234)
        captured: dict[str, Any] = {}

        async def capture(url: str, *, json: dict[str, Any]) -> httpx.Response:
            captured["json"] = json
            return _http_response()

        with patch.object(provider._client, "post", side_effect=capture):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert captured["json"]["max_tokens"] == 1234

    async def test_structured_requests_ask_for_json_mode(self) -> None:
        class Out(BaseModel):
            value: str

        provider = NVIDIAProvider(api_key="k")
        captured: dict[str, Any] = {}

        async def capture(url: str, *, json: dict[str, Any]) -> httpx.Response:
            captured["json"] = json
            return _http_response(content=json_lib.dumps({"value": "ok"}))

        with patch.object(provider._client, "post", side_effect=capture):
            await provider.generate_structured(
                system="s",
                instruction="i",
                context_blocks=_blocks(),
                document_title="d.pdf",
                output_model=Out,
            )

        assert captured["json"]["response_format"] == {"type": "json_object"}

    async def test_structured_prompt_describes_the_target_schema(self) -> None:
        """JSON mode only guarantees valid JSON, not our shape — the schema must be in
        the prompt or the model has no way to know what fields are expected."""

        class Out(BaseModel):
            total_amount: str

        provider = NVIDIAProvider(api_key="k")
        captured: dict[str, Any] = {}

        async def capture(url: str, *, json: dict[str, Any]) -> httpx.Response:
            captured["json"] = json
            return _http_response(content=json_lib.dumps({"total_amount": "$1"}))

        with patch.object(provider._client, "post", side_effect=capture):
            await provider.generate_structured(
                system="s",
                instruction="i",
                context_blocks=_blocks(),
                document_title="d.pdf",
                output_model=Out,
            )

        prompt = captured["json"]["messages"][1]["content"]
        assert "total_amount" in prompt


class TestResponseHandling:
    async def test_text_is_returned_and_stripped(self) -> None:
        provider = NVIDIAProvider(api_key="k")

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
        provider = NVIDIAProvider(api_key="k")

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
        provider = NVIDIAProvider(api_key="k")
        response = _http_response(usage={"prompt_tokens": 321, "completion_tokens": 45})

        with patch.object(provider._client, "post", AsyncMock(return_value=response)):
            result = await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert result.input_tokens == 321
        assert result.output_tokens == 45

    async def test_missing_usage_defaults_to_zero(self) -> None:
        provider = NVIDIAProvider(api_key="k")
        response = httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            request=httpx.Request("POST", "https://example.test/chat/completions"),
        )

        with patch.object(provider._client, "post", AsyncMock(return_value=response)):
            result = await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert result.input_tokens == 0
        assert result.output_tokens == 0

    async def test_empty_choices_does_not_crash(self) -> None:
        """A malformed reply must degrade to empty text, not an unhandled IndexError."""
        provider = NVIDIAProvider(api_key="k")
        response = httpx.Response(
            status_code=200,
            json={"choices": []},
            request=httpx.Request("POST", "https://example.test/chat/completions"),
        )

        with patch.object(provider._client, "post", AsyncMock(return_value=response)):
            result = await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert result.text == ""

    async def test_a_content_filter_finish_reason_becomes_a_clear_error(self) -> None:
        """The OpenAI-compatible equivalent of a refusal: HTTP 200 with nothing usable
        in it. Without this check the user would see a blank result and no explanation."""
        provider = NVIDIAProvider(api_key="k")
        response = _http_response(content="", finish_reason="content_filter")

        with (
            patch.object(provider._client, "post", AsyncMock(return_value=response)),
            pytest.raises(AIUnavailableError) as exc_info,
        ):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert "declined" in exc_info.value.message

    async def test_structured_output_is_parsed_and_validated(self) -> None:
        class Out(BaseModel):
            name: str
            value: int

        provider = NVIDIAProvider(api_key="k")
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

        provider = NVIDIAProvider(api_key="k")
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

        provider = NVIDIAProvider(api_key="k")
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
        provider = NVIDIAProvider(api_key="k")
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
        provider = NVIDIAProvider(api_key="k")
        response = _http_response(status_code=403, content="")

        with (
            patch.object(provider._client, "post", AsyncMock(return_value=response)),
            pytest.raises(AIConfigurationError),
        ):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

    async def test_429_becomes_unavailable_after_exhausting_retries(self) -> None:
        provider = NVIDIAProvider(api_key="k", max_retries=1)
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
        provider = NVIDIAProvider(api_key="k", max_retries=2)
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
        provider = NVIDIAProvider(api_key="k", max_retries=1)
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
        provider = NVIDIAProvider(api_key="k")
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
        provider = NVIDIAProvider(api_key="k", max_retries=0)

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
        provider = NVIDIAProvider(api_key="k", max_retries=0)

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
        provider = NVIDIAProvider(api_key="k", max_retries=0)

        with (
            patch.object(provider._client, "post", AsyncMock(side_effect=RuntimeError("boom"))),
            pytest.raises(AIUnavailableError),
        ):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

    async def test_an_api_key_never_appears_in_an_error_message(self) -> None:
        """The single most important safety test in this file."""
        provider = NVIDIAProvider(api_key="nvapi-SUPERSECRET123")
        response = _http_response(status_code=401, content="invalid key nvapi-SUPERSECRET123")

        with (
            patch.object(provider._client, "post", AsyncMock(return_value=response)),
            pytest.raises(AIConfigurationError) as exc_info,
        ):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert "SUPERSECRET" not in exc_info.value.message
        assert "nvapi-" not in exc_info.value.message

    async def test_upstream_response_body_is_not_passed_through(self) -> None:
        provider = NVIDIAProvider(api_key="k")
        response = _http_response(
            status_code=400,
            content="internal routing failure on node gpu-cluster-7 model deepseek-internal-x",
        )

        with (
            patch.object(provider._client, "post", AsyncMock(return_value=response)),
            pytest.raises(AIUnavailableError) as exc_info,
        ):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert "gpu-cluster-7" not in exc_info.value.message
        assert "deepseek-internal-x" not in exc_info.value.message

    async def test_no_bare_exception_types_are_ever_raised(self) -> None:
        """Every failure path must produce one of our own exception types, never let an
        httpx or json exception escape to the caller."""
        provider = NVIDIAProvider(api_key="k", max_retries=0)

        with (
            patch.object(
                provider._client, "post", AsyncMock(side_effect=httpx.ConnectTimeout("x"))
            ),
            pytest.raises((AIConfigurationError, AIUnavailableError, InvalidModelResponseError)),
        ):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )
