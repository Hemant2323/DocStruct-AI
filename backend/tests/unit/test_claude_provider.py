"""Tests for the Claude provider, with the SDK mocked.

Never touches the network. What is being tested here is *our* code: how a request is
assembled, how a reply is read, and — most of all — that every kind of failure becomes a
typed error carrying a message that is safe to show a user.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import anthropic
import pytest
from pydantic import BaseModel

from app.ai.providers.claude import ClaudeProvider, _extract_text_and_citations
from app.core.config import Settings
from app.core.exceptions import (
    AIConfigurationError,
    AIUnavailableError,
    InvalidModelResponseError,
)
from app.models.domain.insight import ContextBlock


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "anthropic_api_key": "test-key-not-real",
        "anthropic_model": "claude-opus-5",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


def _blocks(count: int = 2) -> list[ContextBlock]:
    return [
        ContextBlock(f"Chunk {i} text.", uuid4(), (uuid4(),), f"page {i + 1}") for i in range(count)
    ]


def _response(
    *, text: str = "Result.", citations: list[Any] | None = None, stop_reason: str = "end_turn"
) -> SimpleNamespace:
    block = SimpleNamespace(type="text", text=text, citations=citations or [])
    return SimpleNamespace(
        content=[block],
        usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        stop_reason=stop_reason,
        stop_details=None,
    )


def _stream_returning(response: Any) -> MagicMock:
    """Build a mock matching the SDK's async streaming context manager."""
    stream = AsyncMock()
    stream.get_final_message = AsyncMock(return_value=response)

    manager = MagicMock()
    manager.__aenter__ = AsyncMock(return_value=stream)
    manager.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=manager)


class TestConfiguration:
    def test_reports_unconfigured_without_a_key(self) -> None:
        provider = ClaudeProvider(_settings(anthropic_api_key=None))
        assert provider.is_configured is False

    def test_reports_configured_with_a_key(self) -> None:
        assert ClaudeProvider(_settings()).is_configured is True

    def test_blank_key_counts_as_unconfigured(self) -> None:
        """An empty string in .env is a common mistake and must not look configured."""
        assert ClaudeProvider(_settings(anthropic_api_key="   ")).is_configured is False

    def test_model_name_comes_from_settings(self) -> None:
        provider = ClaudeProvider(_settings(anthropic_model="claude-opus-5"))
        assert provider.model_name == "claude-opus-5"

    async def test_generating_without_a_key_is_a_clear_error(self) -> None:
        provider = ClaudeProvider(_settings(anthropic_api_key=None))

        with pytest.raises(AIConfigurationError) as exc_info:
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert exc_info.value.code == "ai_not_configured"


class TestRequestConstruction:
    async def test_content_is_sent_as_an_ordered_custom_content_document(self) -> None:
        """Order is the citation coordinate system, so it must be preserved exactly."""
        provider = ClaudeProvider(_settings())
        blocks = _blocks(3)
        captured: dict[str, Any] = {}

        def capture(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return _stream_returning(_response())()

        with patch.object(provider._client.beta.messages, "stream", side_effect=capture):
            await provider.generate(
                system="s", instruction="i", context_blocks=blocks, document_title="report.pdf"
            )

        document = captured["messages"][0]["content"][0]
        assert document["type"] == "document"
        assert document["source"]["type"] == "content"
        assert [item["text"] for item in document["source"]["content"]] == [
            block.text for block in blocks
        ]
        assert document["title"] == "report.pdf"

    async def test_citations_are_requested_for_prose(self) -> None:
        provider = ClaudeProvider(_settings())
        captured: dict[str, Any] = {}

        def capture(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return _stream_returning(_response())()

        with patch.object(provider._client.beta.messages, "stream", side_effect=capture):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert captured["messages"][0]["content"][0]["citations"] == {"enabled": True}

    async def test_adaptive_thinking_and_effort_are_set(self) -> None:
        """budget_tokens is removed on this model family and would be a 400."""
        provider = ClaudeProvider(_settings(ai_effort="high"))
        captured: dict[str, Any] = {}

        def capture(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return _stream_returning(_response())()

        with patch.object(provider._client.beta.messages, "stream", side_effect=capture):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert captured["thinking"] == {"type": "adaptive"}
        assert "budget_tokens" not in captured["thinking"]
        assert captured["output_config"]["effort"] == "high"

    async def test_prompt_caching_is_marked_when_enabled(self) -> None:
        provider = ClaudeProvider(_settings(ai_prompt_caching=True))
        captured: dict[str, Any] = {}

        def capture(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return _stream_returning(_response())()

        with patch.object(provider._client.beta.messages, "stream", side_effect=capture):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(6), document_title="d.pdf"
            )

        assert captured["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    async def test_caching_is_skipped_when_disabled(self) -> None:
        provider = ClaudeProvider(_settings(ai_prompt_caching=False))
        captured: dict[str, Any] = {}

        def capture(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return _stream_returning(_response())()

        with patch.object(provider._client.beta.messages, "stream", side_effect=capture):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(6), document_title="d.pdf"
            )

        assert "cache_control" not in captured["messages"][0]["content"][0]

    async def test_structured_requests_never_ask_for_citations(self) -> None:
        """The API rejects the combination with a 400, so this must not be sent."""

        class Out(BaseModel):
            value: str

        provider = ClaudeProvider(_settings())
        captured: dict[str, Any] = {}

        async def capture(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(
                parsed_output=Out(value="x"),
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
                stop_reason="end_turn",
            )

        with patch.object(provider._client.messages, "parse", side_effect=capture):
            await provider.generate_structured(
                system="s",
                instruction="i",
                context_blocks=_blocks(),
                document_title="d.pdf",
                output_model=Out,
            )

        assert "citations" not in captured["messages"][0]["content"][0]


class TestResponseHandling:
    async def test_text_and_citations_are_returned(self) -> None:
        provider = ClaudeProvider(_settings())
        citation = SimpleNamespace(
            type="content_block_location",
            cited_text="Cited passage.",
            document_index=0,
            start_block_index=1,
            end_block_index=2,
        )

        with patch.object(
            provider._client.beta.messages,
            "stream",
            _stream_returning(_response(text="The answer.", citations=[citation])),
        ):
            result = await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert result.text == "The answer."
        assert len(result.raw_citations) == 1
        assert result.raw_citations[0].start_block_index == 1
        assert result.raw_citations[0].quoted_text == "Cited passage."
        assert result.input_tokens == 100

    def test_multiple_text_blocks_are_joined_in_order(self) -> None:
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="First part. ", citations=[]),
                SimpleNamespace(type="text", text="Second part.", citations=[]),
            ]
        )

        text, _ = _extract_text_and_citations(response)

        assert text == "First part. Second part."

    def test_non_text_blocks_are_ignored(self) -> None:
        """Thinking blocks are not part of the answer."""
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="reasoning"),
                SimpleNamespace(type="text", text="The answer.", citations=[]),
            ]
        )

        text, _ = _extract_text_and_citations(response)

        assert text == "The answer."

    def test_unmappable_citation_types_are_ignored(self) -> None:
        """A location type we cannot map must not be recorded as if we could."""
        citation = SimpleNamespace(
            type="page_location", cited_text="x", start_page_number=1, end_page_number=2
        )
        response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="A", citations=[citation])]
        )

        _, citations = _extract_text_and_citations(response)

        assert citations == []

    async def test_a_refusal_becomes_a_clear_error(self) -> None:
        """A refusal is HTTP 200 with an empty-looking body; without this the user
        would see a blank result and no explanation."""
        provider = ClaudeProvider(_settings())

        with (
            patch.object(
                provider._client.beta.messages,
                "stream",
                _stream_returning(_response(text="", stop_reason="refusal")),
            ),
            pytest.raises(AIUnavailableError) as exc_info,
        ):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert "declined" in exc_info.value.message

    async def test_missing_structured_output_is_reported(self) -> None:
        class Out(BaseModel):
            value: str

        provider = ClaudeProvider(_settings())

        async def no_parsed(**_: Any) -> Any:
            return SimpleNamespace(
                parsed_output=None,
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                stop_reason="end_turn",
            )

        with (
            patch.object(provider._client.messages, "parse", side_effect=no_parsed),
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

    @pytest.mark.parametrize(
        ("sdk_error", "expected", "expected_code"),
        [
            (
                anthropic.AuthenticationError(
                    "invalid x-api-key sk-ant-SECRET",
                    response=MagicMock(status_code=401),
                    body=None,
                ),
                AIConfigurationError,
                "ai_not_configured",
            ),
            (
                anthropic.RateLimitError(
                    "rate limited", response=MagicMock(status_code=429), body=None
                ),
                AIUnavailableError,
                "ai_unavailable",
            ),
            (
                anthropic.APITimeoutError(request=MagicMock()),
                AIUnavailableError,
                "ai_unavailable",
            ),
            (
                anthropic.APIConnectionError(request=MagicMock()),
                AIUnavailableError,
                "ai_unavailable",
            ),
            (
                anthropic.BadRequestError(
                    "context too long", response=MagicMock(status_code=400), body=None
                ),
                AIUnavailableError,
                "ai_unavailable",
            ),
            (
                anthropic.InternalServerError(
                    "server error", response=MagicMock(status_code=500), body=None
                ),
                AIUnavailableError,
                "ai_unavailable",
            ),
            (RuntimeError("something unexpected"), AIUnavailableError, "ai_unavailable"),
        ],
    )
    async def test_sdk_errors_become_typed_errors(
        self, sdk_error: Exception, expected: type[Exception], expected_code: str
    ) -> None:
        provider = ClaudeProvider(_settings())

        with (
            patch.object(provider._client.beta.messages, "stream", side_effect=sdk_error),
            pytest.raises(expected) as exc_info,
        ):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert exc_info.value.code == expected_code  # type: ignore[attr-defined]

    async def test_an_api_key_never_appears_in_an_error_message(self) -> None:
        """The single most important safety test in this file."""
        provider = ClaudeProvider(_settings(anthropic_api_key="sk-ant-SUPERSECRET123"))
        leaky = anthropic.AuthenticationError(
            "invalid x-api-key: sk-ant-SUPERSECRET123",
            response=MagicMock(status_code=401),
            body=None,
        )

        with (
            patch.object(provider._client.beta.messages, "stream", side_effect=leaky),
            pytest.raises(AIConfigurationError) as exc_info,
        ):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert "SUPERSECRET" not in exc_info.value.message
        assert "sk-ant" not in exc_info.value.message

    async def test_upstream_detail_is_not_passed_through(self) -> None:
        provider = ClaudeProvider(_settings())
        leaky = anthropic.BadRequestError(
            "prompt is too long: 900000 tokens > 200000 maximum for model claude-internal-x",
            response=MagicMock(status_code=400),
            body=None,
        )

        with (
            patch.object(provider._client.beta.messages, "stream", side_effect=leaky),
            pytest.raises(AIUnavailableError) as exc_info,
        ):
            await provider.generate(
                system="s", instruction="i", context_blocks=_blocks(), document_title="d.pdf"
            )

        assert "claude-internal-x" not in exc_info.value.message
        assert "900000" not in exc_info.value.message
