"""Tests for provider selection: which ``LLMProvider`` ``AI_PROVIDER`` builds.

Calls ``app.api.deps._build_provider`` directly rather than going through FastAPI's
dependency injection or a live app — the branching logic is the only thing being tested
here, and it does not need a request, a database, or a network to exercise.

No test in this file makes a network call: constructing a provider never does, for either
implementation.
"""

from __future__ import annotations

from app.ai.providers.claude import ClaudeProvider
from app.ai.providers.gemini import GeminiProvider
from app.ai.providers.nvidia import NVIDIAProvider
from app.api.deps import _build_provider
from app.core.config import Settings


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "anthropic_api_key": "claude-key",
        "nvidia_api_key": "nvidia-key",
        "gemini_api_key": "gemini-key",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


class TestProviderSelection:
    def test_defaults_to_claude(self) -> None:
        """AI_PROVIDER defaults to "claude", so existing behaviour is unchanged for
        anyone who has not set it."""
        provider = _build_provider(_settings())

        assert isinstance(provider, ClaudeProvider)

    def test_explicit_claude_builds_claude_provider(self) -> None:
        provider = _build_provider(_settings(ai_provider="claude"))

        assert isinstance(provider, ClaudeProvider)

    def test_nvidia_builds_nvidia_provider(self) -> None:
        provider = _build_provider(_settings(ai_provider="nvidia"))

        assert isinstance(provider, NVIDIAProvider)

    def test_gemini_builds_gemini_provider(self) -> None:
        provider = _build_provider(_settings(ai_provider="gemini"))

        assert isinstance(provider, GeminiProvider)

    def test_nvidia_provider_receives_its_own_settings(self) -> None:
        """The two providers must not share configuration by accident."""
        settings = _settings(
            ai_provider="nvidia",
            nvidia_api_key="nvidia-key",
            nvidia_model="deepseek-ai/deepseek-v4-flash-0731",
        )

        provider = _build_provider(settings)

        assert provider.model_name == "deepseek-ai/deepseek-v4-flash-0731"

    def test_claude_provider_receives_its_own_settings(self) -> None:
        settings = _settings(ai_provider="claude", anthropic_model="claude-opus-5")

        provider = _build_provider(settings)

        assert provider.model_name == "claude-opus-5"

    def test_gemini_provider_receives_its_own_settings(self) -> None:
        """The three providers must not share configuration by accident."""
        settings = _settings(
            ai_provider="gemini", gemini_api_key="gemini-key", gemini_model="gemini-2.5-pro"
        )

        provider = _build_provider(settings)

        assert provider.model_name == "gemini-2.5-pro"

    def test_nvidia_provider_is_unconfigured_without_its_own_key(self) -> None:
        """Selecting NVIDIA does not borrow the Anthropic key, or vice versa."""
        settings = _settings(
            ai_provider="nvidia", nvidia_api_key=None, anthropic_api_key="claude-key"
        )

        provider = _build_provider(settings)

        assert provider.is_configured is False

    def test_claude_provider_is_unconfigured_without_its_own_key(self) -> None:
        settings = _settings(
            ai_provider="claude", anthropic_api_key=None, nvidia_api_key="nvidia-key"
        )

        provider = _build_provider(settings)

        assert provider.is_configured is False

    def test_gemini_provider_is_unconfigured_without_its_own_key(self) -> None:
        """Selecting Gemini does not borrow the Anthropic or NVIDIA key."""
        settings = _settings(
            ai_provider="gemini",
            gemini_api_key=None,
            anthropic_api_key="claude-key",
            nvidia_api_key="nvidia-key",
        )

        provider = _build_provider(settings)

        assert provider.is_configured is False

    def test_switching_provider_changes_the_returned_type(self) -> None:
        """Confirms the branch is genuinely live, not just returning one hard-coded type."""
        claude = _build_provider(_settings(ai_provider="claude"))
        nvidia = _build_provider(_settings(ai_provider="nvidia"))
        gemini = _build_provider(_settings(ai_provider="gemini"))

        assert len({type(claude), type(nvidia), type(gemini)}) == 3
