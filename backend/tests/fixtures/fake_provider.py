"""A stand-in language model provider for tests.

**No automated test ever calls the real Anthropic API** (CLAUDE.md §7): it costs money,
takes seconds, and returns something different every run. This fake implements the same
interface and returns whatever a test tells it to.

It also records what it was asked, so tests can assert on the context that *would* have
been sent — which is how the "did the right content reach the model" questions get
answered without a network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from app.ai.providers.base import LLMProvider
from app.models.domain.insight import ContextBlock, ModelResult, RawCitation


@dataclass
class RecordedCall:
    """What the provider was asked to do."""

    system: str
    instruction: str
    context_blocks: list[ContextBlock]
    document_title: str
    enable_citations: bool = True
    output_model: type[BaseModel] | None = None


@dataclass
class FakeLLMProvider(LLMProvider):
    """A provider that returns canned results and remembers its calls.

    Args:
        text: What :meth:`generate` should return.
        raw_citations: Citation spans to report alongside it.
        parsed: What :meth:`generate_structured` should return.
        error: If set, raised instead of returning — for testing failure paths.
        configured: Whether the provider reports itself as usable.
    """

    text: str = "A generated result."
    raw_citations: tuple[RawCitation, ...] = ()
    parsed: Any = None
    error: Exception | None = None
    configured: bool = True
    model: str = "fake-model"

    calls: list[RecordedCall] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def is_configured(self) -> bool:
        return self.configured

    async def generate(
        self,
        *,
        system: str,
        instruction: str,
        context_blocks: list[ContextBlock],
        document_title: str,
        enable_citations: bool = True,
    ) -> ModelResult:
        self.calls.append(
            RecordedCall(
                system=system,
                instruction=instruction,
                context_blocks=list(context_blocks),
                document_title=document_title,
                enable_citations=enable_citations,
            )
        )
        if self.error is not None:
            raise self.error

        return ModelResult(
            text=self.text,
            raw_citations=self.raw_citations,
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
        )

    async def generate_structured(
        self,
        *,
        system: str,
        instruction: str,
        context_blocks: list[ContextBlock],
        document_title: str,
        output_model: type[BaseModel],
    ) -> ModelResult:
        self.calls.append(
            RecordedCall(
                system=system,
                instruction=instruction,
                context_blocks=list(context_blocks),
                document_title=document_title,
                enable_citations=False,
                output_model=output_model,
            )
        )
        if self.error is not None:
            raise self.error

        return ModelResult(
            parsed=self.parsed,
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
        )

    @property
    def last_call(self) -> RecordedCall:
        """The most recent call, for assertions."""
        if not self.calls:
            raise AssertionError("The provider was never called")
        return self.calls[-1]

    @property
    def call_count(self) -> int:
        return len(self.calls)
