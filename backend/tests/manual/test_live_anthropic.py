"""Manual verification against the real Anthropic API.

**Excluded from the normal test suite.** Every test here is marked ``live`` and
``pytest.ini`` deselects that marker, because these tests cost money, take seconds, need
network access, and return something different every run — all of which make them useless
as automated regression tests (CLAUDE.md §7).

Run them by hand when you want to confirm the integration genuinely works::

    # from backend/, with ANTHROPIC_API_KEY set in the environment or .env
    pytest -m live -v

They skip themselves if no API key is configured, so running them without one is
harmless rather than a wall of failures.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.ai.citations import resolve_citations
from app.ai.prompts import extract, summarize
from app.ai.providers.claude import ClaudeProvider
from app.ai.schemas import KeyFactsExtraction
from app.core.config import Settings, get_settings
from app.models.domain.content import BlockType, ContentBlock, Provenance
from app.models.domain.insight import ContextBlock

pytestmark = pytest.mark.live


CONTRACT_TEXT = [
    "SERVICE AGREEMENT between Acme Ltd and Globex Inc, dated 14 March 2025.",
    "Invoices are due within 30 days of receipt. Late payment incurs interest at 2% per month.",
    "The total contract value is $45,200.00 for the twelve month term.",
    "Either party may terminate this agreement with 60 days written notice.",
]


@pytest.fixture
def live_settings() -> Settings:
    """Real settings, or skip if no key is configured."""
    get_settings.cache_clear()
    settings = get_settings()
    if not settings.ai_configured:
        pytest.skip("ANTHROPIC_API_KEY is not set — skipping live API verification")
    return settings


@pytest.fixture
def context_blocks() -> list[ContextBlock]:
    """A small document, one context block per paragraph."""
    return [
        ContextBlock(
            text=text, chunk_id=uuid4(), block_ids=(uuid4(),), location_label=f"page {i + 1}"
        )
        for i, text in enumerate(CONTRACT_TEXT)
    ]


@pytest.fixture
def blocks_by_id(context_blocks: list[ContextBlock]) -> dict:
    """Provenance for each block, as the parser would have recorded it."""
    return {
        block.block_ids[0]: ContentBlock(
            block_type=BlockType.PARAGRAPH,
            text=block.text,
            order=index,
            provenance=Provenance.for_page(index + 1),
        )
        for index, block in enumerate(context_blocks)
    }


async def test_summary_against_the_real_api(
    live_settings: Settings, context_blocks: list[ContextBlock], blocks_by_id: dict
) -> None:
    """Confirms the request shape is accepted and citations come back mappable."""
    provider = ClaudeProvider(live_settings)

    result = await provider.generate(
        system=summarize.SYSTEM,
        instruction=summarize.INSTRUCTION,
        context_blocks=context_blocks,
        document_title="agreement.txt",
        enable_citations=True,
    )

    assert result.text.strip(), "The model returned nothing"
    print(f"\n--- SUMMARY ({result.input_tokens} in / {result.output_tokens} out) ---")
    print(result.text)

    citations = resolve_citations(result.raw_citations, context_blocks, blocks_by_id)
    print(f"\n--- CITATIONS ({len(citations)}) ---")
    for citation in citations:
        print(f"  [{citation.location_label}] {citation.quoted_text[:70]}")

    # The point of the exercise: real citations that resolve to real locations.
    assert citations, "No citations were returned — the citation feature is not working"
    assert any(citation.is_verified for citation in citations)

    # Grounding check: the model must not have invented an amount.
    assert "45,200" in result.text or "45200" in result.text


async def test_extraction_against_the_real_api(
    live_settings: Settings, context_blocks: list[ContextBlock]
) -> None:
    """Confirms structured output validates and quotes are genuinely verbatim."""
    from app.ai.citations import verify_quote

    provider = ClaudeProvider(live_settings)

    result = await provider.generate_structured(
        system=extract.SYSTEM,
        instruction=extract.INSTRUCTION,
        context_blocks=context_blocks,
        document_title="agreement.txt",
        output_model=KeyFactsExtraction,
    )

    parsed = result.parsed
    assert isinstance(parsed, KeyFactsExtraction)
    assert parsed.facts, "No facts were extracted"

    print(f"\n--- EXTRACTION ({len(parsed.facts)} facts) ---")
    verified = 0
    for fact in parsed.facts:
        citation = verify_quote(fact.source_quote, context_blocks, {})
        found = citation is not None
        verified += int(found)
        marker = "verified" if found else "UNVERIFIED"
        print(f"  {fact.name}: {fact.value}  [{marker}]")

    # If the model routinely fabricates quotes, the extraction prompt needs work.
    assert verified > 0, "No extracted quote could be found in the source text"


async def test_the_api_key_is_never_echoed(live_settings: Settings) -> None:
    """A deliberate authentication failure must not surface the key."""
    from app.core.exceptions import AIConfigurationError

    broken = live_settings.model_copy(update={"anthropic_api_key": "sk-ant-invalid-key-xyz"})
    provider = ClaudeProvider(broken)

    with pytest.raises(AIConfigurationError) as exc_info:
        await provider.generate(
            system="Say hello.",
            instruction="Say hello.",
            context_blocks=[ContextBlock("hello", uuid4(), (uuid4(),), "page 1")],
            document_title="test.txt",
        )

    assert "sk-ant" not in exc_info.value.message
    assert "invalid-key-xyz" not in exc_info.value.message
