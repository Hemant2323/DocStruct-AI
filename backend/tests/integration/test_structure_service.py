"""Tests for `AnalysisService.structure()`'s two input modes (Milestone 4 Step 4).

Real ingestion is not exercised here — a document is created directly with real chunks
and content blocks (exactly enough for `_prepare_context`/`verify_quote` to have genuine
stored text to check a model's claimed quote against), then `AnalysisService` is built
directly against a real database session and a fake provider, and called without going
through HTTP. This is a service-level test, not an API test: `app/api/schemas/structure.py`
and `app/api/routes/analysis.py` are untouched by Step 4, and are not exercised here.

**No test here calls a real API** (CLAUDE.md §7) — everything goes through
`FakeLLMProvider`, extended locally with `_SequencedFakeProvider` for the two calls
(schema inference, then multi-record extraction) the instruction-driven path makes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.dynamic_schema import InferredStructureField, InferredStructureSchema
from app.core.config import Settings
from app.core.exceptions import (
    AIUnavailableError,
    InsufficientContentError,
    InvalidStructureSchemaError,
    ValidationError,
)
from app.models.domain.content import BlockType, ContentBlock, Provenance, ProvenanceKind
from app.models.domain.document import Document, ProcessingStatus, SupportedFileType
from app.models.domain.insight import ContextBlock, ModelResult
from app.repositories.ai_result_repo import AIResultRepository
from app.repositories.block_repo import ContentBlockRepository
from app.repositories.chunk_repo import ChunkRepository
from app.repositories.document_repo import DocumentRepository
from app.services.analysis_service import AnalysisService
from tests.fixtures.fake_provider import FakeLLMProvider

OWNER = "owner-a"

DOCUMENT_TEXT = (
    "PAYMENT REPORT\n\n"
    "Account holder Arjun Malhotra banks with State Bank of India. Amount paid: 2100.\n\n"
    "Account holder Person 2 banks with HDFC Bank. Amount paid: 4500."
)

SCHEMA = {"account_holder": "string", "bank_name": "string", "amount": "number"}


@dataclass
class _SequencedFakeProvider(FakeLLMProvider):
    """A `FakeLLMProvider` that returns a different `.parsed` value on each successive
    `generate_structured()` call, in the order given.

    Needed for the instruction-driven path, which makes two structured calls (schema
    inference, then multi-record extraction) and must return a different result for
    each. Kept local to this test file rather than added to the shared fixture, since no
    other test currently needs sequencing.
    """

    parsed_sequence: list[Any] = field(default_factory=list)

    async def generate_structured(
        self,
        *,
        system: str,
        instruction: str,
        context_blocks: list[ContextBlock],
        document_title: str,
        output_model: type[BaseModel],
    ) -> ModelResult:
        if self.parsed_sequence:
            self.parsed = self.parsed_sequence.pop(0)
        return await super().generate_structured(
            system=system,
            instruction=instruction,
            context_blocks=context_blocks,
            document_title=document_title,
            output_model=output_model,
        )


def _field_result(value: Any, source_quote: str) -> SimpleNamespace:
    """One extracted field's value, in the shape `build_dynamic_record_list_model()`
    produces per record — mirrors `test_structure_api.py`'s `_field()` helper."""
    return SimpleNamespace(value=value, source_quote=source_quote)


def _record(account_holder: Any, bank_name: Any, amount: Any) -> SimpleNamespace:
    return SimpleNamespace(account_holder=account_holder, bank_name=bank_name, amount=amount)


#: The extraction result two genuine, verifiable payment records — the example from the
#: M4 spec.
TWO_RECORD_RESULT = SimpleNamespace(
    records=[
        _record(
            _field_result("Arjun Malhotra", "Account holder Arjun Malhotra"),
            _field_result("State Bank of India", "State Bank of India"),
            _field_result(2100, "Amount paid: 2100"),
        ),
        _record(
            _field_result("Person 2", "Account holder Person 2"),
            _field_result("HDFC Bank", "banks with HDFC Bank"),
            _field_result(4500, "Amount paid: 4500"),
        ),
    ]
)

#: A schema-inference result matching SCHEMA, field for field.
INFERRED_SCHEMA_RESULT = InferredStructureSchema(
    fields=[
        InferredStructureField(name="account_holder", type="string"),
        InferredStructureField(name="bank_name", type="string"),
        InferredStructureField(name="amount", type="number"),
    ]
)

INSTRUCTION = (
    "Create a payment report with one row per transaction. Include account holder, bank and amount."
)


@pytest.fixture
async def document(db_session: AsyncSession) -> Document:
    """A ready document with real chunks and content blocks, so a model's claimed quote
    has genuine stored text to be verified against — what `_prepare_context` and
    `verify_quote` need, without going through the full parse/ingest pipeline."""
    documents = DocumentRepository(db_session)
    created = await documents.create(
        owner_id=OWNER,
        original_filename="payments.txt",
        file_type=SupportedFileType.TXT,
        size_bytes=len(DOCUMENT_TEXT),
        status=ProcessingStatus.READY,
    )

    blocks = ContentBlockRepository(db_session)
    block_ids = await blocks.replace_for_document(
        created.id,
        [
            ContentBlock(
                block_type=BlockType.PARAGRAPH,
                text=DOCUMENT_TEXT,
                order=0,
                provenance=Provenance(kind=ProvenanceKind.PAGE, label="page 1", page=1),
            )
        ],
    )

    chunks = ChunkRepository(db_session)
    await chunks.replace_for_document(created.id, [(DOCUMENT_TEXT, block_ids, 50, "page 1")])

    await db_session.flush()
    return created


@pytest.fixture
def make_service(
    db_session: AsyncSession, settings: Settings
) -> Callable[[FakeLLMProvider], AnalysisService]:
    """A factory so each test can bind whichever provider (plain fake or sequenced) it
    needs, while sharing the same real repositories/session."""

    def _make(provider: FakeLLMProvider) -> AnalysisService:
        return AnalysisService(
            provider=provider,
            settings=settings,
            documents=DocumentRepository(db_session),
            chunks=ChunkRepository(db_session),
            blocks=ContentBlockRepository(db_session),
            results=AIResultRepository(db_session),
        )

    return _make


class TestInputModeValidation:
    async def test_neither_schema_nor_instruction_is_rejected(
        self,
        make_service: Callable[[FakeLLMProvider], AnalysisService],
        document: Document,
        fake_provider: FakeLLMProvider,
    ) -> None:
        service = make_service(fake_provider)
        with pytest.raises(ValidationError):
            await service.structure(document.id, owner_id=OWNER)

    async def test_both_schema_and_instruction_is_rejected(
        self,
        make_service: Callable[[FakeLLMProvider], AnalysisService],
        document: Document,
        fake_provider: FakeLLMProvider,
    ) -> None:
        service = make_service(fake_provider)
        with pytest.raises(ValidationError):
            await service.structure(
                document.id, owner_id=OWNER, schema=SCHEMA, instruction=INSTRUCTION
            )


class TestSchemaMode:
    """MODE A: a caller-supplied schema — the existing/manual path (test #1, #8)."""

    async def test_manual_schema_returns_multiple_records(
        self,
        make_service: Callable[[FakeLLMProvider], AnalysisService],
        document: Document,
        fake_provider: FakeLLMProvider,
    ) -> None:
        fake_provider.parsed = TWO_RECORD_RESULT
        service = make_service(fake_provider)

        insight = (await service.structure(document.id, owner_id=OWNER, schema=SCHEMA)).insight

        assert len(insight.records) == 2
        assert fake_provider.last_call.output_model.__name__ == "DynamicStructureRecordList"

    async def test_fields_is_not_populated_with_flattened_records(
        self,
        make_service: Callable[[FakeLLMProvider], AnalysisService],
        document: Document,
        fake_provider: FakeLLMProvider,
    ) -> None:
        """Explicit guard for the exact regression Step 4 must avoid: records must never
        end up flattened into the single-row `fields` tuple."""
        fake_provider.parsed = TWO_RECORD_RESULT
        service = make_service(fake_provider)

        insight = (await service.structure(document.id, owner_id=OWNER, schema=SCHEMA)).insight

        assert insight.fields == ()

    async def test_invalid_schema_is_rejected_before_any_provider_call(
        self,
        make_service: Callable[[FakeLLMProvider], AnalysisService],
        document: Document,
        fake_provider: FakeLLMProvider,
    ) -> None:
        """Existing behaviour (test #8): schema validation is still fail-fast, before
        the document or the model are ever touched."""
        service = make_service(fake_provider)

        with pytest.raises(InvalidStructureSchemaError):
            await service.structure(document.id, owner_id=OWNER, schema={"Bad Name": "string"})

        assert fake_provider.call_count == 0

    async def test_empty_record_list_is_insufficient_content(
        self,
        make_service: Callable[[FakeLLMProvider], AnalysisService],
        document: Document,
        fake_provider: FakeLLMProvider,
    ) -> None:
        fake_provider.parsed = SimpleNamespace(records=[])
        service = make_service(fake_provider)

        with pytest.raises(InsufficientContentError):
            await service.structure(document.id, owner_id=OWNER, schema=SCHEMA)

    async def test_a_record_response_with_no_records_attribute_is_also_insufficient(
        self,
        make_service: Callable[[FakeLLMProvider], AnalysisService],
        document: Document,
        fake_provider: FakeLLMProvider,
    ) -> None:
        fake_provider.parsed = SimpleNamespace()  # no `records` at all
        service = make_service(fake_provider)

        with pytest.raises(InsufficientContentError):
            await service.structure(document.id, owner_id=OWNER, schema=SCHEMA)


class TestRecordCitations:
    async def test_verified_field_citation_is_resolved(
        self,
        make_service: Callable[[FakeLLMProvider], AnalysisService],
        document: Document,
        fake_provider: FakeLLMProvider,
    ) -> None:
        fake_provider.parsed = TWO_RECORD_RESULT
        service = make_service(fake_provider)

        insight = (await service.structure(document.id, owner_id=OWNER, schema=SCHEMA)).insight

        first_record = {field.name: field for field in insight.records[0]}
        citation = first_record["bank_name"].citation
        assert citation is not None
        assert citation.is_verified
        assert citation.quoted_text == "State Bank of India"

    async def test_unverified_quote_leaves_the_field_without_a_citation(
        self,
        make_service: Callable[[FakeLLMProvider], AnalysisService],
        document: Document,
        fake_provider: FakeLLMProvider,
    ) -> None:
        """A fabricated quote must produce a field with no verified location, not a
        nearby one — the same rule single-record structure already enforces."""
        fake_provider.parsed = SimpleNamespace(
            records=[
                _record(
                    _field_result("Arjun Malhotra", "Account holder Arjun Malhotra"),
                    _field_result("State Bank of India", "State Bank of India"),
                    _field_result(2100, "A fee applies that this document never states"),
                )
            ]
        )
        service = make_service(fake_provider)

        insight = (await service.structure(document.id, owner_id=OWNER, schema=SCHEMA)).insight

        first_record = {field.name: field for field in insight.records[0]}
        assert first_record["amount"].citation is None
        assert any("could not be traced" in warning for warning in insight.warnings)


class TestCaching:
    async def test_cached_multi_record_result_preserves_records(
        self,
        make_service: Callable[[FakeLLMProvider], AnalysisService],
        document: Document,
        fake_provider: FakeLLMProvider,
    ) -> None:
        """The exact regression identified after Step 3: `_mark_cached()` must copy
        `records`, not silently drop them on the cached-result path."""
        fake_provider.parsed = TWO_RECORD_RESULT
        service = make_service(fake_provider)

        first = (await service.structure(document.id, owner_id=OWNER, schema=SCHEMA)).insight
        calls_after_first = fake_provider.call_count
        second = (await service.structure(document.id, owner_id=OWNER, schema=SCHEMA)).insight

        assert first.from_cache is False
        assert second.from_cache is True
        assert second.records == first.records
        assert len(second.records) == 2
        assert fake_provider.call_count == calls_after_first


class TestInstructionMode:
    """MODE B: a natural-language instruction, resolved via schema inference before
    extraction (test #2, #3, #9)."""

    async def test_instruction_infers_a_schema_then_returns_multiple_records(
        self,
        make_service: Callable[[FakeLLMProvider], AnalysisService],
        document: Document,
    ) -> None:
        provider = _SequencedFakeProvider(
            parsed_sequence=[INFERRED_SCHEMA_RESULT, TWO_RECORD_RESULT]
        )
        service = make_service(provider)

        result = await service.structure(document.id, owner_id=OWNER, instruction=INSTRUCTION)

        assert len(result.insight.records) == 2
        # The resolved schema is handed back, not left inside the service — this is what
        # the API reports as `schema_used`, and it cannot be reconstructed from records.
        assert result.schema == SCHEMA
        assert provider.call_count == 2
        # First call is inference (no schema-shaped output model yet); second is the
        # actual multi-record extraction, once the schema has been resolved.
        assert provider.calls[0].output_model is InferredStructureSchema
        assert provider.calls[1].output_model.__name__ == "DynamicStructureRecordList"

    async def test_inferred_schema_is_validated_before_extraction(
        self,
        make_service: Callable[[FakeLLMProvider], AnalysisService],
        document: Document,
    ) -> None:
        """An inferred schema gets no special trust: an invalid field name from the
        inference call must be rejected the same way a caller-supplied one would be,
        and must never reach the extraction call."""
        invalid_inference = InferredStructureSchema(
            fields=[InferredStructureField(name="Bad Name", type="string")]
        )
        provider = _SequencedFakeProvider(parsed_sequence=[invalid_inference])
        service = make_service(provider)

        with pytest.raises(InvalidStructureSchemaError):
            await service.structure(document.id, owner_id=OWNER, instruction=INSTRUCTION)

        assert provider.call_count == 1

    async def test_instruction_mode_records_have_no_fields_flattening(
        self,
        make_service: Callable[[FakeLLMProvider], AnalysisService],
        document: Document,
    ) -> None:
        provider = _SequencedFakeProvider(
            parsed_sequence=[INFERRED_SCHEMA_RESULT, TWO_RECORD_RESULT]
        )
        service = make_service(provider)

        insight = (
            await service.structure(document.id, owner_id=OWNER, instruction=INSTRUCTION)
        ).insight

        assert insight.fields == ()

    async def test_schema_inference_failure_propagates(
        self,
        make_service: Callable[[FakeLLMProvider], AnalysisService],
        document: Document,
        fake_provider: FakeLLMProvider,
    ) -> None:
        """A provider failure during the inference call must propagate as-is, not
        become a silently empty or invalid structure result."""
        fake_provider.error = AIUnavailableError("The AI service is temporarily unavailable.")
        service = make_service(fake_provider)

        with pytest.raises(AIUnavailableError):
            await service.structure(document.id, owner_id=OWNER, instruction=INSTRUCTION)

    async def test_instruction_resolves_to_the_same_cache_fingerprint_as_the_equivalent_schema(
        self,
        make_service: Callable[[FakeLLMProvider], AnalysisService],
        document: Document,
    ) -> None:
        """The resolved schema is the sole basis for the cache, per Part 4: a request
        made by instruction and one made by the equivalent explicit schema must hit the
        same cached result."""
        provider = _SequencedFakeProvider(
            parsed_sequence=[INFERRED_SCHEMA_RESULT, TWO_RECORD_RESULT]
        )
        service = make_service(provider)

        by_instruction = (
            await service.structure(document.id, owner_id=OWNER, instruction=INSTRUCTION)
        ).insight
        calls_after_instruction = provider.call_count
        by_schema = (await service.structure(document.id, owner_id=OWNER, schema=SCHEMA)).insight

        assert by_instruction.from_cache is False
        assert by_schema.from_cache is True
        assert by_schema.records == by_instruction.records
        # The schema-mode request never had to load the document or call the model —
        # cache-checking a known schema needs neither.
        assert provider.call_count == calls_after_instruction


class TestExistingBehaviorUnaffected:
    async def test_a_different_schema_is_not_served_from_cache(
        self,
        make_service: Callable[[FakeLLMProvider], AnalysisService],
        document: Document,
        fake_provider: FakeLLMProvider,
    ) -> None:
        fake_provider.parsed = TWO_RECORD_RESULT
        service = make_service(fake_provider)

        first = (await service.structure(document.id, owner_id=OWNER, schema=SCHEMA)).insight
        calls_after_first = fake_provider.call_count
        second = (
            await service.structure(document.id, owner_id=OWNER, schema={"amount": "number"})
        ).insight

        assert first.from_cache is False
        assert second.from_cache is False
        assert fake_provider.call_count == calls_after_first + 1
