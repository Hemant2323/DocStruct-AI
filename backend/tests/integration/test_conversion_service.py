"""Tests for `ConversionService` — the orchestration boundary.

**No test here calls a real API** (CLAUDE.md §7). Everything below runs against real
repositories, a real database session and a real `AnalysisService`, with only the model
replaced. That combination is deliberate: the point of this phase is the wiring, and
faking the pieces it wires would test nothing.

The service makes four model calls in a fixed order — schema inference and extraction
(inside `AnalysisService.structure`), then planning, then correction proposal. The fake
below returns a different reply for each, and `provider.calls` then records exactly what
was asked for, which is how call order is proven without mocks.

What these tests do *not* do is re-check the correction engine, the assembler or the
renderer. Those have their own suites from earlier phases. What is checked here is that
the service routes data between them correctly, and never around them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.conversion_schema import (
    PlannedDocument,
    ProposedCorrectionList,
    ProposedCorrectionModel,
)
from app.ai.dynamic_schema import InferredStructureField, InferredStructureSchema
from app.core.config import Settings
from app.core.exceptions import (
    AIUnavailableError,
    InsufficientContentError,
    InvalidModelResponseError,
    NotFoundError,
    ValidationError,
)
from app.models.domain.content import BlockType, ContentBlock, Provenance, ProvenanceKind
from app.models.domain.conversion import ValueOrigin
from app.models.domain.document import Document, ProcessingStatus, SupportedFileType
from app.models.domain.insight import ContextBlock, ModelResult
from app.repositories.ai_result_repo import AIResultRepository
from app.repositories.block_repo import ContentBlockRepository
from app.repositories.chunk_repo import ChunkRepository
from app.repositories.document_repo import DocumentRepository
from app.services.analysis_service import AnalysisService
from app.services.conversion_service import ConversionService
from tests.fixtures.fake_provider import FakeLLMProvider

OWNER = "owner-a"

DOCUMENT_TEXT = (
    "CUSTOMER MASTER\n\n"
    "Account holder Rajesh Kumr Sharma banks with State Bank of India. "
    "Account type: sav. KYC status: Pending. Balance: 145320.\n\n"
    "Account holder Priya Sharma banks with HDFC Bank. "
    "Account type: SAVINGS. KYC status: Completed. Balance: 45320.\n\n"
    "Account holder Rahul Verma banks with ICICI Bank. "
    "Account type: Current. KYC status: Pending. Balance: 3200."
)

INSTRUCTION = (
    "Create a clean customer master report. Group customers by account type, "
    "standardize account statuses, put pending KYC issues in a separate section, "
    "preserve important details, and do not invent missing information."
)

SCHEMA_FIELDS = ("account_holder", "account_type", "kyc_status", "balance")


@dataclass
class _SequencedFakeProvider(FakeLLMProvider):
    """Returns a different `.parsed` on each successive structured call, in order."""

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


def _value(value: Any, quote: str) -> SimpleNamespace:
    return SimpleNamespace(value=value, source_quote=quote)


def _inferred_schema() -> InferredStructureSchema:
    return InferredStructureSchema(
        fields=[
            InferredStructureField(name="account_holder", type="string"),
            InferredStructureField(name="account_type", type="string"),
            InferredStructureField(name="kyc_status", type="string"),
            InferredStructureField(name="balance", type="number"),
        ]
    )


def _extraction() -> SimpleNamespace:
    """Three customers, deliberately inconsistent: three spellings of savings, one typo."""
    return SimpleNamespace(
        records=[
            SimpleNamespace(
                account_holder=_value("Rajesh Kumr Sharma", "Account holder Rajesh Kumr Sharma"),
                account_type=_value("sav", "Account type: sav"),
                kyc_status=_value("Pending", "KYC status: Pending"),
                balance=_value(145320, "Balance: 145320"),
            ),
            SimpleNamespace(
                account_holder=_value("Priya Sharma", "Account holder Priya Sharma"),
                account_type=_value("SAVINGS", "Account type: SAVINGS"),
                kyc_status=_value("Completed", "KYC status: Completed"),
                balance=_value(45320, "Balance: 45320"),
            ),
            SimpleNamespace(
                account_holder=_value("Rahul Verma", "Account holder Rahul Verma"),
                account_type=_value("Current", "Account type: Current"),
                kyc_status=_value("Pending", "KYC status: Pending"),
                balance=_value(3200, "Balance: 3200"),
            ),
        ]
    )


def _plan() -> PlannedDocument:
    return PlannedDocument.model_validate(
        {
            "title": "Customer Master Report",
            "sections": [
                {
                    "heading": "Customers",
                    "kind": "table",
                    "columns": ["account_holder", "account_type", "balance"],
                    "group_by": "account_type",
                    "order_by": [{"field": "balance", "descending": True, "numeric": True}],
                },
                {
                    "heading": "Pending KYC",
                    "kind": "table",
                    "columns": ["account_holder", "kyc_status"],
                    "filters": [{"field": "kyc_status", "mode": "equals", "values": ["Pending"]}],
                },
            ],
            "labels": [{"field": "account_holder", "label": "Account Holder"}],
            "vocabularies": [{"field": "account_type", "values": ["Savings", "Current", "Salary"]}],
        }
    )


def _proposal(
    *,
    record_index: int,
    field_name: str,
    original: str,
    corrected: str,
    kind: str,
    confidence: str,
    reason: str = "test",
) -> ProposedCorrectionModel:
    return ProposedCorrectionModel(
        record_index=record_index,
        field=field_name,
        original=original,
        corrected=corrected,
        kind=kind,
        confidence=confidence,
        reason=reason,
    )


def _proposals() -> ProposedCorrectionList:
    """One safe fix per messy value, one uncertain typo, and one outright hallucination."""
    return ProposedCorrectionList(
        corrections=[
            _proposal(
                record_index=0,
                field_name="account_type",
                original="sav",
                corrected="Savings",
                kind="vocabulary",
                confidence="high",
                reason="standardized account type",
            ),
            _proposal(
                record_index=1,
                field_name="account_type",
                original="SAVINGS",
                corrected="Savings",
                kind="vocabulary",
                confidence="high",
                reason="standardized account type",
            ),
            _proposal(
                record_index=0,
                field_name="account_holder",
                original="Rajesh Kumr Sharma",
                corrected="Rajesh Kumar Sharma",
                kind="spelling",
                confidence="medium",
                reason="possible spelling correction",
            ),
            _proposal(
                record_index=2,
                field_name="account_holder",
                original="Rahul Verma",
                corrected="Neha Kapoor",
                kind="spelling",
                confidence="high",
                reason="hallucinated rename",
            ),
        ]
    )


def _full_sequence() -> list[Any]:
    """The four structured replies one conversion consumes, in order."""
    return [_inferred_schema(), _extraction(), _plan(), _proposals()]


@pytest.fixture
async def document(db_session: AsyncSession) -> Document:
    """A ready document with real blocks and chunks, so quote verification has genuine
    stored text to check against."""
    documents = DocumentRepository(db_session)
    created = await documents.create(
        owner_id=OWNER,
        original_filename="customers.txt",
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
    await ChunkRepository(db_session).replace_for_document(
        created.id, [(DOCUMENT_TEXT, block_ids, 60, "page 1")]
    )
    await db_session.flush()
    return created


@pytest.fixture
def build_service(db_session: AsyncSession, settings: Settings):  # type: ignore[no-untyped-def]
    def _build(provider: FakeLLMProvider) -> ConversionService:
        analysis = AnalysisService(
            provider=provider,
            settings=settings,
            documents=DocumentRepository(db_session),
            chunks=ChunkRepository(db_session),
            blocks=ContentBlockRepository(db_session),
            results=AIResultRepository(db_session),
        )
        return ConversionService(analysis=analysis, provider=provider, settings=settings)

    return _build


class TestSuccessfulConversion:
    async def test_a_conversion_produces_a_document_and_its_markdown(
        self, build_service, document: Document
    ) -> None:
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        assert result.document.title == "Customer Master Report"
        assert result.markdown.startswith("# Customer Master Report")
        assert result.source_record_count == 3

    async def test_the_four_ai_calls_happen_in_the_designed_order(
        self, build_service, document: Document
    ) -> None:
        """Schema inference, extraction, planning, correction proposal — proven by what
        each call asked the provider to produce."""
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        assert provider.call_count == 4
        assert provider.calls[0].output_model is InferredStructureSchema
        assert provider.calls[1].output_model.__name__ == "DynamicStructureRecordList"
        assert provider.calls[2].output_model is PlannedDocument
        assert provider.calls[3].output_model is ProposedCorrectionList

    async def test_the_model_is_never_asked_to_write_the_document(
        self, build_service, document: Document
    ) -> None:
        """The core architectural rule: every model call is schema-bound to a plan, a
        correction list, or extraction — never to prose or a finished document."""
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        assert all(call.output_model is not None for call in provider.calls)
        # `generate` (free prose) is never used; every call went through the structured path.
        assert all(call.enable_citations is False for call in provider.calls)

    async def test_the_instruction_reaches_both_planning_and_correction(
        self, build_service, document: Document
    ) -> None:
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        assert INSTRUCTION in provider.calls[2].instruction
        assert INSTRUCTION in provider.calls[3].instruction

    async def test_token_usage_sums_across_the_stages(
        self, build_service, document: Document
    ) -> None:
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        # The fake reports 100/50 per call; extraction contributes two calls' worth.
        assert result.input_tokens > 100
        assert result.output_tokens > 50
        assert result.model == provider.model_name


class TestCorrectionsGoThroughTheValidator:
    async def test_a_high_confidence_valid_correction_is_applied(
        self, build_service, document: Document
    ) -> None:
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        applied = {(c.record_index, c.field, c.corrected) for c in result.review.applied}
        assert (0, "account_type", "Savings") in applied
        assert (1, "account_type", "Savings") in applied

    async def test_a_medium_confidence_correction_remains_a_suggestion(
        self, build_service, document: Document
    ) -> None:
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        suggested = {(c.field, c.corrected) for c in result.review.suggestions}
        assert ("account_holder", "Rajesh Kumar Sharma") in suggested
        assert all(c.applied is False for c in result.review.suggestions)

    async def test_a_confident_hallucination_is_rejected(
        self, build_service, document: Document
    ) -> None:
        """The model claimed HIGH confidence renaming one customer to another. Python
        refuses, and the refusal is recorded rather than silently dropped."""
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        assert result.rejected_correction_count == 1
        assert result.review.rejected[0].proposal.corrected == "Neha Kapoor"
        assert result.review.rejected[0].reason

    async def test_assembly_receives_reviewed_corrections_not_raw_proposals(
        self, build_service, document: Document
    ) -> None:
        """Four proposals went in; the document shows two changes, keeps one original
        with a suggestion beside it, and never shows the rejected rename."""
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        holders: set[str] = set()
        corrected_types: set[str] = set()
        for section in result.document.sections:
            for record in section.records:
                holder = record.field("account_holder")
                if holder is not None:
                    holders.add(holder.value.text)
                account_type = record.field("account_type")
                if account_type is not None and account_type.value.origin is ValueOrigin.CORRECTED:
                    corrected_types.add(account_type.value.text)

        assert "Neha Kapoor" not in holders  # rejected
        assert "Rahul Verma" in holders  # original preserved
        assert "Rajesh Kumr Sharma" in holders  # suggestion did not alter it
        assert corrected_types == {"Savings"}

    async def test_the_declared_vocabulary_reaches_validation(
        self, build_service, document: Document
    ) -> None:
        """Without the plan's vocabulary these two corrections could not be proved, so
        the fact that they applied is evidence the vocabulary was passed through."""
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        assert result.vocabularies == {"account_type": ("Savings", "Current", "Salary")}
        assert len([c for c in result.review.applied if c.field == "account_type"]) == 2

    async def test_a_vocabulary_the_plan_never_declared_cannot_apply(
        self, build_service, document: Document
    ) -> None:
        """An AI-proposed vocabulary value does not become truth on the model's say-so:
        with no vocabulary in the plan, the same corrections are refused."""
        plan_without_vocabulary = PlannedDocument.model_validate(
            {
                "title": "Customer Master Report",
                "sections": [{"heading": "Customers", "kind": "table"}],
            }
        )
        provider = _SequencedFakeProvider(
            parsed_sequence=[
                _inferred_schema(),
                _extraction(),
                plan_without_vocabulary,
                _proposals(),
            ]
        )
        service = build_service(provider)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        assert result.review.applied == ()
        assert any("no vocabulary is declared" in r.reason for r in result.review.rejected)

    async def test_proposals_are_generated_against_the_original_records(
        self, build_service, document: Document
    ) -> None:
        """The correction call must see "sav", not an already-normalized value —
        otherwise its anchors could never match."""
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        correction_call = provider.calls[3].instruction
        assert "account_type=sav" in correction_call
        assert "account_type=SAVINGS" in correction_call


class TestPlanReachesAssembly:
    async def test_grouping_from_the_plan_is_applied(
        self, build_service, document: Document
    ) -> None:
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        headings = [section.heading for section in result.document.sections]
        # Both savings variants normalized first, so they group together.
        assert "Customers — Savings" in headings
        assert "Customers — Current" in headings

    async def test_filtering_from_the_plan_is_applied(
        self, build_service, document: Document
    ) -> None:
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        pending = next(s for s in result.document.sections if s.heading == "Pending KYC")
        assert len(pending.records) == 2
        assert {r.source_record_index for r in pending.records} == {0, 2}

    async def test_ordering_from_the_plan_is_applied(
        self, build_service, document: Document
    ) -> None:
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        savings = next(s for s in result.document.sections if s.heading.endswith("Savings"))
        balances = [r.field("balance").value.text for r in savings.records]  # type: ignore[union-attr]
        assert balances == ["145320", "45320"]  # descending, numeric

    async def test_labels_from_the_plan_are_applied(
        self, build_service, document: Document
    ) -> None:
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        record = result.document.sections[0].records[0]
        assert record.field("account_holder").label == "Account Holder"  # type: ignore[union-attr]

    async def test_a_field_the_plan_invented_never_reaches_the_document(
        self, build_service, document: Document
    ) -> None:
        plan_with_invention = PlannedDocument.model_validate(
            {
                "title": "Report",
                "sections": [
                    {
                        "heading": "Customers",
                        "kind": "table",
                        "columns": ["account_holder", "credit_score"],
                    }
                ],
            }
        )
        provider = _SequencedFakeProvider(
            parsed_sequence=[_inferred_schema(), _extraction(), plan_with_invention, _proposals()]
        )
        service = build_service(provider)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        assert result.document.sections[0].columns == ("account_holder",)
        # It must not appear as a column or a value. It *should* still appear in the
        # warnings, which explain that it was removed — so the body is checked, not the
        # whole document.
        body = result.markdown.split("## Warnings")[0]
        assert "credit_score" not in body
        assert "Credit Score" not in body
        assert any("credit_score" in warning for warning in result.warnings)


class TestRenderingAndProvenance:
    async def test_the_markdown_is_structured_not_prose(
        self, build_service, document: Document
    ) -> None:
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        assert "## Customers — Savings" in result.markdown
        assert "## Pending KYC" in result.markdown
        assert "| Account Holder |" in result.markdown
        assert "## Changes & Corrections" in result.markdown

    async def test_citations_survive_into_the_document(
        self, build_service, document: Document
    ) -> None:
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        verified = [
            item.value
            for section in result.document.sections
            for record in section.records
            for item in record.fields
            if item.value.is_verified
        ]
        assert verified, "quotes matched the stored text, so some values must be verified"
        assert "page 1" in result.markdown

    async def test_no_internal_identifiers_leak_into_the_markdown(
        self, build_service, document: Document
    ) -> None:
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        for leak in ("chunk_id", "block_ids", "correction_id"):
            assert leak not in result.markdown

    async def test_no_invented_information_appears(self, build_service, document: Document) -> None:
        """Every displayed value is either source text or a recorded correction of it."""
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        applied_ids = {c.id for c in result.review.applied}
        for section in result.document.sections:
            for record in section.records:
                for item in record.fields:
                    value = item.value
                    if value.origin is ValueOrigin.CORRECTED:
                        assert value.correction_id in applied_ids
                        assert value.original_text is not None
                    else:
                        assert value.origin in (ValueOrigin.SOURCE, ValueOrigin.ABSENT)


class TestNoDuplicateMerging:
    async def test_every_source_record_survives_as_its_own_record(
        self, build_service, document: Document
    ) -> None:
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        grouped = [s for s in result.document.sections if s.heading.startswith("Customers")]
        indices = {r.source_record_index for s in grouped for r in s.records}
        assert indices == {0, 1, 2}

    async def test_no_record_reports_being_merged(self, build_service, document: Document) -> None:
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        assert all(
            record.merged_from == ()
            for section in result.document.sections
            for record in section.records
        )


class TestFailures:
    @pytest.mark.parametrize("bad", ["", "   ", "\n\t "])
    async def test_an_empty_instruction_is_rejected_before_any_ai_call(
        self, build_service, document: Document, bad: str
    ) -> None:
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        with pytest.raises(ValidationError):
            await service.convert(document.id, bad, owner_id=OWNER)

        assert provider.call_count == 0

    async def test_an_overly_long_instruction_is_rejected(
        self, build_service, document: Document
    ) -> None:
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        with pytest.raises(ValidationError):
            await service.convert(document.id, "a" * 2001, owner_id=OWNER)

    async def test_a_missing_document_is_reported(self, build_service) -> None:
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        with pytest.raises(NotFoundError):
            await service.convert(uuid4(), INSTRUCTION, owner_id=OWNER)

    async def test_another_owners_document_is_not_found(
        self, build_service, document: Document
    ) -> None:
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        with pytest.raises(NotFoundError):
            await service.convert(document.id, INSTRUCTION, owner_id="someone-else")

    async def test_a_plan_with_no_usable_sections_is_reported_not_faked(
        self, build_service, document: Document
    ) -> None:
        """Inventing a fallback layout would be deciding for the user what their
        document is."""
        empty_plan = PlannedDocument.model_validate({"title": "Report", "sections": []})
        provider = _SequencedFakeProvider(
            parsed_sequence=[_inferred_schema(), _extraction(), empty_plan, _proposals()]
        )
        service = build_service(provider)

        with pytest.raises(InvalidModelResponseError):
            await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

    async def test_a_plan_whose_sections_are_all_invalid_is_reported(
        self, build_service, document: Document
    ) -> None:
        unusable = PlannedDocument.model_validate(
            {"title": "Report", "sections": [{"heading": "Customers", "kind": "spreadsheet"}]}
        )
        provider = _SequencedFakeProvider(
            parsed_sequence=[_inferred_schema(), _extraction(), unusable, _proposals()]
        )
        service = build_service(provider)

        with pytest.raises(InvalidModelResponseError):
            await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

    async def test_a_provider_failure_propagates(self, build_service, document: Document) -> None:
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        provider.error = AIUnavailableError("The AI service is temporarily unavailable.")
        service = build_service(provider)

        with pytest.raises(AIUnavailableError):
            await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

    async def test_an_extraction_that_finds_nothing_is_reported(
        self, build_service, document: Document
    ) -> None:
        """No records means nothing to organize — reported rather than rendered as an
        empty report that looks like a real one."""
        provider = _SequencedFakeProvider(
            parsed_sequence=[_inferred_schema(), SimpleNamespace(records=[])]
        )
        service = build_service(provider)

        # Raised by the extraction stage, before conversion has anything to organize.
        with pytest.raises(InsufficientContentError):
            await service.convert(document.id, INSTRUCTION, owner_id=OWNER)


class TestWarnings:
    async def test_rejected_corrections_are_summarised_in_the_warnings(
        self, build_service, document: Document
    ) -> None:
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        assert any("rejected as unsafe" in warning for warning in result.warnings)
        assert any("rejected as unsafe" in warning for warning in result.document.warnings)

    async def test_plan_mapping_warnings_are_surfaced(
        self, build_service, document: Document
    ) -> None:
        plan_with_invention = PlannedDocument.model_validate(
            {
                "title": "Report",
                "sections": [
                    {
                        "heading": "Customers",
                        "kind": "table",
                        "columns": ["account_holder", "credit_score"],
                    }
                ],
            }
        )
        provider = _SequencedFakeProvider(
            parsed_sequence=[_inferred_schema(), _extraction(), plan_with_invention, _proposals()]
        )
        service = build_service(provider)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        assert any("credit_score" in warning for warning in result.warnings)


class TestEndToEnd:
    async def test_the_whole_pipeline_on_the_banking_fixture(
        self, build_service, document: Document
    ) -> None:
        """One realistic run: messy customer data in, organized cited report out, with
        every change accounted for and the unsafe one refused."""
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        # Organized as planned.
        headings = [s.heading for s in result.document.sections]
        assert headings == [
            "Customers — Current",
            "Customers — Savings",
            "Pending KYC",
        ]

        # Two safe corrections applied, one uncertain kept as a suggestion, one refused.
        assert result.applied_correction_count == 2
        assert result.suggestion_count == 1
        assert result.rejected_correction_count == 1

        # The audit is legible in the output.
        assert "**sav** → **Savings**" in result.markdown
        assert "### Possible corrections (not applied)" in result.markdown
        assert "Rajesh Kumar Sharma" in result.markdown  # as a suggestion only

        # Nothing invented: the rejected rename appears nowhere.
        assert "Neha Kapoor" not in result.markdown

        # Provenance survived.
        assert "page 1" in result.markdown

        # Every record accounted for.
        assert result.source_record_count == 3
        assert result.document.source_record_count == 3

    async def test_the_same_inputs_produce_the_same_document(
        self, build_service, document: Document
    ) -> None:
        """Determinism holds across the whole service: the model's replies are the only
        variable, and with those fixed the output is byte-identical.

        ``refresh=True`` on both runs so each performs a full extraction. Without it the
        second run would hit the extraction cache and make one call fewer — see
        ``test_a_repeat_conversion_reuses_the_cached_extraction``.
        """
        first = await build_service(
            _SequencedFakeProvider(parsed_sequence=_full_sequence())
        ).convert(document.id, INSTRUCTION, owner_id=OWNER, refresh=True)
        second = await build_service(
            _SequencedFakeProvider(parsed_sequence=_full_sequence())
        ).convert(document.id, INSTRUCTION, owner_id=OWNER, refresh=True)

        assert first.markdown == second.markdown
        assert first.document == second.document

    async def test_a_repeat_conversion_reuses_the_cached_extraction(
        self, build_service, document: Document
    ) -> None:
        """Only the extraction is cached — by machinery that already existed. Planning
        and correction proposal always run fresh, so a second conversion costs three
        calls rather than four, and says so through ``records_from_cache``.
        """
        first = await build_service(
            _SequencedFakeProvider(parsed_sequence=_full_sequence())
        ).convert(document.id, INSTRUCTION, owner_id=OWNER)
        assert first.records_from_cache is False

        # The extraction reply is absent this time: the cache supplies those records.
        cached_provider = _SequencedFakeProvider(
            parsed_sequence=[_inferred_schema(), _plan(), _proposals()]
        )
        second = await build_service(cached_provider).convert(
            document.id, INSTRUCTION, owner_id=OWNER
        )

        assert second.records_from_cache is True
        assert cached_provider.call_count == 3
        assert second.markdown == first.markdown

    async def test_source_records_are_never_mutated(
        self, build_service, document: Document
    ) -> None:
        """A conversion reads the extraction; it must not rewrite it. Running twice
        would otherwise produce different results the second time."""
        provider = _SequencedFakeProvider(parsed_sequence=_full_sequence())
        service = build_service(provider)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        # The correction call saw the originals, and the applied corrections still name
        # them as their "original" — proving nothing rewrote the source in place.
        originals = {c.original for c in result.review.applied}
        assert originals == {"sav", "SAVINGS"}
