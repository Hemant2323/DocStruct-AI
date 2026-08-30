"""Tests for the two AI calls conversion makes, and for the boundary they sit on.

**No test here calls a real API** (CLAUDE.md §7) — every one goes through the existing
``FakeLLMProvider``, which needed no change to represent these outputs since its
``parsed`` field is already ``Any``.

The tests that matter most are in ``TestProposalsAreNotTrusted``. They wire a model's
output into ``review_corrections`` and check that the deterministic engine, not the
model, decides what happens — including the case where the model is confidently wrong.
The engine's own rules are covered by ``test_corrections.py`` and are not repeated here;
what is tested is the *integration*: that proposals arrive in a shape the validator
accepts, and that nothing bypasses it.
"""

from __future__ import annotations

import pytest

from app.ai.assembly import assemble_document
from app.ai.conversion_planner import (
    generate_correction_proposals,
    generate_document_plan,
)
from app.ai.conversion_schema import (
    PlannedDocument,
    ProposedCorrectionList,
    ProposedCorrectionModel,
    vocabulary_arguments,
)
from app.ai.corrections import review_corrections
from app.core.exceptions import AIConfigurationError, InvalidModelResponseError
from app.models.domain.conversion import Confidence, CorrectionKind, ValueOrigin
from app.models.domain.insight import ExtractedField
from tests.fixtures.fake_provider import FakeLLMProvider

SCHEMA = {
    "account_holder": "string",
    "account_type": "string",
    "kyc_status": "string",
    "balance": "number",
}

INSTRUCTION = (
    "Create a clean customer master report. Standardize account types, put pending KYC "
    "issues in a separate section, and do not invent missing information."
)


def _records() -> list[list[ExtractedField]]:
    return [
        [
            ExtractedField(name="account_holder", value="Rajesh Kumr Sharma"),
            ExtractedField(name="account_type", value="sav"),
            ExtractedField(name="kyc_status", value="Pending"),
            ExtractedField(name="balance", value="145320"),
        ],
        [
            ExtractedField(name="account_holder", value="Priya Sharma"),
            ExtractedField(name="account_type", value="SAVINGS"),
            ExtractedField(name="kyc_status", value="Completed"),
            ExtractedField(name="balance", value="45320"),
        ],
    ]


def _planned_document() -> PlannedDocument:
    return PlannedDocument.model_validate(
        {
            "title": "Customer Master Report",
            "sections": [
                {
                    "heading": "Customers",
                    "kind": "table",
                    "columns": ["account_holder", "account_type", "balance"],
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


def _proposals(*items: ProposedCorrectionModel) -> ProposedCorrectionList:
    return ProposedCorrectionList(corrections=list(items))


def _proposal(
    *,
    record_index: int = 0,
    field: str = "account_type",
    original: str = "sav",
    corrected: str = "Savings",
    kind: str = "vocabulary",
    confidence: str = "high",
    reason: str = "standardized account type",
) -> ProposedCorrectionModel:
    return ProposedCorrectionModel(
        record_index=record_index,
        field=field,
        original=original,
        corrected=corrected,
        kind=kind,
        confidence=confidence,
        reason=reason,
    )


class TestGenerateDocumentPlan:
    async def test_a_plan_is_returned_as_typed_sections(self) -> None:
        provider = FakeLLMProvider(parsed=_planned_document())

        result = await generate_document_plan(
            provider, instruction=INSTRUCTION, schema=SCHEMA, records=_records()
        )

        assert result.plan.title == "Customer Master Report"
        assert [s.heading for s in result.plan.sections] == ["Customers", "Pending KYC"]
        assert result.plan.labels == {"account_holder": "Account Holder"}

    async def test_declared_vocabularies_come_back_for_the_validator(self) -> None:
        provider = FakeLLMProvider(parsed=_planned_document())

        result = await generate_document_plan(
            provider, instruction=INSTRUCTION, schema=SCHEMA, records=_records()
        )

        assert result.vocabularies == {"account_type": ("Savings", "Current", "Salary")}

    async def test_the_planner_is_asked_for_a_plan_not_a_document(self) -> None:
        """The output model is the schema-bound plan, so the provider cannot be asked
        to write prose here even if the prompt were wrong."""
        provider = FakeLLMProvider(parsed=_planned_document())

        await generate_document_plan(
            provider, instruction=INSTRUCTION, schema=SCHEMA, records=_records()
        )

        assert provider.last_call.output_model is PlannedDocument
        assert provider.last_call.enable_citations is False

    async def test_the_available_fields_reach_the_model(self) -> None:
        provider = FakeLLMProvider(parsed=_planned_document())

        await generate_document_plan(
            provider, instruction=INSTRUCTION, schema=SCHEMA, records=_records()
        )

        sent = provider.last_call.instruction
        for name in SCHEMA:
            assert name in sent
        assert INSTRUCTION in sent

    async def test_only_a_sample_of_records_is_sent(self) -> None:
        """Planning needs the shape of the data, not all of it — otherwise the planning
        call grows with the document."""
        provider = FakeLLMProvider(parsed=_planned_document())
        many = _records() * 20

        await generate_document_plan(provider, instruction=INSTRUCTION, schema=SCHEMA, records=many)

        sent = provider.last_call.instruction
        assert sent.count("- record ") <= 3
        assert f"{len(many)} record(s) in total" in sent

    async def test_a_field_the_planner_invented_is_stripped(self) -> None:
        """End to end at this boundary: a model asking for a column nobody extracted
        does not get one."""
        planned = PlannedDocument.model_validate(
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
        provider = FakeLLMProvider(parsed=planned)

        result = await generate_document_plan(
            provider, instruction=INSTRUCTION, schema=SCHEMA, records=_records()
        )

        assert result.plan.sections[0].columns == ("account_holder",)
        assert any("credit_score" in warning for warning in result.warnings)

    async def test_token_usage_is_reported(self) -> None:
        provider = FakeLLMProvider(parsed=_planned_document())

        result = await generate_document_plan(
            provider, instruction=INSTRUCTION, schema=SCHEMA, records=_records()
        )

        assert result.input_tokens == 100
        assert result.output_tokens == 50

    async def test_a_missing_reply_is_reported_not_guessed_at(self) -> None:
        provider = FakeLLMProvider(parsed=None)

        with pytest.raises(InvalidModelResponseError):
            await generate_document_plan(
                provider, instruction=INSTRUCTION, schema=SCHEMA, records=_records()
            )

    async def test_a_reply_of_the_wrong_shape_is_rejected(self) -> None:
        provider = FakeLLMProvider(parsed=ProposedCorrectionList())

        with pytest.raises(InvalidModelResponseError):
            await generate_document_plan(
                provider, instruction=INSTRUCTION, schema=SCHEMA, records=_records()
            )

    async def test_an_unconfigured_provider_is_reported_clearly(self) -> None:
        provider = FakeLLMProvider(parsed=_planned_document(), configured=False)

        with pytest.raises(AIConfigurationError):
            await generate_document_plan(
                provider, instruction=INSTRUCTION, schema=SCHEMA, records=_records()
            )


class TestGenerateCorrectionProposals:
    async def test_proposals_are_returned_as_typed_values(self) -> None:
        provider = FakeLLMProvider(parsed=_proposals(_proposal()))

        result = await generate_correction_proposals(
            provider, instruction=INSTRUCTION, records=_records()
        )

        [proposal] = result.proposals
        assert proposal.kind is CorrectionKind.VOCABULARY
        assert proposal.confidence is Confidence.HIGH
        assert proposal.record_index == 0
        assert proposal.field == "account_type"
        assert proposal.original == "sav"
        assert proposal.corrected == "Savings"
        assert proposal.reason == "standardized account type"

    async def test_the_output_model_carries_only_corrections(self) -> None:
        """The reply cannot contain the document, which is what keeps it small on a
        large source."""
        provider = FakeLLMProvider(parsed=_proposals(_proposal()))

        await generate_correction_proposals(provider, instruction=INSTRUCTION, records=_records())

        assert provider.last_call.output_model is ProposedCorrectionList
        assert set(ProposedCorrectionList.model_fields) == {"corrections"}

    async def test_every_record_is_shown_with_its_index(self) -> None:
        """Unlike planning, correcting needs the whole set: one of three spellings of
        the same word is invisible in a sample."""
        provider = FakeLLMProvider(parsed=_proposals())

        await generate_correction_proposals(provider, instruction=INSTRUCTION, records=_records())

        sent = provider.last_call.instruction
        assert "record 0:" in sent
        assert "record 1:" in sent

    async def test_declared_vocabularies_reach_the_model(self) -> None:
        provider = FakeLLMProvider(parsed=_proposals())

        await generate_correction_proposals(
            provider,
            instruction=INSTRUCTION,
            records=_records(),
            vocabularies={"account_type": ["Savings", "Current"]},
        )

        sent = provider.last_call.instruction
        assert "Savings" in sent
        assert "Current" in sent

    async def test_without_vocabularies_the_model_is_told_not_to_propose_them(self) -> None:
        provider = FakeLLMProvider(parsed=_proposals())

        await generate_correction_proposals(provider, instruction=INSTRUCTION, records=_records())

        assert "do not propose any 'vocabulary' corrections" in provider.last_call.instruction

    async def test_an_empty_proposal_list_is_a_valid_answer(self) -> None:
        provider = FakeLLMProvider(parsed=_proposals())

        result = await generate_correction_proposals(
            provider, instruction=INSTRUCTION, records=_records()
        )

        assert result.proposals == ()
        assert result.warnings == ()

    async def test_multiple_proposals_are_all_returned(self) -> None:
        provider = FakeLLMProvider(
            parsed=_proposals(
                _proposal(record_index=0),
                _proposal(record_index=1, original="SAVINGS"),
            )
        )

        result = await generate_correction_proposals(
            provider, instruction=INSTRUCTION, records=_records()
        )

        assert [p.record_index for p in result.proposals] == [0, 1]

    async def test_a_missing_reply_is_reported(self) -> None:
        provider = FakeLLMProvider(parsed=None)

        with pytest.raises(InvalidModelResponseError):
            await generate_correction_proposals(
                provider, instruction=INSTRUCTION, records=_records()
            )

    async def test_an_unconfigured_provider_is_reported_clearly(self) -> None:
        provider = FakeLLMProvider(parsed=_proposals(), configured=False)

        with pytest.raises(AIConfigurationError):
            await generate_correction_proposals(
                provider, instruction=INSTRUCTION, records=_records()
            )


class TestProposalsAreNotTrusted:
    """The boundary that matters: what a model proposes is input to the validator, and
    the validator — not the model — decides what happens."""

    async def test_generating_proposals_changes_nothing_by_itself(self) -> None:
        records = _records()
        provider = FakeLLMProvider(parsed=_proposals(_proposal()))

        await generate_correction_proposals(provider, instruction=INSTRUCTION, records=records)

        # The generator has no access to a document and no way to apply anything.
        assert records[0][1].value == "sav"

    async def test_proposals_feed_straight_into_the_validator(self) -> None:
        records = _records()
        provider = FakeLLMProvider(parsed=_proposals(_proposal()))

        result = await generate_correction_proposals(
            provider, instruction=INSTRUCTION, records=records
        )
        review = review_corrections(
            result.proposals,
            records,
            vocabularies=vocabulary_arguments({"account_type": ("Savings", "Current")}),
        )

        assert len(review.applied) == 1
        assert review.applied[0].corrected == "Savings"

    async def test_a_confident_hallucination_is_rejected_by_python(self) -> None:
        """The model claims HIGH confidence renaming one person to another. Python
        refuses regardless — this is the single most important test of the whole
        AI boundary."""
        records = _records()
        provider = FakeLLMProvider(
            parsed=_proposals(
                _proposal(
                    record_index=1,
                    field="account_holder",
                    original="Priya Sharma",
                    corrected="Neha Kapoor",
                    kind="spelling",
                    confidence="high",
                    reason="looks like a typo",
                )
            )
        )

        result = await generate_correction_proposals(
            provider, instruction=INSTRUCTION, records=records
        )
        review = review_corrections(result.proposals, records)

        assert result.proposals[0].confidence is Confidence.HIGH  # the model said so
        assert review.applied == ()  # Python disagreed
        assert review.suggestions == ()
        assert len(review.rejected) == 1

    async def test_a_medium_confidence_proposal_stays_a_suggestion(self) -> None:
        records = _records()
        provider = FakeLLMProvider(
            parsed=_proposals(
                _proposal(
                    record_index=0,
                    field="account_holder",
                    original="Rajesh Kumr Sharma",
                    corrected="Rajesh Kumar Sharma",
                    kind="spelling",
                    confidence="medium",
                    reason="possible spelling correction",
                )
            )
        )

        result = await generate_correction_proposals(
            provider, instruction=INSTRUCTION, records=records
        )
        review = review_corrections(result.proposals, records)

        assert review.applied == ()
        assert len(review.suggestions) == 1

    async def test_a_vocabulary_proposal_needs_the_plan_to_have_declared_one(self) -> None:
        """The two calls compose: without the plan's vocabulary there is nothing to
        check a vocabulary fix against, so it cannot be applied."""
        records = _records()
        provider = FakeLLMProvider(parsed=_proposals(_proposal()))

        result = await generate_correction_proposals(
            provider, instruction=INSTRUCTION, records=records
        )

        without = review_corrections(result.proposals, records)
        with_vocab = review_corrections(
            result.proposals,
            records,
            vocabularies=vocabulary_arguments({"account_type": ("Savings",)}),
        )

        assert without.applied == ()
        assert len(with_vocab.applied) == 1

    async def test_the_whole_pipeline_runs_with_only_the_model_planning(self) -> None:
        """Plan -> propose -> validate -> assemble, with the model contributing only the
        first two and touching no value directly."""
        records = _records()
        planner = FakeLLMProvider(parsed=_planned_document())
        plan_result = await generate_document_plan(
            planner, instruction=INSTRUCTION, schema=SCHEMA, records=records
        )

        proposer = FakeLLMProvider(
            parsed=_proposals(
                _proposal(record_index=0, original="sav"),
                _proposal(record_index=1, original="SAVINGS"),
                _proposal(
                    record_index=1,
                    field="account_holder",
                    original="Priya Sharma",
                    corrected="Neha Kapoor",
                    kind="spelling",
                    confidence="high",
                    reason="hallucinated rename",
                ),
            )
        )
        proposal_result = await generate_correction_proposals(
            proposer,
            instruction=INSTRUCTION,
            records=records,
            vocabularies=vocabulary_arguments(plan_result.vocabularies),
        )

        review = review_corrections(
            proposal_result.proposals,
            records,
            vocabularies=vocabulary_arguments(plan_result.vocabularies),
        )
        document = assemble_document(records, schema=SCHEMA, plan=plan_result.plan, review=review)

        # Both account types normalized; the rename refused.
        assert len(review.applied) == 2
        assert len(review.rejected) == 1
        assert document.title == "Customer Master Report"

        holders = {
            record.field("account_holder").value.text  # type: ignore[union-attr]
            for section in document.sections
            for record in section.records
        }
        assert "Neha Kapoor" not in holders
        assert "Priya Sharma" in holders

        customers = document.sections[0]
        types = {
            record.field("account_type").value  # type: ignore[union-attr]
            for record in customers.records
        }
        assert all(value.text == "Savings" for value in types)
        assert all(value.origin is ValueOrigin.CORRECTED for value in types)
