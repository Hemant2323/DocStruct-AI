"""Tests for the AI-output schemas and their mapping onto the typed conversion core.

Pure logic, no network and no provider — this is the boundary where a model's loose
reply becomes the strict types ``assembly`` and ``corrections`` accept, so it can be
tested by handing it the reply directly.

The question most of these ask: *can a model widen its own permissions here?* It must not
be able to name a column that was never extracted, invent a section kind, or smuggle a
correction whose kind decides its own proof.
"""

from __future__ import annotations

import pytest

from app.ai.assembly import MatchMode
from app.ai.conversion_schema import (
    DEFAULT_TITLE,
    PlannedDocument,
    ProposedCorrectionList,
    ProposedCorrectionModel,
    to_document_plan,
    to_proposed_corrections,
    vocabulary_arguments,
)
from app.models.domain.conversion import Confidence, CorrectionKind, SectionKind

SCHEMA = {
    "account_holder": "string",
    "account_type": "string",
    "balance": "number",
    "kyc_status": "string",
}


def _plan(**kwargs: object) -> PlannedDocument:
    return PlannedDocument.model_validate({"title": "Customer Master Report", **kwargs})


class TestSectionKinds:
    def test_a_table_section_maps(self) -> None:
        planned = _plan(sections=[{"heading": "Customers", "kind": "table"}])

        mapping = to_document_plan(planned, schema=SCHEMA)

        assert mapping.plan.sections[0].kind is SectionKind.TABLE

    def test_a_detail_section_maps(self) -> None:
        planned = _plan(sections=[{"heading": "Customers", "kind": "detail"}])

        mapping = to_document_plan(planned, schema=SCHEMA)

        assert mapping.plan.sections[0].kind is SectionKind.DETAIL

    def test_a_note_section_maps_with_its_prose(self) -> None:
        planned = _plan(
            sections=[{"heading": "Notes", "kind": "note", "note": "Prepared for review."}]
        )

        mapping = to_document_plan(planned, schema=SCHEMA)

        assert mapping.plan.sections[0].kind is SectionKind.NOTE
        assert mapping.plan.sections[0].note == "Prepared for review."

    def test_kind_is_case_insensitive(self) -> None:
        planned = _plan(sections=[{"heading": "Customers", "kind": "TABLE"}])

        mapping = to_document_plan(planned, schema=SCHEMA)

        assert mapping.plan.sections[0].kind is SectionKind.TABLE

    def test_an_unknown_kind_drops_the_section_rather_than_guessing(self) -> None:
        """The three kinds render differently; picking one would be inventing layout."""
        planned = _plan(sections=[{"heading": "Customers", "kind": "grid"}])

        mapping = to_document_plan(planned, schema=SCHEMA)

        assert mapping.plan.sections == ()
        assert any("not a known section kind" in warning for warning in mapping.warnings)

    def test_multiple_sections_keep_their_order(self) -> None:
        planned = _plan(
            sections=[
                {"heading": "Savings", "kind": "table"},
                {"heading": "Pending KYC", "kind": "table"},
                {"heading": "Notes", "kind": "note", "note": "x"},
            ]
        )

        mapping = to_document_plan(planned, schema=SCHEMA)

        assert [s.heading for s in mapping.plan.sections] == [
            "Savings",
            "Pending KYC",
            "Notes",
        ]


class TestFieldReferencesAreClosed:
    """A plan may organise the fields that exist. It may not conjure new ones."""

    def test_only_known_columns_survive(self) -> None:
        planned = _plan(
            sections=[
                {
                    "heading": "Customers",
                    "kind": "table",
                    "columns": ["account_holder", "credit_score", "balance"],
                }
            ]
        )

        mapping = to_document_plan(planned, schema=SCHEMA)

        assert mapping.plan.sections[0].columns == ("account_holder", "balance")
        assert any("credit_score" in warning for warning in mapping.warnings)

    @pytest.mark.parametrize(
        "invented", ["credit_score", "annual_income", "branch_manager", "account_opening_reason"]
    )
    def test_invented_fields_never_reach_the_plan(self, invented: str) -> None:
        planned = _plan(sections=[{"heading": "Customers", "kind": "table", "columns": [invented]}])

        mapping = to_document_plan(planned, schema=SCHEMA)

        assert mapping.plan.sections[0].columns == ()
        assert any(invented in warning for warning in mapping.warnings)

    def test_grouping_by_an_unknown_field_is_removed(self) -> None:
        planned = _plan(
            sections=[{"heading": "Customers", "kind": "table", "group_by": "credit_score"}]
        )

        mapping = to_document_plan(planned, schema=SCHEMA)

        assert mapping.plan.sections[0].group_by is None
        assert any("grouping" in warning for warning in mapping.warnings)

    def test_a_filter_on_an_unknown_field_is_removed(self) -> None:
        planned = _plan(
            sections=[
                {
                    "heading": "Customers",
                    "kind": "table",
                    "filters": [{"field": "credit_score", "mode": "equals", "values": ["x"]}],
                }
            ]
        )

        mapping = to_document_plan(planned, schema=SCHEMA)

        assert mapping.plan.sections[0].filters == ()

    def test_ordering_by_an_unknown_field_is_removed(self) -> None:
        planned = _plan(
            sections=[
                {
                    "heading": "Customers",
                    "kind": "table",
                    "order_by": [{"field": "credit_score"}],
                }
            ]
        )

        mapping = to_document_plan(planned, schema=SCHEMA)

        assert mapping.plan.sections[0].order_by == ()

    def test_a_label_for_an_unknown_field_is_ignored(self) -> None:
        planned = _plan(
            labels=[
                {"field": "account_holder", "label": "Account Holder"},
                {"field": "credit_score", "label": "Credit Score"},
            ]
        )

        mapping = to_document_plan(planned, schema=SCHEMA)

        assert mapping.plan.labels == {"account_holder": "Account Holder"}

    def test_a_vocabulary_for_an_unknown_field_is_ignored(self) -> None:
        planned = _plan(vocabularies=[{"field": "credit_score", "values": ["Good", "Bad"]}])

        mapping = to_document_plan(planned, schema=SCHEMA)

        assert mapping.vocabularies == {}


class TestGroupingFilteringOrdering:
    def test_grouping_on_a_known_field_is_kept(self) -> None:
        planned = _plan(
            sections=[{"heading": "Customers", "kind": "table", "group_by": "account_type"}]
        )

        mapping = to_document_plan(planned, schema=SCHEMA)

        assert mapping.plan.sections[0].group_by == "account_type"

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            ("equals", MatchMode.EQUALS),
            ("not_equals", MatchMode.NOT_EQUALS),
            ("present", MatchMode.PRESENT),
            ("absent", MatchMode.ABSENT),
        ],
    )
    def test_every_match_mode_maps(self, mode: str, expected: MatchMode) -> None:
        planned = _plan(
            sections=[
                {
                    "heading": "Customers",
                    "kind": "table",
                    "filters": [{"field": "kyc_status", "mode": mode, "values": ["Pending"]}],
                }
            ]
        )

        mapping = to_document_plan(planned, schema=SCHEMA)

        assert mapping.plan.sections[0].filters[0].mode is expected

    def test_an_unknown_match_mode_drops_the_filter(self) -> None:
        planned = _plan(
            sections=[
                {
                    "heading": "Customers",
                    "kind": "table",
                    "filters": [{"field": "kyc_status", "mode": "contains", "values": ["x"]}],
                }
            ]
        )

        mapping = to_document_plan(planned, schema=SCHEMA)

        assert mapping.plan.sections[0].filters == ()
        assert any("not a known mode" in warning for warning in mapping.warnings)

    def test_ordering_flags_are_carried(self) -> None:
        planned = _plan(
            sections=[
                {
                    "heading": "Customers",
                    "kind": "table",
                    "order_by": [{"field": "balance", "descending": True, "numeric": True}],
                }
            ]
        )

        mapping = to_document_plan(planned, schema=SCHEMA)

        key = mapping.plan.sections[0].order_by[0]
        assert (key.field, key.descending, key.numeric) == ("balance", True, True)

    def test_a_filter_keeps_its_values(self) -> None:
        planned = _plan(
            sections=[
                {
                    "heading": "Pending KYC",
                    "kind": "table",
                    "filters": [
                        {"field": "kyc_status", "mode": "equals", "values": ["Pending", "Failed"]}
                    ],
                }
            ]
        )

        mapping = to_document_plan(planned, schema=SCHEMA)

        assert mapping.plan.sections[0].filters[0].values == ("Pending", "Failed")


class TestPlanDocumentLevel:
    def test_the_title_is_carried(self) -> None:
        mapping = to_document_plan(_plan(), schema=SCHEMA)

        assert mapping.plan.title == "Customer Master Report"

    def test_a_blank_title_falls_back_and_warns(self) -> None:
        planned = PlannedDocument.model_validate({"title": "   "})

        mapping = to_document_plan(planned, schema=SCHEMA)

        assert mapping.plan.title == DEFAULT_TITLE
        assert any("no title" in warning for warning in mapping.warnings)

    def test_a_blank_heading_falls_back_to_a_positional_name(self) -> None:
        planned = _plan(sections=[{"heading": "  ", "kind": "table"}])

        mapping = to_document_plan(planned, schema=SCHEMA)

        assert mapping.plan.sections[0].heading == "Section 1"

    def test_declared_vocabularies_are_returned_separately_from_layout(self) -> None:
        """Vocabularies constrain corrections, not layout, so they travel beside the
        plan rather than inside it."""
        planned = _plan(
            vocabularies=[
                {"field": "account_type", "values": ["Savings", "Current", "Salary"]},
                {"field": "kyc_status", "values": ["Completed", "Pending"]},
            ]
        )

        mapping = to_document_plan(planned, schema=SCHEMA)

        assert mapping.vocabularies == {
            "account_type": ("Savings", "Current", "Salary"),
            "kyc_status": ("Completed", "Pending"),
        }

    def test_empty_vocabulary_values_are_discarded(self) -> None:
        planned = _plan(vocabularies=[{"field": "account_type", "values": ["  ", ""]}])

        mapping = to_document_plan(planned, schema=SCHEMA)

        assert mapping.vocabularies == {}

    def test_an_empty_plan_maps_without_error(self) -> None:
        mapping = to_document_plan(PlannedDocument(), schema=SCHEMA)

        assert mapping.plan.sections == ()
        assert mapping.plan.title == DEFAULT_TITLE

    def test_vocabulary_arguments_shapes_values_for_the_validator(self) -> None:
        assert vocabulary_arguments({"account_type": ("Savings",)}) == {"account_type": ["Savings"]}


class TestCorrectionProposals:
    def test_a_valid_proposal_maps(self) -> None:
        payload = ProposedCorrectionList(
            corrections=[
                ProposedCorrectionModel(
                    record_index=3,
                    field="account_type",
                    original="sav",
                    corrected="Savings",
                    kind="vocabulary",
                    confidence="high",
                    reason="standardized account type",
                )
            ]
        )

        mapping = to_proposed_corrections(payload)

        [proposal] = mapping.proposals
        assert proposal.record_index == 3
        assert proposal.field == "account_type"
        assert proposal.original == "sav"
        assert proposal.corrected == "Savings"
        assert proposal.kind is CorrectionKind.VOCABULARY
        assert proposal.confidence is Confidence.HIGH
        assert proposal.reason == "standardized account type"

    @pytest.mark.parametrize("kind", [item.value for item in CorrectionKind])
    def test_every_supported_kind_maps(self, kind: str) -> None:
        payload = ProposedCorrectionList(
            corrections=[
                ProposedCorrectionModel(
                    record_index=0,
                    field="f",
                    original="a",
                    corrected="b",
                    kind=kind,
                    confidence="high",
                    reason="r",
                )
            ]
        )

        mapping = to_proposed_corrections(payload)

        assert mapping.proposals[0].kind is CorrectionKind(kind)

    @pytest.mark.parametrize("confidence", [item.value for item in Confidence])
    def test_every_confidence_level_maps(self, confidence: str) -> None:
        payload = ProposedCorrectionList(
            corrections=[
                ProposedCorrectionModel(
                    record_index=0,
                    field="f",
                    original="a",
                    corrected="b",
                    kind="spelling",
                    confidence=confidence,
                    reason="r",
                )
            ]
        )

        mapping = to_proposed_corrections(payload)

        assert mapping.proposals[0].confidence is Confidence(confidence)

    def test_duplicate_merge_is_not_a_supported_kind(self) -> None:
        """Deferred out of v1 — a model asking for it must not get it."""
        payload = ProposedCorrectionList(
            corrections=[
                ProposedCorrectionModel(
                    record_index=0,
                    field="f",
                    original="a",
                    corrected="b",
                    kind="duplicate_merge",
                    confidence="high",
                    reason="merge these",
                )
            ]
        )

        mapping = to_proposed_corrections(payload)

        assert mapping.proposals == ()
        assert any("not a known correction kind" in w for w in mapping.warnings)

    def test_an_unknown_kind_is_dropped(self) -> None:
        payload = ProposedCorrectionList(
            corrections=[
                ProposedCorrectionModel(
                    record_index=0,
                    field="f",
                    original="a",
                    corrected="b",
                    kind="rewrite_everything",
                    confidence="high",
                    reason="r",
                )
            ]
        )

        mapping = to_proposed_corrections(payload)

        assert mapping.proposals == ()

    def test_an_unknown_confidence_is_downgraded_to_low_not_dropped(self) -> None:
        """Low can never be applied, so the proposal survives for the user to see while
        being unable to change anything — safer than discarding information."""
        payload = ProposedCorrectionList(
            corrections=[
                ProposedCorrectionModel(
                    record_index=0,
                    field="f",
                    original="a",
                    corrected="b",
                    kind="spelling",
                    confidence="extremely certain",
                    reason="r",
                )
            ]
        )

        mapping = to_proposed_corrections(payload)

        assert mapping.proposals[0].confidence is Confidence.LOW
        assert any("treated as low" in w for w in mapping.warnings)

    def test_multiple_proposals_keep_their_order(self) -> None:
        payload = ProposedCorrectionList(
            corrections=[
                ProposedCorrectionModel(
                    record_index=index,
                    field="account_type",
                    original="sav",
                    corrected="Savings",
                    kind="vocabulary",
                    confidence="high",
                    reason="r",
                )
                for index in range(3)
            ]
        )

        mapping = to_proposed_corrections(payload)

        assert [p.record_index for p in mapping.proposals] == [0, 1, 2]

    def test_a_negative_record_index_is_refused_by_the_schema(self) -> None:
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            ProposedCorrectionModel(
                record_index=-1,
                field="f",
                original="a",
                corrected="b",
                kind="spelling",
                confidence="high",
                reason="r",
            )

    def test_an_empty_correction_list_maps_cleanly(self) -> None:
        mapping = to_proposed_corrections(ProposedCorrectionList())

        assert mapping.proposals == ()
        assert mapping.warnings == ()

    def test_mapping_does_not_validate_correctness(self) -> None:
        """This boundary only decides what is *representable*. Whether a correction is
        right stays with review_corrections, the single authority — so an obviously
        hallucinated rename maps through here untouched, to be rejected there."""
        payload = ProposedCorrectionList(
            corrections=[
                ProposedCorrectionModel(
                    record_index=0,
                    field="account_holder",
                    original="Priya Sharma",
                    corrected="Neha Kapoor",
                    kind="spelling",
                    confidence="high",
                    reason="hallucinated rename",
                )
            ]
        )

        mapping = to_proposed_corrections(payload)

        assert len(mapping.proposals) == 1
