"""Tests for document assembly.

Pure logic, no network and no database. Assembly is where a validated correction finally
touches a value, so these tests care most about two things: that an applied correction
keeps its whole trail intact (corrected text, original text, citation, audit id), and
that a *suggestion* changes nothing at all.

The plan is built by hand here. That is the point of the design — assembly takes a typed
plan and cannot tell whether a person or a model produced it.
"""

from __future__ import annotations

import pytest

from app.ai.assembly import (
    DocumentPlan,
    MatchMode,
    OrderKey,
    RecordFilter,
    SectionPlan,
    assemble_document,
    humanize_field_name,
)
from app.ai.corrections import review_corrections
from app.models.domain.content import Provenance, ProvenanceKind
from app.models.domain.conversion import (
    Confidence,
    CorrectionKind,
    ProposedCorrection,
    SectionKind,
    ValueOrigin,
)
from app.models.domain.insight import Citation, ExtractedField

SCHEMA = {
    "name": "string",
    "account_type": "string",
    "status": "string",
    "opening_date": "date",
    "balance": "number",
}

VOCABULARIES = {
    "account_type": ["Savings", "Current", "Salary"],
    "status": ["Completed", "Pending", "Failed"],
}


def _citation(label: str) -> Citation:
    return Citation(
        quoted_text=f"source text at {label}",
        provenance=Provenance(kind=ProvenanceKind.ROW_RANGE, label=label, row_start=2, row_end=2),
    )


def _records() -> list[list[ExtractedField]]:
    return [
        [
            ExtractedField(name="name", value="Rajesh Kumr Sharma", citation=_citation("row 2")),
            ExtractedField(name="account_type", value="sav", citation=_citation("row 2")),
            ExtractedField(name="status", value="DONE", citation=_citation("row 2")),
            ExtractedField(name="opening_date", value="2026-08-25", citation=_citation("row 2")),
            ExtractedField(name="balance", value="145320", citation=_citation("row 2")),
        ],
        [
            ExtractedField(name="name", value="Priya Sharma", citation=_citation("row 3")),
            ExtractedField(name="account_type", value="SAVINGS", citation=_citation("row 3")),
            ExtractedField(name="status", value="done", citation=_citation("row 3")),
            ExtractedField(name="opening_date", value="25/08/2026", citation=_citation("row 3")),
            # "balance" deliberately absent for this record.
        ],
    ]


def _plan(*sections: SectionPlan, title: str = "Customer Master Report") -> DocumentPlan:
    return DocumentPlan(title=title, sections=tuple(sections))


def _empty_review():  # type: ignore[no-untyped-def]
    return review_corrections([], _records())


def _review_with(*proposals: ProposedCorrection):  # type: ignore[no-untyped-def]
    return review_corrections(proposals, _records(), vocabularies=VOCABULARIES)


def _assemble(plan: DocumentPlan, review=None, warnings=()):  # type: ignore[no-untyped-def]
    return assemble_document(
        _records(),
        schema=SCHEMA,
        plan=plan,
        review=review if review is not None else _empty_review(),
        warnings=warnings,
    )


class TestValues:
    def test_source_values_are_preserved_verbatim(self) -> None:
        document = _assemble(_plan(SectionPlan(heading="Customers")))

        record = document.sections[0].records[0]
        name = record.field("name")
        assert name is not None
        assert name.value.text == "Rajesh Kumr Sharma"
        assert name.value.origin is ValueOrigin.SOURCE
        assert name.value.original_text is None
        assert name.value.correction_id is None

    def test_an_applied_correction_replaces_the_value_and_keeps_the_trail(self) -> None:
        review = _review_with(
            ProposedCorrection(
                kind=CorrectionKind.VOCABULARY,
                record_index=0,
                field="account_type",
                original="sav",
                corrected="Savings",
                reason="standardized account type",
                confidence=Confidence.HIGH,
            )
        )
        document = _assemble(_plan(SectionPlan(heading="Customers")), review)

        item = document.sections[0].records[0].field("account_type")
        assert item is not None
        assert item.value.text == "Savings"
        assert item.value.origin is ValueOrigin.CORRECTED
        assert item.value.original_text == "sav"
        assert item.value.correction_id == review.applied[0].id

    def test_a_corrected_value_keeps_the_original_citation(self) -> None:
        """The citation describes what the source says; the correction describes how the
        shown value differs from it. Losing the citation would break the audit chain."""
        review = _review_with(
            ProposedCorrection(
                kind=CorrectionKind.VOCABULARY,
                record_index=0,
                field="account_type",
                original="sav",
                corrected="Savings",
                reason="standardized",
                confidence=Confidence.HIGH,
            )
        )
        document = _assemble(_plan(SectionPlan(heading="Customers")), review)

        item = document.sections[0].records[0].field("account_type")
        assert item is not None
        assert item.value.citation is not None
        assert item.value.citation.is_verified
        assert item.value.citation.location_label == "row 2"

    def test_a_suggestion_leaves_the_value_untouched(self) -> None:
        """The whole promise of the feature: an uncertain fix does not alter the data."""
        review = _review_with(
            ProposedCorrection(
                kind=CorrectionKind.SPELLING,
                record_index=0,
                field="name",
                original="Rajesh Kumr Sharma",
                corrected="Rajesh Kumar Sharma",
                reason="possible spelling correction",
                confidence=Confidence.MEDIUM,
            )
        )
        document = _assemble(_plan(SectionPlan(heading="Customers")), review)

        item = document.sections[0].records[0].field("name")
        assert item is not None
        assert item.value.text == "Rajesh Kumr Sharma"
        assert item.value.origin is ValueOrigin.SOURCE
        assert document.suggestions[0].corrected == "Rajesh Kumar Sharma"
        assert document.corrections == ()

    def test_an_absent_field_stays_absent_rather_than_invented(self) -> None:
        document = _assemble(_plan(SectionPlan(heading="Customers")))

        item = document.sections[0].records[1].field("balance")
        assert item is not None
        assert item.value.origin is ValueOrigin.ABSENT
        assert item.value.text == ""
        assert item.value.citation is None

    def test_labels_default_to_a_humanized_field_name(self) -> None:
        document = _assemble(_plan(SectionPlan(heading="Customers")))

        item = document.sections[0].records[0].field("account_type")
        assert item is not None
        assert item.name == "account_type"  # key never rewritten
        assert item.label == "Account Type"

    def test_plan_labels_override_the_default(self) -> None:
        plan = DocumentPlan(
            title="Report",
            sections=(SectionPlan(heading="Customers"),),
            labels={"name": "Account Holder"},
        )
        document = _assemble(plan)

        item = document.sections[0].records[0].field("name")
        assert item is not None
        assert item.label == "Account Holder"


class TestRecordsAndLineage:
    def test_every_record_is_kept(self) -> None:
        document = _assemble(_plan(SectionPlan(heading="Customers")))

        assert len(document.sections[0].records) == 2
        assert document.source_record_count == 2

    def test_source_record_index_is_preserved(self) -> None:
        document = _assemble(_plan(SectionPlan(heading="Customers")))

        assert [r.source_record_index for r in document.sections[0].records] == [0, 1]

    def test_no_duplicate_merging_happens(self) -> None:
        """Two records that look similar stay two records — merging is deferred."""
        document = _assemble(_plan(SectionPlan(heading="Customers")))

        assert len(document.sections[0].records) == 2
        assert all(record.merged_from == () for record in document.sections[0].records)


class TestSections:
    def test_multiple_sections_are_supported(self) -> None:
        document = _assemble(
            _plan(
                SectionPlan(heading="Customers"),
                SectionPlan(heading="Notes", kind=SectionKind.NOTE, note="Prepared for review."),
            )
        )

        assert [section.heading for section in document.sections] == ["Customers", "Notes"]
        assert document.sections[1].kind is SectionKind.NOTE
        assert document.sections[1].note == "Prepared for review."

    def test_a_note_section_carries_no_records(self) -> None:
        document = _assemble(
            _plan(SectionPlan(heading="Notes", kind=SectionKind.NOTE, note="Text."))
        )

        assert document.sections[0].records == ()

    def test_columns_default_to_the_schema(self) -> None:
        document = _assemble(_plan(SectionPlan(heading="Customers")))

        assert document.sections[0].columns == tuple(SCHEMA)

    def test_a_section_may_choose_its_own_columns(self) -> None:
        document = _assemble(_plan(SectionPlan(heading="Customers", columns=("name", "balance"))))

        assert document.sections[0].columns == ("name", "balance")

    def test_a_detail_section_is_supported(self) -> None:
        document = _assemble(_plan(SectionPlan(heading="Customers", kind=SectionKind.DETAIL)))

        assert document.sections[0].kind is SectionKind.DETAIL
        assert len(document.sections[0].records) == 2

    def test_an_empty_section_is_dropped_by_default(self) -> None:
        """A "Pending KYC" heading with nothing under it reads like a failure."""
        document = _assemble(
            _plan(
                SectionPlan(
                    heading="Failed",
                    filters=(
                        RecordFilter(field="status", mode=MatchMode.EQUALS, values=("Failed",)),
                    ),
                )
            )
        )

        assert document.sections == ()

    def test_an_empty_section_can_be_kept_when_asked_for(self) -> None:
        document = _assemble(
            _plan(
                SectionPlan(
                    heading="Failed",
                    filters=(
                        RecordFilter(field="status", mode=MatchMode.EQUALS, values=("Failed",)),
                    ),
                    include_empty=True,
                )
            )
        )

        assert len(document.sections) == 1
        assert document.sections[0].records == ()


class TestFiltering:
    def test_equals_filter_keeps_matching_records(self) -> None:
        document = _assemble(
            _plan(
                SectionPlan(
                    heading="Savings",
                    filters=(
                        RecordFilter(
                            field="account_type", mode=MatchMode.EQUALS, values=("SAVINGS",)
                        ),
                    ),
                )
            )
        )

        assert [r.source_record_index for r in document.sections[0].records] == [1]

    def test_filtering_is_case_insensitive(self) -> None:
        """Source data is messy by definition — a case-sensitive filter would drop the
        very rows the user is trying to organize."""
        document = _assemble(
            _plan(
                SectionPlan(
                    heading="Savings",
                    filters=(
                        RecordFilter(
                            field="account_type", mode=MatchMode.EQUALS, values=("savings",)
                        ),
                    ),
                )
            )
        )

        assert len(document.sections[0].records) == 1

    def test_not_equals_filter_excludes_matching_records(self) -> None:
        document = _assemble(
            _plan(
                SectionPlan(
                    heading="Non-savings",
                    filters=(
                        RecordFilter(
                            field="account_type", mode=MatchMode.NOT_EQUALS, values=("SAVINGS",)
                        ),
                    ),
                )
            )
        )

        assert [r.source_record_index for r in document.sections[0].records] == [0]

    def test_absent_filter_selects_records_missing_a_field(self) -> None:
        document = _assemble(
            _plan(
                SectionPlan(
                    heading="Missing balance",
                    filters=(RecordFilter(field="balance", mode=MatchMode.ABSENT),),
                )
            )
        )

        assert [r.source_record_index for r in document.sections[0].records] == [1]

    def test_present_filter_selects_records_having_a_field(self) -> None:
        document = _assemble(
            _plan(
                SectionPlan(
                    heading="Has balance",
                    filters=(RecordFilter(field="balance", mode=MatchMode.PRESENT),),
                )
            )
        )

        assert [r.source_record_index for r in document.sections[0].records] == [0]

    def test_a_filter_sees_the_corrected_value(self) -> None:
        """Normalizing "sav" to "Savings" and then filtering on "Savings" must find it —
        the filter reads what the document will show."""
        review = _review_with(
            ProposedCorrection(
                kind=CorrectionKind.VOCABULARY,
                record_index=0,
                field="account_type",
                original="sav",
                corrected="Savings",
                reason="standardized",
                confidence=Confidence.HIGH,
            )
        )
        document = _assemble(
            _plan(
                SectionPlan(
                    heading="Savings",
                    filters=(
                        RecordFilter(
                            field="account_type", mode=MatchMode.EQUALS, values=("Savings",)
                        ),
                    ),
                )
            ),
            review,
        )

        assert [r.source_record_index for r in document.sections[0].records] == [0, 1]


class TestGrouping:
    def test_group_by_produces_one_section_per_value(self) -> None:
        document = _assemble(_plan(SectionPlan(heading="Customers", group_by="account_type")))

        # Group order is deterministic: by group value, case-insensitively. "sav"
        # precedes "SAVINGS" because it is a prefix of it, not because of its casing —
        # which is the property that matters, since the same input must always produce
        # the same section order.
        headings = [section.heading for section in document.sections]
        assert headings == ["Customers — sav", "Customers — SAVINGS"]

    def test_grouping_after_correction_collapses_variants(self) -> None:
        """The feature working end to end: three spellings of savings become one group
        once the corrections are applied."""
        review = _review_with(
            ProposedCorrection(
                kind=CorrectionKind.VOCABULARY,
                record_index=0,
                field="account_type",
                original="sav",
                corrected="Savings",
                reason="standardized",
                confidence=Confidence.HIGH,
            ),
            ProposedCorrection(
                kind=CorrectionKind.VOCABULARY,
                record_index=1,
                field="account_type",
                original="SAVINGS",
                corrected="Savings",
                reason="standardized",
                confidence=Confidence.HIGH,
            ),
        )
        document = _assemble(
            _plan(SectionPlan(heading="Customers", group_by="account_type")), review
        )

        assert len(document.sections) == 1
        assert document.sections[0].heading == "Customers — Savings"
        assert len(document.sections[0].records) == 2

    def test_records_missing_the_group_field_are_collected_last(self) -> None:
        document = _assemble(_plan(SectionPlan(heading="Customers", group_by="balance")))

        assert document.sections[-1].heading == "Customers — Not stated"
        assert document.sections[-1].records[0].source_record_index == 1


class TestOrdering:
    def test_text_ordering(self) -> None:
        document = _assemble(
            _plan(SectionPlan(heading="Customers", order_by=(OrderKey(field="name"),)))
        )

        names = [r.field("name").value.text for r in document.sections[0].records]  # type: ignore[union-attr]
        assert names == ["Priya Sharma", "Rajesh Kumr Sharma"]

    def test_descending_ordering(self) -> None:
        document = _assemble(
            _plan(
                SectionPlan(
                    heading="Customers", order_by=(OrderKey(field="name", descending=True),)
                )
            )
        )

        names = [r.field("name").value.text for r in document.sections[0].records]  # type: ignore[union-attr]
        assert names == ["Rajesh Kumr Sharma", "Priya Sharma"]

    def test_numeric_ordering_beats_text_ordering(self) -> None:
        """As text, "145320" sorts before "45320" — which is not what sorting a balance
        means to anyone."""
        records = [
            [ExtractedField(name="balance", value="145320")],
            [ExtractedField(name="balance", value="45320")],
        ]
        document = assemble_document(
            records,
            schema={"balance": "number"},
            plan=_plan(
                SectionPlan(
                    heading="By balance", order_by=(OrderKey(field="balance", numeric=True),)
                )
            ),
            review=review_corrections([], records),
        )

        values = [r.field("balance").value.text for r in document.sections[0].records]  # type: ignore[union-attr]
        assert values == ["45320", "145320"]

    def test_records_missing_the_sort_field_sort_last(self) -> None:
        document = _assemble(
            _plan(
                SectionPlan(
                    heading="Customers", order_by=(OrderKey(field="balance", numeric=True),)
                )
            )
        )

        assert [r.source_record_index for r in document.sections[0].records] == [0, 1]

    def test_a_non_numeric_value_does_not_break_numeric_ordering(self) -> None:
        records = [
            [ExtractedField(name="balance", value="n/a")],
            [ExtractedField(name="balance", value="500")],
        ]
        document = assemble_document(
            records,
            schema={"balance": "number"},
            plan=_plan(
                SectionPlan(
                    heading="By balance", order_by=(OrderKey(field="balance", numeric=True),)
                )
            ),
            review=review_corrections([], records),
        )

        values = [r.field("balance").value.text for r in document.sections[0].records]  # type: ignore[union-attr]
        assert values == ["500", "n/a"]


class TestDocumentLevel:
    def test_title_schema_and_warnings_are_carried(self) -> None:
        document = _assemble(
            _plan(SectionPlan(heading="Customers"), title="KYC Report"),
            warnings=("Only 2 of 30 rows were converted.",),
        )

        assert document.title == "KYC Report"
        assert document.schema_used == SCHEMA
        assert document.warnings == ("Only 2 of 30 rows were converted.",)

    def test_applied_and_suggested_corrections_are_both_carried(self) -> None:
        review = _review_with(
            ProposedCorrection(
                kind=CorrectionKind.VOCABULARY,
                record_index=0,
                field="account_type",
                original="sav",
                corrected="Savings",
                reason="standardized",
                confidence=Confidence.HIGH,
            ),
            ProposedCorrection(
                kind=CorrectionKind.SPELLING,
                record_index=0,
                field="name",
                original="Rajesh Kumr Sharma",
                corrected="Rajesh Kumar Sharma",
                reason="possible spelling",
                confidence=Confidence.MEDIUM,
            ),
        )
        document = _assemble(_plan(SectionPlan(heading="Customers")), review)

        assert document.applied_correction_count == 1
        assert len(document.suggestions) == 1
        assert document.suggestions_for(0, "name")[0].corrected == "Rajesh Kumar Sharma"
        assert document.suggestions_for(1, "name") == ()

    def test_a_rejected_correction_never_reaches_the_document(self) -> None:
        review = _review_with(
            ProposedCorrection(
                kind=CorrectionKind.VOCABULARY,
                record_index=0,
                field="account_type",
                original="sav",
                corrected="Investment",  # outside the vocabulary
                reason="bogus",
                confidence=Confidence.HIGH,
            )
        )
        document = _assemble(_plan(SectionPlan(heading="Customers")), review)

        item = document.sections[0].records[0].field("account_type")
        assert item is not None
        assert item.value.text == "sav"
        assert document.corrections == ()
        assert document.suggestions == ()


class TestPurity:
    def test_identical_input_produces_identical_output(self) -> None:
        plan = _plan(
            SectionPlan(heading="Customers", group_by="account_type", order_by=(OrderKey("name"),))
        )

        assert _assemble(plan) == _assemble(plan)

    def test_inputs_are_not_mutated(self) -> None:
        records = _records()
        before = records[0][1].value

        assemble_document(
            records,
            schema=SCHEMA,
            plan=_plan(SectionPlan(heading="Customers")),
            review=review_corrections([], records),
        )

        assert records[0][1].value == before

    def test_an_applied_correction_without_an_id_is_refused(self) -> None:
        """A changed value that cannot be traced to an audit entry is the silent edit
        this feature exists to prevent, so assembly fails loudly."""
        from app.ai.corrections import CorrectionReview
        from app.models.domain.conversion import Correction

        broken = CorrectionReview(
            applied=(
                Correction(
                    id="",
                    kind=CorrectionKind.VOCABULARY,
                    record_index=0,
                    field="account_type",
                    original="sav",
                    corrected="Savings",
                    reason="no id",
                    confidence=Confidence.HIGH,
                    applied=True,
                ),
            )
        )

        with pytest.raises(ValueError, match="no id"):
            _assemble(_plan(SectionPlan(heading="Customers")), broken)


class TestHumanizeFieldName:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("account_holder", "Account Holder"),
            ("name", "Name"),
            ("days_until_deadline", "Days Until Deadline"),
            ("transaction_id", "Transaction Id"),
        ],
    )
    def test_snake_case_becomes_title_case(self, raw: str, expected: str) -> None:
        assert humanize_field_name(raw) == expected
