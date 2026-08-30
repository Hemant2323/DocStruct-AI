"""Conversion must work on documents of any kind, not the ones it was built against.

The conversion feature was developed against customer/account data, and every other
conversion test in this suite still uses that shape. That is a real risk: an assumption
about banking data could take hold anywhere in the pipeline and no existing test would
notice, because no existing test feeds it anything else.

So these tests deliberately use two document shapes that share nothing with that data,
and nothing with each other:

* **Bridge inspection log** — many short records, few fields, values written
  inconsistently. Exercises tables, grouping, filtering and vocabulary normalisation.
* **Herbarium specimen catalogue** — few records, many fields, some fields simply not
  recorded for some specimens. Exercises detail sections, notes, and the handling of
  information a document genuinely does not contain.

Neither is a customer record, and nothing in the pipeline is told which is which.

**No test here calls a real model** (CLAUDE.md §7). Where a model's output is needed it
is written out as the literal structure a model would have returned, so what is under
test is the deterministic half — which is the half that decides anything.
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
)
from app.ai.conversion_schema import PlannedDocument, to_document_plan
from app.ai.corrections import review_corrections
from app.ai.prompts import conversion_corrections, conversion_plan
from app.models.domain.conversion import (
    Confidence,
    CorrectionKind,
    ProposedCorrection,
    SectionKind,
    ValueOrigin,
)
from app.models.domain.insight import ExtractedField

# --------------------------------------------------------------------------------------
# Shape 1: an inspection log. Many records, few fields, inconsistent values.
# --------------------------------------------------------------------------------------

INSPECTION_SCHEMA = {
    "structure_id": "string",
    "span_name": "string",
    "region": "string",
    "condition": "string",
    "inspected_on": "date",
}

INSPECTION_RECORDS = (
    (
        ExtractedField(name="structure_id", value="BR-1044", citation=None),
        ExtractedField(name="span_name", value="North Ridge Crossing", citation=None),
        ExtractedField(name="region", value="Highlands", citation=None),
        ExtractedField(name="condition", value="Good", citation=None),
        ExtractedField(name="inspected_on", value="2026-03-14", citation=None),
    ),
    (
        ExtractedField(name="structure_id", value="BR-1071", citation=None),
        ExtractedField(name="span_name", value="Miller Creek", citation=None),
        ExtractedField(name="region", value="Highlands", citation=None),
        # Written three different ways across the log — the case a vocabulary fixes.
        ExtractedField(name="condition", value="gd", citation=None),
        ExtractedField(name="inspected_on", value="2026-04-02", citation=None),
    ),
    (
        ExtractedField(name="structure_id", value="BR-2210", citation=None),
        ExtractedField(name="span_name", value="Eastgate Viaduct", citation=None),
        ExtractedField(name="region", value="Coastal", citation=None),
        ExtractedField(name="condition", value="POOR", citation=None),
        ExtractedField(name="inspected_on", value="2026-01-09", citation=None),
    ),
    (
        ExtractedField(name="structure_id", value="BR-2288", citation=None),
        ExtractedField(name="span_name", value="Saltmarsh Bridge", citation=None),
        ExtractedField(name="region", value="Coastal", citation=None),
        ExtractedField(name="condition", value="Fair", citation=None),
        ExtractedField(name="inspected_on", value="2026-02-27", citation=None),
    ),
)

CONDITION_VOCABULARY = {"condition": ("Good", "Fair", "Poor")}

# --------------------------------------------------------------------------------------
# Shape 2: a specimen catalogue. Few records, many fields, real gaps in the data.
# --------------------------------------------------------------------------------------

SPECIMEN_SCHEMA = {
    "accession": "string",
    "taxon": "string",
    "collector": "string",
    "collected_on": "date",
    "locality": "string",
    "elevation_m": "integer",
    "habitat": "string",
    "notes": "string",
}

SPECIMEN_RECORDS = (
    (
        ExtractedField(name="accession", value="HB-00412", citation=None),
        ExtractedField(name="taxon", value="Saxifraga oppositifolia", citation=None),
        ExtractedField(name="collector", value="E. Whitmore", citation=None),
        ExtractedField(name="collected_on", value="1974-06-11", citation=None),
        ExtractedField(name="locality", value="Corrie Dubh", citation=None),
        ExtractedField(name="elevation_m", value="880", citation=None),
        ExtractedField(name="habitat", value="wet　rock ledge", citation=None),
        ExtractedField(name="notes", value="Flowering.", citation=None),
    ),
    (
        ExtractedField(name="accession", value="HB-00519", citation=None),
        ExtractedField(name="taxon", value="Dryas octopetala", citation=None),
        ExtractedField(name="collector", value="E. Whitmore", citation=None),
        ExtractedField(name="collected_on", value="1974-07-03", citation=None),
        ExtractedField(name="locality", value="Corrie Dubh", citation=None),
        # elevation_m and habitat were never recorded for this specimen. They are
        # absent from the record entirely, which is how extraction represents "the
        # document does not say".
        ExtractedField(name="notes", value="Fruiting.", citation=None),
    ),
)


class TestInspectionLogConverts:
    """A table-shaped document nothing in the pipeline was designed around."""

    def test_groups_and_orders_without_any_knowledge_of_the_subject(self) -> None:
        plan = DocumentPlan(
            title="Bridge Inspection Summary",
            sections=(
                SectionPlan(
                    heading="Inspections",
                    kind=SectionKind.TABLE,
                    columns=("structure_id", "span_name", "condition", "inspected_on"),
                    group_by="region",
                    order_by=(OrderKey(field="structure_id"),),
                ),
            ),
            labels={"structure_id": "Structure", "span_name": "Span"},
        )

        document = assemble_document(
            INSPECTION_RECORDS,
            schema=INSPECTION_SCHEMA,
            plan=plan,
            review=review_corrections((), INSPECTION_RECORDS),
        )

        # One section per distinct region, in stable order.
        headings = [section.heading for section in document.sections]
        assert headings == ["Inspections — Coastal", "Inspections — Highlands"]
        assert all(section.kind is SectionKind.TABLE for section in document.sections)

        coastal = document.sections[0]
        assert [record.source_record_index for record in coastal.records] == [2, 3]
        assert coastal.columns == ("structure_id", "span_name", "condition", "inspected_on")

        # The plan's labels are honoured; the field names themselves are untouched.
        first = coastal.records[0]
        by_name = {field.name: field for field in first.fields}
        assert by_name["structure_id"].label == "Structure"
        assert by_name["condition"].label == "Condition"

    def test_vocabulary_normalisation_works_on_a_vocabulary_it_was_never_taught(
        self,
    ) -> None:
        """The canonical set comes from the plan, not from anything hardcoded."""
        proposals = [
            ProposedCorrection(
                kind=CorrectionKind.VOCABULARY,
                record_index=1,
                field="condition",
                original="gd",
                corrected="Good",
                reason="Abbreviated form of a canonical condition rating.",
                confidence=Confidence.HIGH,
            ),
            ProposedCorrection(
                kind=CorrectionKind.CASE_NORMALIZATION,
                record_index=2,
                field="condition",
                original="POOR",
                corrected="Poor",
                reason="Upper-case form of a canonical condition rating.",
                confidence=Confidence.HIGH,
            ),
        ]

        review = review_corrections(
            proposals, INSPECTION_RECORDS, vocabularies=CONDITION_VOCABULARY
        )

        assert len(review.applied) == 2
        assert not review.rejected

        document = assemble_document(
            INSPECTION_RECORDS,
            schema=INSPECTION_SCHEMA,
            plan=DocumentPlan(
                title="Bridge Inspection Summary",
                sections=(SectionPlan(heading="Inspections", kind=SectionKind.TABLE),),
            ),
            review=review,
        )

        values = {
            record.source_record_index: {field.name: field.value for field in record.fields}
            for record in document.sections[0].records
        }
        corrected = values[1]["condition"]
        assert corrected.text == "Good"
        assert corrected.origin is ValueOrigin.CORRECTED
        # The original survives alongside the change, with the audit entry that
        # justifies it. This is what stops a correction being a silent rewrite.
        assert corrected.original_text == "gd"
        assert corrected.correction_id

    def test_a_filter_on_a_domain_specific_value_still_works(self) -> None:
        plan = DocumentPlan(
            title="Bridge Inspection Summary",
            sections=(
                SectionPlan(
                    heading="Needs attention",
                    kind=SectionKind.TABLE,
                    filters=(
                        RecordFilter(field="condition", mode=MatchMode.EQUALS, values=("Poor",)),
                    ),
                ),
            ),
        )

        review = review_corrections(
            [
                ProposedCorrection(
                    kind=CorrectionKind.CASE_NORMALIZATION,
                    record_index=2,
                    field="condition",
                    original="POOR",
                    corrected="Poor",
                    reason="Canonical casing.",
                    confidence=Confidence.HIGH,
                )
            ],
            INSPECTION_RECORDS,
            vocabularies=CONDITION_VOCABULARY,
        )

        document = assemble_document(
            INSPECTION_RECORDS,
            schema=INSPECTION_SCHEMA,
            plan=plan,
            review=review,
        )

        # The filter sees the corrected value, so the record it is meant to catch is
        # caught even though the source wrote it as "POOR".
        assert [r.source_record_index for r in document.sections[0].records] == [2]


class TestSpecimenCatalogueConverts:
    """A detail-shaped document with genuine gaps — structurally unlike shape 1."""

    def test_detail_sections_and_notes_render_for_a_wide_record(self) -> None:
        plan = DocumentPlan(
            title="Herbarium Accessions",
            sections=(
                SectionPlan(
                    heading="Scope",
                    kind=SectionKind.NOTE,
                    note="Accessions collected during the 1974 highland survey.",
                ),
                SectionPlan(
                    heading="Specimens",
                    kind=SectionKind.DETAIL,
                    columns=(
                        "accession",
                        "taxon",
                        "collector",
                        "collected_on",
                        "locality",
                        "elevation_m",
                        "habitat",
                    ),
                ),
            ),
            labels={"elevation_m": "Elevation (m)"},
        )

        document = assemble_document(
            SPECIMEN_RECORDS,
            schema=SPECIMEN_SCHEMA,
            plan=plan,
            review=review_corrections((), SPECIMEN_RECORDS),
        )

        note, specimens = document.sections
        assert note.kind is SectionKind.NOTE
        assert note.note == "Accessions collected during the 1974 highland survey."
        assert note.records == ()

        assert specimens.kind is SectionKind.DETAIL
        assert len(specimens.records) == 2

    def test_a_field_the_document_never_states_stays_absent(self) -> None:
        """Missing information remains missing — never inferred from a sibling record.

        The first specimen records an elevation and the second does not. A system that
        filled the gap from context would be inventing a measurement, which for a
        scientific record is worse than a blank.
        """
        plan = DocumentPlan(
            title="Herbarium Accessions",
            sections=(
                SectionPlan(
                    heading="Specimens",
                    kind=SectionKind.DETAIL,
                    columns=("accession", "elevation_m", "habitat"),
                ),
            ),
        )

        document = assemble_document(
            SPECIMEN_RECORDS,
            schema=SPECIMEN_SCHEMA,
            plan=plan,
            review=review_corrections((), SPECIMEN_RECORDS),
        )

        second = document.sections[0].records[1]
        by_name = {field.name: field.value for field in second.fields}

        assert by_name["elevation_m"].origin is ValueOrigin.ABSENT
        assert by_name["elevation_m"].text == ""
        assert by_name["habitat"].origin is ValueOrigin.ABSENT
        # The record that does state one is unaffected by its neighbour's gap.
        first = document.sections[0].records[0]
        assert {f.name: f.value.text for f in first.fields}["elevation_m"] == "880"

    def test_no_value_carries_a_location_that_was_not_resolved(self) -> None:
        """Provenance is never fabricated, whatever the document is about."""
        document = assemble_document(
            SPECIMEN_RECORDS,
            schema=SPECIMEN_SCHEMA,
            plan=DocumentPlan(
                title="Herbarium Accessions",
                sections=(SectionPlan(heading="Specimens", kind=SectionKind.DETAIL),),
            ),
            review=review_corrections((), SPECIMEN_RECORDS),
        )

        # These fixtures carry `citation=None` throughout, so nothing may claim a source.
        for record in document.sections[0].records:
            for field in record.fields:
                assert field.value.citation is None

    def test_whitespace_correction_survives_an_ideographic_space(self) -> None:
        """A non-ASCII space is still whitespace. Documents are not all ASCII."""
        proposal = ProposedCorrection(
            kind=CorrectionKind.WHITESPACE,
            record_index=0,
            field="habitat",
            original="wet　rock ledge",
            corrected="wet rock ledge",
            reason="Ideographic space normalised to a regular space.",
            confidence=Confidence.HIGH,
        )

        review = review_corrections([proposal], SPECIMEN_RECORDS)

        # Whether the validator accepts this is its own decision; what must hold is
        # that it decides deterministically and never silently half-applies.
        assert len(review.applied) + len(review.suggestions) + len(review.rejected) == 1
        if review.applied:
            assert review.applied[0].corrected == "wet rock ledge"


class TestSafetyHoldsOnUnfamiliarData:
    """The guarantees must not depend on recognising the subject matter."""

    def test_a_field_that_was_never_extracted_is_stripped_from_the_plan(self) -> None:
        """Unknown fields are rejected — the closed set is the extracted schema."""
        planned = PlannedDocument.model_validate(
            {
                "title": "Bridge Inspection Summary",
                "sections": [
                    {
                        "heading": "Inspections",
                        "kind": "table",
                        # `load_rating` was never extracted. A planner that assumes a
                        # familiar field exists must not be able to conjure it.
                        "columns": ["structure_id", "load_rating", "condition"],
                        "group_by": "owner_authority",
                        "order_by": [{"field": "inspected_on"}],
                    }
                ],
                "labels": [{"field": "load_rating", "label": "Load Rating"}],
                "vocabularies": [
                    {"field": "condition", "values": ["Good", "Fair", "Poor"]},
                    {"field": "load_rating", "values": ["A", "B"]},
                ],
            }
        )

        mapping = to_document_plan(planned, schema=INSPECTION_SCHEMA)
        section = mapping.plan.sections[0]

        assert section.columns == ("structure_id", "condition")
        assert section.group_by is None
        assert "load_rating" not in mapping.plan.labels
        assert "load_rating" not in mapping.vocabularies
        assert "condition" in mapping.vocabularies
        # Every removal is reported rather than silently swallowed.
        assert any("load_rating" in warning for warning in mapping.warnings)
        assert any("owner_authority" in warning for warning in mapping.warnings)

    def test_a_confident_proposal_that_replaces_one_entity_with_another_is_refused(
        self,
    ) -> None:
        """Confidence is not authority, on any subject.

        Renaming one taxon to a different taxon is the botanical equivalent of renaming
        one customer to another: a change no amount of claimed certainty should buy.
        """
        proposal = ProposedCorrection(
            kind=CorrectionKind.SPELLING,
            record_index=1,
            field="taxon",
            original="Dryas octopetala",
            corrected="Silene acaulis",
            reason="Looks like the intended species.",
            confidence=Confidence.HIGH,
        )

        review = review_corrections([proposal], SPECIMEN_RECORDS)

        assert not review.applied
        assert len(review.rejected) == 1
        assert review.rejected[0].reason

    def test_a_medium_confidence_proposal_leaves_the_source_text_standing(self) -> None:
        proposal = ProposedCorrection(
            kind=CorrectionKind.SPELLING,
            record_index=0,
            field="locality",
            original="Corrie Dubh",
            corrected="Corrie Dubb",
            reason="Possible transcription slip.",
            confidence=Confidence.MEDIUM,
        )

        review = review_corrections([proposal], SPECIMEN_RECORDS)
        document = assemble_document(
            SPECIMEN_RECORDS,
            schema=SPECIMEN_SCHEMA,
            plan=DocumentPlan(
                title="Herbarium Accessions",
                sections=(SectionPlan(heading="Specimens", kind=SectionKind.DETAIL),),
            ),
            review=review,
        )

        assert not review.applied
        assert len(review.suggestions) == 1

        first = document.sections[0].records[0]
        locality = {field.name: field.value for field in first.fields}["locality"]
        assert locality.text == "Corrie Dubh"
        assert locality.origin is ValueOrigin.SOURCE


class TestPromptsCarryNoSubjectArea:
    """A regression guard, because this is exactly how domain bias creeps back.

    The prompts are the one place a subject area can be smuggled in without any code
    changing: an example about accounts teaches the planner that documents are about
    accounts. These assertions are deliberately about the *shipped prompt text*, not
    about behaviour, because the text is the thing that drifts.
    """

    #: Terms that would tell a model what kind of document to expect. Each was present
    #: in an earlier revision of these prompts.
    LOADED_TERMS = (
        "bank",
        "account",
        "customer",
        "invoice",
        "balance",
        "savings",
        "transaction",
        "payment",
        "kyc",
    )

    @pytest.mark.parametrize(
        "prompt_text",
        [
            pytest.param(conversion_plan.SYSTEM, id="plan-system"),
            pytest.param(conversion_corrections.SYSTEM, id="corrections-system"),
        ],
    )
    def test_system_prompt_names_no_subject_area(self, prompt_text: str) -> None:
        lowered = prompt_text.lower()
        found = [term for term in self.LOADED_TERMS if term in lowered]
        assert not found, f"subject-specific terms leaked into a prompt: {found}"

    def test_built_instructions_contain_only_the_document_s_own_vocabulary(self) -> None:
        """What reaches the model is the user's words plus the extracted field names."""
        instruction = conversion_plan.build_instruction(
            "Group the entries by region and list anything in poor condition separately.",
            INSPECTION_SCHEMA,
            INSPECTION_RECORDS,
        )

        for name in INSPECTION_SCHEMA:
            assert name in instruction
        assert "Group the entries by region" in instruction
        lowered = instruction.lower()
        assert not [term for term in self.LOADED_TERMS if term in lowered]

    def test_corrections_instruction_refuses_vocabulary_work_without_a_declared_set(
        self,
    ) -> None:
        """No declared vocabulary means no vocabulary corrections — for any field."""
        instruction = conversion_corrections.build_instruction(
            "Tidy the condition ratings.", INSPECTION_RECORDS, vocabularies=None
        )
        assert "do not propose any 'vocabulary' corrections" in instruction

        with_vocabulary = conversion_corrections.build_instruction(
            "Tidy the condition ratings.",
            INSPECTION_RECORDS,
            vocabularies=CONDITION_VOCABULARY,
        )
        assert '"Good", "Fair", "Poor"' in with_vocabulary
