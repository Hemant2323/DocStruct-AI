"""Tests for the `Insight` domain model's multi-record structure support (Milestone 4).

Pure logic, no network and no database — constructing dataclasses and reading their
attributes back is deterministic Python. Persistence round-tripping is covered separately
by ``tests/integration/test_ai_result_repository.py``, since that genuinely needs a real
database to exercise correctly.
"""

from __future__ import annotations

from uuid import uuid4

from app.models.domain.content import Provenance, ProvenanceKind
from app.models.domain.insight import Citation, ExtractedField, Insight, InsightKind

DOCUMENT_ID = uuid4()


def _verified_citation(label: str = "page 1") -> Citation:
    return Citation(
        quoted_text="State Bank of India",
        provenance=Provenance(kind=ProvenanceKind.PAGE, label=label, page=1),
        chunk_id=uuid4(),
        block_ids=(uuid4(),),
    )


class TestBackwardCompatibility:
    """Every existing Insight construction — summary, explanation, extraction — must
    keep working unmodified: `records` needs a safe default, not a required argument."""

    def test_insight_can_be_created_without_supplying_records(self) -> None:
        insight = Insight(document_id=DOCUMENT_ID, kind=InsightKind.SUMMARY, content="hi")
        assert insight.records == ()

    def test_existing_fields_only_construction_still_works(self) -> None:
        """Mirrors how `/extraction` and today's `/structure` build an Insight — fields
        populated, records left at its default."""
        insight = Insight(
            document_id=DOCUMENT_ID,
            kind=InsightKind.EXTRACTION,
            fields=(ExtractedField(name="total", value="45200.0"),),
        )
        assert insight.fields == (ExtractedField(name="total", value="45200.0"),)
        assert insight.records == ()


class TestMultiRecordStructure:
    def test_insight_can_contain_multiple_records(self) -> None:
        record_one = (ExtractedField(name="account_holder", value="Arjun Malhotra"),)
        record_two = (ExtractedField(name="account_holder", value="Person 2"),)

        insight = Insight(
            document_id=DOCUMENT_ID,
            kind=InsightKind.EXTRACTION,
            records=(record_one, record_two),
        )

        assert len(insight.records) == 2
        assert insight.records[0] == record_one
        assert insight.records[1] == record_two

    def test_each_record_can_contain_multiple_fields(self) -> None:
        record = (
            ExtractedField(name="account_holder", value="Arjun Malhotra"),
            ExtractedField(name="bank_name", value="State Bank of India"),
            ExtractedField(name="amount", value="2100"),
        )

        insight = Insight(document_id=DOCUMENT_ID, kind=InsightKind.EXTRACTION, records=(record,))

        assert len(insight.records[0]) == 3
        assert [field.name for field in insight.records[0]] == [
            "account_holder",
            "bank_name",
            "amount",
        ]

    def test_record_citations_survive_domain_representation(self) -> None:
        citation = _verified_citation()
        record = (ExtractedField(name="bank_name", value="State Bank of India", citation=citation),)

        insight = Insight(document_id=DOCUMENT_ID, kind=InsightKind.EXTRACTION, records=(record,))

        resolved = insight.records[0][0].citation
        assert resolved is not None
        assert resolved.is_verified
        assert resolved.quoted_text == "State Bank of India"
        assert resolved.location_label == "page 1"

    def test_unverified_field_citation_within_a_record_stays_none(self) -> None:
        """A field the model's quote could not be verified for must not gain a location
        just because it lives inside a record — same honesty rule as single-row fields."""
        record = (ExtractedField(name="amount", value="2100", citation=None),)

        insight = Insight(document_id=DOCUMENT_ID, kind=InsightKind.EXTRACTION, records=(record,))

        assert insight.records[0][0].citation is None

    def test_fields_and_records_coexist_without_interfering(self) -> None:
        """`fields` and `records` are independent attributes — populating one must not
        change or clear the other."""
        insight = Insight(
            document_id=DOCUMENT_ID,
            kind=InsightKind.EXTRACTION,
            fields=(ExtractedField(name="single", value="row"),),
            records=((ExtractedField(name="multi", value="record"),),),
        )

        assert insight.fields == (ExtractedField(name="single", value="row"),)
        assert insight.records == ((ExtractedField(name="multi", value="record"),),)
