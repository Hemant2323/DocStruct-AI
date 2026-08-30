"""Tests for `AIResultRepository`'s multi-record structure support (Milestone 4), against
a real database.

Serialisation round-tripping is exactly the kind of thing a mock would happily confirm
while the real JSON column did something else — mirrors the structure of
``test_chunk_embedding_repository.py``. Existing `fields`-only behaviour (summary,
explanation, extraction) is already covered elsewhere (``test_export_api.py`` exercises
`AIResultRepository` via the API); these tests focus specifically on what Step 3 added.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.domain.content import Provenance, ProvenanceKind
from app.models.domain.document import Document, SupportedFileType
from app.models.domain.insight import Citation, ExtractedField, Insight, InsightKind
from app.repositories.ai_result_repo import AIResultRepository
from app.repositories.document_repo import DocumentRepository

OWNER = "owner-a"


@pytest.fixture
async def document(db_session: AsyncSession) -> Document:
    """A persisted document for a result to hang off."""
    repository = DocumentRepository(db_session)
    created = await repository.create(
        owner_id=OWNER,
        original_filename="payments.pdf",
        file_type=SupportedFileType.PDF,
        size_bytes=1024,
    )
    await db_session.flush()
    return created


@pytest.fixture
def repository(db_session: AsyncSession) -> AIResultRepository:
    return AIResultRepository(db_session)


def _verified_citation(label: str = "page 1") -> Citation:
    return Citation(
        quoted_text="State Bank of India",
        provenance=Provenance(kind=ProvenanceKind.PAGE, label=label, page=1),
        chunk_id=uuid4(),
        block_ids=(uuid4(),),
    )


class TestRecordsRoundTrip:
    async def test_save_and_get_preserves_multiple_records(
        self, repository: AIResultRepository, document: Document
    ) -> None:
        record_one = (
            ExtractedField(name="account_holder", value="Arjun Malhotra"),
            ExtractedField(name="bank_name", value="State Bank of India"),
        )
        record_two = (
            ExtractedField(name="account_holder", value="Person 2"),
            ExtractedField(name="bank_name", value="HDFC Bank"),
        )
        insight = Insight(
            document_id=document.id,
            kind=InsightKind.EXTRACTION,
            records=(record_one, record_two),
        )

        await repository.save(insight, owner_id=OWNER, fingerprint="fp-records")
        reloaded = await repository.get(document.id, owner_id=OWNER, fingerprint="fp-records")

        assert reloaded is not None
        assert len(reloaded.records) == 2
        assert reloaded.records[0] == record_one
        assert reloaded.records[1] == record_two

    async def test_record_citations_survive_the_round_trip(
        self, repository: AIResultRepository, document: Document
    ) -> None:
        citation = _verified_citation()
        record = (ExtractedField(name="bank_name", value="State Bank of India", citation=citation),)
        insight = Insight(document_id=document.id, kind=InsightKind.EXTRACTION, records=(record,))

        await repository.save(insight, owner_id=OWNER, fingerprint="fp-citation")
        reloaded = await repository.get(document.id, owner_id=OWNER, fingerprint="fp-citation")

        assert reloaded is not None
        resolved = reloaded.records[0][0].citation
        assert resolved is not None
        assert resolved.is_verified
        assert resolved.quoted_text == "State Bank of India"
        assert resolved.location_label == "page 1"
        assert resolved.chunk_id == citation.chunk_id
        assert resolved.block_ids == citation.block_ids

    async def test_existing_fields_are_unaffected_when_records_are_also_present(
        self, repository: AIResultRepository, document: Document
    ) -> None:
        insight = Insight(
            document_id=document.id,
            kind=InsightKind.EXTRACTION,
            fields=(ExtractedField(name="single_row_field", value="unchanged"),),
            records=((ExtractedField(name="record_field", value="new"),),),
        )

        await repository.save(insight, owner_id=OWNER, fingerprint="fp-both")
        reloaded = await repository.get(document.id, owner_id=OWNER, fingerprint="fp-both")

        assert reloaded is not None
        assert reloaded.fields == (ExtractedField(name="single_row_field", value="unchanged"),)
        assert reloaded.records == ((ExtractedField(name="record_field", value="new"),),)

    async def test_a_result_saved_without_records_deserializes_to_an_empty_tuple(
        self, repository: AIResultRepository, document: Document
    ) -> None:
        """Exercises the exact shape every pre-Step-3 saved result has: no `records`
        ever supplied, mirroring a summary/explanation/extraction result saved before
        this column existed."""
        insight = Insight(
            document_id=document.id,
            kind=InsightKind.SUMMARY,
            content="A summary with no records at all.",
        )

        await repository.save(insight, owner_id=OWNER, fingerprint="fp-no-records")
        reloaded = await repository.get(document.id, owner_id=OWNER, fingerprint="fp-no-records")

        assert reloaded is not None
        assert reloaded.records == ()
        assert reloaded.content == "A summary with no records at all."

    async def test_list_for_document_includes_records(
        self, repository: AIResultRepository, document: Document
    ) -> None:
        record = (ExtractedField(name="account_holder", value="Person 1"),)
        insight = Insight(document_id=document.id, kind=InsightKind.EXTRACTION, records=(record,))
        await repository.save(insight, owner_id=OWNER, fingerprint="fp-list")

        results = await repository.list_for_document(document.id, owner_id=OWNER)

        assert len(results) == 1
        assert results[0].records == (record,)
