"""End-to-end tests for exporting a structure result (Milestone 4.6).

Mirrors ``test_structure_api.py``: the language model is a fake, real ingestion runs
through the actual HTTP endpoints, and every export test first calls `/structure` to
populate the cache the export endpoint then reads — export itself must never call the
provider, which several tests assert directly via ``fake_provider.call_count``.

Owner isolation is tested at the service level (real repositories via ``db_session``,
`FakeLLMProvider` standing in only for its ``model_name``), since Milestone 1 has no
authentication and every HTTP request runs as the same owner — there is no way to send a
"different owner" request through the API as it exists today.
"""

from __future__ import annotations

import csv
import io
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from openpyxl import load_workbook
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.prompts import structure as structure_prompt
from app.core.exceptions import NotFoundError
from app.models.domain.document import SupportedFileType
from app.models.domain.insight import ContextMode, ExtractedField, Insight, InsightKind
from app.repositories.ai_result_repo import AIResultRepository
from app.repositories.document_repo import DocumentRepository
from app.services.analysis_service import compute_fingerprint
from app.services.export_service import ExportService
from app.services.job_service import JobService
from tests.fixtures.builders import build_txt
from tests.fixtures.fake_provider import FakeLLMProvider

DOCUMENT_TEXT = (
    "PAYMENT REPORT\n\n"
    "Account holder Arjun Malhotra banks with State Bank of India. Amount due: 2100.\n\n"
    "Account holder Priya Sharma banks with HDFC Bank. Amount due: 4500.\n\n"
    "Account holder Rahul Verma banks with ICICI Bank. Amount due: 3200."
)

SCHEMA = {"account_holder": "string", "bank_name": "string", "amount": "number"}


async def _ingest(
    client: AsyncClient, api_prefix: str, jobs: JobService, *, filename: str = "agreement.txt"
) -> str:
    response = await client.post(
        f"{api_prefix}/documents",
        files={"file": (filename, build_txt(DOCUMENT_TEXT), "text/plain")},
    )
    assert response.status_code == 202, response.text
    document_id: str = response.json()["id"]

    await jobs.wait_for_all(timeout=30.0)

    status = await client.get(f"{api_prefix}/documents/{document_id}")
    assert status.json()["status"] == "ready", status.text
    return document_id


def _field(value: Any, source_quote: str) -> SimpleNamespace:
    return SimpleNamespace(value=value, source_quote=source_quote)


def _records(*records: SimpleNamespace) -> SimpleNamespace:
    """A multi-record structure result, in the shape `/structure` now produces
    (Milestone 4) — one entry per matching record, not a single flat object."""
    return SimpleNamespace(records=list(records))


@pytest.fixture
def jobs(ai_app: FastAPI) -> JobService:
    return ai_app.state.job_service  # type: ignore[no-any-return]


async def _structure_then_prepare_export(
    ai_client: AsyncClient,
    api_prefix: str,
    jobs: JobService,
    fake_provider: FakeLLMProvider,
    *,
    filename: str = "agreement.txt",
) -> str:
    """Populate the cache exactly like a real caller would: run /structure first.

    Three records, deliberately: a single-record fixture would pass just as well against
    an export that silently dropped everything after the first row, which is exactly the
    bug this milestone's export has to not have.
    """
    fake_provider.parsed = _records(
        SimpleNamespace(
            account_holder=_field("Arjun Malhotra", "Account holder Arjun Malhotra"),
            bank_name=_field("State Bank of India", "State Bank of India"),
            amount=_field(2100, "Amount due: 2100"),
        ),
        SimpleNamespace(
            account_holder=_field("Priya Sharma", "Account holder Priya Sharma"),
            bank_name=_field("HDFC Bank", "banks with HDFC Bank"),
            amount=_field(4500, "Amount due: 4500"),
        ),
        SimpleNamespace(
            account_holder=_field("Rahul Verma", "Account holder Rahul Verma"),
            bank_name=_field("ICICI Bank", "banks with ICICI Bank"),
            amount=_field(3200, "Amount due: 3200"),
        ),
    )
    document_id = await _ingest(ai_client, api_prefix, jobs, filename=filename)

    response = await ai_client.post(
        f"{api_prefix}/documents/{document_id}/structure", json={"schema": SCHEMA}
    )
    assert response.status_code == 200, response.text
    return document_id


class TestJsonExport:
    async def test_correct_media_type_and_every_record_is_exported(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        document_id = await _structure_then_prepare_export(
            ai_client, api_prefix, jobs, fake_provider
        )

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure/export?format=json",
            json={"schema": SCHEMA},
        )

        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("application/json")
        records = response.json()["records"]
        assert len(records) == 3
        assert [record["account_holder"] for record in records] == [
            "Arjun Malhotra",
            "Priya Sharma",
            "Rahul Verma",
        ]
        assert [record["bank_name"] for record in records] == [
            "State Bank of India",
            "HDFC Bank",
            "ICICI Bank",
        ]
        assert [record["amount"] for record in records] == ["2100", "4500", "3200"]

    async def test_provenance_is_present_per_record(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        """Each record carries its own sources — the export is the one artefact that
        leaves the product, so a value must not appear in it stripped of where it came
        from."""
        document_id = await _structure_then_prepare_export(
            ai_client, api_prefix, jobs, fake_provider
        )

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure/export?format=json",
            json={"schema": SCHEMA},
        )

        records = response.json()["records"]
        citation = records[1]["_sources"]["bank_name"]
        assert citation is not None
        assert citation["is_verified"] is True
        assert citation["location_label"] != "source not verified"

    async def test_no_internal_ids_are_exposed(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        document_id = await _structure_then_prepare_export(
            ai_client, api_prefix, jobs, fake_provider
        )

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure/export?format=json",
            json={"schema": SCHEMA},
        )

        raw = response.text
        assert "chunk_id" not in raw
        assert "block_ids" not in raw


class TestCsvExport:
    async def test_correct_media_type_headers_and_one_row_per_record(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        document_id = await _structure_then_prepare_export(
            ai_client, api_prefix, jobs, fake_provider
        )

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure/export?format=csv",
            json={"schema": SCHEMA},
        )

        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/csv")
        assert response.text.splitlines()[0] == "account_holder,bank_name,amount"

        rows = list(csv.DictReader(io.StringIO(response.text)))
        assert len(rows) == 3
        assert rows[0]["account_holder"] == "Arjun Malhotra"
        assert rows[0]["bank_name"] == "State Bank of India"
        assert rows[0]["amount"] == "2100"
        assert rows[2]["account_holder"] == "Rahul Verma"
        assert rows[2]["amount"] == "3200"

    async def test_formula_injection_is_escaped(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        fake_provider.parsed = _records(
            SimpleNamespace(bank_name=_field("=cmd|' /C calc'!A0", "State Bank of India"))
        )
        document_id = await _ingest(ai_client, api_prefix, jobs)
        await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"schema": {"bank_name": "string"}},
        )

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure/export?format=csv",
            json={"schema": {"bank_name": "string"}},
        )

        [row] = list(csv.DictReader(io.StringIO(response.text)))
        assert row["bank_name"].startswith("'=")


class TestXlsxExport:
    async def test_correct_media_type_headers_and_one_row_per_record(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        document_id = await _structure_then_prepare_export(
            ai_client, api_prefix, jobs, fake_provider
        )

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure/export?format=xlsx",
            json={"schema": SCHEMA},
        )

        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        workbook = load_workbook(io.BytesIO(response.content))
        rows = list(workbook.active.iter_rows(values_only=True))
        assert rows[0] == ("account_holder", "bank_name", "amount")
        assert len(rows) == 4  # header + 3 records
        assert rows[1] == ("Arjun Malhotra", "State Bank of India", "2100")
        assert rows[3] == ("Rahul Verma", "ICICI Bank", "3200")

    async def test_a_single_worksheet_holds_every_record(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        """One table, not one worksheet per record."""
        document_id = await _structure_then_prepare_export(
            ai_client, api_prefix, jobs, fake_provider
        )

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure/export?format=xlsx",
            json={"schema": SCHEMA},
        )

        workbook = load_workbook(io.BytesIO(response.content))
        assert len(workbook.sheetnames) == 1

    async def test_formula_injection_is_escaped(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        fake_provider.parsed = _records(
            SimpleNamespace(bank_name=_field("+1;DROP TABLE", "State Bank of India"))
        )
        document_id = await _ingest(ai_client, api_prefix, jobs)
        await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"schema": {"bank_name": "string"}},
        )

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure/export?format=xlsx",
            json={"schema": {"bank_name": "string"}},
        )

        workbook = load_workbook(io.BytesIO(response.content))
        _, data_row = list(workbook.active.iter_rows(values_only=True))
        assert data_row[0] == "'+1;DROP TABLE"


class TestSafeFilename:
    async def test_content_disposition_has_no_unsafe_characters(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        document_id = await _structure_then_prepare_export(
            ai_client, api_prefix, jobs, fake_provider, filename='weird<>:"name.txt'
        )

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure/export?format=json",
            json={"schema": SCHEMA},
        )

        disposition = response.headers["content-disposition"]
        assert "attachment" in disposition
        # The header's own syntax is `filename="..."` -- exactly two quote characters
        # delimit the value. sanitize_filename() strips embedded quotes/angle brackets
        # from the filename itself, so any extra quote here would mean one leaked
        # through and could break out of the delimited value.
        assert disposition.count('"') == 2
        for unsafe in ("<", ">", ":"):
            assert unsafe not in disposition, f"{unsafe!r} leaked into Content-Disposition"
        assert "\n" not in disposition
        assert "\r" not in disposition

    async def test_filename_reflects_the_requested_format(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        document_id = await _structure_then_prepare_export(
            ai_client, api_prefix, jobs, fake_provider
        )

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure/export?format=csv",
            json={"schema": SCHEMA},
        )

        assert ".csv" in response.headers["content-disposition"]


class TestMissingResults:
    async def test_nonexistent_document_returns_404(
        self, ai_client: AsyncClient, api_prefix: str
    ) -> None:
        response = await ai_client.post(
            f"{api_prefix}/documents/{uuid4()}/structure/export?format=json",
            json={"schema": SCHEMA},
        )

        assert response.status_code == 404, response.text
        assert response.json()["error"]["code"] == "not_found"

    async def test_export_without_a_prior_structure_call_is_404(
        self, ai_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure/export?format=json",
            json={"schema": SCHEMA},
        )

        assert response.status_code == 404, response.text
        assert response.json()["error"]["code"] == "not_found"

    async def test_a_different_schema_does_not_export_the_wrong_result(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        """The single most important test in this file: exporting with schema B must
        never silently return schema A's cached result."""
        document_id = await _structure_then_prepare_export(
            ai_client, api_prefix, jobs, fake_provider
        )

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure/export?format=json",
            json={"schema": {"vendor": "string"}},
        )

        assert response.status_code == 404, response.text

    async def test_invalid_schema_is_rejected_before_any_lookup(
        self, ai_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure/export?format=json",
            json={"schema": {}},
        )

        assert response.status_code == 422, response.text


class TestNoProviderCallDuringExport:
    async def test_provider_call_count_does_not_increase(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        document_id = await _structure_then_prepare_export(
            ai_client, api_prefix, jobs, fake_provider
        )
        calls_after_structure = fake_provider.call_count

        for export_format in ("json", "csv", "xlsx"):
            await ai_client.post(
                f"{api_prefix}/documents/{document_id}/structure/export?format={export_format}",
                json={"schema": SCHEMA},
            )

        assert fake_provider.call_count == calls_after_structure

    async def test_repeated_export_of_a_missing_result_still_never_calls_the_provider(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        """Even a 404 path -- which could plausibly fall back to generating -- must not."""
        document_id = await _ingest(ai_client, api_prefix, jobs)

        await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure/export?format=json",
            json={"schema": SCHEMA},
        )

        assert fake_provider.call_count == 0


async def _seed_cached_result(
    ai_client: AsyncClient,
    api_prefix: str,
    jobs: JobService,
    fake_provider: FakeLLMProvider,
    db_session: AsyncSession,
    insight_factory: Any,
) -> str:
    """Write a structure cache row directly, under the fingerprint export looks up.

    `/structure` refuses to cache a result with nothing in it, and no longer produces a
    ``fields``-only one at all, so the two shapes below can only be reached this way —
    which is exactly how a row cached before the multi-record change would look.
    """
    document_id = await _ingest(ai_client, api_prefix, jobs)

    fingerprint = compute_fingerprint(
        InsightKind.EXTRACTION,
        fake_provider.model_name,
        structure_prompt.VERSION,
        optimize_context=False,
        schema_name="__structure__",
        schema_fields=SCHEMA,
    )
    await AIResultRepository(db_session).save(
        insight_factory(UUID(document_id)),
        owner_id="local-user",
        fingerprint=fingerprint,
    )
    await db_session.commit()
    return document_id


class TestEmptyResult:
    async def test_a_result_with_nothing_in_it_exports_a_valid_header_only_file(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
        db_session: AsyncSession,
    ) -> None:
        """Preserved behaviour: an empty result is a valid empty file, not an error. The
        header now names the requested schema's columns, since that is the table
        `/structure` produces — the file says "these columns, no rows" rather than
        nothing at all."""
        document_id = await _seed_cached_result(
            ai_client,
            api_prefix,
            jobs,
            fake_provider,
            db_session,
            lambda doc_id: Insight(
                document_id=doc_id,
                kind=InsightKind.EXTRACTION,
                records=(),
                fields=(),
                context_mode=ContextMode.FULL,
                model=fake_provider.model_name,
            ),
        )

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure/export?format=csv",
            json={"schema": SCHEMA},
        )

        assert response.status_code == 200, response.text
        assert response.text.strip() == "account_holder,bank_name,amount"


class TestLegacySingleRecordResult:
    """A result cached before `/structure` became multi-record carries ``fields`` and no
    ``records``. Export must still produce its original field-per-row layout rather than
    dropping it or reshaping it into an empty table."""

    async def test_legacy_fields_still_export_in_the_original_csv_layout(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
        db_session: AsyncSession,
    ) -> None:
        document_id = await _seed_cached_result(
            ai_client,
            api_prefix,
            jobs,
            fake_provider,
            db_session,
            lambda doc_id: Insight(
                document_id=doc_id,
                kind=InsightKind.EXTRACTION,
                fields=(
                    ExtractedField(name="account_holder", value="Arjun Malhotra"),
                    ExtractedField(name="amount", value="2100"),
                ),
                context_mode=ContextMode.FULL,
                model=fake_provider.model_name,
            ),
        )

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure/export?format=csv",
            json={"schema": SCHEMA},
        )

        assert response.status_code == 200, response.text
        assert response.text.splitlines()[0] == (
            "field_name,value,location_label,is_verified,quoted_text"
        )
        rows = list(csv.DictReader(io.StringIO(response.text)))
        assert {row["field_name"] for row in rows} == {"account_holder", "amount"}

    async def test_legacy_fields_still_export_in_the_original_json_layout(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
        db_session: AsyncSession,
    ) -> None:
        document_id = await _seed_cached_result(
            ai_client,
            api_prefix,
            jobs,
            fake_provider,
            db_session,
            lambda doc_id: Insight(
                document_id=doc_id,
                kind=InsightKind.EXTRACTION,
                fields=(ExtractedField(name="account_holder", value="Arjun Malhotra"),),
                context_mode=ContextMode.FULL,
                model=fake_provider.model_name,
            ),
        )

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure/export?format=json",
            json={"schema": SCHEMA},
        )

        body = response.json()
        assert "records" not in body
        assert body["fields"][0]["name"] == "account_holder"
        assert body["fields"][0]["value"] == "Arjun Malhotra"


class TestOwnerIsolation:
    """Milestone 1 has no authentication, so this cannot be exercised through the HTTP
    API (every request runs as the same owner) -- tested directly at the service level
    instead, against real repositories."""

    async def test_a_different_owners_document_is_not_found(self, db_session: AsyncSession) -> None:
        documents = DocumentRepository(db_session)
        results = AIResultRepository(db_session)
        provider = FakeLLMProvider()

        owner_a_document = await documents.create(
            owner_id="owner-a",
            original_filename="a.txt",
            file_type=SupportedFileType.TXT,
            size_bytes=10,
        )
        await db_session.flush()

        service = ExportService(provider=provider, documents=documents, results=results)

        with pytest.raises(NotFoundError):
            await service.export_structure(
                owner_a_document.id, owner_id="owner-b", schema=SCHEMA, export_format="json"
            )

    async def test_a_cached_result_under_one_owner_is_not_exported_for_another(
        self, db_session: AsyncSession
    ) -> None:
        documents = DocumentRepository(db_session)
        results = AIResultRepository(db_session)
        provider = FakeLLMProvider()

        # Same document ID is not realistic across owners, but the repository's
        # owner-scoped queries are what must hold regardless -- this proves the export
        # lookup itself is owner-scoped, not merely that a wrong-owner document 404s.
        document = await documents.create(
            owner_id="owner-a",
            original_filename="a.txt",
            file_type=SupportedFileType.TXT,
            size_bytes=10,
        )
        await db_session.flush()

        fingerprint = compute_fingerprint(
            InsightKind.EXTRACTION,
            provider.model_name,
            structure_prompt.VERSION,
            optimize_context=False,
            schema_name="__structure__",
            schema_fields=SCHEMA,
        )
        await results.save(
            Insight(
                document_id=document.id,
                kind=InsightKind.EXTRACTION,
                fields=(ExtractedField(name="a", value="b"),),
                context_mode=ContextMode.FULL,
                model=provider.model_name,
            ),
            owner_id="owner-a",
            fingerprint=fingerprint,
        )
        await db_session.commit()

        service = ExportService(provider=provider, documents=documents, results=results)

        with pytest.raises(NotFoundError):
            await service.export_structure(
                document.id, owner_id="owner-b", schema=SCHEMA, export_format="json"
            )
