"""End-to-end tests for the document structuring endpoint (Milestone 4).

The language model is replaced by a fake — **no test here calls a real API**. Everything
else is real: a document is uploaded and ingested through the actual HTTP endpoints,
exactly like ``test_analysis_api.py``, which this file otherwise mirrors closely.

``FakeLLMProvider.generate_structured()`` does not validate ``parsed`` against the output
model it was asked for, so a plain ``SimpleNamespace`` stands in for a dynamic structure
result — the service reads it with ``getattr()``, so this is a faithful stand-in without
needing to import ``build_dynamic_record_list_model`` into the test itself. The shape it
must mimic is the record-list one: ``{"records": [ {field: {value, source_quote}}, ... ]}``.

Instruction-mode tests need two different structured replies from one provider (schema
inference, then extraction), which the shared fake does not sequence — hence the small
``_SequencedFakeProvider`` below, mirroring the one in ``test_structure_service.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from pydantic import BaseModel

from app.ai.dynamic_schema import InferredStructureField, InferredStructureSchema
from app.ai.schemas import ExtractedFact, KeyFactsExtraction
from app.api.deps import get_llm_provider
from app.core.exceptions import AIUnavailableError, InvalidModelResponseError
from app.models.domain.insight import ContextBlock, ModelResult
from app.services.job_service import JobService
from tests.fixtures.builders import build_txt
from tests.fixtures.fake_provider import FakeLLMProvider

DOCUMENT_TEXT = (
    "SERVICE AGREEMENT\n\n"
    "This agreement is between Acme Ltd and Globex Inc.\n\n"
    "Invoice number INV-1001 was issued on 2026-03-14.\n\n"
    "The total contract value is $45,200.00 for the year.\n\n"
    "Invoices are due within 30 days of receipt."
)


@dataclass
class _SequencedFakeProvider(FakeLLMProvider):
    """A `FakeLLMProvider` returning a different `.parsed` on each successive
    `generate_structured()` call — needed for instruction mode's two calls."""

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


async def _ingest(client: AsyncClient, api_prefix: str, jobs: JobService) -> str:
    response = await client.post(
        f"{api_prefix}/documents",
        files={"file": ("agreement.txt", build_txt(DOCUMENT_TEXT), "text/plain")},
    )
    assert response.status_code == 202, response.text
    document_id: str = response.json()["id"]

    await jobs.wait_for_all(timeout=30.0)

    status = await client.get(f"{api_prefix}/documents/{document_id}")
    assert status.json()["status"] == "ready", status.text
    return document_id


def _field(value: Any, source_quote: str) -> SimpleNamespace:
    """One requested field's value, in the shape each record carries."""
    return SimpleNamespace(value=value, source_quote=source_quote)


def _records(*records: SimpleNamespace) -> SimpleNamespace:
    """A multi-record structure result, in the shape
    `build_dynamic_record_list_model()` produces."""
    return SimpleNamespace(records=list(records))


@pytest.fixture
def jobs(ai_app: FastAPI) -> JobService:
    return ai_app.state.job_service  # type: ignore[no-any-return]


class TestValidStructureRequest:
    async def test_successful_structured_extraction(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        fake_provider.parsed = _records(
            SimpleNamespace(
                invoice_number=_field("INV-1001", "Invoice number INV-1001 was issued"),
                total=_field(45200.0, "The total contract value is $45,200.00 for the year."),
            )
        )
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"schema": {"invoice_number": "string", "total": "number"}},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["kind"] == "extraction"
        assert body["record_count"] == 1
        by_name = {field["name"]: field for field in body["records"][0]}
        assert by_name["invoice_number"]["value"] == "INV-1001"
        assert by_name["total"]["value"] == "45200.0"

    async def test_every_record_is_returned_not_only_the_first(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        """The single most important test of the multi-record contract: three matching
        entities must produce three records, not one."""
        fake_provider.parsed = _records(
            SimpleNamespace(vendor=_field("Acme Ltd", "Acme Ltd and Globex")),
            SimpleNamespace(vendor=_field("Globex Inc", "Acme Ltd and Globex Inc.")),
            SimpleNamespace(vendor=_field("Third Party", "Invoice number INV-1001")),
        )
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"schema": {"vendor": "string"}},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["record_count"] == 3
        assert len(body["records"]) == 3
        assert [record[0]["value"] for record in body["records"]] == [
            "Acme Ltd",
            "Globex Inc",
            "Third Party",
        ]

    async def test_records_are_not_flattened_into_a_single_record(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        fake_provider.parsed = _records(
            SimpleNamespace(
                vendor=_field("Acme Ltd", "Acme Ltd and Globex"),
                total=_field(45200.0, "The total contract value is $45,200.00 for the year."),
            ),
            SimpleNamespace(
                vendor=_field("Globex Inc", "Acme Ltd and Globex Inc."),
                total=_field(100.0, "Invoice number INV-1001 was issued"),
            ),
        )
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"schema": {"vendor": "string", "total": "number"}},
        )

        body = response.json()
        assert len(body["records"]) == 2
        # Each record keeps its own two fields; nothing is merged across records.
        assert all(len(record) == 2 for record in body["records"])

    async def test_multiple_field_types(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        fake_provider.parsed = _records(
            SimpleNamespace(
                invoice_number=_field("INV-1001", "Invoice number INV-1001"),
                total=_field(45200.0, "The total contract value is $45,200.00 for the year."),
                is_paid=_field(False, "Invoices are due within 30 days of receipt."),
                due_date=_field("2026-04-13", "Invoices are due within 30 days of receipt."),
            )
        )
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={
                "schema": {
                    "invoice_number": "string",
                    "total": "number",
                    "is_paid": "boolean",
                    "due_date": "date",
                }
            },
        )

        assert response.status_code == 200, response.text
        names = {field["name"] for field in response.json()["records"][0]}
        assert names == {"invoice_number", "total", "is_paid", "due_date"}

    async def test_only_the_requested_fields_are_sent_to_the_provider(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        fake_provider.parsed = _records(
            SimpleNamespace(vendor=_field("Acme Ltd", "Acme Ltd and Globex"))
        )
        document_id = await _ingest(ai_client, api_prefix, jobs)

        await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"schema": {"vendor": "string"}},
        )

        assert fake_provider.last_call.enable_citations is False
        assert fake_provider.last_call.output_model.__name__ == "DynamicStructureRecordList"


class TestSchemaUsed:
    async def test_manual_schema_is_echoed_back_unchanged(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        schema = {"invoice_number": "string", "total": "number"}
        fake_provider.parsed = _records(
            SimpleNamespace(invoice_number=_field("INV-1001", "Invoice number INV-1001"))
        )
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure", json={"schema": schema}
        )

        assert response.json()["schema_used"] == schema

    async def test_inferred_schema_is_returned_for_an_instruction(
        self,
        ai_app: FastAPI,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
    ) -> None:
        """Instruction mode's whole point: the caller never named the fields, so the
        schema arrived at must be reported back — it is also what `/structure/export`
        needs to look this result up later."""
        provider = _SequencedFakeProvider(
            parsed_sequence=[
                InferredStructureSchema(
                    fields=[
                        InferredStructureField(name="vendor", type="string"),
                        InferredStructureField(name="total", type="number"),
                    ]
                ),
                _records(SimpleNamespace(vendor=_field("Acme Ltd", "Acme Ltd and Globex"))),
            ]
        )
        ai_app.dependency_overrides[get_llm_provider] = lambda: provider
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"instruction": "List every vendor with its total."},
        )

        assert response.status_code == 200, response.text
        assert response.json()["schema_used"] == {"vendor": "string", "total": "number"}


class TestInstructionMode:
    async def test_instruction_request_succeeds_and_returns_records(
        self,
        ai_app: FastAPI,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
    ) -> None:
        provider = _SequencedFakeProvider(
            parsed_sequence=[
                InferredStructureSchema(
                    fields=[InferredStructureField(name="vendor", type="string")]
                ),
                _records(
                    SimpleNamespace(vendor=_field("Acme Ltd", "Acme Ltd and Globex")),
                    SimpleNamespace(vendor=_field("Globex Inc", "Acme Ltd and Globex Inc.")),
                ),
            ]
        )
        ai_app.dependency_overrides[get_llm_provider] = lambda: provider
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"instruction": "One row per vendor mentioned in the agreement."},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["record_count"] == 2
        assert provider.call_count == 2

    async def test_an_instruction_that_infers_an_invalid_schema_is_rejected(
        self,
        ai_app: FastAPI,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
    ) -> None:
        """An inferred schema gets no special trust — a bad field name from the model is
        rejected exactly like a caller-supplied one, as a 422."""
        provider = _SequencedFakeProvider(
            parsed_sequence=[
                InferredStructureSchema(
                    fields=[InferredStructureField(name="Bad Name", type="string")]
                )
            ]
        )
        ai_app.dependency_overrides[get_llm_provider] = lambda: provider
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"instruction": "Anything at all."},
        )

        assert response.status_code == 422, response.text
        assert response.json()["error"]["code"] == "invalid_structure_schema"


class TestRequestInputValidation:
    async def test_neither_schema_nor_instruction_is_rejected(
        self, ai_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(f"{api_prefix}/documents/{document_id}/structure", json={})

        assert response.status_code == 422, response.text

    async def test_both_schema_and_instruction_is_rejected(
        self, ai_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        """Never silently preferring one: a caller who sent both would otherwise get a
        result that ignores half of what they asked for, without being told."""
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"schema": {"vendor": "string"}, "instruction": "List the vendors."},
        )

        assert response.status_code == 422, response.text

    async def test_blank_instruction_is_rejected(
        self, ai_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"instruction": "   "},
        )

        assert response.status_code == 422, response.text

    async def test_an_overly_long_instruction_is_rejected(
        self, ai_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"instruction": "a" * 2001},
        )

        assert response.status_code == 422, response.text


class TestSourceQuoteVerification:
    async def test_a_real_quote_is_verified(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        fake_provider.parsed = _records(
            SimpleNamespace(
                total=_field(45200.0, "The total contract value is $45,200.00 for the year.")
            )
        )
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"schema": {"total": "number"}},
        )

        citation = response.json()["records"][0][0]["citation"]
        assert citation is not None
        assert citation["is_verified"] is True

    async def test_each_record_keeps_its_own_citation(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        """Citations must survive per record — one record's verified source must never
        stand in for another's."""
        fake_provider.parsed = _records(
            SimpleNamespace(vendor=_field("Acme Ltd", "Acme Ltd and Globex Inc.")),
            SimpleNamespace(vendor=_field("Invented Co", "A clause this document never states")),
        )
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"schema": {"vendor": "string"}},
        )

        body = response.json()
        assert body["records"][0][0]["citation"]["is_verified"] is True
        assert body["records"][1][0]["citation"] is None

    async def test_an_invented_quote_gets_no_source(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        """The single most important test in this file: a fabricated quote must produce
        a field with **no** verified location, not a nearby one."""
        fake_provider.parsed = _records(
            SimpleNamespace(penalty=_field("15% per month", "A late fee of 15% per month applies."))
        )
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"schema": {"penalty": "string"}},
        )

        body = response.json()
        assert body["records"][0][0]["citation"] is None
        assert any("could not be traced" in warning for warning in body["warnings"])

    async def test_a_field_the_model_omits_is_left_out_not_invented(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        fake_provider.parsed = _records(
            SimpleNamespace(
                invoice_number=_field("INV-1001", "Invoice number INV-1001")
                # "vendor" requested but not present -- the model found nothing for it.
            )
        )
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"schema": {"invoice_number": "string", "vendor": "string"}},
        )

        names = {field["name"] for field in response.json()["records"][0]}
        assert names == {"invoice_number"}


class TestMissingOrUnreadyDocument:
    async def test_nonexistent_document_returns_404(
        self, ai_client: AsyncClient, api_prefix: str
    ) -> None:
        response = await ai_client.post(
            f"{api_prefix}/documents/{uuid4()}/structure",
            json={"schema": {"a": "string"}},
        )

        assert response.status_code == 404, response.text
        assert response.json()["error"]["code"] == "not_found"

    async def test_no_records_found_is_reported_as_insufficient_content(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        fake_provider.parsed = _records()  # the model found no matching records at all
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"schema": {"invoice_number": "string"}},
        )

        assert response.status_code == 422, response.text
        assert response.json()["error"]["code"] == "insufficient_content"


class TestProviderBehavior:
    async def test_provider_not_configured_is_reported_clearly(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        document_id = await _ingest(ai_client, api_prefix, jobs)
        fake_provider.configured = False

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"schema": {"a": "string"}},
        )

        assert response.status_code == 500, response.text
        assert response.json()["error"]["code"] == "ai_not_configured"

    async def test_provider_failure_is_safe_for_the_client(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        document_id = await _ingest(ai_client, api_prefix, jobs)
        fake_provider.error = AIUnavailableError("The AI service is temporarily unavailable.")

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"schema": {"a": "string"}},
        )

        assert response.status_code == 500, response.text
        assert "request_id" in response.json()["error"]

    async def test_malformed_structured_response_is_reported_not_repaired(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        document_id = await _ingest(ai_client, api_prefix, jobs)
        fake_provider.error = InvalidModelResponseError(
            "The AI returned a response that did not match the expected format."
        )

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"schema": {"a": "string"}},
        )

        assert response.status_code == 500, response.text


class TestSchemaRejection:
    async def test_empty_schema_is_rejected(
        self, ai_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure", json={"schema": {}}
        )

        assert response.status_code == 422, response.text

    async def test_invalid_field_name_is_rejected(
        self, ai_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"schema": {"Invoice-Number": "string"}},
        )

        assert response.status_code == 422, response.text

    async def test_reserved_field_name_is_rejected(
        self, ai_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"schema": {"model_config": "string"}},
        )

        assert response.status_code == 422, response.text

    async def test_unsupported_type_is_rejected(
        self, ai_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"schema": {"tags": "array"}},
        )

        assert response.status_code == 422, response.text

    async def test_too_many_fields_is_rejected(
        self, ai_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"schema": {f"field_{i}": "string" for i in range(41)}},
        )

        assert response.status_code == 422, response.text


class TestCaching:
    async def test_a_second_identical_request_is_served_from_cache(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        fake_provider.parsed = _records(
            SimpleNamespace(vendor=_field("Acme Ltd", "Acme Ltd and Globex"))
        )
        document_id = await _ingest(ai_client, api_prefix, jobs)
        schema = {"vendor": "string"}

        first = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure", json={"schema": schema}
        )
        calls_after_first = fake_provider.call_count
        second = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure", json={"schema": schema}
        )

        assert first.json()["from_cache"] is False
        assert second.json()["from_cache"] is True
        assert fake_provider.call_count == calls_after_first

    async def test_a_cached_result_still_returns_its_records(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        """Records must survive the cache round trip — they are stored in their own
        column and rebuilt on read, so a cache hit must not come back empty."""
        fake_provider.parsed = _records(
            SimpleNamespace(vendor=_field("Acme Ltd", "Acme Ltd and Globex")),
            SimpleNamespace(vendor=_field("Globex Inc", "Acme Ltd and Globex Inc.")),
        )
        document_id = await _ingest(ai_client, api_prefix, jobs)
        schema = {"vendor": "string"}

        first = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure", json={"schema": schema}
        )
        second = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure", json={"schema": schema}
        )

        assert second.json()["from_cache"] is True
        assert second.json()["records"] == first.json()["records"]
        assert second.json()["record_count"] == 2
        assert second.json()["schema_used"] == schema

    async def test_refresh_bypasses_the_cache(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        fake_provider.parsed = _records(
            SimpleNamespace(vendor=_field("Acme Ltd", "Acme Ltd and Globex"))
        )
        document_id = await _ingest(ai_client, api_prefix, jobs)
        schema = {"vendor": "string"}

        await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure", json={"schema": schema}
        )
        calls_after_first = fake_provider.call_count
        refreshed = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"schema": schema, "refresh": True},
        )

        assert refreshed.json()["from_cache"] is False
        assert fake_provider.call_count == calls_after_first + 1

    async def test_field_order_does_not_bypass_the_cache(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        fake_provider.parsed = _records(
            SimpleNamespace(
                a=_field("x", "Acme Ltd and Globex"), b=_field("y", "Acme Ltd and Globex")
            )
        )
        document_id = await _ingest(ai_client, api_prefix, jobs)

        await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"schema": {"a": "string", "b": "string"}},
        )
        calls_after_first = fake_provider.call_count
        second = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"schema": {"b": "string", "a": "string"}},
        )

        assert second.json()["from_cache"] is True
        assert fake_provider.call_count == calls_after_first

    async def test_a_different_schema_is_not_served_from_cache(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        fake_provider.parsed = _records(
            SimpleNamespace(vendor=_field("Acme Ltd", "Acme Ltd and Globex"))
        )
        document_id = await _ingest(ai_client, api_prefix, jobs)

        await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"schema": {"vendor": "string"}},
        )
        calls_after_first = fake_provider.call_count
        fake_provider.parsed = _records(
            SimpleNamespace(total=_field(45200.0, "$45,200.00 for the year"))
        )
        second = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"schema": {"total": "number"}},
        )

        assert second.json()["from_cache"] is False
        assert fake_provider.call_count == calls_after_first + 1


class TestExistingExtractionIsUnaffected:
    async def test_named_extraction_still_works_unchanged(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        """The multi-record /structure contract must not disturb /extraction's existing
        single-record response shape."""
        fake_provider.parsed = KeyFactsExtraction(
            facts=[
                ExtractedFact(
                    name="total_value",
                    value="$45,200.00",
                    source_quote="The total contract value is $45,200.00 for the year.",
                )
            ]
        )
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(f"{api_prefix}/documents/{document_id}/extraction")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["kind"] == "extraction"
        assert body["fields"][0]["name"] == "total_value"
        # /extraction keeps using `fields` and gains no `records` key.
        assert "records" not in body
        assert fake_provider.last_call.output_model is KeyFactsExtraction

    async def test_structure_and_extraction_are_cached_independently(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        document_id = await _ingest(ai_client, api_prefix, jobs)

        fake_provider.parsed = KeyFactsExtraction(
            facts=[ExtractedFact(name="a", value="b", source_quote="Acme Ltd and Globex")]
        )
        await ai_client.post(f"{api_prefix}/documents/{document_id}/extraction")
        calls_after_extraction = fake_provider.call_count

        fake_provider.parsed = _records(
            SimpleNamespace(vendor=_field("Acme Ltd", "Acme Ltd and Globex"))
        )
        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/structure",
            json={"schema": {"vendor": "string"}},
        )

        assert response.json()["from_cache"] is False
        assert fake_provider.call_count == calls_after_extraction + 1
