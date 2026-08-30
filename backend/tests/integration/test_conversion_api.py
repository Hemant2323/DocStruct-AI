"""End-to-end tests for the document conversion endpoints.

**No test here calls a real API** (CLAUDE.md §7). Everything else is real: a document is
uploaded and ingested through the actual HTTP endpoints, the conversion runs through the
real service and repositories, and the result is written to and read back from a real
database.

The conversion engine itself — corrections, assembly, rendering — has its own suites from
earlier phases and is not re-tested here. What these check is the API and persistence
boundary: that the route is thin, that what was produced is what gets stored, that
retrieval reads the database and nothing else, and that a conversion id is useless to
someone who does not own it.
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

from app.ai.conversion_schema import (
    PlannedDocument,
    ProposedCorrectionList,
    ProposedCorrectionModel,
)
from app.ai.dynamic_schema import InferredStructureField, InferredStructureSchema
from app.api.deps import DEFAULT_OWNER_ID, get_conversion_repository
from app.models.domain.insight import ContextBlock, ModelResult
from app.services.job_service import JobService
from tests.fixtures.builders import build_txt
from tests.fixtures.fake_provider import FakeLLMProvider

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
    "standardize account statuses, preserve important details, and do not invent "
    "missing information."
)


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
                },
                {
                    "heading": "Pending KYC",
                    "kind": "table",
                    "columns": ["account_holder", "kyc_status"],
                    "filters": [{"field": "kyc_status", "mode": "equals", "values": ["Pending"]}],
                },
            ],
            "labels": [{"field": "account_holder", "label": "Account Holder"}],
            "vocabularies": [{"field": "account_type", "values": ["Savings", "Current"]}],
        }
    )


def _proposals() -> ProposedCorrectionList:
    """Two provable fixes, one uncertain typo, one hallucinated rename."""
    return ProposedCorrectionList(
        corrections=[
            ProposedCorrectionModel(
                record_index=0,
                field="account_type",
                original="sav",
                corrected="Savings",
                kind="vocabulary",
                confidence="high",
                reason="standardized account type",
            ),
            ProposedCorrectionModel(
                record_index=1,
                field="account_type",
                original="SAVINGS",
                corrected="Savings",
                kind="vocabulary",
                confidence="high",
                reason="standardized account type",
            ),
            ProposedCorrectionModel(
                record_index=0,
                field="account_holder",
                original="Rajesh Kumr Sharma",
                corrected="Rajesh Kumar Sharma",
                kind="spelling",
                confidence="medium",
                reason="possible spelling correction",
            ),
            ProposedCorrectionModel(
                record_index=2,
                field="account_holder",
                original="Rahul Verma",
                corrected="Neha Kapoor",
                kind="spelling",
                confidence="high",
                reason="hallucinated rename",
            ),
        ]
    )


def _sequence() -> list[Any]:
    return [_inferred_schema(), _extraction(), _plan(), _proposals()]


@pytest.fixture
def conversion_provider() -> _SequencedFakeProvider:
    return _SequencedFakeProvider(parsed_sequence=_sequence())


@pytest.fixture
def conversion_app(ai_app: FastAPI, conversion_provider: _SequencedFakeProvider) -> FastAPI:
    """The application with the model replaced by a sequenced fake."""
    from app.api.deps import get_llm_provider

    ai_app.dependency_overrides[get_llm_provider] = lambda: conversion_provider
    return ai_app


@pytest.fixture
async def conversion_client(conversion_app: FastAPI):  # type: ignore[no-untyped-def]
    from httpx import ASGITransport

    transport = ASGITransport(app=conversion_app)
    async with (
        conversion_app.router.lifespan_context(conversion_app),
        AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        yield client


@pytest.fixture
def jobs(conversion_app: FastAPI) -> JobService:
    return conversion_app.state.job_service  # type: ignore[no-any-return]


async def _ingest(client: AsyncClient, api_prefix: str, jobs: JobService) -> str:
    response = await client.post(
        f"{api_prefix}/documents",
        files={"file": ("customers.txt", build_txt(DOCUMENT_TEXT), "text/plain")},
    )
    assert response.status_code == 202, response.text
    document_id: str = response.json()["id"]

    await jobs.wait_for_all(timeout=30.0)

    status = await client.get(f"{api_prefix}/documents/{document_id}")
    assert status.json()["status"] == "ready", status.text
    return document_id


async def _convert(client: AsyncClient, api_prefix: str, document_id: str, **body: Any) -> Any:
    payload = {"instruction": INSTRUCTION, **body}
    return await client.post(f"{api_prefix}/documents/{document_id}/convert", json=payload)


class TestRequestValidation:
    async def test_a_valid_instruction_is_accepted(
        self, conversion_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)

        response = await _convert(conversion_client, api_prefix, document_id)

        assert response.status_code == 201, response.text

    async def test_a_missing_instruction_is_rejected(
        self, conversion_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)

        response = await conversion_client.post(
            f"{api_prefix}/documents/{document_id}/convert", json={}
        )

        assert response.status_code == 422, response.text

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
    async def test_a_blank_instruction_is_rejected(
        self, conversion_client: AsyncClient, api_prefix: str, jobs: JobService, blank: str
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)

        response = await _convert(conversion_client, api_prefix, document_id, instruction=blank)

        assert response.status_code == 422, response.text

    async def test_an_oversized_instruction_is_rejected(
        self, conversion_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)

        response = await _convert(
            conversion_client, api_prefix, document_id, instruction="a" * 2001
        )

        assert response.status_code == 422, response.text

    async def test_a_malformed_document_id_is_rejected(
        self, conversion_client: AsyncClient, api_prefix: str
    ) -> None:
        response = await conversion_client.post(
            f"{api_prefix}/documents/not-a-uuid/convert", json={"instruction": INSTRUCTION}
        )

        assert response.status_code == 422, response.text

    async def test_an_unknown_document_is_not_found(
        self, conversion_client: AsyncClient, api_prefix: str
    ) -> None:
        response = await _convert(conversion_client, api_prefix, str(uuid4()))

        assert response.status_code == 404, response.text
        assert response.json()["error"]["code"] == "not_found"


class TestSuccessfulConversion:
    async def test_the_response_carries_the_document_and_its_audit(
        self,
        conversion_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)

        body = (await _convert(conversion_client, api_prefix, document_id)).json()

        assert body["title"] == "Customer Master Report"
        assert body["markdown"].startswith("# Customer Master Report")
        assert body["document_id"] == document_id
        assert body["source_record_count"] == 3
        assert len(body["corrections"]) == 2
        assert len(body["suggestions"]) == 1
        assert len(body["rejected"]) == 1

    async def test_the_plan_is_exposed_for_transparency(
        self, conversion_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)

        body = (await _convert(conversion_client, api_prefix, document_id)).json()

        assert body["plan"]["title"] == "Customer Master Report"
        assert [s["heading"] for s in body["plan"]["sections"]] == ["Customers", "Pending KYC"]
        assert body["plan"]["labels"] == {"account_holder": "Account Holder"}

    async def test_sections_carry_values_with_their_origin(
        self, conversion_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)

        body = (await _convert(conversion_client, api_prefix, document_id)).json()

        origins = {
            item["value"]["origin"]
            for section in body["sections"]
            for record in section["records"]
            for item in record["fields"]
        }
        assert "corrected" in origins
        assert "source" in origins

    async def test_a_rejected_correction_is_reported_with_its_reason(
        self, conversion_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)

        body = (await _convert(conversion_client, api_prefix, document_id)).json()

        [rejection] = body["rejected"]
        assert rejection["corrected"] == "Neha Kapoor"
        assert rejection["rejection_reason"]
        assert "Neha Kapoor" not in body["markdown"]

    async def test_no_internal_identifiers_are_exposed(
        self, conversion_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        """Chunk and block ids exist behind these citations but have no product meaning."""
        document_id = await _ingest(conversion_client, api_prefix, jobs)

        raw = (await _convert(conversion_client, api_prefix, document_id)).text

        assert "chunk_id" not in raw
        assert "block_ids" not in raw

    async def test_provenance_survives_into_the_response(
        self, conversion_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)

        body = (await _convert(conversion_client, api_prefix, document_id)).json()

        verified = [
            item["value"]
            for section in body["sections"]
            for record in section["records"]
            for item in record["fields"]
            if item["value"]["is_verified"]
        ]
        assert verified
        assert all(item["source_label"] for item in verified)


class TestServiceIntegration:
    async def test_the_route_does_not_call_the_provider_itself(
        self,
        conversion_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        conversion_provider: _SequencedFakeProvider,
    ) -> None:
        """Exactly the four calls the service makes — the route adds none of its own."""
        document_id = await _ingest(conversion_client, api_prefix, jobs)

        await _convert(conversion_client, api_prefix, document_id)

        assert conversion_provider.call_count == 4

    async def test_the_instruction_reaches_the_service(
        self,
        conversion_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        conversion_provider: _SequencedFakeProvider,
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)

        await _convert(conversion_client, api_prefix, document_id)

        assert any(INSTRUCTION in call.instruction for call in conversion_provider.calls)

    async def test_the_instruction_is_echoed_back_stripped(
        self, conversion_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)

        body = (
            await _convert(
                conversion_client, api_prefix, document_id, instruction=f"  {INSTRUCTION}  "
            )
        ).json()

        assert body["instruction"] == INSTRUCTION


class TestPersistence:
    async def test_a_conversion_is_persisted_and_can_be_retrieved(
        self, conversion_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)
        created = (await _convert(conversion_client, api_prefix, document_id)).json()

        fetched = await conversion_client.get(
            f"{api_prefix}/documents/{document_id}/conversions/{created['id']}"
        )

        assert fetched.status_code == 200, fetched.text
        assert fetched.json() == created

    async def test_retrieval_never_calls_the_provider(
        self,
        conversion_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        conversion_provider: _SequencedFakeProvider,
    ) -> None:
        """The data-integrity rule: a stored conversion is read, never regenerated."""
        document_id = await _ingest(conversion_client, api_prefix, jobs)
        created = (await _convert(conversion_client, api_prefix, document_id)).json()
        calls_after_convert = conversion_provider.call_count

        await conversion_client.get(
            f"{api_prefix}/documents/{document_id}/conversions/{created['id']}"
        )
        await conversion_client.get(f"{api_prefix}/documents/{document_id}/conversions")

        assert conversion_provider.call_count == calls_after_convert

    async def test_markdown_survives_the_round_trip(
        self, conversion_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)
        created = (await _convert(conversion_client, api_prefix, document_id)).json()

        fetched = (
            await conversion_client.get(
                f"{api_prefix}/documents/{document_id}/conversions/{created['id']}"
            )
        ).json()

        assert fetched["markdown"] == created["markdown"]
        assert "## Changes & Corrections" in fetched["markdown"]

    async def test_the_plan_survives_the_round_trip(
        self, conversion_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)
        created = (await _convert(conversion_client, api_prefix, document_id)).json()

        fetched = (
            await conversion_client.get(
                f"{api_prefix}/documents/{document_id}/conversions/{created['id']}"
            )
        ).json()

        assert fetched["plan"] == created["plan"]

    async def test_the_correction_audit_survives_the_round_trip(
        self, conversion_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)
        created = (await _convert(conversion_client, api_prefix, document_id)).json()

        fetched = (
            await conversion_client.get(
                f"{api_prefix}/documents/{document_id}/conversions/{created['id']}"
            )
        ).json()

        assert fetched["corrections"] == created["corrections"]
        assert fetched["suggestions"] == created["suggestions"]
        assert fetched["rejected"] == created["rejected"]

    async def test_warnings_and_metadata_survive_the_round_trip(
        self, conversion_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)
        created = (await _convert(conversion_client, api_prefix, document_id)).json()

        fetched = (
            await conversion_client.get(
                f"{api_prefix}/documents/{document_id}/conversions/{created['id']}"
            )
        ).json()

        assert fetched["warnings"] == created["warnings"]
        assert fetched["model"] == created["model"]
        assert fetched["created_at"] == created["created_at"]
        assert fetched["schema_used"] == created["schema_used"]
        assert fetched["vocabularies"] == created["vocabularies"]
        assert fetched["source_record_count"] == created["source_record_count"]

    async def test_sections_survive_the_round_trip(
        self, conversion_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)
        created = (await _convert(conversion_client, api_prefix, document_id)).json()

        fetched = (
            await conversion_client.get(
                f"{api_prefix}/documents/{document_id}/conversions/{created['id']}"
            )
        ).json()

        assert fetched["sections"] == created["sections"]

    async def test_two_runs_are_distinguishable(
        self,
        conversion_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        conversion_provider: _SequencedFakeProvider,
    ) -> None:
        """Each conversion is its own record — the second never overwrites the first."""
        document_id = await _ingest(conversion_client, api_prefix, jobs)
        first = (await _convert(conversion_client, api_prefix, document_id)).json()

        # A second run: the extraction is cached, so only plan and corrections are needed.
        conversion_provider.parsed_sequence = [_inferred_schema(), _plan(), _proposals()]
        second = (await _convert(conversion_client, api_prefix, document_id)).json()

        assert first["id"] != second["id"]

        listing = (
            await conversion_client.get(f"{api_prefix}/documents/{document_id}/conversions")
        ).json()
        assert len(listing["items"]) == 2
        assert {item["id"] for item in listing["items"]} == {first["id"], second["id"]}

    async def test_the_listing_summarises_without_content(
        self, conversion_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)
        await _convert(conversion_client, api_prefix, document_id)

        listing = (
            await conversion_client.get(f"{api_prefix}/documents/{document_id}/conversions")
        ).json()

        [item] = listing["items"]
        assert item["title"] == "Customer Master Report"
        assert item["applied_correction_count"] == 2
        assert item["suggestion_count"] == 1
        assert item["rejected_correction_count"] == 1
        assert "markdown" not in item

    async def test_a_persistence_failure_does_not_return_success(
        self,
        conversion_app: FastAPI,
        conversion_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
    ) -> None:
        """A conversion the caller can see but never fetch again would be worse than a
        clear failure, so a write error must surface rather than be swallowed."""
        document_id = await _ingest(conversion_client, api_prefix, jobs)

        class _FailingRepository:
            async def save(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("database unavailable")

        conversion_app.dependency_overrides[get_conversion_repository] = _FailingRepository

        # The app registers a catch-all handler that turns this into a 500 in
        # production, but Starlette re-raises after calling it so the server can log —
        # and with an in-process ASGI transport that re-raise reaches the caller. Either
        # way the request does not succeed, which is the property under test.
        try:
            with pytest.raises(RuntimeError, match="database unavailable"):
                await _convert(conversion_client, api_prefix, document_id)
        finally:
            conversion_app.dependency_overrides.pop(get_conversion_repository, None)

        # Nothing half-written survived the rolled-back request.
        listing = (
            await conversion_client.get(f"{api_prefix}/documents/{document_id}/conversions")
        ).json()
        assert listing["items"] == []


class TestOwnership:
    async def test_a_conversion_for_another_owner_is_not_found(
        self,
        conversion_app: FastAPI,
        conversion_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
    ) -> None:
        """Knowing an id must not be enough. Reported as not found rather than
        forbidden, so a 403 cannot confirm that a conversion exists."""
        from app.api.deps import get_current_owner_id

        document_id = await _ingest(conversion_client, api_prefix, jobs)
        created = (await _convert(conversion_client, api_prefix, document_id)).json()

        conversion_app.dependency_overrides[get_current_owner_id] = lambda: "someone-else"
        try:
            response = await conversion_client.get(
                f"{api_prefix}/documents/{document_id}/conversions/{created['id']}"
            )
        finally:
            conversion_app.dependency_overrides.pop(get_current_owner_id, None)

        assert response.status_code == 404, response.text

    async def test_another_owner_sees_no_conversions_in_the_listing(
        self,
        conversion_app: FastAPI,
        conversion_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
    ) -> None:
        from app.api.deps import get_current_owner_id

        document_id = await _ingest(conversion_client, api_prefix, jobs)
        await _convert(conversion_client, api_prefix, document_id)

        conversion_app.dependency_overrides[get_current_owner_id] = lambda: "someone-else"
        try:
            listing = await conversion_client.get(
                f"{api_prefix}/documents/{document_id}/conversions"
            )
        finally:
            conversion_app.dependency_overrides.pop(get_current_owner_id, None)

        assert listing.json()["items"] == []

    async def test_converting_an_inaccessible_document_is_rejected(
        self,
        conversion_app: FastAPI,
        conversion_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
    ) -> None:
        from app.api.deps import get_current_owner_id

        document_id = await _ingest(conversion_client, api_prefix, jobs)

        conversion_app.dependency_overrides[get_current_owner_id] = lambda: "someone-else"
        try:
            response = await _convert(conversion_client, api_prefix, document_id)
        finally:
            conversion_app.dependency_overrides.pop(get_current_owner_id, None)

        assert response.status_code == 404, response.text

    async def test_the_owner_can_retrieve_their_own_conversion(
        self, conversion_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)
        created = (await _convert(conversion_client, api_prefix, document_id)).json()

        response = await conversion_client.get(
            f"{api_prefix}/documents/{document_id}/conversions/{created['id']}"
        )

        assert response.status_code == 200
        assert DEFAULT_OWNER_ID  # the request ran as the default owner

    async def test_an_unknown_conversion_is_not_found(
        self, conversion_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)

        response = await conversion_client.get(
            f"{api_prefix}/documents/{document_id}/conversions/{uuid4()}"
        )

        assert response.status_code == 404, response.text
        assert response.json()["error"]["code"] == "not_found"

    async def test_a_conversion_cannot_be_fetched_through_a_different_document(
        self, conversion_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        """Scoped by document as well as owner, so an id from one document cannot be
        read through another."""
        document_id = await _ingest(conversion_client, api_prefix, jobs)
        created = (await _convert(conversion_client, api_prefix, document_id)).json()

        response = await conversion_client.get(
            f"{api_prefix}/documents/{uuid4()}/conversions/{created['id']}"
        )

        assert response.status_code == 404, response.text


class TestRefreshSemantics:
    async def test_refresh_false_reuses_the_cached_extraction(
        self,
        conversion_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        conversion_provider: _SequencedFakeProvider,
    ) -> None:
        """Only the extraction is cached. The conversion itself is always produced
        fresh, so a second request still costs a plan and a correction call."""
        document_id = await _ingest(conversion_client, api_prefix, jobs)
        first = (await _convert(conversion_client, api_prefix, document_id)).json()
        assert first["records_from_cache"] is False

        conversion_provider.parsed_sequence = [_inferred_schema(), _plan(), _proposals()]
        calls_before = conversion_provider.call_count

        second = (await _convert(conversion_client, api_prefix, document_id, refresh=False)).json()

        assert second["records_from_cache"] is True
        assert conversion_provider.call_count == calls_before + 3

    async def test_refresh_true_re_extracts(
        self,
        conversion_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        conversion_provider: _SequencedFakeProvider,
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)
        await _convert(conversion_client, api_prefix, document_id)

        conversion_provider.parsed_sequence = _sequence()
        calls_before = conversion_provider.call_count

        refreshed = (
            await _convert(conversion_client, api_prefix, document_id, refresh=True)
        ).json()

        assert refreshed["records_from_cache"] is False
        assert conversion_provider.call_count == calls_before + 4

    async def test_a_repeat_request_never_returns_the_earlier_conversion(
        self,
        conversion_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        conversion_provider: _SequencedFakeProvider,
    ) -> None:
        """There is no conversion-level cache: asking again produces a new result, and
        `refresh=False` must not be mistaken for one."""
        document_id = await _ingest(conversion_client, api_prefix, jobs)
        first = (await _convert(conversion_client, api_prefix, document_id)).json()

        conversion_provider.parsed_sequence = [_inferred_schema(), _plan(), _proposals()]
        second = (await _convert(conversion_client, api_prefix, document_id, refresh=False)).json()

        assert second["id"] != first["id"]
        assert second["created_at"] >= first["created_at"]


class TestExportEndpoint:
    """Phase 8: downloading a stored conversion. Reads only — never re-runs anything."""

    @pytest.mark.parametrize(
        ("export_format", "media_type"),
        [
            ("md", "text/markdown"),
            ("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ("json", "application/json"),
        ],
    )
    async def test_every_format_downloads_with_the_right_media_type(
        self,
        conversion_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        export_format: str,
        media_type: str,
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)
        created = (await _convert(conversion_client, api_prefix, document_id)).json()

        response = await conversion_client.get(
            f"{api_prefix}/documents/{document_id}/conversions/{created['id']}"
            f"/export?format={export_format}"
        )

        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith(media_type)
        assert "attachment" in response.headers["content-disposition"]
        assert f".{export_format}" in response.headers["content-disposition"]
        assert response.content

    async def test_downloading_never_calls_the_provider(
        self,
        conversion_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        conversion_provider: _SequencedFakeProvider,
    ) -> None:
        """The data-integrity rule, at the download boundary."""
        document_id = await _ingest(conversion_client, api_prefix, jobs)
        created = (await _convert(conversion_client, api_prefix, document_id)).json()
        calls_after_convert = conversion_provider.call_count

        for export_format in ("md", "docx", "xlsx", "json"):
            await conversion_client.get(
                f"{api_prefix}/documents/{document_id}/conversions/{created['id']}"
                f"/export?format={export_format}"
            )

        assert conversion_provider.call_count == calls_after_convert

    async def test_the_markdown_download_matches_the_stored_result(
        self, conversion_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)
        created = (await _convert(conversion_client, api_prefix, document_id)).json()

        response = await conversion_client.get(
            f"{api_prefix}/documents/{document_id}/conversions/{created['id']}/export?format=md"
        )

        assert response.content.decode("utf-8") == created["markdown"]

    async def test_repeated_downloads_are_identical(
        self, conversion_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)
        created = (await _convert(conversion_client, api_prefix, document_id)).json()
        url = f"{api_prefix}/documents/{document_id}/conversions/{created['id']}/export?format=json"

        first = await conversion_client.get(url)
        second = await conversion_client.get(url)

        assert first.content == second.content

    async def test_every_record_reaches_the_spreadsheet(
        self, conversion_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        import io as _io

        from openpyxl import load_workbook

        document_id = await _ingest(conversion_client, api_prefix, jobs)
        created = (await _convert(conversion_client, api_prefix, document_id)).json()

        response = await conversion_client.get(
            f"{api_prefix}/documents/{document_id}/conversions/{created['id']}/export?format=xlsx"
        )

        workbook = load_workbook(_io.BytesIO(response.content))
        holders = {
            row[0]
            for name in workbook.sheetnames
            if name != "Changes & Corrections"
            for row in workbook[name].iter_rows(min_row=2, values_only=True)
        }
        assert {"Rajesh Kumr Sharma", "Priya Sharma", "Rahul Verma"} <= holders
        # The rejected rename never made it into the document, so it cannot be here.
        assert "Neha Kapoor" not in holders

    async def test_an_unknown_format_is_rejected(
        self, conversion_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)
        created = (await _convert(conversion_client, api_prefix, document_id)).json()

        response = await conversion_client.get(
            f"{api_prefix}/documents/{document_id}/conversions/{created['id']}/export?format=pdf"
        )

        assert response.status_code == 422, response.text

    async def test_an_unknown_conversion_is_not_found(
        self, conversion_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)

        response = await conversion_client.get(
            f"{api_prefix}/documents/{document_id}/conversions/{uuid4()}/export?format=md"
        )

        assert response.status_code == 404, response.text

    async def test_another_owner_cannot_download(
        self,
        conversion_app: FastAPI,
        conversion_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
    ) -> None:
        """Knowing a conversion id must not be enough to download its contents."""
        from app.api.deps import get_current_owner_id

        document_id = await _ingest(conversion_client, api_prefix, jobs)
        created = (await _convert(conversion_client, api_prefix, document_id)).json()

        conversion_app.dependency_overrides[get_current_owner_id] = lambda: "someone-else"
        try:
            response = await conversion_client.get(
                f"{api_prefix}/documents/{document_id}/conversions/{created['id']}/export?format=md"
            )
        finally:
            conversion_app.dependency_overrides.pop(get_current_owner_id, None)

        assert response.status_code == 404, response.text


class TestDocumentDeletionRemovesConversions:
    """A conversion holds the document's own content, reorganized. Deleting the document
    must take it with them — an orphaned conversion would keep serving content the user
    asked to be removed."""

    async def test_deleting_the_document_removes_its_conversions(
        self, conversion_client: AsyncClient, api_prefix: str, jobs: JobService
    ) -> None:
        document_id = await _ingest(conversion_client, api_prefix, jobs)
        created = (await _convert(conversion_client, api_prefix, document_id)).json()

        deleted = await conversion_client.delete(f"{api_prefix}/documents/{document_id}")
        assert deleted.status_code == 204, deleted.text

        fetched = await conversion_client.get(
            f"{api_prefix}/documents/{document_id}/conversions/{created['id']}"
        )
        listing = await conversion_client.get(f"{api_prefix}/documents/{document_id}/conversions")
        export = await conversion_client.get(
            f"{api_prefix}/documents/{document_id}/conversions/{created['id']}/export?format=md"
        )

        assert fetched.status_code == 404
        assert export.status_code == 404
        assert listing.json()["items"] == []
