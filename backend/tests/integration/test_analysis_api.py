"""End-to-end tests for the AI analysis endpoints.

The language model is replaced by a fake — **no test here calls the real API**. Everything
else is real: a document is uploaded, ingested by the Milestone 2 pipeline, and analysed
through the actual HTTP endpoints.

The tests that matter most are the provenance ones. They prove that a source location
shown to a user was resolved from stored content, and that anything unresolvable is
reported as unverified rather than dressed up with a plausible page number.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.ai.schemas import ExtractedFact, KeyFactsExtraction
from app.core.exceptions import AIUnavailableError
from app.models.domain.insight import RawCitation
from app.services.job_service import JobService
from tests.fixtures.builders import build_txt
from tests.fixtures.fake_provider import FakeLLMProvider

DOCUMENT_TEXT = (
    "SERVICE AGREEMENT\n\n"
    "This agreement is between Acme Ltd and Globex Inc.\n\n"
    "Invoices are due within 30 days of receipt.\n\n"
    "The total contract value is $45,200.00 for the year.\n\n"
    "Either party may terminate with 60 days written notice."
)


async def _ingest(client: AsyncClient, api_prefix: str, jobs: JobService) -> str:
    """Upload and fully process a document, returning its id."""
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


@pytest.fixture
def jobs(ai_app: FastAPI) -> JobService:
    return ai_app.state.job_service  # type: ignore[no-any-return]


class TestSummary:
    async def test_produces_a_summary(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        fake_provider.text = "This is a service agreement between two companies."
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(f"{api_prefix}/documents/{document_id}/summary")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["kind"] == "summary"
        assert body["content"] == "This is a service agreement between two companies."
        assert body["context_mode"] == "full"
        assert body["from_cache"] is False

    async def test_the_whole_document_reaches_the_model(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        """A summary is only trustworthy if the model actually saw the content."""
        document_id = await _ingest(ai_client, api_prefix, jobs)

        await ai_client.post(f"{api_prefix}/documents/{document_id}/summary")

        sent = " ".join(block.text for block in fake_provider.last_call.context_blocks)
        assert "$45,200.00" in sent
        assert "60 days written notice" in sent
        assert fake_provider.last_call.document_title == "agreement.txt"

    async def test_citations_are_requested_for_prose(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        document_id = await _ingest(ai_client, api_prefix, jobs)

        await ai_client.post(f"{api_prefix}/documents/{document_id}/summary")

        assert fake_provider.last_call.enable_citations is True

    async def test_an_empty_reply_is_rejected(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        fake_provider.text = "   "
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(f"{api_prefix}/documents/{document_id}/summary")

        assert response.status_code == 500
        assert response.json()["error"]["code"] == "invalid_model_response"


class TestExplanation:
    async def test_produces_an_explanation(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        fake_provider.text = "In simple terms, two companies agreed to work together."
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(f"{api_prefix}/documents/{document_id}/explanation")

        assert response.status_code == 200
        assert response.json()["kind"] == "explanation"

    async def test_summary_and_explanation_use_different_prompts(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        """Otherwise the two features would be the same feature twice."""
        document_id = await _ingest(ai_client, api_prefix, jobs)

        await ai_client.post(f"{api_prefix}/documents/{document_id}/summary")
        summary_prompt = fake_provider.last_call.system

        await ai_client.post(f"{api_prefix}/documents/{document_id}/explanation")
        explanation_prompt = fake_provider.last_call.system

        assert summary_prompt != explanation_prompt
        assert "do not know the subject area" in explanation_prompt


class TestProvenance:
    async def test_a_citation_resolves_to_a_real_location(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        """The chain, end to end: model index -> our chunk -> stored provenance."""
        fake_provider.text = "Invoices are due in 30 days."
        fake_provider.raw_citations = (RawCitation("Invoices are due within 30 days.", 0, 1),)
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(f"{api_prefix}/documents/{document_id}/summary")

        citations = response.json()["citations"]
        assert len(citations) == 1
        assert citations[0]["is_verified"] is True
        # A real location from the TXT parser, not something the model produced.
        assert "line" in citations[0]["location_label"]
        assert citations[0]["block_ids"]

    async def test_an_out_of_range_citation_is_reported_unverified(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        """Never point at the wrong place; say the source could not be confirmed."""
        fake_provider.text = "Something."
        fake_provider.raw_citations = (RawCitation("Invented passage.", 999, 1000),)
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(f"{api_prefix}/documents/{document_id}/summary")
        body = response.json()

        assert body["citations"][0]["is_verified"] is False
        assert body["citations"][0]["location_label"] == "source not verified"
        assert body["verified_citation_count"] == 0
        assert any("could not be traced" in warning for warning in body["warnings"])

    async def test_no_location_is_ever_invented(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        """Whatever the model claims, a shown location came from our own records."""
        fake_provider.text = "See page 47 of the agreement."
        fake_provider.raw_citations = (RawCitation("nonexistent text", 500, 501),)
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(f"{api_prefix}/documents/{document_id}/summary")

        for citation in response.json()["citations"]:
            if not citation["is_verified"]:
                assert citation["location_label"] == "source not verified"


class TestExtraction:
    async def test_extracts_facts_with_verified_sources(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        fake_provider.parsed = KeyFactsExtraction(
            document_type="service agreement",
            facts=[
                ExtractedFact(
                    name="total_value",
                    value="$45,200.00",
                    source_quote="The total contract value is $45,200.00 for the year.",
                ),
                ExtractedFact(
                    name="payment_terms",
                    value="30 days",
                    source_quote="Invoices are due within 30 days of receipt.",
                ),
            ],
        )
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(f"{api_prefix}/documents/{document_id}/extraction")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["kind"] == "extraction"
        assert len(body["fields"]) == 2

        by_name = {field["name"]: field for field in body["fields"]}
        assert by_name["total_value"]["value"] == "$45,200.00"
        assert by_name["total_value"]["citation"]["is_verified"] is True

    async def test_an_invented_quote_gets_no_source(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        """The most important extraction test.

        A fabricated quote must produce a field with **no** location, not a nearby one.
        """
        fake_provider.parsed = KeyFactsExtraction(
            facts=[
                ExtractedFact(
                    name="late_fee",
                    value="15% per month",
                    source_quote="A late fee of 15% per month applies to overdue invoices.",
                ),
            ],
        )
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(f"{api_prefix}/documents/{document_id}/extraction")
        body = response.json()

        assert body["fields"][0]["citation"] is None
        assert any("could not be traced" in warning for warning in body["warnings"])

    async def test_extraction_never_requests_citations(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        """Combining citations with a schema is a 400 from the API."""
        fake_provider.parsed = KeyFactsExtraction(
            facts=[
                ExtractedFact(name="a", value="b", source_quote="Invoices are due within 30 days.")
            ]
        )
        document_id = await _ingest(ai_client, api_prefix, jobs)

        await ai_client.post(f"{api_prefix}/documents/{document_id}/extraction")

        assert fake_provider.last_call.enable_citations is False
        assert fake_provider.last_call.output_model is KeyFactsExtraction

    async def test_no_extractable_facts_is_a_clear_message(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        fake_provider.parsed = KeyFactsExtraction(facts=[])
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(f"{api_prefix}/documents/{document_id}/extraction")

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "insufficient_content"

    async def test_unknown_extraction_type_is_rejected(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
    ) -> None:
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/extraction",
            json={"extraction_type": "not_a_real_schema"},
        )

        assert response.status_code == 422


class TestCaching:
    async def test_a_second_request_is_served_from_cache(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        """Without this every page view would bill another AI call."""
        document_id = await _ingest(ai_client, api_prefix, jobs)

        first = await ai_client.post(f"{api_prefix}/documents/{document_id}/summary")
        calls_after_first = fake_provider.call_count
        second = await ai_client.post(f"{api_prefix}/documents/{document_id}/summary")

        assert first.json()["from_cache"] is False
        assert second.json()["from_cache"] is True
        assert second.json()["content"] == first.json()["content"]
        assert fake_provider.call_count == calls_after_first

    async def test_refresh_forces_a_new_call(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        document_id = await _ingest(ai_client, api_prefix, jobs)

        await ai_client.post(f"{api_prefix}/documents/{document_id}/summary")
        before = fake_provider.call_count

        fake_provider.text = "A different summary."
        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/summary", json={"refresh": True}
        )

        assert fake_provider.call_count == before + 1
        assert response.json()["content"] == "A different summary."

    async def test_optimized_and_full_results_are_cached_separately(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        """They are different analyses, so one must not be served in place of the other."""
        document_id = await _ingest(ai_client, api_prefix, jobs)

        await ai_client.post(f"{api_prefix}/documents/{document_id}/summary")
        before = fake_provider.call_count

        await ai_client.post(
            f"{api_prefix}/documents/{document_id}/summary", json={"optimize_context": True}
        )

        assert fake_provider.call_count == before + 1

    async def test_previous_results_are_listed(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
    ) -> None:
        document_id = await _ingest(ai_client, api_prefix, jobs)
        await ai_client.post(f"{api_prefix}/documents/{document_id}/summary")
        await ai_client.post(f"{api_prefix}/documents/{document_id}/explanation")

        response = await ai_client.get(f"{api_prefix}/documents/{document_id}/insights")

        kinds = {item["kind"] for item in response.json()["items"]}
        assert kinds == {"summary", "explanation"}

    async def test_listing_never_triggers_an_ai_call(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        document_id = await _ingest(ai_client, api_prefix, jobs)

        await ai_client.get(f"{api_prefix}/documents/{document_id}/insights")

        assert fake_provider.call_count == 0


class TestContextOptimization:
    async def test_small_documents_are_sent_whole_even_when_asked(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        """Optimizing a short document only loses detail."""
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.post(
            f"{api_prefix}/documents/{document_id}/summary", json={"optimize_context": True}
        )

        assert response.json()["context_mode"] == "full"

    async def test_original_content_is_untouched_by_optimization(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
    ) -> None:
        """The guarantee that matters: analysing never alters the user's data."""
        document_id = await _ingest(ai_client, api_prefix, jobs)

        before = (await ai_client.get(f"{api_prefix}/documents/{document_id}/content")).json()

        await ai_client.post(
            f"{api_prefix}/documents/{document_id}/summary", json={"optimize_context": True}
        )

        after = (await ai_client.get(f"{api_prefix}/documents/{document_id}/content")).json()
        assert after == before

    async def test_context_preview_costs_nothing(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        document_id = await _ingest(ai_client, api_prefix, jobs)

        response = await ai_client.get(f"{api_prefix}/documents/{document_id}/context-preview")

        assert response.status_code == 200
        body = response.json()
        assert body["chunk_count"] >= 1
        assert body["fits_in_full"] is True
        # A small document should not be offered optimization.
        assert body["optimization_available"] is False
        assert fake_provider.call_count == 0


class TestErrorHandling:
    async def test_missing_api_key_is_reported_clearly(
        self,
        app: FastAPI,
        api_prefix: str,
    ) -> None:
        """The real provider with no key — the default state of a fresh checkout."""
        transport = ASGITransport(app=app)
        async with (
            app.router.lifespan_context(app),
            AsyncClient(transport=transport, base_url="http://testserver") as client,
        ):
            upload = await client.post(
                f"{api_prefix}/documents",
                files={"file": ("a.txt", build_txt(DOCUMENT_TEXT), "text/plain")},
            )
            document_id = upload.json()["id"]
            await app.state.job_service.wait_for_all(timeout=30.0)

            response = await client.post(f"{api_prefix}/documents/{document_id}/summary")

        assert response.status_code == 500
        body = response.json()
        assert body["error"]["code"] == "ai_not_configured"
        assert body["error"]["request_id"]

    async def test_provider_failure_is_safe_for_the_client(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
        fake_provider: FakeLLMProvider,
    ) -> None:
        document_id = await _ingest(ai_client, api_prefix, jobs)
        fake_provider.error = AIUnavailableError("The AI service is busy right now.")

        response = await ai_client.post(f"{api_prefix}/documents/{document_id}/summary")

        assert response.status_code == 500
        text = response.text
        assert "Traceback" not in text
        assert ".py" not in text
        assert "anthropic" not in text.lower()

    async def test_analysing_an_unprocessed_document_is_refused(
        self,
        ai_client: AsyncClient,
        api_prefix: str,
    ) -> None:
        """Asking the model to interpret nothing invites it to invent something."""
        upload = await ai_client.post(
            f"{api_prefix}/documents",
            files={"file": ("a.txt", build_txt(DOCUMENT_TEXT), "text/plain")},
        )
        document_id = upload.json()["id"]

        response = await ai_client.post(f"{api_prefix}/documents/{document_id}/summary")

        # Either still processing, or already finished if the task won the race — both
        # are acceptable, but it must never be an unhandled failure.
        assert response.status_code in {200, 422}

    async def test_unknown_document_returns_404(
        self, ai_client: AsyncClient, api_prefix: str
    ) -> None:
        from uuid import uuid4

        response = await ai_client.post(f"{api_prefix}/documents/{uuid4()}/summary")

        assert response.status_code == 404

    async def test_another_owners_document_cannot_be_analysed(
        self,
        ai_app: FastAPI,
        ai_client: AsyncClient,
        api_prefix: str,
        jobs: JobService,
    ) -> None:
        from app.api.deps import get_current_owner_id

        document_id = await _ingest(ai_client, api_prefix, jobs)

        ai_app.dependency_overrides[get_current_owner_id] = lambda: "someone-else"
        try:
            response = await ai_client.post(f"{api_prefix}/documents/{document_id}/summary")
        finally:
            ai_app.dependency_overrides.pop(get_current_owner_id, None)

        assert response.status_code == 404


class TestAIStatus:
    async def test_reports_unconfigured_without_a_key(
        self, client: AsyncClient, api_prefix: str
    ) -> None:
        response = await client.get(f"{api_prefix}/ai/status")

        assert response.status_code == 200
        assert response.json()["configured"] is False

    async def test_never_exposes_the_key(
        self, client: AsyncClient, api_prefix: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The key is server-side only — not even a prefix may leave the server."""
        from app.core.config import get_settings

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SUPERSECRET123")
        get_settings.cache_clear()
        try:
            response = await client.get(f"{api_prefix}/ai/status")
        finally:
            get_settings.cache_clear()

        text = response.text
        assert "SUPERSECRET" not in text
        assert "sk-ant" not in text

    async def test_reports_the_model(self, client: AsyncClient, api_prefix: str) -> None:
        assert response_model(await client.get(f"{api_prefix}/ai/status")) == "claude-opus-5"


class TestAIStatusFollowsTheActiveProvider:
    """The endpoint must describe whichever provider is actually configured.

    It used to read `settings.ai_configured` and `settings.anthropic_model`, both of
    which are about Anthropic specifically. A server running `AI_PROVIDER=gemini` with a
    Gemini key therefore reported `configured: false` and Anthropic's model name — and the
    frontend, which disables every AI button on `configured: false`, presented a fully
    working server as one with no AI at all.

    No fake provider here on purpose. These tests go through the real
    `deps._build_provider`, because provider *selection* is the thing that was broken;
    overriding the provider would step over the bug.
    """

    @staticmethod
    def _configure(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
        """Point the process at a provider, the way a deployment's environment would."""
        from app.core.config import get_settings

        for name, value in env.items():
            monkeypatch.setenv(name, value)
        get_settings.cache_clear()

    async def test_gemini_with_a_key_reports_configured(
        self, app: FastAPI, api_prefix: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The exact deployment case that was broken."""
        self._configure(
            monkeypatch,
            AI_PROVIDER="gemini",
            GEMINI_API_KEY="test-key-not-real",
            GEMINI_MODEL="gemini-3.6-flash",
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as http_client:
            response = await http_client.get(f"{api_prefix}/ai/status")

        assert response.status_code == 200
        assert response.json()["configured"] is True
        assert response.json()["model"] == "gemini-3.6-flash"

    async def test_gemini_without_a_key_reports_unconfigured(
        self, app: FastAPI, api_prefix: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fix must not turn the report into "always configured" — an unusable
        server still has to say so, or the buttons stop being a warning at all."""
        self._configure(monkeypatch, AI_PROVIDER="gemini", GEMINI_API_KEY="")

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as http_client:
            response = await http_client.get(f"{api_prefix}/ai/status")

        assert response.json()["configured"] is False

    async def test_an_anthropic_key_does_not_make_gemini_look_configured(
        self, app: FastAPI, api_prefix: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The inverse of the original bug, and the reason it has its own test: reading
        the wrong provider's settings can report a *false positive* just as easily as a
        false negative. A key for a provider that is not in use configures nothing."""
        self._configure(
            monkeypatch,
            AI_PROVIDER="gemini",
            GEMINI_API_KEY="",
            ANTHROPIC_API_KEY="sk-ant-not-a-real-key",
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as http_client:
            response = await http_client.get(f"{api_prefix}/ai/status")

        assert response.json()["configured"] is False
        assert response.json()["model"] != "claude-opus-5"

    async def test_the_key_never_leaves_the_server_on_the_gemini_path(
        self, app: FastAPI, api_prefix: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guarantee the original endpoint carried, re-established on the new path."""
        self._configure(monkeypatch, AI_PROVIDER="gemini", GEMINI_API_KEY="AIzaSySUPERSECRET123")

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as http_client:
            response = await http_client.get(f"{api_prefix}/ai/status")

        assert "SUPERSECRET" not in response.text
        assert "AIzaSy" not in response.text

    async def test_the_response_shape_is_unchanged(
        self, app: FastAPI, api_prefix: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Backward compatibility: the frontend reads exactly these two fields."""
        self._configure(monkeypatch, AI_PROVIDER="gemini", GEMINI_API_KEY="test-key-not-real")

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as http_client:
            response = await http_client.get(f"{api_prefix}/ai/status")

        body = response.json()
        assert set(body) == {"configured", "model"}
        assert isinstance(body["configured"], bool)
        assert isinstance(body["model"], str)

    async def test_status_agrees_with_what_the_services_check(
        self, app: FastAPI, api_prefix: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The property that makes this fix structural rather than cosmetic.

        Services gate every call on `provider.is_configured`. Reporting the same value
        means the endpoint and the requests it gates cannot disagree — which is exactly
        what happened when one read settings and the other read the provider.
        """
        from app.api.deps import _build_provider
        from app.core.config import get_settings

        self._configure(monkeypatch, AI_PROVIDER="gemini", GEMINI_API_KEY="test-key-not-real")

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as http_client:
            response = await http_client.get(f"{api_prefix}/ai/status")

        provider = _build_provider(get_settings())
        assert response.json()["configured"] is provider.is_configured
        assert response.json()["model"] == provider.model_name


def response_model(response: Any) -> str:
    model: str = response.json()["model"]
    return model
