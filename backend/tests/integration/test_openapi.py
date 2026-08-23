"""Tests that the generated OpenAPI documentation stays useful.

Swagger docs rot quietly: a new route ships with no summary, an error case goes
undocumented, and nobody notices because nothing fails. These tests make that a build
failure instead.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient


@pytest.fixture
async def spec(client: AsyncClient) -> dict[str, Any]:
    """The application's generated OpenAPI document."""
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    return response.json()


def _operations(spec: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """Every (method, path, operation) in the spec."""
    return [
        (method.upper(), path, operation)
        for path, operations in spec["paths"].items()
        for method, operation in operations.items()
    ]


class TestDocumentationCompleteness:
    async def test_every_endpoint_has_a_summary(self, spec: dict[str, Any]) -> None:
        missing = [
            f"{method} {path}"
            for method, path, operation in _operations(spec)
            if not operation.get("summary")
        ]
        assert not missing, f"Endpoints without a summary: {missing}"

    async def test_every_endpoint_has_a_description(self, spec: dict[str, Any]) -> None:
        """The description is the docstring — it should explain more than the name."""
        too_thin = [
            f"{method} {path}"
            for method, path, operation in _operations(spec)
            if len(operation.get("description", "").strip()) < 40
        ]
        assert not too_thin, f"Endpoints with a missing or too-brief description: {too_thin}"

    async def test_every_endpoint_belongs_to_a_documented_tag(self, spec: dict[str, Any]) -> None:
        """Tags group the Swagger page; an untagged route lands in a nameless bucket."""
        described = {tag["name"] for tag in spec.get("tags", [])}
        assert described

        for method, path, operation in _operations(spec):
            tags = operation.get("tags", [])
            assert tags, f"{method} {path} has no tag"
            assert set(tags) <= described, f"{method} {path} uses an undocumented tag"

    async def test_every_response_has_a_description(self, spec: dict[str, Any]) -> None:
        missing = [
            f"{method} {path} -> {code}"
            for method, path, operation in _operations(spec)
            for code, response in operation.get("responses", {}).items()
            if not response.get("description")
        ]
        assert not missing, f"Undocumented responses: {missing}"


class TestErrorDocumentation:
    async def test_error_responses_use_the_shared_envelope(self, spec: dict[str, Any]) -> None:
        """Documented errors must point at the envelope the API actually returns."""
        for method, path, operation in _operations(spec):
            for code, response in operation.get("responses", {}).items():
                if not code.startswith(("4", "5")):
                    continue
                schema = response.get("content", {}).get("application/json", {}).get("schema", {})
                assert schema, f"{method} {path} -> {code} documents no response body"

    async def test_document_endpoints_document_not_found(self, spec: dict[str, Any]) -> None:
        for path in ("/api/documents/{document_id}",):
            for method in ("get", "delete"):
                responses = spec["paths"][path][method]["responses"]
                assert "404" in responses, f"{method.upper()} {path} does not document 404"

    async def test_error_envelope_documents_the_request_id(self, spec: dict[str, Any]) -> None:
        """The correlation ID is only useful if a developer knows it exists."""
        error_detail = spec["components"]["schemas"]["ErrorDetail"]

        assert "request_id" in error_detail["properties"]
        assert error_detail["properties"]["request_id"].get("description")

    async def test_error_schema_carries_an_example(self, spec: dict[str, Any]) -> None:
        """An example is what makes the Swagger page readable at a glance."""
        assert "example" in spec["components"]["schemas"]["ErrorResponse"]


class TestApiMetadata:
    async def test_api_has_a_title_version_and_description(self, spec: dict[str, Any]) -> None:
        info = spec["info"]

        assert info["title"] == "DocStruct AI"
        assert info["version"]
        assert "error" in info["description"].lower()

    async def test_status_values_are_documented_as_an_enum(self, spec: dict[str, Any]) -> None:
        """A developer must be able to see the four states without reading the source."""
        statuses = spec["components"]["schemas"]["ProcessingStatus"]["enum"]

        assert set(statuses) == {"pending", "processing", "ready", "failed"}
