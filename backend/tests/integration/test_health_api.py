"""Tests for the health and readiness endpoints."""

from __future__ import annotations

from httpx import AsyncClient


class TestHealth:
    async def test_health_returns_ok(self, client: AsyncClient, api_prefix: str) -> None:
        response = await client.get(f"{api_prefix}/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["app"] == "DocStruct AI"
        assert body["environment"] == "test"
        assert body["version"]

    async def test_health_does_not_touch_the_database(
        self, client: AsyncClient, api_prefix: str
    ) -> None:
        """Liveness must stay green during a database outage.

        If it did not, a database blip would make a supervisor restart a perfectly
        healthy application process and throw away in-flight work.
        """
        from app.core import database

        async def _fail() -> bool:
            raise AssertionError("liveness must not check the database")

        original = database.check_database_connection
        database.check_database_connection = _fail  # type: ignore[assignment]
        try:
            response = await client.get(f"{api_prefix}/health")
        finally:
            database.check_database_connection = original  # type: ignore[assignment]

        assert response.status_code == 200


class TestReadiness:
    async def test_ready_when_database_reachable(
        self, client: AsyncClient, api_prefix: str
    ) -> None:
        response = await client.get(f"{api_prefix}/ready")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert body["checks"]["database"] is True

    async def test_returns_503_when_database_unreachable(
        self, client: AsyncClient, api_prefix: str
    ) -> None:
        """A failed dependency check must surface in the status code, not just the body,
        so a load balancer can act on it without parsing JSON."""
        from app.api.routes import health as health_module

        async def _unreachable() -> bool:
            return False

        original = health_module.check_database_connection
        health_module.check_database_connection = _unreachable  # type: ignore[assignment]
        try:
            response = await client.get(f"{api_prefix}/ready")
        finally:
            health_module.check_database_connection = original  # type: ignore[assignment]

        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "not_ready"
        assert body["checks"]["database"] is False


class TestApiSurface:
    async def test_unknown_route_uses_the_standard_error_envelope(
        self, client: AsyncClient, api_prefix: str
    ) -> None:
        response = await client.get(f"{api_prefix}/does-not-exist")

        assert response.status_code == 404
        assert "error" in response.json()

    async def test_openapi_schema_is_served_outside_production(self, client: AsyncClient) -> None:
        response = await client.get("/openapi.json")

        assert response.status_code == 200
        assert "paths" in response.json()
