"""Shared API schemas: health, readiness, and the error envelope."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Liveness: the process is up and serving requests.

    Deliberately checks nothing external. A liveness probe that depends on the database
    will report the *application* as dead during a database blip and get it restarted,
    which does not help and loses in-flight work.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "app": "DocStruct AI",
                "version": "0.1.0",
                "environment": "development",
            }
        }
    )

    status: Literal["ok"] = "ok"
    app: str = Field(description="Application name, from the APP_NAME setting.")
    version: str = Field(description="API version.")
    environment: str = Field(description="development, test, or production.")


class ReadinessResponse(BaseModel):
    """Readiness: the application's dependencies are reachable.

    Returned with HTTP 200 when ready and 503 when not, so a load balancer can act on
    the status code without parsing the body.
    """

    model_config = ConfigDict(
        json_schema_extra={"example": {"status": "ready", "checks": {"database": True}}}
    )

    status: Literal["ready", "not_ready"]
    checks: dict[str, bool] = Field(description="Per-dependency result; True means healthy.")


class ErrorDetail(BaseModel):
    """The body of every error response.

    A single, predictable error shape means the frontend has one error path rather than
    one per endpoint.
    """

    code: str = Field(
        description=(
            "Stable machine-readable error code. Branch on this rather than on the "
            "message text, which may be reworded."
        )
    )
    message: str = Field(description="Human-readable description, safe to show to an end user.")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Structured context, such as which fields failed validation. Never contains "
            "secrets, stack traces, or document content."
        ),
    )
    request_id: str | None = Field(
        default=None,
        description=(
            "Correlation ID for this request, matching the X-Request-ID response header. "
            "Quote it when reporting a problem: the server log holds the full technical "
            "detail under this ID."
        ),
    )


class ErrorResponse(BaseModel):
    """Top-level error envelope returned by every failing request."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": {
                    "code": "not_found",
                    "message": "Document not found",
                    "details": {"document_id": "0b6c1f3e-6a4d-4c58-9a1f-2b7d8e5c4a90"},
                    "request_id": "9f1c2d3e4a5b6c7d8e9f0a1b2c3d4e5f",
                }
            }
        }
    )

    error: ErrorDetail
