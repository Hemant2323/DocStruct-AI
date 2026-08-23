"""Reusable OpenAPI response declarations.

FastAPI documents the success response automatically from a route's return type, but it
cannot know which errors a route can produce. Declaring them here and reusing the entries
keeps the generated Swagger page accurate without repeating the same dictionary in every
route, and keeps every documented error pointing at the one real error envelope.
"""

from __future__ import annotations

from typing import Any

from fastapi import status

from app.api.schemas.common import ErrorResponse

#: Error responses, keyed by status code, ready to spread into a route's `responses=`.
ERROR_RESPONSES: dict[int, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {
        "model": ErrorResponse,
        "description": (
            "The document does not exist, or belongs to another owner. The two cases are "
            "deliberately indistinguishable so the API cannot be used to discover which "
            "document IDs exist."
        ),
    },
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "model": ErrorResponse,
        "description": (
            "The request failed validation — for example a malformed UUID, or a `limit` "
            "outside the allowed range. `details.fields` names each offending field."
        ),
    },
    status.HTTP_413_CONTENT_TOO_LARGE: {
        "model": ErrorResponse,
        "description": (
            "The uploaded file is larger than the configured limit. The limit is "
            "enforced while reading, so an oversized upload is cut off rather than "
            "buffered in full."
        ),
    },
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {
        "model": ErrorResponse,
        "description": (
            "The file's actual content is not a supported format. Detection is based on "
            "the bytes, not the filename, so renaming a file does not change this."
        ),
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "model": ErrorResponse,
        "description": (
            "An unexpected server error. The response carries only a generic message and "
            "a `request_id`; the full technical detail is in the server log under that ID."
        ),
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": "A dependency the application needs is currently unreachable.",
    },
}


def error_responses(*status_codes: int) -> dict[int | str, dict[str, Any]]:
    """Return the declarations for the given status codes.

    Args:
        *status_codes: Status codes the route can return, e.g. ``404, 422``.

    Raises:
        KeyError: If a status code has no declaration above. Failing loudly here beats
            silently generating documentation with a missing error case.
    """
    return {code: ERROR_RESPONSES[code] for code in status_codes}
