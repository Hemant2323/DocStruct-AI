"""Translation of exceptions into HTTP responses.

The status-code mapping lives here and nowhere else. Services raise domain errors that
describe *what went wrong*; this module decides how that looks over HTTP.

Three rules, all from CLAUDE.md:

* Nothing is swallowed. Every handled exception is logged before a response is returned.
* Unexpected exceptions never leak internals. The traceback goes to the log; the client
  gets a generic message. A stack trace in an HTTP response is an information leak.
* Every error response carries a ``request_id`` matching the log entry, so a safe
  response and a debuggable log are not a trade-off.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.middleware import REQUEST_ID_HEADER
from app.core.context import get_request_id
from app.core.exceptions import (
    ConfigurationError,
    DocStructError,
    FileTooLargeError,
    NotFoundError,
    ProcessingError,
    StorageError,
    UnsupportedFileTypeError,
    ValidationError,
)

logger = logging.getLogger(__name__)

#: Returned instead of the real message when a server-side error would otherwise expose
#: internal detail. Deliberately vague to the client; the log holds the truth.
GENERIC_SERVER_ERROR_MESSAGE = (
    "Something went wrong on our end. The problem has been logged. "
    "If it keeps happening, quote the request ID below."
)

#: Domain error -> HTTP status. Subclasses are matched before their parents.
_STATUS_MAP: list[tuple[type[DocStructError], int]] = [
    (NotFoundError, status.HTTP_404_NOT_FOUND),
    (FileTooLargeError, status.HTTP_413_CONTENT_TOO_LARGE),
    (UnsupportedFileTypeError, status.HTTP_415_UNSUPPORTED_MEDIA_TYPE),
    (ValidationError, status.HTTP_422_UNPROCESSABLE_CONTENT),
    (ProcessingError, status.HTTP_500_INTERNAL_SERVER_ERROR),
    (StorageError, status.HTTP_500_INTERNAL_SERVER_ERROR),
    (ConfigurationError, status.HTTP_500_INTERNAL_SERVER_ERROR),
]


def _status_for(exc: DocStructError) -> int:
    """Return the HTTP status for a domain error, defaulting to 500."""
    for error_type, http_status in _STATUS_MAP:
        if isinstance(exc, error_type):
            return http_status
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def _request_id_for(request: Request) -> str | None:
    """Return this request's correlation ID.

    Read from the ASGI scope first: it survives an exception propagating past the
    middleware that set it, which the context variable may not.
    """
    request_id = getattr(request.state, "request_id", None)
    if isinstance(request_id, str):
        return request_id
    return get_request_id()


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build the standard error envelope, stamped with the correlation ID."""
    request_id = _request_id_for(request)

    response_headers = dict(headers or {})
    if request_id:
        response_headers[REQUEST_ID_HEADER] = request_id

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": details or {},
                "request_id": request_id,
            }
        },
        headers=response_headers or None,
    )


async def docstruct_error_handler(request: Request, exc: DocStructError) -> JSONResponse:
    """Handle a deliberately raised domain error."""
    http_status = _status_for(exc)
    is_server_error = http_status >= 500

    # Client mistakes are routine; server-side failures are not. Log accordingly so the
    # error log stays meaningful.
    log = logger.error if is_server_error else logger.warning
    log(
        "%s on %s %s: %s",
        type(exc).__name__,
        request.method,
        request.url.path,
        exc.message,
        exc_info=is_server_error,
    )

    if is_server_error:
        # A 5xx means something broke inside the application. Even though these messages
        # are written by us, withholding them keeps internal wording (paths, backends,
        # component names) out of the client's hands as the codebase grows.
        return _error_response(
            request,
            status_code=http_status,
            code=exc.code,
            message=GENERIC_SERVER_ERROR_MESSAGE,
        )

    return _error_response(
        request,
        status_code=http_status,
        code=exc.code,
        message=exc.message,
        details=exc.details,
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle FastAPI/Pydantic request-validation failures.

    Reshaped into the same envelope as every other error so clients have one error path.
    Only the field location and the validator's message are returned — never the
    submitted value, which could be a credential pasted into the wrong field.
    """
    fields = [
        {
            "field": ".".join(str(part) for part in error.get("loc", ())),
            "message": error.get("msg", "invalid value"),
        }
        for error in exc.errors()
    ]

    logger.warning(
        "Request validation failed on %s %s: %d field(s)",
        request.method,
        request.url.path,
        len(fields),
    )

    return _error_response(
        request,
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        code="validation_error",
        message="Request validation failed",
        details={"fields": fields},
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle framework-raised HTTP exceptions (404 for unknown routes, 405, ...)."""
    if exc.status_code >= 500:
        logger.error(
            "HTTP %d on %s %s", exc.status_code, request.method, request.url.path, exc_info=True
        )
        return _error_response(
            request,
            status_code=exc.status_code,
            code="internal_error",
            message=GENERIC_SERVER_ERROR_MESSAGE,
            headers=getattr(exc, "headers", None),
        )

    return _error_response(
        request,
        status_code=exc.status_code,
        code="http_error",
        message=str(exc.detail),
        headers=getattr(exc, "headers", None),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last resort for exceptions nothing else claimed.

    The full traceback is logged; the response says only that something failed. This is
    the opposite of silently ignoring an error — it is recorded in full, just not shown
    to the caller. The exception type is deliberately kept out of the response: even a
    class name discloses which libraries and internals are in play.
    """
    logger.exception("Unhandled %s on %s %s", type(exc).__name__, request.method, request.url.path)
    return _error_response(
        request,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="internal_error",
        message=GENERIC_SERVER_ERROR_MESSAGE,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach every handler to the application."""
    app.add_exception_handler(DocStructError, docstruct_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)
