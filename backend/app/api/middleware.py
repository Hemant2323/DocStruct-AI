"""HTTP middleware.

Middleware wraps every request, so anything that must happen for *all* endpoints
(correlation IDs, access logging) belongs here rather than being repeated in each route.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.context import set_request_id

logger = logging.getLogger(__name__)

#: Header carrying the correlation ID back to the client.
REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign every request a correlation ID and log how it finished.

    The ID is generated server-side and never taken from the incoming request. Trusting a
    client-supplied header would let a caller forge or repeat IDs, which would corrupt the
    logs that this exists to make trustworthy.

    The ID is published three ways:

    * ``request.state.request_id`` — read by the exception handlers. This lives on the
      ASGI scope, so it survives an exception propagating past this middleware.
    * A context variable — read by the logging filter, so every log line emitted while
      handling the request carries the ID without anything being threaded through.
    * The ``X-Request-ID`` response header — so a user can quote it in a bug report.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = uuid4().hex
        request.state.request_id = request_id
        set_request_id(request_id)

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Logged here so the timing is recorded even for a failed request. The
            # exception is re-raised untouched: the registered handlers own the response,
            # and swallowing it here would hide the failure.
            duration_ms = (time.perf_counter() - started) * 1000
            logger.warning(
                "%s %s failed after %.1fms",
                request.method,
                request.url.path,
                duration_ms,
            )
            raise

        duration_ms = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id

        # Only the method, path, status, and duration — never query strings, headers, or
        # bodies, which can carry credentials or document content.
        logger.info(
            "%s %s -> %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response
