"""Per-request context.

Holds a correlation ID for the request currently being handled, so that log lines and
the error returned to the client can be tied together.

This is what lets an error response stay safe *and* debuggable: the client sees only an
opaque ID, while the server log holds the full traceback under that same ID. A user can
report "request 3f2a…" and the exact failure is one search away.

A ``ContextVar`` is used rather than a global because each request runs in its own
asyncio task, and a plain global would be overwritten by whichever request ran last.
"""

from __future__ import annotations

from contextvars import ContextVar

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def set_request_id(request_id: str) -> None:
    """Record the correlation ID for the request being handled."""
    _request_id.set(request_id)


def get_request_id() -> str | None:
    """Return the current request's correlation ID, if one has been set."""
    return _request_id.get()
