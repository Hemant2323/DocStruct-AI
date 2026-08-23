"""Logging configuration.

A single ``configure_logging()`` call sets up the root logger at startup. Everywhere else
uses ``logging.getLogger(__name__)`` and never touches handler configuration.

Per CLAUDE.md: log identifiers and metadata, never document content, file contents, or
credentials.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from app.core.context import get_request_id

# `%(request_id)s` is supplied by _RequestIdFilter below, so every line emitted while
# handling a request can be tied back to the response the client received.
_LOG_FORMAT = "%(asctime)s %(levelname)-8s [%(request_id)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

#: Shown in place of a correlation ID outside of a request (startup, shutdown, scripts).
NO_REQUEST_ID = "-"

_configured = False


class _RequestIdFilter(logging.Filter):
    """Attaches the current request's correlation ID to every log record.

    A filter is used rather than passing the ID to each logging call: it applies to
    records from libraries too (uvicorn, SQLAlchemy), and it cannot be forgotten.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or NO_REQUEST_ID
        # Filters can drop records by returning False; this one only annotates.
        return True


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for the process.

    Idempotent: calling it more than once (for example from both the app factory and a
    test fixture) will not attach duplicate handlers that double every log line.

    Args:
        level: Log level name, e.g. ``"INFO"`` or ``"DEBUG"``.
    """
    global _configured

    root = logging.getLogger()
    root.setLevel(level)

    if _configured:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
    # The filter is attached to the handler, not the logger, so it also annotates
    # records that propagate up from third-party loggers.
    handler.addFilter(_RequestIdFilter())
    root.addHandler(handler)

    # uvicorn installs its own handlers; let records propagate to ours instead of
    # being emitted twice in two different formats.
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(noisy).handlers.clear()
        logging.getLogger(noisy).propagate = True

    _configured = True


def get_logger(name: str, **context: Any) -> logging.LoggerAdapter[logging.Logger]:
    """Return a logger that prefixes every message with stable context.

    Args:
        name: Logger name, normally ``__name__``.
        **context: Key/value pairs appended to each message. Identifiers and metadata
            only — never content or secrets.
    """
    logger = logging.getLogger(name)
    return _ContextAdapter(logger, context)


class _ContextAdapter(logging.LoggerAdapter[logging.Logger]):
    """Appends ``key=value`` context to each log message."""

    def process(self, msg: Any, kwargs: Any) -> tuple[Any, Any]:
        if not self.extra:
            return msg, kwargs
        suffix = " ".join(f"{key}={value}" for key, value in self.extra.items())
        return f"{msg} [{suffix}]", kwargs
