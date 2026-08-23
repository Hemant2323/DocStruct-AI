"""Typed domain exceptions.

Business logic raises these instead of HTTP exceptions, which keeps the service layer
free of HTTP concerns. A single handler in ``app.api.errors`` translates them into HTTP
responses, so the status-code mapping lives in exactly one place.

Per CLAUDE.md: errors are never silently swallowed. Every failure either becomes one of
these or propagates.
"""

from __future__ import annotations

from typing import Any


class DocStructError(Exception):
    """Base class for every error this application raises deliberately.

    Args:
        message: Human-readable description, safe to return to the client.
        details: Optional structured context (field names, limits, identifiers).
            Must never contain document content, credentials, or personal data.
    """

    #: Stable machine-readable code the frontend can branch on.
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(DocStructError):
    """A requested resource does not exist, or is not visible to this owner."""

    code = "not_found"


class ValidationError(DocStructError):
    """Input failed a business rule (as opposed to a schema/type check)."""

    code = "validation_error"


class UnsupportedFileTypeError(ValidationError):
    """The uploaded file's type has no registered parser."""

    code = "unsupported_file_type"


class FileTooLargeError(ValidationError):
    """The uploaded file exceeds the configured size limit."""

    code = "file_too_large"


class StorageError(DocStructError):
    """Reading from or writing to the file store failed."""

    code = "storage_error"


class ProcessingError(DocStructError):
    """Document processing failed after the file was accepted."""

    code = "processing_error"


class ConfigurationError(DocStructError):
    """The application is misconfigured — a required setting is missing or invalid."""

    code = "configuration_error"
