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

    #: Whether this error's own message may be shown to the client on a 5xx.
    #:
    #: Server-side errors normally return a generic sentence so that internal detail
    #: cannot escape. A few carry a message that is both safe and genuinely useful —
    #: "AI features are not configured", "the AI service is busy" — and telling the user
    #: "something went wrong" instead would hide something they can act on.
    #:
    #: Only set this to True for a message written by hand with no interpolation of
    #: upstream text, paths, or identifiers. It has no effect on 4xx errors, whose
    #: messages are always shown.
    expose_message: bool = False

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


class ProcessingError(DocStructError):
    """Document processing failed after the file was accepted."""

    code = "processing_error"


class CorruptFileError(ValidationError):
    """The file is the declared type but is damaged and cannot be read."""

    code = "corrupt_file"


class EncryptedFileError(ValidationError):
    """The file is password-protected, so its content cannot be reached."""

    code = "encrypted_file"


class EmptyDocumentError(ValidationError):
    """The file was readable but contained no extractable content."""

    code = "no_content"


class ScannedDocumentError(ValidationError):
    """The file is a scan or images, so text extraction would need OCR.

    Raised rather than returning an empty result: a scanned PDF that silently produced
    nothing would look identical to a successfully processed blank document.
    """

    code = "requires_ocr"


class DocumentTooComplexError(ValidationError):
    """The file exceeds a processing limit — too many pages, rows, or blocks."""

    code = "too_complex"


class ParserError(ProcessingError):
    """Parsing failed for a reason that is not the user's fault.

    Distinct from the errors above: those describe a problem with the file, which the
    user can act on. This one means our code or a parsing library failed, so it is a
    server-side error and its detail must not reach the client.
    """

    code = "parser_error"


class AIError(ProcessingError):
    """Base class for failures of the AI layer.

    A ``ProcessingError`` (5xx-class), so ``app.api.errors`` returns a generic message
    and the technical detail stays in the log. Model provider names, prompts and
    upstream error text must never reach the client.
    """

    code = "ai_error"


class AIConfigurationError(AIError):
    """The AI provider is not configured — typically no API key is set.

    Distinct from a transient failure: retrying will not help, and the user needs to be
    told that the feature is unavailable rather than broken.
    """

    code = "ai_not_configured"
    #: Safe to show: written by hand, no upstream text interpolated.
    expose_message = True


class AIUnavailableError(AIError):
    """The model could not be reached, or refused, or timed out.

    Transient by nature, so the message tells the user to try again shortly.
    """

    code = "ai_unavailable"
    #: Safe to show: written by hand, no upstream text interpolated.
    expose_message = True


class InvalidModelResponseError(AIError):
    """The model replied, but not in the shape the request required.

    Raised when structured output fails schema validation or arrives unparseable.
    Surfacing this rather than guessing at a repair keeps invented data out of results.
    """

    code = "invalid_model_response"
    #: Safe to show: written by hand, no upstream text interpolated.
    expose_message = True


class OutputTruncatedError(InvalidModelResponseError):
    """The reply hit the output token ceiling and stopped mid-answer.

    A subclass rather than a separate error because it *is* an unusable response and every
    caller that does not care about the distinction should keep treating it as one — the
    HTTP status, the exposed message and the logging are all inherited unchanged.

    What the distinction buys is one decision. "The reply did not match the schema" is a
    dead end: retrying it differently would produce the same mismatch. "The reply ran out
    of room" is not — it says the request asked for more than one answer can hold, which
    is a thing the caller can act on by asking for less at a time. Conversion uses exactly
    that to fall back from one call to windowed extraction. Catching the parent class
    there instead would re-run genuine schema failures as well, at full cost, to no end.
    """

    code = "output_truncated"


class ContextTooLargeError(ValidationError):
    """The document is too large to send even after optimization.

    A ``ValidationError`` (4xx-class) because it is a property of the document the user
    chose, and the message tells them what they can do about it.
    """

    code = "context_too_large"


class InsufficientContentError(ValidationError):
    """There is not enough extracted content to analyse.

    Raised instead of asking the model to summarise nothing, which would invite it to
    invent something plausible.
    """

    code = "insufficient_content"


class InvalidStructureSchemaError(ValidationError):
    """A user-requested structure schema (Milestone 4.5) is malformed or unsafe.

    Covers every rejection reason — empty schema, too many fields, a bad field name, an
    unsupported type — under one code. ``details`` carries which check failed.
    """

    code = "invalid_structure_schema"


class EmbeddingsNotAvailableError(ValidationError):
    """The document has content, but no usable embedding exists for it yet.

    Distinct from :class:`InsufficientContentError`: the document itself is fine, but
    retrieval has nothing to search because embeddings were never computed, or every one
    that exists is stale (its chunk's text has since changed) or otherwise unusable.
    """

    code = "embeddings_not_available"


class StorageError(DocStructError):
    """Reading from or writing to the file store failed."""

    code = "storage_error"


class ConfigurationError(DocStructError):
    """The application is misconfigured — a required setting is missing or invalid."""

    code = "configuration_error"
