"""File type detection and archive safety checks.

**The extension is a hint, never a decision.** A file called ``report.pdf`` can contain
anything at all, so acceptance is decided by looking at the bytes.

Detection is hand-written rather than delegated to ``python-magic``. That library needs
the native ``libmagic`` binary, which is awkward to install on Windows and would become a
setup obstacle for this project. We only support five formats, and identifying them takes
a signature check plus, for the two ZIP-based formats, a peek inside the archive.
"""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass

from charset_normalizer import from_bytes

from app.core.exceptions import UnsupportedFileTypeError, ValidationError
from app.models.domain.document import SupportedFileType

#: Magic bytes at the start of a PDF.
_PDF_SIGNATURE = b"%PDF-"

#: Local file header of a ZIP archive. DOCX and XLSX are both ZIP containers.
_ZIP_SIGNATURE = b"PK\x03\x04"
#: Empty and spanned ZIP archives use different headers; treated as ZIP-shaped but
#: unusable, so they produce a clear "corrupt" message rather than "unsupported".
_ZIP_EMPTY_SIGNATURES = (b"PK\x05\x06", b"PK\x07\x08")

#: Entries that identify what an OOXML archive actually is.
_DOCX_MARKER = "word/document.xml"
_XLSX_MARKER = "xl/workbook.xml"

#: How many bytes to examine when deciding whether content is text and, if so, whether
#: it is delimited. Enough to be representative, small enough to stay fast.
_TEXT_SAMPLE_BYTES = 64 * 1024

#: Minimum confidence for treating sniffed content as CSV when there is no extension
#: hint. Deliberately conservative: mistaking prose for CSV produces nonsense columns,
#: whereas treating a CSV as prose still yields readable text.
_CSV_MIN_ROWS = 2


@dataclass(frozen=True, slots=True)
class ArchiveLimits:
    """Ceilings applied to ZIP-based formats (DOCX, XLSX)."""

    max_uncompressed_bytes: int
    max_entries: int


def detect_file_type(content: bytes, *, filename: str | None = None) -> SupportedFileType:
    """Identify a file's real type from its content.

    Args:
        content: The file's bytes.
        filename: The client-supplied name. Used **only** to break the genuine tie
            between CSV and TXT, which are both plain text and cannot be distinguished
            by signature. It is never trusted to override what the bytes say.

    Returns:
        The detected type.

    Raises:
        UnsupportedFileTypeError: If the content is not one of the supported formats.
        ValidationError: If the content is recognisably a supported container but
            damaged beyond reading.
    """
    if not content:
        raise ValidationError("The uploaded file is empty")

    if content.startswith(_PDF_SIGNATURE):
        return SupportedFileType.PDF

    if content.startswith(_ZIP_SIGNATURE):
        return _detect_ooxml_type(content)

    if content.startswith(_ZIP_EMPTY_SIGNATURES):
        raise ValidationError(
            "This file looks like an empty or damaged Office document and cannot be read"
        )

    text = _decode_text(content)
    if text is None:
        raise UnsupportedFileTypeError(
            "This file type is not supported. Supported formats are PDF, DOCX, TXT, CSV and XLSX.",
            details={"supported": [file_type.value for file_type in SupportedFileType]},
        )

    return _resolve_text_type(text, filename=filename)


def _detect_ooxml_type(content: bytes) -> SupportedFileType:
    """Tell DOCX from XLSX by looking at what the archive contains."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = set(archive.namelist())
    except zipfile.BadZipFile as exc:
        raise ValidationError(
            "This file appears to be a damaged Office document and cannot be read"
        ) from exc

    if _DOCX_MARKER in names:
        return SupportedFileType.DOCX
    if _XLSX_MARKER in names:
        return SupportedFileType.XLSX

    raise UnsupportedFileTypeError(
        "This looks like a ZIP archive but not a Word or Excel document. Supported "
        "formats are PDF, DOCX, TXT, CSV and XLSX.",
    )


def _decode_text(content: bytes) -> str | None:
    """Decode content as text, or return ``None`` if it is not text at all.

    Encoding is *detected*, never assumed. A Windows-authored file is frequently cp1252,
    and decoding it as UTF-8 either fails or produces mojibake that would silently poison
    every summary and citation built from it later.
    """
    sample = content[:_TEXT_SAMPLE_BYTES]

    # A NUL byte in the first block is the strongest single signal of binary content;
    # no text encoding we support produces one in normal prose.
    if b"\x00" in sample:
        return None

    best_match = from_bytes(sample).best()
    if best_match is None:
        return None

    try:
        return content.decode(best_match.encoding, errors="strict")
    except (UnicodeDecodeError, LookupError):
        # The sample decoded but the whole file did not: a truncated multi-byte
        # sequence at the sample boundary is normal, genuine binary is not.
        try:
            return content.decode(best_match.encoding, errors="replace")
        except LookupError:
            return None


def _resolve_text_type(text: str, *, filename: str | None) -> SupportedFileType:
    """Decide whether text content is CSV or plain TXT.

    This is the one place an extension legitimately matters. CSV *is* text, so no
    signature can separate them; a file of comma-containing prose and a real CSV are
    only distinguishable by intent. The extension expresses that intent, and the
    structure check confirms or overrules it.
    """
    extension_hint = SupportedFileType.from_filename(filename) if filename else None
    looks_delimited = _looks_like_csv(text)

    if extension_hint is SupportedFileType.CSV:
        # Trust an explicit .csv unless it plainly is not delimited, in which case
        # treating it as text still extracts something useful rather than failing.
        return SupportedFileType.CSV if looks_delimited else SupportedFileType.TXT

    if extension_hint is SupportedFileType.TXT:
        return SupportedFileType.TXT

    # No usable hint: fall back to structure. Bias toward TXT, because reading a CSV as
    # prose degrades gracefully while reading prose as a CSV produces nonsense columns.
    return SupportedFileType.CSV if looks_delimited else SupportedFileType.TXT


def _looks_like_csv(text: str) -> bool:
    """Return whether text has a consistent delimited structure."""
    sample = text[:_TEXT_SAMPLE_BYTES]
    lines = [line for line in sample.splitlines() if line.strip()]

    if len(lines) < _CSV_MIN_ROWS:
        return False

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return False

    # The sniffer is willing to find a delimiter in almost anything, so confirm that the
    # rows genuinely share a column count and that there is more than one column.
    reader = csv.reader(lines[:20], dialect)
    try:
        counts = [len(row) for row in reader]
    except csv.Error:
        return False

    if not counts or counts[0] < 2:
        return False

    return len(set(counts)) == 1


def check_archive_safety(content: bytes, limits: ArchiveLimits) -> None:
    """Reject ZIP-based documents that would expand to an unreasonable size.

    DOCX and XLSX files are ZIP archives, so a small upload can declare gigabytes of
    uncompressed content — a "zip bomb". The declared sizes are read from the archive's
    directory **without extracting anything**, so the check itself is cheap and safe.

    Args:
        content: The archive bytes.
        limits: Ceilings to enforce.

    Raises:
        ValidationError: If the archive is damaged or exceeds a limit.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            entries = archive.infolist()
    except zipfile.BadZipFile as exc:
        raise ValidationError(
            "This file appears to be a damaged Office document and cannot be read"
        ) from exc

    if len(entries) > limits.max_entries:
        raise ValidationError(
            "This document contains too many internal parts to process safely",
            details={"max_entries": limits.max_entries},
        )

    total_uncompressed = sum(entry.file_size for entry in entries)
    if total_uncompressed > limits.max_uncompressed_bytes:
        raise ValidationError(
            "This document expands to too much content to process safely",
            details={"max_uncompressed_bytes": limits.max_uncompressed_bytes},
        )
