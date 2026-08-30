"""Upload security helpers.

These run before anything expensive touches an uploaded file. The guiding rule is that a
hostile or merely enormous file must cost us a bounded amount of memory and fail with a
clear message, never exhaust the process.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import quote

from app.core.exceptions import FileTooLargeError, ValidationError

#: Read size when streaming an upload. Large enough to be efficient, small enough that
#: the overshoot past the limit before we notice is negligible.
CHUNK_SIZE = 64 * 1024

#: Longest filename we keep for display. Anything longer is truncated, not rejected —
#: the name is cosmetic, so a long one is not a reason to refuse the upload.
MAX_FILENAME_LENGTH = 255

#: Characters that must never survive into a stored filename: path separators, the
#: parent-directory marker, and anything that could break out of a log line or a
#: terminal. The filename is never used as a path, so this is defence in depth for
#: display contexts.
_UNSAFE_FILENAME_CHARS = re.compile(r'[\x00-\x1f\x7f<>:"/\\|?*]')


class AsyncByteReader(Protocol):
    """The part of Starlette's ``UploadFile`` this module needs.

    Depending on a protocol rather than the concrete class keeps ``core`` free of web
    framework imports and makes these functions testable with a plain fake.
    """

    async def read(self, size: int = -1) -> bytes: ...


@dataclass(frozen=True, slots=True)
class ReadResult:
    """The outcome of reading an upload.

    Attributes:
        content: The file's bytes.
        size_bytes: How many bytes were read.
        sha256: Hex digest of the content, computed during the read so the bytes are
            only walked once.
    """

    content: bytes
    size_bytes: int
    sha256: str


async def read_upload_within_limit(source: AsyncByteReader, max_bytes: int) -> ReadResult:
    """Read an upload into memory, refusing to exceed ``max_bytes``.

    The limit is enforced **while reading**, not afterwards. Checking ``Content-Length``
    instead would be trivially defeated: the header is supplied by the client and may be
    absent or a lie. Reading in chunks and stopping the moment the cap is passed means a
    10 GB upload costs us one chunk more than the limit, not 10 GB.

    Args:
        source: Anything with an async ``read(size)``, i.e. a Starlette ``UploadFile``.
        max_bytes: Hard ceiling on the number of bytes accepted.

    Returns:
        The content, its size, and its SHA-256 digest.

    Raises:
        FileTooLargeError: If the upload exceeds ``max_bytes``.
        ValidationError: If the upload is empty.
    """
    chunks: list[bytes] = []
    total = 0
    digest = hashlib.sha256()

    while True:
        chunk = await source.read(CHUNK_SIZE)
        if not chunk:
            break

        total += len(chunk)
        if total > max_bytes:
            # Drop what we have rather than holding it while raising.
            chunks.clear()
            raise FileTooLargeError(
                f"File exceeds the maximum upload size of {max_bytes // (1024 * 1024)} MB",
                details={"max_bytes": max_bytes},
            )

        digest.update(chunk)
        chunks.append(chunk)

    if total == 0:
        raise ValidationError("The uploaded file is empty")

    return ReadResult(content=b"".join(chunks), size_bytes=total, sha256=digest.hexdigest())


def sanitize_filename(filename: str | None) -> str:
    """Return a filename that is safe to store as display metadata and to log.

    This is **not** what protects the filesystem — files are stored under a generated
    UUID, so no user-controlled string ever reaches a path (see ``storage/local.py``).
    This protects everything downstream that shows the name: log lines, HTTP headers, and
    the browser. Stripping control characters stops log-injection, and stripping path
    separators means a name like ``../../etc/passwd`` cannot mislead anyone reading it.

    Args:
        filename: The client-supplied name, which may be missing or hostile.

    Returns:
        A non-empty, single-line, reasonably short filename.
    """
    if not filename:
        return "untitled"

    # Normalize first: otherwise a combining sequence could reconstitute a stripped
    # character after the fact.
    cleaned = unicodedata.normalize("NFKC", filename)
    cleaned = _UNSAFE_FILENAME_CHARS.sub("", cleaned)

    # Collapse whitespace (including any newline that survived) into single spaces.
    cleaned = " ".join(cleaned.split())

    # Leading dots hide files on Unix and leading/trailing spaces confuse display.
    cleaned = cleaned.strip(". ")

    if not cleaned:
        return "untitled"

    if len(cleaned) > MAX_FILENAME_LENGTH:
        # Keep the extension, which is the part a human uses to recognise the file.
        stem, dot, extension = cleaned.rpartition(".")
        if dot and len(extension) <= 10:
            keep = MAX_FILENAME_LENGTH - len(extension) - 1
            cleaned = f"{stem[:keep]}.{extension}"
        else:
            cleaned = cleaned[:MAX_FILENAME_LENGTH]

    return cleaned


def content_disposition_attachment(filename: str) -> str:
    """Build a ``Content-Disposition`` header that survives a non-ASCII filename.

    HTTP header values are latin-1. A perfectly reasonable filename — a report titled in
    Devanagari, a spreadsheet a user uploaded with a Chinese name — is therefore not
    directly expressible, and interpolating one raises ``UnicodeEncodeError`` while the
    response is being sent. The user sees a failed download for a file that was built
    correctly.

    RFC 6266 solves this with two parameters: a plain ``filename`` holding an ASCII
    approximation for old clients, and ``filename*`` holding the real name percent-encoded
    as UTF-8, which every current browser prefers. Emitting both means the download works
    everywhere and keeps its real name where that is supported.

    The name is sanitised first, so control characters, quotes and path separators are
    already gone before either parameter is built.
    """
    safe = sanitize_filename(filename)

    # Keep the extension on the ASCII fallback: a name that transliterates to nothing
    # should still arrive as "download.xlsx" rather than as an extensionless blob.
    stem, dot, extension = safe.rpartition(".")
    if not dot:
        stem, extension = safe, ""
    ascii_stem = stem.encode("ascii", "ignore").decode("ascii").strip(". ") or "download"
    fallback = f"{ascii_stem}.{extension}" if extension else ascii_stem

    encoded = quote(safe, safe="")
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{encoded}"
