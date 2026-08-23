"""Plain text parser.

Splits a text file into paragraph blocks on blank lines, tracking line numbers so every
block can say exactly where it came from.
"""

from __future__ import annotations

import re

from charset_normalizer import from_bytes

from app.core.exceptions import DocumentTooComplexError, EmptyDocumentError
from app.ingestion.parsers.base import DocumentParser, ParserLimits
from app.models.domain.content import (
    BlockType,
    ContentBlock,
    ParsedDocument,
    ParseStats,
    Provenance,
)
from app.models.domain.document import SupportedFileType

#: A line that is probably a heading: short, no trailing sentence punctuation, and either
#: all-caps or numbered ("1.", "2.3"). Plain text has no markup, so this is a heuristic —
#: recorded as such via `is_exact=False` on the provenance rather than presented as fact.
_NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*[.)]?\s+\S")
_MAX_HEADING_LENGTH = 80


class TxtParser(DocumentParser):
    """Extracts blocks from a plain text file."""

    file_type = SupportedFileType.TXT

    def parse(self, content: bytes, limits: ParserLimits) -> ParsedDocument:
        text, encoding, replaced = _decode(content)
        warnings: list[str] = []

        if replaced:
            # Surfaced rather than swallowed: the user should know some characters were
            # unreadable instead of silently receiving corrupted text.
            warnings.append(
                f"Some characters could not be decoded using {encoding} and were replaced."
            )

        lines = text.splitlines()
        blocks = self._blocks_from_lines(lines, limits)

        if not blocks:
            raise EmptyDocumentError("This text file contains no readable content")

        return ParsedDocument(
            blocks=blocks,
            file_type=self.file_type,
            stats=ParseStats(
                rows=len(lines),
                characters=sum(block.char_count for block in blocks),
            ),
            warnings=warnings,
        )

    def _blocks_from_lines(self, lines: list[str], limits: ParserLimits) -> list[ContentBlock]:
        """Group consecutive non-blank lines into blocks, tracking line numbers."""
        blocks: list[ContentBlock] = []
        buffer: list[str] = []
        buffer_start = 1  # Line numbers are 1-indexed for humans.

        def flush(end_line: int) -> None:
            """Turn the buffered lines into a block."""
            if not buffer:
                return

            joined = "\n".join(buffer).strip()
            if not joined:
                buffer.clear()
                return

            if len(blocks) >= limits.max_blocks:
                raise DocumentTooComplexError(
                    "This document contains too many sections to process",
                    details={"max_blocks": limits.max_blocks},
                )

            is_heading = _looks_like_heading(buffer)
            provenance = Provenance.for_lines(buffer_start, end_line)
            if is_heading:
                # Heading detection in plain text is inference, not fact. The location is
                # exact but the classification is not, so the flag records the doubt.
                provenance = Provenance(
                    kind=provenance.kind,
                    label=provenance.label,
                    is_exact=False,
                    line_start=provenance.line_start,
                    line_end=provenance.line_end,
                )

            blocks.append(
                ContentBlock(
                    block_type=BlockType.HEADING if is_heading else BlockType.PARAGRAPH,
                    text=joined,
                    order=len(blocks),
                    provenance=provenance,
                    level=1 if is_heading else None,
                )
            )
            buffer.clear()

        for line_number, line in enumerate(lines, start=1):
            if line.strip():
                if not buffer:
                    buffer_start = line_number
                buffer.append(line)
            else:
                flush(line_number - 1)

        flush(len(lines))
        return blocks


def _decode(content: bytes) -> tuple[str, str, bool]:
    """Decode bytes to text, detecting the encoding.

    The encoding is detected rather than assumed. Text files authored on Windows are
    frequently cp1252, and forcing UTF-8 either raises or produces mojibake — which would
    then flow into every summary and citation built from this document.

    Returns:
        ``(text, encoding_name, characters_were_replaced)``.
    """
    best_match = from_bytes(content).best()

    if best_match is not None:
        encoding = best_match.encoding
        try:
            return content.decode(encoding), encoding, False
        except (UnicodeDecodeError, LookupError):
            pass
    else:
        encoding = "utf-8"

    # Last resort. Replacing is better than failing outright — the user gets most of
    # their document plus an explicit warning that some characters were lost.
    try:
        return content.decode(encoding, errors="replace"), encoding, True
    except LookupError:
        return content.decode("utf-8", errors="replace"), "utf-8", True


def _looks_like_heading(lines: list[str]) -> bool:
    """Return whether a group of lines is probably a heading.

    Only single short lines qualify: a heading that wraps is indistinguishable from a
    sentence, and guessing wrong turns body text into structure.
    """
    if len(lines) != 1:
        return False

    line = lines[0].strip()
    if not line or len(line) > _MAX_HEADING_LENGTH:
        return False

    # A trailing full stop marks a sentence, not a title.
    if line.endswith((".", ",", ";", ":")):
        return bool(_NUMBERED_HEADING.match(line)) and line.endswith(":")

    if _NUMBERED_HEADING.match(line):
        return True

    # ALL CAPS with at least one letter.
    letters = [character for character in line if character.isalpha()]
    return bool(letters) and all(character.isupper() for character in letters)
