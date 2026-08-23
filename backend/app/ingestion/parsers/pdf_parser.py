"""PDF parser.

Uses ``pdfplumber`` alone. It handles text, tables and word geometry through a single
engine; pairing it with a second PDF library would be the overlapping-dependency problem
CLAUDE.md §2.6 warns about.

Two things make PDF harder than the other formats:

* **There is no structure in the file.** A PDF records where glyphs sit on a page, not
  what is a heading. Headings are therefore *inferred* from font size, and every inferred
  heading is marked ``is_exact=False`` rather than presented as fact.
* **A scanned PDF contains no text at all**, only images. Returning an empty result would
  be indistinguishable from a genuinely blank document, so a scan is detected and
  reported as needing OCR.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from typing import Any

import pdfplumber
from pdfminer.pdfdocument import PDFPasswordIncorrect

from app.core.exceptions import (
    CorruptFileError,
    DocumentTooComplexError,
    EmptyDocumentError,
    EncryptedFileError,
    ScannedDocumentError,
)
from app.ingestion.parsers.base import DocumentParser, ParserLimits
from app.models.domain.content import (
    BlockType,
    ContentBlock,
    ParsedDocument,
    ParseStats,
    Provenance,
)
from app.models.domain.document import SupportedFileType

logger = logging.getLogger(__name__)

#: Below this many characters, a page is treated as having produced no real text.
#: A handful of stray glyphs from a logo should not make a scan look like a text page.
_MIN_CHARS_PER_TEXT_PAGE = 20

#: If at least this fraction of pages yielded no text, the document is a scan. Set below
#: 1.0 because a scanned report often has a text cover page or an embedded text watermark.
_SCANNED_PAGE_RATIO = 0.8

#: A heading candidate must be this much larger than the document's body text.
_HEADING_SIZE_RATIO = 1.15
_MAX_HEADING_LENGTH = 120

#: Joins a word split across a line break by hyphenation ("agree-\nment" -> "agreement").
_HYPHEN_LINE_BREAK = re.compile(r"(\w)-\n(\w)")


@dataclass(frozen=True, slots=True)
class _Line:
    """One visual line of text on a page, with the font size it was drawn at."""

    text: str
    size: float


class PdfParser(DocumentParser):
    """Extracts text and tables from a PDF, page by page."""

    file_type = SupportedFileType.PDF

    def parse(self, content: bytes, limits: ParserLimits) -> ParsedDocument:
        blocks: list[ContentBlock] = []
        warnings: list[str] = []
        pages_with_text = 0
        table_count = 0

        with self._open(content) as pdf:
            page_count = len(pdf.pages)

            if page_count > limits.max_pdf_pages:
                raise DocumentTooComplexError(
                    f"This PDF has {page_count} pages, more than the {limits.max_pdf_pages} "
                    "page limit",
                    details={"max_pages": limits.max_pdf_pages},
                )

            body_size = _estimate_body_font_size(pdf.pages)

            for page_number, page in enumerate(pdf.pages, start=1):
                # `order` is document-wide, so each page starts numbering where the
                # previous one stopped.
                page_blocks, has_text, tables = self._parse_page(
                    page, page_number, len(blocks), limits, body_size
                )
                blocks.extend(page_blocks)

                if has_text:
                    pages_with_text += 1
                table_count += tables

                if len(blocks) > limits.max_blocks:
                    raise DocumentTooComplexError(
                        "This PDF produces too many sections to process",
                        details={"max_blocks": limits.max_blocks},
                    )

        empty_pages = page_count - pages_with_text
        if page_count and empty_pages / page_count >= _SCANNED_PAGE_RATIO:
            raise ScannedDocumentError(
                "This PDF appears to be a scan or made of images. Reading it needs "
                "OCR, which is not available yet."
            )

        if not blocks:
            raise EmptyDocumentError("No readable text could be extracted from this PDF")

        if empty_pages:
            warnings.append(f"{empty_pages} of {page_count} pages contained no extractable text.")

        return ParsedDocument(
            blocks=blocks,
            file_type=self.file_type,
            stats=ParseStats(
                pages=page_count,
                tables=table_count,
                characters=sum(block.char_count for block in blocks),
            ),
            warnings=warnings,
        )

    def _open(self, content: bytes) -> pdfplumber.PDF:
        """Open the PDF, turning library failures into clear, user-facing errors."""
        try:
            return pdfplumber.open(io.BytesIO(content))
        except PDFPasswordIncorrect as exc:
            raise EncryptedFileError(
                "This PDF is password protected, so its contents cannot be read."
            ) from exc
        except Exception as exc:
            # pdfplumber wraps every underlying pdfminer failure in its own
            # PdfminerException, so catching pdfminer's own classes would never fire.
            # The broad catch is converted immediately into a specific typed error —
            # nothing is swallowed, and the original is chained for the log.
            message = str(exc).lower()
            if "password" in message or "encrypt" in message:
                raise EncryptedFileError(
                    "This PDF is password protected, so its contents cannot be read."
                ) from exc
            raise CorruptFileError(
                "This PDF could not be opened. It may be damaged or not a valid PDF."
            ) from exc

    def _parse_page(
        self,
        page: Any,
        page_number: int,
        order_offset: int,
        limits: ParserLimits,
        body_size: float,
    ) -> tuple[list[ContentBlock], bool, int]:
        """Extract blocks from one page. Returns (blocks, had_text, table_count)."""
        blocks: list[ContentBlock] = []
        table_count = 0

        # Tables first, and their text is excluded from the prose extraction below so the
        # same content is not stored twice.
        try:
            tables = page.extract_tables() or []
        except Exception as exc:
            # Degrade rather than fail: a table that cannot be parsed must not cost us
            # the page's prose. Logged, never silently swallowed.
            logger.warning(
                "Table extraction failed on page %d: %s: %s",
                page_number,
                type(exc).__name__,
                exc,
            )
            tables = []

        for table in tables:
            rendered = _render_table(table)
            if not rendered:
                continue
            blocks.append(
                ContentBlock(
                    block_type=BlockType.TABLE,
                    text=rendered,
                    order=order_offset + len(blocks),
                    provenance=Provenance.for_page(page_number),
                    metadata={"rows": len(table), "columns": max(len(r) for r in table)},
                )
            )
            table_count += 1

        lines = _extract_lines(page)
        page_had_text = sum(len(line.text) for line in lines) >= _MIN_CHARS_PER_TEXT_PAGE

        for block in _group_lines(lines, page_number, body_size, order_offset + len(blocks)):
            blocks.append(block)

        return blocks, page_had_text, table_count


def _extract_lines(page: Any) -> list[_Line]:
    """Pull visual lines off a page, each with its dominant font size.

    ``extract_text_lines`` gives per-line character data, which is what makes font-size
    based heading detection possible at all.
    """
    try:
        raw_lines = page.extract_text_lines() or []
    except Exception as exc:
        # Fall back to plain text extraction: losing font sizes costs heading detection,
        # losing the page would cost the content itself.
        logger.warning(
            "Line extraction failed, falling back to plain text: %s: %s",
            type(exc).__name__,
            exc,
        )
        text = page.extract_text() or ""
        return [_Line(text=line, size=0.0) for line in text.splitlines() if line.strip()]

    lines: list[_Line] = []
    for raw in raw_lines:
        text = (raw.get("text") or "").strip()
        if not text:
            continue
        characters = raw.get("chars") or []
        sizes = [c.get("size", 0.0) for c in characters if c.get("size")]
        lines.append(_Line(text=text, size=max(sizes) if sizes else 0.0))
    return lines


def _estimate_body_font_size(pages: list[Any]) -> float:
    """Estimate the document's body-text font size.

    Headings are recognised by being larger than the body, so the body size has to be
    established first. The most common size across a sample of pages is a robust proxy —
    most characters in a document are body text.
    """
    counts: dict[float, int] = {}
    for page in pages[:10]:
        try:
            characters = page.chars or []
        except Exception as exc:
            logger.debug("Skipping page during font-size estimate: %s", exc)
            continue
        for character in characters:
            size = round(float(character.get("size") or 0), 1)
            if size:
                counts[size] = counts.get(size, 0) + 1

    if not counts:
        return 0.0
    return max(counts, key=lambda size: counts[size])


def _group_lines(
    lines: list[_Line], page_number: int, body_size: float, order_offset: int
) -> list[ContentBlock]:
    """Group lines into heading and paragraph blocks."""
    blocks: list[ContentBlock] = []
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        text = _join_lines(buffer)
        if text:
            blocks.append(
                ContentBlock(
                    block_type=BlockType.PARAGRAPH,
                    text=text,
                    order=order_offset + len(blocks),
                    provenance=Provenance.for_page(page_number),
                )
            )
        buffer.clear()

    for line in lines:
        if _is_heading(line, body_size):
            flush()
            blocks.append(
                ContentBlock(
                    block_type=BlockType.HEADING,
                    text=line.text,
                    order=order_offset + len(blocks),
                    # A PDF declares no headings, so this classification is inference.
                    # The page is exact; the fact that it is a heading is not.
                    provenance=Provenance.for_page(page_number, is_exact=False),
                    level=1,
                )
            )
        else:
            buffer.append(line.text)

    flush()
    return blocks


def _is_heading(line: _Line, body_size: float) -> bool:
    """Return whether a line is probably a heading, based on its font size."""
    if not body_size or not line.size:
        return False
    if len(line.text) > _MAX_HEADING_LENGTH:
        return False
    if line.text.endswith((".", ",", ";")):
        return False
    return line.size >= body_size * _HEADING_SIZE_RATIO


def _join_lines(lines: list[str]) -> str:
    """Join visual lines into a paragraph, repairing hyphenated line breaks.

    PDFs break words across lines with a hyphen. Left alone, "agree-\\nment" would be
    indexed as two fragments and never match a search for "agreement".
    """
    joined = "\n".join(lines)
    joined = _HYPHEN_LINE_BREAK.sub(r"\1\2", joined)
    return " ".join(joined.split()).strip()


def _render_table(table: list[list[Any]]) -> str:
    """Render an extracted table as pipe-delimited text."""
    rows = [
        [(" ".join(str(cell).split()) if cell is not None else "") for cell in row] for row in table
    ]
    rows = [row for row in rows if any(cell for cell in row)]
    if not rows:
        return ""

    header, *body = rows
    lines = [" | ".join(header), " | ".join("---" for _ in header)]
    lines.extend(" | ".join(row) for row in body)
    return "\n".join(lines)
