"""Word document parser.

DOCX gives the best structure of any format here, and gets it for free: heading styles
(``Heading 1``, ``Heading 2``, ...) are declared in the file rather than inferred, so the
section hierarchy is *fact* rather than a guess. That is why DOCX provenance carries a
``heading_path`` and why its ``is_exact`` stays ``True``.

Paragraphs and tables are walked in true document order by reading the underlying XML
body, because ``document.paragraphs`` and ``document.tables`` are separate sequences that
lose their interleaving.
"""

from __future__ import annotations

import io
import re
import zipfile

import docx
from docx.document import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.core.exceptions import (
    CorruptFileError,
    DocumentTooComplexError,
    EmptyDocumentError,
)
from app.ingestion.parsers.base import DocumentParser, ParserLimits
from app.models.domain.content import (
    BlockType,
    ContentBlock,
    ParsedDocument,
    ParseStats,
    Provenance,
    ProvenanceKind,
)
from app.models.domain.document import SupportedFileType

#: Matches the built-in heading styles, capturing the level. Localised style names are
#: not matched, which is a known limitation rather than a silent wrong answer: an
#: unmatched heading simply becomes a paragraph.
_HEADING_STYLE = re.compile(r"^heading\s*(\d+)$", re.IGNORECASE)

#: Styles that mark a list item.
_LIST_STYLE = re.compile(r"(list|bullet)", re.IGNORECASE)


class DocxParser(DocumentParser):
    """Extracts blocks from a Word document."""

    file_type = SupportedFileType.DOCX

    def parse(self, content: bytes, limits: ParserLimits) -> ParsedDocument:
        document = self._open(content)

        blocks: list[ContentBlock] = []
        warnings: list[str] = []
        heading_path: list[str] = []
        paragraph_index = 0
        table_count = 0

        for element in _iter_body(document):
            if len(blocks) >= limits.max_blocks:
                raise DocumentTooComplexError(
                    "This document contains too many sections to process",
                    details={"max_blocks": limits.max_blocks},
                )

            if isinstance(element, Paragraph):
                block = self._paragraph_block(element, paragraph_index, len(blocks), heading_path)
                paragraph_index += 1
                if block is None:
                    continue

                if block.block_type is BlockType.HEADING:
                    _update_heading_path(heading_path, block.level or 1, block.text)
                    # The heading names its own section, so its path is the ancestors.
                    block = ContentBlock(
                        block_type=block.block_type,
                        text=block.text,
                        order=block.order,
                        provenance=Provenance.for_paragraph(
                            paragraph_index - 1,
                            heading_path=tuple(heading_path[:-1]),
                        ),
                        level=block.level,
                        metadata=block.metadata,
                    )

                blocks.append(block)

            elif isinstance(element, Table):
                table_block = self._table_block(
                    element, len(blocks), heading_path, limits, warnings
                )
                if table_block is not None:
                    blocks.append(table_block)
                    table_count += 1

        if not blocks:
            raise EmptyDocumentError("This document contains no readable text")

        return ParsedDocument(
            blocks=blocks,
            file_type=self.file_type,
            stats=ParseStats(
                tables=table_count,
                characters=sum(block.char_count for block in blocks),
            ),
            warnings=warnings,
        )

    def _open(self, content: bytes) -> DocxDocument:
        """Open the document, converting library failures into a clear message."""
        try:
            return docx.Document(io.BytesIO(content))
        except (zipfile.BadZipFile, KeyError, ValueError, OSError) as exc:
            raise CorruptFileError(
                "This Word document could not be opened. It may be damaged or password protected."
            ) from exc

    def _paragraph_block(
        self,
        paragraph: Paragraph,
        paragraph_index: int,
        order: int,
        heading_path: list[str],
    ) -> ContentBlock | None:
        """Convert one paragraph to a block, or ``None`` if it is empty."""
        text = paragraph.text.strip()
        if not text:
            return None

        style_name = (paragraph.style.name if paragraph.style is not None else "") or ""
        heading_match = _HEADING_STYLE.match(style_name.strip())

        if heading_match:
            return ContentBlock(
                block_type=BlockType.HEADING,
                text=text,
                order=order,
                provenance=Provenance.for_paragraph(
                    paragraph_index, heading_path=tuple(heading_path)
                ),
                level=int(heading_match.group(1)),
            )

        block_type = BlockType.LIST_ITEM if _LIST_STYLE.search(style_name) else BlockType.PARAGRAPH
        return ContentBlock(
            block_type=block_type,
            text=text,
            order=order,
            provenance=Provenance.for_paragraph(paragraph_index, heading_path=tuple(heading_path)),
        )

    def _table_block(
        self,
        table: Table,
        order: int,
        heading_path: list[str],
        limits: ParserLimits,
        warnings: list[str],
    ) -> ContentBlock | None:
        """Convert a table to a single block.

        The whole table becomes one block because it is atomic: the chunker must never
        split it, and rows separated from their header row are meaningless.
        """
        rows: list[list[str]] = []
        for row in table.rows[: limits.max_rows]:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            rows.append(cells[: limits.max_columns])

        if not rows or not any(any(cell for cell in row) for row in rows):
            return None

        if len(table.rows) > limits.max_rows:
            warnings.append(f"A table was truncated at {limits.max_rows:,} rows.")

        header, *body = rows
        lines = [" | ".join(header)]
        lines.append(" | ".join("---" for _ in header))
        lines.extend(" | ".join(row) for row in body)

        return ContentBlock(
            block_type=BlockType.TABLE,
            text="\n".join(lines),
            order=order,
            # A table has no paragraph index of its own; its position in the document is
            # what locates it, and the heading path says which section it belongs to.
            provenance=Provenance(
                kind=ProvenanceKind.PARAGRAPH,
                label=f"table under '{heading_path[-1]}'" if heading_path else "table",
                # A table carries no paragraph index of its own, so the location is
                # known only to section granularity. Recorded honestly rather than
                # invented (CLAUDE.md §4.3).
                is_exact=False,
                heading_path=tuple(heading_path),
            ),
            metadata={"rows": len(rows), "columns": len(header)},
        )


def _iter_body(document: DocxDocument) -> list[Paragraph | Table]:
    """Yield paragraphs and tables in true document order.

    ``document.paragraphs`` and ``document.tables`` are separate lists, so using them
    would place every table after every paragraph and destroy the reading order — and
    with it the meaning of ``order`` on each block. Walking the XML body preserves it.
    """
    body = document.element.body
    items: list[Paragraph | Table] = []

    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            items.append(Paragraph(child, document))
        elif child.tag == qn("w:tbl"):
            items.append(Table(child, document))

    return items


def _update_heading_path(heading_path: list[str], level: int, text: str) -> None:
    """Maintain the stack of enclosing section headings.

    A level-2 heading replaces any deeper entries and sits under the current level-1, so
    a paragraph can later report the full path it lives under.
    """
    # Trim to the parent depth, padding if a level was skipped (H1 straight to H3).
    del heading_path[level - 1 :]
    while len(heading_path) < level - 1:
        heading_path.append("")
    heading_path.append(text)
