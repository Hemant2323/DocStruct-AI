"""Domain models for extracted document content.

The single most important idea in the system lives here: **every parser, whatever the
file format, produces the same shape.** A PDF, a spreadsheet and a Word document all come
out as an ordered list of :class:`ContentBlock`.

That is what makes the rest of the platform format-agnostic. The normalizer, the chunker,
and later the summarizer and question-answerer all operate on ``ContentBlock`` and never
learn what a page or a worksheet is. It is also why adding a new file format costs one
new parser file and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.models.domain.document import SupportedFileType


class BlockType(StrEnum):
    """What kind of content a block holds.

    These values are persisted and sent over the API, so changing one is a breaking
    change. A test pins them for that reason.
    """

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    #: Column names, inferred types and a data profile for a CSV file or one worksheet.
    #: Lets a question about structure ("what columns are there?") be answered without
    #: reading a single data row.
    SCHEMA = "schema"
    #: A batch of consecutive rows from a CSV file or worksheet.
    SHEET_ROW = "sheet_row"
    CAPTION = "caption"

    @property
    def is_tabular(self) -> bool:
        """True for blocks that represent structured rows rather than prose."""
        return self in {BlockType.TABLE, BlockType.SCHEMA, BlockType.SHEET_ROW}

    @property
    def is_atomic(self) -> bool:
        """True when splitting the block would destroy its meaning.

        Half a table is worse than no table, so the chunker never splits these.
        """
        return self in {BlockType.TABLE, BlockType.SCHEMA}


class ProvenanceKind(StrEnum):
    """Which addressing scheme a block's source location uses.

    Each format can only describe positions in its own terms — a PDF has pages, a
    spreadsheet has sheets and cells — so the kind says how to read the other fields.
    """

    PAGE = "page"
    PARAGRAPH = "paragraph"
    LINE_RANGE = "line_range"
    ROW_RANGE = "row_range"
    CELL_RANGE = "cell_range"
    #: The source location is genuinely unknown. Used rather than inventing a number.
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where in the original file a block came from.

    **This is mandatory, not decoration.** Citations are an MVP requirement, and a
    citation is only possible when every piece of text can be traced back to its origin.
    A parser that drops provenance is a broken parser (CLAUDE.md §4.3).

    Only the fields relevant to ``kind`` are populated; the rest stay ``None``. Never
    guess a value — use :meth:`unknown` if the location genuinely cannot be determined.

    Attributes:
        kind: How to interpret the location fields.
        label: Human-readable location, e.g. ``"page 4"`` or ``"rows 120-140"``.
            Precomputed by the parser, which is the only component that understands its
            own format. Without this the frontend would need a five-way branch to render
            a citation.
        is_exact: ``False`` when the location is approximate — for example a PDF heading
            inferred from font size. Honest uncertainty beats a confident wrong number.
        heading_path: Enclosing section headings, outermost first. Turns a citation into
            "contract.docx, Payment Terms > Late Fees" instead of "paragraph 143".
    """

    kind: ProvenanceKind
    label: str
    is_exact: bool = True

    # --- Page-based (PDF) --------------------------------------------------
    page: int | None = None

    # --- Paragraph-based (DOCX) --------------------------------------------
    paragraph_index: int | None = None

    # --- Line-based (TXT) --------------------------------------------------
    line_start: int | None = None
    line_end: int | None = None

    # --- Row-based (CSV, XLSX) ---------------------------------------------
    row_start: int | None = None
    row_end: int | None = None

    # --- Sheet/cell-based (XLSX) -------------------------------------------
    sheet_name: str | None = None
    cell_range: str | None = None

    # --- Structural context (any format) -----------------------------------
    heading_path: tuple[str, ...] = ()

    @classmethod
    def for_page(cls, page: int, *, is_exact: bool = True) -> Provenance:
        """Provenance for a PDF page (1-indexed)."""
        return cls(
            kind=ProvenanceKind.PAGE,
            label=f"page {page}",
            page=page,
            is_exact=is_exact,
        )

    @classmethod
    def for_paragraph(cls, index: int, *, heading_path: tuple[str, ...] = ()) -> Provenance:
        """Provenance for a DOCX paragraph (0-indexed internally, 1-indexed in the label)."""
        return cls(
            kind=ProvenanceKind.PARAGRAPH,
            label=f"paragraph {index + 1}",
            paragraph_index=index,
            heading_path=heading_path,
        )

    @classmethod
    def for_lines(cls, start: int, end: int) -> Provenance:
        """Provenance for a range of lines in a text file (1-indexed, inclusive)."""
        label = f"line {start}" if start == end else f"lines {start}-{end}"
        return cls(
            kind=ProvenanceKind.LINE_RANGE,
            label=label,
            line_start=start,
            line_end=end,
        )

    @classmethod
    def for_rows(cls, start: int, end: int) -> Provenance:
        """Provenance for a range of CSV rows (1-indexed, inclusive, excluding header)."""
        label = f"row {start}" if start == end else f"rows {start}-{end}"
        return cls(
            kind=ProvenanceKind.ROW_RANGE,
            label=label,
            row_start=start,
            row_end=end,
        )

    @classmethod
    def for_cells(
        cls, sheet_name: str, cell_range: str, *, row_start: int, row_end: int
    ) -> Provenance:
        """Provenance for a rectangular region of a worksheet."""
        return cls(
            kind=ProvenanceKind.CELL_RANGE,
            label=f"{sheet_name}!{cell_range}",
            sheet_name=sheet_name,
            cell_range=cell_range,
            row_start=row_start,
            row_end=row_end,
        )

    @classmethod
    def unknown(cls, reason: str) -> Provenance:
        """Provenance for content whose source location cannot be determined.

        Deliberately explicit. CLAUDE.md forbids inventing a plausible-looking number,
        so an honest "unknown" is the only alternative.
        """
        return cls(
            kind=ProvenanceKind.UNKNOWN,
            label=f"location unknown ({reason})",
            is_exact=False,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the JSON database column, omitting unset fields."""
        data: dict[str, Any] = {
            "kind": self.kind.value,
            "label": self.label,
            "is_exact": self.is_exact,
        }
        optional = {
            "page": self.page,
            "paragraph_index": self.paragraph_index,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "row_start": self.row_start,
            "row_end": self.row_end,
            "sheet_name": self.sheet_name,
            "cell_range": self.cell_range,
        }
        data.update({key: value for key, value in optional.items() if value is not None})
        if self.heading_path:
            data["heading_path"] = list(self.heading_path)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Provenance:
        """Rebuild from the JSON database column."""
        return cls(
            kind=ProvenanceKind(data["kind"]),
            label=data["label"],
            is_exact=bool(data.get("is_exact", True)),
            page=data.get("page"),
            paragraph_index=data.get("paragraph_index"),
            line_start=data.get("line_start"),
            line_end=data.get("line_end"),
            row_start=data.get("row_start"),
            row_end=data.get("row_end"),
            sheet_name=data.get("sheet_name"),
            cell_range=data.get("cell_range"),
            heading_path=tuple(data.get("heading_path", ())),
        )


@dataclass(frozen=True, slots=True)
class ContentBlock:
    """One piece of extracted content, in the canonical shape every parser produces.

    Attributes:
        block_type: What kind of content this is.
        text: The extracted text, normalized.
        order: Position within the document, starting at 0. Defines reading order.
        provenance: Where it came from in the source file. Never empty.
        level: Heading depth (1 = top level) for headings, otherwise ``None``.
        metadata: Format-specific extras — column types for a schema block, row counts
            for a table. Never carries secrets or credentials.
    """

    block_type: BlockType
    text: str
    order: int
    provenance: Provenance
    level: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        """Length of the text, used by the chunker to size chunks."""
        return len(self.text)

    def with_order(self, order: int) -> ContentBlock:
        """Return a copy at a new position.

        Blocks are frozen, so the normalizer renumbers by building new ones rather than
        mutating — which keeps a parser's output an immutable record of what it saw.
        """
        return ContentBlock(
            block_type=self.block_type,
            text=self.text,
            order=order,
            provenance=self.provenance,
            level=self.level,
            metadata=self.metadata,
        )

    def with_text(self, text: str) -> ContentBlock:
        """Return a copy with different text, preserving order and provenance."""
        return ContentBlock(
            block_type=self.block_type,
            text=text,
            order=self.order,
            provenance=self.provenance,
            level=self.level,
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class ParseStats:
    """What the parser saw, for display and for spotting silent truncation.

    Attributes:
        pages: PDF pages read.
        sheets: worksheets read.
        rows: data rows read across all sheets/files.
        tables: tables detected.
        characters: total extracted characters.
    """

    pages: int = 0
    sheets: int = 0
    rows: int = 0
    tables: int = 0
    characters: int = 0

    @property
    def page_count(self) -> int | None:
        """A single "size" figure for display: pages for documents, sheets for workbooks."""
        return self.pages or self.sheets or None


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Everything a parser extracted from one file.

    Attributes:
        blocks: The content, in reading order.
        file_type: Which format produced it.
        stats: What the parser saw.
        warnings: Non-fatal problems — "3 pages had no extractable text". These are
            **surfaced, never swallowed**: a partial success must not look like a clean
            one (CLAUDE.md §2.2).
    """

    blocks: list[ContentBlock]
    file_type: SupportedFileType
    stats: ParseStats = field(default_factory=ParseStats)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True when nothing usable was extracted."""
        return not any(block.text.strip() for block in self.blocks)

    @property
    def total_characters(self) -> int:
        return sum(block.char_count for block in self.blocks)
