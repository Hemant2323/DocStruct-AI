"""Shared logic for tabular formats (CSV and XLSX).

Both formats describe the same thing — rows of typed values under named columns — so the
type inference, profiling and row-batching live here and each parser only handles reading
its own container.

The important idea is that **tabular files are treated as data, not prose.** A 50,000-row
spreadsheet cannot be handed to a model row by row, and would be useless if it were. So
each table produces:

* one ``schema`` block describing the columns, their inferred types and a data profile,
  which answers structural questions ("what columns are there?", "what date range?")
  without reading a single data row; plus
* ``sheet_row`` blocks holding batches of rows, which serve lookups of specific records.

A single row would be too small to be a useful retrieval unit, which is why they are
batched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Any

#: Values that mean "no value" in exported data.
_NULL_TOKENS = {"", "na", "n/a", "null", "none", "-", "--", "nan"}

_INTEGER = re.compile(r"^[+-]?\d{1,15}$")
_DECIMAL = re.compile(r"^[+-]?(\d{1,15}(\.\d+)?|\.\d+)$")
_BOOLEAN = {"true", "false", "yes", "no", "y", "n", "0", "1"}
_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y", "%d %b %Y")

#: Rows sampled when inferring a column's type. Enough to be representative without
#: walking a very large file twice.
_TYPE_SAMPLE_ROWS = 200


class ColumnType(StrEnum):
    """The inferred type of a column."""

    INTEGER = "integer"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    DATE = "date"
    TEXT = "text"
    EMPTY = "empty"


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """What we know about one column.

    Attributes:
        name: The column header.
        column_type: Inferred type.
        null_count: Values that were blank or a null token.
        distinct_sample: Up to a handful of example values, for the schema block.
        minimum / maximum: Range, for numeric and date columns only.
    """

    name: str
    column_type: ColumnType
    null_count: int
    distinct_sample: list[str]
    minimum: str | None = None
    maximum: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        """Compact form for the block's metadata column."""
        data: dict[str, Any] = {
            "name": self.name,
            "type": self.column_type.value,
            "null_count": self.null_count,
        }
        if self.minimum is not None:
            data["min"] = self.minimum
        if self.maximum is not None:
            data["max"] = self.maximum
        return data


@dataclass(frozen=True, slots=True)
class TableProfile:
    """A whole table's structure and shape."""

    columns: list[ColumnProfile]
    row_count: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "column_count": len(self.columns),
            "columns": [column.to_metadata() for column in self.columns],
        }

    def describe(self, *, title: str) -> str:
        """Render the profile as readable text for the schema block.

        Written as prose-with-structure rather than raw JSON because this text is what a
        language model will read in a later milestone, and it is also what a person sees
        when inspecting extracted content.
        """
        lines = [
            title,
            f"{self.row_count} data rows, {len(self.columns)} columns.",
            "",
            "Columns:",
        ]
        for column in self.columns:
            parts = [f"  - {column.name} ({column.column_type.value}"]
            if column.minimum is not None and column.maximum is not None:
                parts.append(f", range {column.minimum} to {column.maximum}")
            if column.null_count:
                parts.append(f", {column.null_count} empty")
            parts.append(")")
            if column.distinct_sample:
                sample = ", ".join(column.distinct_sample[:3])
                parts.append(f" e.g. {sample}")
            lines.append("".join(parts))
        return "\n".join(lines)


def is_null(value: str) -> bool:
    """Return whether a raw cell value means 'no value'."""
    return value.strip().lower() in _NULL_TOKENS


def infer_column_type(values: list[str]) -> ColumnType:
    """Infer a column's type from its values.

    A column is only given a specific type when **every** non-null sampled value fits it.
    One stray word in a numeric column means the column is text — being conservative here
    matters, because a wrong type would make the profile's stated range a lie.
    """
    present = [value.strip() for value in values if not is_null(value)]
    if not present:
        return ColumnType.EMPTY

    if all(_INTEGER.match(value) for value in present):
        return ColumnType.INTEGER
    if all(_DECIMAL.match(value) for value in present):
        return ColumnType.DECIMAL
    if all(value.lower() in _BOOLEAN for value in present):
        return ColumnType.BOOLEAN
    if all(_parse_date(value) is not None for value in present):
        return ColumnType.DATE
    return ColumnType.TEXT


def _parse_date(value: str) -> date | None:
    """Try the common date layouts. Returns ``None`` if none match."""
    for date_format in _DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), date_format).date()
        except ValueError:
            continue
    return None


def profile_table(headers: list[str], rows: list[list[str]]) -> TableProfile:
    """Build a profile of a table from its headers and rows."""
    columns: list[ColumnProfile] = []

    for index, header in enumerate(headers):
        values = [row[index] if index < len(row) else "" for row in rows]
        sample = values[:_TYPE_SAMPLE_ROWS]
        column_type = infer_column_type(sample)

        present = [value.strip() for value in values if not is_null(value)]
        null_count = len(values) - len(present)

        minimum, maximum = _range_for(column_type, present)

        # Preserve first-seen order so the sample reads like the file, not like a set.
        seen: list[str] = []
        for value in present:
            if value not in seen:
                seen.append(value)
            if len(seen) >= 5:
                break

        columns.append(
            ColumnProfile(
                name=header or f"column_{index + 1}",
                column_type=column_type,
                null_count=null_count,
                distinct_sample=seen,
                minimum=minimum,
                maximum=maximum,
            )
        )

    return TableProfile(columns=columns, row_count=len(rows))


def _range_for(column_type: ColumnType, values: list[str]) -> tuple[str | None, str | None]:
    """Return the min and max of a column, for types where a range is meaningful."""
    if not values:
        return None, None

    if column_type in (ColumnType.INTEGER, ColumnType.DECIMAL):
        try:
            numbers = [float(value) for value in values]
        except ValueError:
            return None, None
        low, high = min(numbers), max(numbers)
        if column_type is ColumnType.INTEGER:
            return str(int(low)), str(int(high))
        return f"{low:g}", f"{high:g}"

    if column_type is ColumnType.DATE:
        dates = [parsed for value in values if (parsed := _parse_date(value)) is not None]
        if not dates:
            return None, None
        return min(dates).isoformat(), max(dates).isoformat()

    return None, None


def render_rows(headers: list[str], rows: list[list[str]]) -> str:
    """Render a batch of rows as a readable, self-describing text block.

    Each row repeats its column names. That is more verbose than a bare table, but it
    means a single retrieved block is understandable on its own — without it, a chunk of
    rows pulled out of context would be a grid of numbers with no headings.
    """
    lines: list[str] = []
    for row in rows:
        pairs = [
            f"{header}: {row[index].strip() if index < len(row) else ''}"
            for index, header in enumerate(headers)
        ]
        lines.append(" | ".join(pairs))
    return "\n".join(lines)


def normalise_headers(raw: list[str], max_columns: int) -> list[str]:
    """Clean header names and make them unique.

    Duplicate or blank headers are common in exported spreadsheets, and both would make
    the rendered rows ambiguous.
    """
    headers: list[str] = []
    seen: dict[str, int] = {}

    for index, value in enumerate(raw[:max_columns]):
        name = " ".join(str(value).split()) if value is not None else ""
        if not name:
            name = f"column_{index + 1}"

        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1

        headers.append(name)

    return headers
