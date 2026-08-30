"""Rendering a converted document as Markdown.

A pure function from :class:`ConvertedDocument` to text: same document in, same string
out, with no clock, randomness or I/O involved. Markdown first because it is both a
deliverable in its own right and the thing the browser preview shows — one renderer,
one thing to keep honest. DOCX, XLSX and JSON are later phases and are deliberately
absent here.

What this renderer is careful about:

- **Changes are visible.** A corrected value is marked, and a value someone merely
  *suspects* is wrong is marked differently, so a reader can tell at a glance what the
  system changed from what it only wants to ask about.
- **Sources use what we already have.** A location is printed only from an existing
  verified :class:`Citation`; there is no new citation notation, and nothing is printed
  as a location that was not resolved from our own stored content.
- **No internals escape.** Chunk and block identifiers exist on a citation but are never
  rendered — they are database bookkeeping, not something a user's report should carry.
- **Markdown cannot be broken by data.** Cell content is escaped, because a value
  containing a pipe would otherwise silently split a row into the wrong columns.
- **A value that spans lines keeps its lines.** Source code, a stanza, a pasted log —
  anything whose line structure carries meaning is rendered in a fenced block rather
  than flattened onto one line. Inside that block the text is emitted **verbatim**: not
  escaped, not re-indented, not trimmed. Escaping is a table's requirement, and applying
  it to code would corrupt the very backslashes it is meant to protect.

Nothing here inspects *what* a value is. There is no language detection and no attempt
to tell code from prose — the only question asked is whether the source had more than
one line, which is a fact about the extracted text rather than a guess about its
meaning.
"""

from __future__ import annotations

from app.models.domain.conversion import (
    ConvertedDocument,
    ConvertedRecord,
    ConvertedSection,
    ConvertedValue,
    SectionKind,
    ValueOrigin,
)

#: Marks a value the system changed. Explained by the legend under Changes & Corrections.
CORRECTED_MARKER = "✎"

#: Marks a value the system suspects but did **not** change. The original text is what
#: is shown; this only says "look at this one".
SUGGESTED_MARKER = "⚠"

_UNVERIFIED_LABEL = "source not verified"


def render_markdown(document: ConvertedDocument) -> str:
    """Render a converted document as Markdown.

    Args:
        document: The assembled document.

    Returns:
        The Markdown text. Encoding is left to the caller, so the export layer decides
        how the bytes are produced rather than this function guessing.
    """
    lines: list[str] = [f"# {_escape_inline(document.title)}", ""]

    for section in document.sections:
        lines.extend(_render_section(section, document))

    lines.extend(_render_corrections(document))
    lines.extend(_render_warnings(document))

    # Collapse the trailing blank lines every block appends, then end with exactly one
    # newline — so the output is stable rather than varying with the last section's kind.
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def _render_section(section: ConvertedSection, document: ConvertedDocument) -> list[str]:
    lines = [f"## {_escape_inline(section.heading)}", ""]

    if section.kind is SectionKind.NOTE:
        if section.note.strip():
            lines.extend([section.note.strip(), ""])
        return lines

    if not section.records:
        # Said plainly rather than left blank: a heading with nothing under it reads
        # like something went wrong, when in fact nothing matched.
        lines.extend(["_No records in this section._", ""])
        if section.note.strip():
            lines.extend([section.note.strip(), ""])
        return lines

    if section.note.strip():
        lines.extend([section.note.strip(), ""])

    if section.kind is SectionKind.DETAIL:
        lines.extend(_render_detail(section, document))
    else:
        lines.extend(_render_table(section, document))

    return lines


def _render_table(section: ConvertedSection, document: ConvertedDocument) -> list[str]:
    labels = _column_labels(section)
    show_source = _any_verified(section)

    header = [*labels]
    if show_source:
        header.append("Source")

    lines = [
        "| " + " | ".join(_escape_cell(name) for name in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]

    for record in section.records:
        cells = [_escape_cell(_display_value(record, name, document)) for name in section.columns]
        if show_source:
            cells.append(_escape_cell(_record_source(record)))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return lines


def _render_detail(section: ConvertedSection, document: ConvertedDocument) -> list[str]:
    """One block per record, each field on its own line.

    Records are separated by a rule when there is more than one, so a section holding
    several self-contained entries reads as several entries rather than one long list.
    A single record needs no divider and does not get one.
    """
    lines: list[str] = []
    for position, record in enumerate(section.records):
        if position > 0:
            lines.extend(["---", ""])

        for name in section.columns:
            item = record.field(name)
            if item is None:
                continue

            label = _escape_inline(item.label)
            source = _value_source(item.value)
            suffix = f"  _(source: {_escape_inline(source)})_" if source else ""
            markers = _markers(record, name, document)
            text = _raw_value(record, name)

            if _is_multiline(text):
                # The label carries the change markers, never the block: appending a
                # glyph inside the fence would alter the text this branch exists to
                # preserve. The value itself is emitted exactly as extracted.
                lines.append(f"- **{label}:**{markers}{suffix}")
                lines.append("")
                lines.extend(_fenced_block(text, indent="  "))
                lines.append("")
                continue

            value = _escape_inline(_display_value(record, name, document))
            lines.append(f"- **{label}:** {value}{suffix}")

        lines.append("")
    return lines


def _fenced_block(text: str, *, indent: str = "") -> list[str]:
    """A fenced code block holding ``text`` unchanged.

    Markdown offers no other construct that survives leading whitespace, so this is what
    "preserve the indentation" means in practice. Two details make it lossless:

    * The fence is longer than the longest run of backticks inside the text, so a value
      that itself contains a fence cannot close this one early.
    * ``indent`` is Markdown list-continuation, stripped by the renderer before the
      block's own content is read. A line already indented four spaces still arrives
      indented four spaces.

    Blank lines are emitted bare rather than as trailing whitespace; a fenced block in a
    list item survives them.
    """
    longest_run = 0
    run = 0
    for character in text:
        run = run + 1 if character == "`" else 0
        longest_run = max(longest_run, run)
    fence = "`" * max(3, longest_run + 1)

    lines = [f"{indent}{fence}"]
    # `splitlines` handles CRLF and lone CR, so a document authored on another platform
    # does not arrive with stray carriage returns embedded in the block.
    lines.extend(f"{indent}{line}" if line else "" for line in text.splitlines())
    lines.append(f"{indent}{fence}")
    return lines


def _is_multiline(text: str) -> bool:
    """Whether the source gave this value more than one line.

    A fact about the extracted text, not a judgement about its kind. Code, verse, an
    address block and a pasted log all answer yes, and all of them are damaged by being
    joined onto one line.
    """
    return "\n" in text or "\r" in text


def _render_corrections(document: ConvertedDocument) -> list[str]:
    if not document.corrections and not document.suggestions:
        return []

    lines = ["## Changes & Corrections", ""]

    if document.corrections:
        for position, correction in enumerate(document.corrections, start=1):
            lines.append(
                f"{position}. **{_escape_inline(correction.original)}** → "
                f"**{_escape_inline(correction.corrected)}**"
            )
            lines.append(
                f"   - Reason: {_escape_inline(correction.reason)}"
                f" · Confidence: {correction.confidence.value.capitalize()}"
                f" · Applied to record {correction.record_index + 1}"
                f", {_escape_inline(correction.field)}"
            )
        lines.append("")

    if document.suggestions:
        lines.extend(["### Possible corrections (not applied)", ""])
        for position, suggestion in enumerate(document.suggestions, start=1):
            lines.append(
                f"{position}. {_escape_inline(suggestion.original)} → "
                f"_{_escape_inline(suggestion.corrected)}_"
            )
            lines.append(
                f"   - Reason: {_escape_inline(suggestion.reason)}"
                f" · Confidence: {suggestion.confidence.value.capitalize()}"
                f" · **Not applied** — record {suggestion.record_index + 1}"
                f", {_escape_inline(suggestion.field)} still shows the original value"
            )
        lines.append("")

    lines.extend(
        [
            f"_{CORRECTED_MARKER} marks a value that was changed. "
            f"{SUGGESTED_MARKER} marks a value with a possible correction that was "
            "**not** applied — the original is shown._",
            "",
        ]
    )
    return lines


def _render_warnings(document: ConvertedDocument) -> list[str]:
    if not document.warnings:
        return []
    lines = ["## Warnings", ""]
    lines.extend(f"- {_escape_inline(warning)}" for warning in document.warnings)
    lines.append("")
    return lines


def _column_labels(section: ConvertedSection) -> list[str]:
    """Column headers, taken from the first record that carries each field."""
    labels: list[str] = []
    for name in section.columns:
        label = name
        for record in section.records:
            item = record.field(name)
            if item is not None:
                label = item.label
                break
        labels.append(label)
    return labels


def _display_value(record: ConvertedRecord, name: str, document: ConvertedDocument) -> str:
    """One value as the reader should see it, with change markers attached."""
    item = record.field(name)
    if item is None:
        return ""

    value = item.value
    if value.origin is ValueOrigin.ABSENT or not value.text:
        return ""

    text = value.text
    if value.origin is ValueOrigin.CORRECTED:
        text = f"{text} {CORRECTED_MARKER}"

    if document.suggestions_for(record.source_record_index, name):
        text = f"{text} {SUGGESTED_MARKER}"

    return text


def _raw_value(record: ConvertedRecord, name: str) -> str:
    """The value's text with no markers and no escaping.

    Used only to decide whether the value spans lines, and to fill a fenced block. An
    absent value yields the empty string, which is not multi-line, so an absent field
    still takes the ordinary single-line path and still renders as blank.
    """
    item = record.field(name)
    if item is None:
        return ""
    value = item.value
    if value.origin is ValueOrigin.ABSENT:
        return ""
    return value.text


def _markers(record: ConvertedRecord, name: str, document: ConvertedDocument) -> str:
    """The change markers for a value, as a suffix for its label.

    Identical in meaning to the markers :func:`_display_value` appends to a single-line
    value — the same two glyphs, driven by the same two conditions. They are attached to
    the label here only because the value is a block that must not be written into.
    """
    item = record.field(name)
    if item is None:
        return ""

    marks = ""
    if item.value.origin is ValueOrigin.CORRECTED:
        marks += f" {CORRECTED_MARKER}"
    if document.suggestions_for(record.source_record_index, name):
        marks += f" {SUGGESTED_MARKER}"
    return marks


def _value_source(value: ConvertedValue) -> str:
    """The location label for one value, or empty when it was never verified.

    Only ever reads ``Citation.location_label``, which the citation itself computes from
    provenance we resolved. Nothing is printed as a source unless the quote was found in
    our own stored content.
    """
    citation = value.citation
    if citation is None or not citation.is_verified:
        return ""
    return citation.location_label


def _record_source(record: ConvertedRecord) -> str:
    """Distinct verified locations behind one record, in first-seen order."""
    seen: list[str] = []
    for item in record.fields:
        label = _value_source(item.value)
        if label and label not in seen:
            seen.append(label)
    return ", ".join(seen) if seen else _UNVERIFIED_LABEL


def _any_verified(section: ConvertedSection) -> bool:
    return any(item.value.is_verified for record in section.records for item in record.fields)


def _escape_inline(text: str) -> str:
    """Neutralise characters that would otherwise restructure the Markdown."""
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _escape_cell(text: str) -> str:
    """Table cells additionally must not be empty, or the column silently collapses."""
    escaped = _escape_inline(text).strip()
    return escaped if escaped else " "
