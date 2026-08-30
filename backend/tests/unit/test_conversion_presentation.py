"""Presentation: a converted document should look like the document it came from.

The conversion pipeline was already shape-correct — the right sections, the right
records — but the renderer flattened every value onto one line. For a table that is
right; for a value the source wrote across several lines it destroys the thing that made
it readable. A C function arrived as one long line with its indentation collapsed into
runs of spaces, which is what "looks like flattened extracted text" meant in practice.

These tests pin the fix and, more importantly, pin what the fix must *not* do. Nothing
here asks what a value means. There is no language detection and no code sniffing: the
only question is whether the extracted text had more than one line, and the answer
changes presentation only. A source-code fixture appears because it is the sharpest case
— indentation is semantic and mangling is obvious — not because the pipeline knows
anything about programs.

**No test here calls a real model** (CLAUDE.md §7).
"""

from __future__ import annotations

from app.ai.assembly import DocumentPlan, SectionPlan, assemble_document
from app.ai.corrections import review_corrections
from app.models.domain.content import Provenance, ProvenanceKind
from app.models.domain.conversion import (
    Confidence,
    ConvertedDocument,
    CorrectionKind,
    ProposedCorrection,
    SectionKind,
    ValueOrigin,
)
from app.models.domain.insight import Citation, ExtractedField
from app.services.conversion_render import (
    CORRECTED_MARKER,
    SUGGESTED_MARKER,
    render_markdown,
)

# --------------------------------------------------------------------------------------
# A section-oriented document whose records carry multi-line values.
#
# Modelled on the five-program C reference used for manual testing: a title, a block of
# source, and an explanation per entry. The *shape* is the fixture — several
# self-contained entries, one field of which spans lines — and that shape occurs in
# recipe books, legal schedules and lab protocols just as readily as in code.
# --------------------------------------------------------------------------------------

BINARY_SEARCH = (
    "#include <stdio.h>\n"
    "int main(void) {\n"
    "    int a[] = {2, 5, 8, 12};\n"
    "    int low = 0, high = 3;\n"
    "    while (low <= high) {\n"
    "        int mid = low + (high - low) / 2;\n"
    "        if (a[mid] == 8) {\n"
    '            printf("Found at %d\\n", mid);\n'
    "            return 0;\n"
    "        }\n"
    "    }\n"
    "    return 1;\n"
    "}"
)

BUBBLE_SORT = (
    "void sort(int a[], int n) {\n"
    "    for (int i = 0; i < n - 1; i++)\n"
    "        for (int j = 0; j < n - i - 1; j++)\n"
    "            if (a[j] > a[j + 1])\n"
    "                swap(&a[j], &a[j + 1]);\n"
    "}"
)

PROGRAM_SCHEMA = {
    "program_title": "string",
    "source_code": "string",
    "explanation": "string",
    "complexity": "string",
}


def _program(
    title: str, code: str, explanation: str, complexity: str | None = None
) -> tuple[ExtractedField, ...]:
    fields = [
        ExtractedField(name="program_title", value=title, citation=None),
        ExtractedField(name="source_code", value=code, citation=None),
        ExtractedField(name="explanation", value=explanation, citation=None),
    ]
    if complexity is not None:
        fields.append(ExtractedField(name="complexity", value=complexity, citation=None))
    return tuple(fields)


PROGRAM_RECORDS = (
    _program(
        "Binary Search",
        BINARY_SEARCH,
        "Halves the remaining range on each comparison.",
        "O(log n)",
    ),
    # No complexity was stated for the second entry — a genuine gap, not a placeholder.
    _program("Bubble Sort", BUBBLE_SORT, "Repeatedly swaps adjacent pairs."),
)


def _render(plan: DocumentPlan, records: tuple[tuple[ExtractedField, ...], ...]) -> str:
    document = assemble_document(
        records,
        schema=PROGRAM_SCHEMA,
        plan=plan,
        review=review_corrections((), records),
    )
    return render_markdown(document)


def _first_fenced_block(markdown: str) -> list[str]:
    """The contents of the first fenced block, with the list indentation removed.

    The two-space prefix is Markdown list continuation, stripped by any renderer before
    the block's own content is read — so removing it here is what a reader sees.
    """
    lines = markdown.splitlines()
    opening = next(i for i, line in enumerate(lines) if line.strip().startswith("```"))
    fence = lines[opening].strip()
    closing = next(i for i in range(opening + 1, len(lines)) if lines[i].strip() == fence)
    return [line[2:] if line.startswith("  ") else line for line in lines[opening + 1 : closing]]


def _detail_plan() -> DocumentPlan:
    return DocumentPlan(
        title="C Programming Reference",
        sections=(SectionPlan(heading="Programs", kind=SectionKind.DETAIL),),
    )


class TestMultilineValuesKeepTheirLines:
    """The fix itself: line structure survives rendering."""

    def test_every_source_line_survives_in_order(self) -> None:
        """The block holds the source exactly: same lines, same order, nothing lost."""
        markdown = _render(_detail_plan(), PROGRAM_RECORDS)

        assert _first_fenced_block(markdown) == BINARY_SEARCH.split("\n")

    def test_indentation_is_preserved_exactly(self) -> None:
        """The whole point. A fence is the only Markdown construct that does this."""
        markdown = _render(_detail_plan(), PROGRAM_RECORDS)

        # Rendered inside a list item, so each line carries the two-space continuation
        # indent that Markdown strips on render — and nothing else.
        assert "  #include <stdio.h>" in markdown
        assert "      int low = 0, high = 3;" in markdown
        assert "              swap(&a[j], &a[j + 1]);" in markdown

    def test_the_value_is_not_escaped_inside_the_block(self) -> None:
        r"""Escaping is a table's requirement. Applying it here would corrupt the code.

        ``printf("Found at %d\n", mid);`` contains a backslash. The inline escaper
        doubles backslashes, which would silently change what the program does.
        """
        markdown = _render(_detail_plan(), PROGRAM_RECORDS)

        assert r'printf("Found at %d\n", mid);' in markdown
        assert r"\\n" not in markdown

    def test_a_value_containing_a_fence_cannot_close_the_block_early(self) -> None:
        """Markdown must not be breakable by data — the rule the table escaping follows."""
        records = (_program("Docs", "Example:\n```\nnot the end\n```\nstill inside", "Prose."),)
        markdown = _render(_detail_plan(), records)

        assert "````" in markdown, "fence was not lengthened to enclose the content"
        assert "still inside" in markdown

    def test_single_line_values_are_untouched_by_the_change(self) -> None:
        markdown = _render(_detail_plan(), PROGRAM_RECORDS)

        assert "- **Program Title:** Binary Search" in markdown
        assert "- **Explanation:** Halves the remaining range on each comparison." in markdown

    def test_crlf_does_not_leave_carriage_returns_in_the_output(self) -> None:
        records = (_program("Windows", "line one\r\nline two\r\nline three", "Prose."),)
        markdown = _render(_detail_plan(), records)

        assert "\r" not in markdown
        assert "  line two" in markdown


class TestTableSectionsStayCompact:
    """A table is a grid. The fix must not turn one into a wall of code."""

    def test_a_table_renders_as_a_table_with_aligned_columns(self) -> None:
        plan = DocumentPlan(
            title="C Programming Reference",
            sections=(
                SectionPlan(
                    heading="Index",
                    kind=SectionKind.TABLE,
                    columns=("program_title", "complexity"),
                ),
            ),
        )
        markdown = _render(plan, PROGRAM_RECORDS)

        assert "| Program Title | Complexity |" in markdown
        assert "| --- | --- |" in markdown
        assert "| Binary Search | O(log n) |" in markdown

        # Every body row has the same number of cells as the header.
        rows = [line for line in markdown.splitlines() if line.startswith("|")]
        widths = {row.count("|") for row in rows}
        assert len(widths) == 1, f"ragged table: {widths}"

    def test_a_multiline_value_in_a_table_does_not_break_the_row(self) -> None:
        """Flattening is correct here: a newline inside a cell would split the row."""
        plan = DocumentPlan(
            title="C Programming Reference",
            sections=(
                SectionPlan(
                    heading="Index",
                    kind=SectionKind.TABLE,
                    columns=("program_title", "source_code"),
                ),
            ),
        )
        markdown = _render(plan, PROGRAM_RECORDS)

        rows = [line for line in markdown.splitlines() if line.startswith("|")]
        assert len(rows) == 4, "header, separator and one row per record"
        assert all(row.count("|") == 3 for row in rows)
        assert "```" not in markdown


class TestSectionAndNoteHierarchy:
    """Headings and prose, for a document that is sections rather than rows."""

    def test_note_sections_render_as_paragraphs_under_their_heading(self) -> None:
        plan = DocumentPlan(
            title="C Programming Reference",
            sections=(
                SectionPlan(
                    heading="About this document",
                    kind=SectionKind.NOTE,
                    note=(
                        "Five independently structured programs, each with source\n"
                        "and an explanation."
                    ),
                ),
                SectionPlan(heading="Programs", kind=SectionKind.DETAIL),
            ),
        )
        markdown = _render(plan, PROGRAM_RECORDS)

        assert "# C Programming Reference" in markdown
        assert "## About this document" in markdown
        assert "## Programs" in markdown
        # The note's own text is emitted as written, not folded into a code block.
        assert "Five independently structured programs" in markdown
        assert markdown.index("## About this document") < markdown.index("## Programs")

    def test_records_in_a_detail_section_are_visibly_separated(self) -> None:
        markdown = _render(_detail_plan(), PROGRAM_RECORDS)

        assert "\n---\n" in markdown
        assert markdown.index("Binary Search") < markdown.index("Bubble Sort")

    def test_a_single_record_gets_no_divider(self) -> None:
        markdown = _render(_detail_plan(), PROGRAM_RECORDS[:1])
        assert "\n---\n" not in markdown


class TestAbsentValuesRemainAbsent:
    """Presentation must never fill a gap."""

    def test_a_field_the_source_never_stated_renders_blank_in_a_detail_section(
        self,
    ) -> None:
        markdown = _render(_detail_plan(), PROGRAM_RECORDS)

        # The second program states no complexity. Its label may appear; a value must
        # not, and certainly not the first program's.
        after_divider = markdown.split("\n---\n", 1)[1]
        assert "O(log n)" not in after_divider
        assert "```" in after_divider, "the second record still renders its code"

    def test_an_absent_value_is_not_treated_as_a_block(self) -> None:
        document = assemble_document(
            PROGRAM_RECORDS,
            schema=PROGRAM_SCHEMA,
            plan=_detail_plan(),
            review=review_corrections((), PROGRAM_RECORDS),
        )
        second = document.sections[0].records[1]
        complexity = {field.name: field.value for field in second.fields}["complexity"]

        assert complexity.origin is ValueOrigin.ABSENT
        assert complexity.text == ""


class TestProvenanceAndAuditAreUnchanged:
    """The presentation pass must not touch what any of this means."""

    def test_a_verified_location_is_shown_beside_a_multiline_value(self) -> None:
        cited = (
            ExtractedField(name="program_title", value="Binary Search", citation=None),
            ExtractedField(
                name="source_code",
                value=BINARY_SEARCH,
                citation=Citation(
                    quoted_text="#include <stdio.h>",
                    # A Citation has no way to be handed a label: the location is
                    # derived from provenance that came out of our own records.
                    provenance=Provenance(kind=ProvenanceKind.PAGE, label="page 1", page=1),
                ),
            ),
        )
        markdown = _render(_detail_plan(), (cited,))

        assert "(source: page 1)" in markdown
        # On the label line, never inside the block.
        label_line = next(line for line in markdown.splitlines() if "Source Code" in line)
        assert "page 1" in label_line

    def test_an_unverified_value_still_claims_no_location(self) -> None:
        markdown = _render(_detail_plan(), PROGRAM_RECORDS)
        assert "(source:" not in markdown

    def test_a_correction_to_a_multiline_value_is_marked_without_altering_the_text(
        self,
    ) -> None:
        """The audit semantics are the same; only where the marker sits differs.

        A glyph appended inside the block would be indistinguishable from part of the
        code, and would mean the rendered text no longer matched the stored value.
        """
        messy = "int main(void) {\n    return 0;\n}"
        tidy = "int main(void) {\n    return 0;\n}"
        records = (
            (
                ExtractedField(name="program_title", value="minimal", citation=None),
                ExtractedField(name="source_code", value=messy, citation=None),
            ),
        )
        review = review_corrections(
            [
                ProposedCorrection(
                    kind=CorrectionKind.CASE_NORMALIZATION,
                    record_index=0,
                    field="program_title",
                    original="minimal",
                    corrected="Minimal",
                    reason="Title case.",
                    confidence=Confidence.HIGH,
                )
            ],
            records,
        )
        document = assemble_document(
            records, schema=PROGRAM_SCHEMA, plan=_detail_plan(), review=review
        )
        markdown = render_markdown(document)

        assert CORRECTED_MARKER in markdown
        assert "## Changes & Corrections" in markdown
        # The untouched block is still byte-identical to the source.
        assert all(f"  {line}" in markdown for line in tidy.split("\n"))

    def test_a_suggestion_marks_the_label_and_leaves_the_block_alone(self) -> None:
        records = (
            (
                ExtractedField(name="program_title", value="Sortt", citation=None),
                ExtractedField(name="source_code", value=BUBBLE_SORT, citation=None),
            ),
        )
        review = review_corrections(
            [
                ProposedCorrection(
                    kind=CorrectionKind.SPELLING,
                    record_index=0,
                    field="source_code",
                    original=BUBBLE_SORT,
                    corrected=BUBBLE_SORT.replace("sort", "sortt"),
                    reason="Possible typo.",
                    confidence=Confidence.MEDIUM,
                )
            ],
            records,
        )
        document: ConvertedDocument = assemble_document(
            records, schema=PROGRAM_SCHEMA, plan=_detail_plan(), review=review
        )
        markdown = render_markdown(document)

        # Whether the validator accepts it is its own business; what matters is that a
        # suggestion never rewrites the value it is about.
        assert not review.applied
        for line in BUBBLE_SORT.split("\n"):
            assert f"  {line}" in markdown
        if review.suggestions:
            label_line = next(line for line in markdown.splitlines() if "Source Code" in line)
            assert SUGGESTED_MARKER in label_line
