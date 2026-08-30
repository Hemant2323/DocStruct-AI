"""Tests for the Markdown renderer.

Pure logic, no network and no database — rendering is a function from a
:class:`ConvertedDocument` to text.

Two properties get the most attention here, because they are what the feature promises:
a reader must be able to see **what was changed** and **what was only suspected**, and
no internal identifier may leak into a document a user downloads.

Documents are built by hand rather than through assembly, so a renderer bug cannot hide
behind an assembly bug.
"""

from __future__ import annotations

from app.models.domain.content import Provenance, ProvenanceKind
from app.models.domain.conversion import (
    Confidence,
    ConvertedDocument,
    ConvertedField,
    ConvertedRecord,
    ConvertedSection,
    ConvertedValue,
    Correction,
    CorrectionKind,
    SectionKind,
    ValueOrigin,
)
from app.models.domain.insight import Citation
from app.services.conversion_render import (
    CORRECTED_MARKER,
    SUGGESTED_MARKER,
    render_markdown,
)


def _citation(label: str = "row 2") -> Citation:
    return Citation(
        quoted_text="Account Holder: Rajesh Kumr Sharma",
        provenance=Provenance(kind=ProvenanceKind.ROW_RANGE, label=label, row_start=2, row_end=2),
    )


def _value(
    text: str,
    *,
    origin: ValueOrigin = ValueOrigin.SOURCE,
    original_text: str | None = None,
    citation: Citation | None = None,
    correction_id: str | None = None,
) -> ConvertedValue:
    return ConvertedValue(
        text=text,
        origin=origin,
        original_text=original_text,
        citation=citation,
        correction_id=correction_id,
    )


def _field(name: str, label: str, value: ConvertedValue) -> ConvertedField:
    return ConvertedField(name=name, label=label, value=value)


def _record(index: int, *fields: ConvertedField) -> ConvertedRecord:
    return ConvertedRecord(fields=tuple(fields), source_record_index=index)


def _simple_record(index: int = 0, *, cited: bool = True) -> ConvertedRecord:
    citation = _citation() if cited else None
    return _record(
        index,
        _field("name", "Name", _value("Rajesh Kumr Sharma", citation=citation)),
        _field("account_type", "Account Type", _value("sav", citation=citation)),
    )


def _table_section(*records: ConvertedRecord, heading: str = "Customers") -> ConvertedSection:
    return ConvertedSection(
        heading=heading,
        kind=SectionKind.TABLE,
        columns=("name", "account_type"),
        records=tuple(records),
    )


def _document(*sections: ConvertedSection, **kwargs: object) -> ConvertedDocument:
    return ConvertedDocument(
        title=str(kwargs.get("title", "Customer Master Report")),
        sections=tuple(sections),
        corrections=tuple(kwargs.get("corrections", ())),  # type: ignore[arg-type]
        suggestions=tuple(kwargs.get("suggestions", ())),  # type: ignore[arg-type]
        source_record_count=int(kwargs.get("source_record_count", 0)),  # type: ignore[arg-type]
        warnings=tuple(kwargs.get("warnings", ())),  # type: ignore[arg-type]
    )


def _correction(
    *,
    id: str = "c1",
    field: str = "account_type",
    original: str = "sav",
    corrected: str = "Savings",
    confidence: Confidence = Confidence.HIGH,
    applied: bool = True,
    record_index: int = 0,
    reason: str = "standardized account type formatting",
) -> Correction:
    return Correction(
        id=id,
        kind=CorrectionKind.VOCABULARY,
        record_index=record_index,
        field=field,
        original=original,
        corrected=corrected,
        reason=reason,
        confidence=confidence,
        applied=applied,
    )


class TestTitleAndStructure:
    def test_the_title_is_rendered_as_a_top_level_heading(self) -> None:
        output = render_markdown(_document(_table_section(_simple_record())))

        assert output.startswith("# Customer Master Report\n")

    def test_section_headings_are_rendered(self) -> None:
        output = render_markdown(_table_section_document())

        assert "## Customers" in output

    def test_multiple_sections_are_rendered_in_order(self) -> None:
        output = render_markdown(
            _document(
                _table_section(_simple_record(0), heading="Savings"),
                _table_section(_simple_record(1), heading="Current"),
            )
        )

        assert output.index("## Savings") < output.index("## Current")

    def test_the_output_ends_with_exactly_one_newline(self) -> None:
        """Stable regardless of which section kind came last, so the same document
        always renders byte-for-byte identically."""
        output = render_markdown(_table_section_document())

        assert output.endswith("\n")
        assert not output.endswith("\n\n")


def _table_section_document() -> ConvertedDocument:
    return _document(_table_section(_simple_record()))


class TestTableSection:
    def test_a_table_has_a_header_row_and_a_separator(self) -> None:
        output = render_markdown(_table_section_document())
        lines = output.splitlines()

        header = next(line for line in lines if line.startswith("| Name"))
        separator = lines[lines.index(header) + 1]
        assert "| Name | Account Type |" in header
        assert set(separator.replace("|", "").replace(" ", "")) == {"-"}

    def test_one_row_per_record(self) -> None:
        output = render_markdown(_document(_table_section(_simple_record(0), _simple_record(1))))

        rows = [line for line in output.splitlines() if line.startswith("| Rajesh")]
        assert len(rows) == 2

    def test_values_appear_in_their_columns(self) -> None:
        output = render_markdown(_table_section_document())

        assert "| Rajesh Kumr Sharma | sav |" in output

    def test_column_headers_use_the_field_label_not_the_key(self) -> None:
        output = render_markdown(_table_section_document())

        assert "Account Type" in output
        assert "| account_type |" not in output

    def test_an_absent_value_renders_as_an_empty_cell(self) -> None:
        record = _record(
            0,
            _field("name", "Name", _value("Priya Sharma", citation=_citation("row 3"))),
            _field("account_type", "Account Type", _value("", origin=ValueOrigin.ABSENT)),
        )
        output = render_markdown(_document(_table_section(record)))

        assert "| Priya Sharma |   |" in output or "| Priya Sharma |  |" in output

    def test_a_source_column_appears_when_values_are_verified(self) -> None:
        output = render_markdown(_table_section_document())

        assert "| Name | Account Type | Source |" in output
        assert "row 2" in output

    def test_no_source_column_when_nothing_is_verified(self) -> None:
        """An unverified table should not carry an empty column implying otherwise."""
        output = render_markdown(_document(_table_section(_simple_record(cited=False))))

        assert "Source" not in output


class TestDetailSection:
    def test_a_detail_section_renders_one_line_per_field(self) -> None:
        section = ConvertedSection(
            heading="Customer",
            kind=SectionKind.DETAIL,
            columns=("name", "account_type"),
            records=(_simple_record(),),
        )
        output = render_markdown(_document(section))

        assert "- **Name:** Rajesh Kumr Sharma" in output
        assert "- **Account Type:** sav" in output

    def test_a_detail_field_shows_its_source(self) -> None:
        section = ConvertedSection(
            heading="Customer",
            kind=SectionKind.DETAIL,
            columns=("name",),
            records=(_simple_record(),),
        )
        output = render_markdown(_document(section))

        assert "_(source: row 2)_" in output

    def test_an_unverified_detail_field_shows_no_source(self) -> None:
        section = ConvertedSection(
            heading="Customer",
            kind=SectionKind.DETAIL,
            columns=("name",),
            records=(_simple_record(cited=False),),
        )
        output = render_markdown(_document(section))

        assert "source:" not in output


class TestNoteSection:
    def test_a_note_section_renders_its_prose(self) -> None:
        section = ConvertedSection(
            heading="Preparation notes",
            kind=SectionKind.NOTE,
            note="Prepared for quarterly review.",
        )
        output = render_markdown(_document(section))

        assert "## Preparation notes" in output
        assert "Prepared for quarterly review." in output

    def test_a_note_section_renders_no_table(self) -> None:
        section = ConvertedSection(heading="Notes", kind=SectionKind.NOTE, note="Text.")
        output = render_markdown(_document(section))

        assert "|" not in output


class TestEmptySection:
    def test_an_empty_section_says_so_rather_than_rendering_nothing(self) -> None:
        """A heading with silence under it reads like a failure; saying "no records"
        makes it clear the section ran and matched nothing."""
        section = ConvertedSection(
            heading="Pending KYC", kind=SectionKind.TABLE, columns=("name",), records=()
        )
        output = render_markdown(_document(section))

        assert "## Pending KYC" in output
        assert "_No records in this section._" in output

    def test_a_document_with_no_sections_still_renders_its_title(self) -> None:
        output = render_markdown(_document())

        assert output.strip() == "# Customer Master Report"


class TestCorrectedValues:
    def test_a_corrected_value_is_marked(self) -> None:
        record = _record(
            0,
            _field("name", "Name", _value("Rajesh Kumr Sharma", citation=_citation())),
            _field(
                "account_type",
                "Account Type",
                _value(
                    "Savings",
                    origin=ValueOrigin.CORRECTED,
                    original_text="sav",
                    citation=_citation(),
                    correction_id="c1",
                ),
            ),
        )
        output = render_markdown(_document(_table_section(record), corrections=(_correction(),)))

        assert f"Savings {CORRECTED_MARKER}" in output

    def test_an_uncorrected_value_is_not_marked(self) -> None:
        output = render_markdown(_table_section_document())

        assert CORRECTED_MARKER not in output


class TestSuggestions:
    def test_a_value_with_a_suggestion_is_marked_but_shows_the_original(self) -> None:
        """The core promise made visible: the original text is what appears, with a
        marker pointing at it — the suggested replacement never silently takes over."""
        suggestion = _correction(
            id="c2",
            field="name",
            original="Rajesh Kumr Sharma",
            corrected="Rajesh Kumar Sharma",
            confidence=Confidence.MEDIUM,
            applied=False,
            reason="possible spelling correction",
        )
        output = render_markdown(
            _document(_table_section(_simple_record()), suggestions=(suggestion,))
        )

        assert f"Rajesh Kumr Sharma {SUGGESTED_MARKER}" in output
        assert "| Rajesh Kumar Sharma |" not in output

    def test_a_suggestion_only_marks_its_own_record(self) -> None:
        suggestion = _correction(
            id="c2",
            field="name",
            original="Rajesh Kumr Sharma",
            corrected="Rajesh Kumar Sharma",
            confidence=Confidence.MEDIUM,
            applied=False,
            record_index=0,
        )
        output = render_markdown(
            _document(
                _table_section(_simple_record(0), _simple_record(1)), suggestions=(suggestion,)
            )
        )

        # Counted across the table rows only: the legend under Changes & Corrections
        # also mentions the marker, and that occurrence is explanatory rather than a
        # mark on a value.
        table_rows = [
            line
            for line in output.splitlines()
            if line.startswith("| Rajesh") or line.startswith("| Priya")
        ]
        assert len(table_rows) == 2
        assert sum(row.count(SUGGESTED_MARKER) for row in table_rows) == 1


class TestChangesAndCorrections:
    def test_applied_corrections_are_listed_with_reason_and_confidence(self) -> None:
        output = render_markdown(
            _document(_table_section(_simple_record()), corrections=(_correction(),))
        )

        assert "## Changes & Corrections" in output
        assert "**sav** → **Savings**" in output
        assert "standardized account type formatting" in output
        assert "Confidence: High" in output

    def test_suggestions_are_listed_separately_and_marked_not_applied(self) -> None:
        suggestion = _correction(
            id="c2",
            field="name",
            original="Rajesh Kumr Sharma",
            corrected="Rajesh Kumar Sharma",
            confidence=Confidence.MEDIUM,
            applied=False,
            reason="possible spelling correction",
        )
        output = render_markdown(
            _document(_table_section(_simple_record()), suggestions=(suggestion,))
        )

        assert "### Possible corrections (not applied)" in output
        assert "**Not applied**" in output
        assert "Confidence: Medium" in output

    def test_applied_and_suggested_are_distinguishable(self) -> None:
        suggestion = _correction(
            id="c2",
            field="name",
            original="Rajesh Kumr Sharma",
            corrected="Rajesh Kumar Sharma",
            confidence=Confidence.MEDIUM,
            applied=False,
        )
        output = render_markdown(
            _document(
                _table_section(_simple_record()),
                corrections=(_correction(),),
                suggestions=(suggestion,),
            )
        )

        applied_at = output.index("## Changes & Corrections")
        suggested_at = output.index("### Possible corrections (not applied)")
        assert applied_at < suggested_at

    def test_a_legend_explains_both_markers(self) -> None:
        output = render_markdown(
            _document(_table_section(_simple_record()), corrections=(_correction(),))
        )

        assert CORRECTED_MARKER in output
        assert SUGGESTED_MARKER in output
        assert "not** applied" in output

    def test_no_corrections_section_when_there_is_nothing_to_report(self) -> None:
        output = render_markdown(_table_section_document())

        assert "Changes & Corrections" not in output


class TestWarnings:
    def test_warnings_are_rendered(self) -> None:
        output = render_markdown(
            _document(
                _table_section(_simple_record()),
                warnings=("Only 30 of 340 rows were converted.",),
            )
        )

        assert "## Warnings" in output
        assert "- Only 30 of 340 rows were converted." in output

    def test_no_warnings_section_when_there_are_none(self) -> None:
        output = render_markdown(_table_section_document())

        assert "## Warnings" not in output


class TestSafety:
    def test_internal_identifiers_are_never_exposed(self) -> None:
        """A citation carries chunk and block ids for our own bookkeeping. A user's
        downloaded report must not."""
        citation = Citation(
            quoted_text="Account Holder: Rajesh Kumr Sharma",
            provenance=Provenance(kind=ProvenanceKind.ROW_RANGE, label="row 2"),
            chunk_id=None,
            block_ids=(),
        )
        record = _record(0, _field("name", "Name", _value("Rajesh Kumr Sharma", citation=citation)))
        section = ConvertedSection(
            heading="Customers", kind=SectionKind.TABLE, columns=("name",), records=(record,)
        )
        output = render_markdown(_document(section))

        assert "chunk_id" not in output
        assert "block_ids" not in output
        assert "correction_id" not in output

    def test_a_pipe_in_a_value_cannot_break_the_table(self) -> None:
        """Unescaped, "A | B" would silently split one cell into two and shift every
        following value into the wrong column."""
        record = _record(
            0,
            _field("name", "Name", _value("Sharma | Rajesh")),
            _field("account_type", "Account Type", _value("sav")),
        )
        output = render_markdown(_document(_table_section(record)))

        row = next(
            line for line in output.splitlines() if "Sharma" in line and line.startswith("|")
        )
        # The pipe inside the value must be escaped, leaving exactly three *delimiters*
        # for two cells. Counting raw pipes would also count the escaped one, so the
        # escape is removed before counting.
        assert "\\|" in row
        assert row.replace("\\|", "").count("|") == 3
        assert "Sharma \\| Rajesh" in row

    def test_a_newline_in_a_value_cannot_break_the_table(self) -> None:
        record = _record(
            0,
            _field("name", "Name", _value("Rajesh\nSharma")),
            _field("account_type", "Account Type", _value("sav")),
        )
        output = render_markdown(_document(_table_section(record)))

        rows = [line for line in output.splitlines() if "Rajesh" in line]
        assert len(rows) == 1

    def test_a_pipe_in_a_title_is_escaped(self) -> None:
        output = render_markdown(_document(title="Report | Draft"))

        assert output.strip() == "# Report \\| Draft"


class TestPurity:
    def test_identical_input_produces_identical_output(self) -> None:
        document = _document(
            _table_section(_simple_record(0), _simple_record(1)),
            corrections=(_correction(),),
            warnings=("A warning.",),
        )

        assert render_markdown(document) == render_markdown(document)

    def test_rendering_does_not_mutate_the_document(self) -> None:
        document = _document(_table_section(_simple_record()), corrections=(_correction(),))
        before = (document.title, document.sections, document.corrections)

        render_markdown(document)

        assert (document.title, document.sections, document.corrections) == before
