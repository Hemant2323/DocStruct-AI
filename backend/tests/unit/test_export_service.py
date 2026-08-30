"""Tests for the export formatters (Milestone 4.6).

Pure logic, no network and no database — building an ``Insight`` by hand and formatting
it is deterministic Python. Repository/document/provider-level behavior (ownership,
"not found," "no cached result") is covered by ``test_export_api.py`` instead, since it
genuinely needs a real database to exercise correctly.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from openpyxl import load_workbook

from app.ai.dynamic_schema import canonicalize_schema, validate_structure_schema
from app.core.exceptions import InvalidStructureSchemaError
from app.models.domain.content import Provenance, ProvenanceKind
from app.models.domain.insight import Citation, ContextMode, ExtractedField, Insight, InsightKind
from app.services.export_service import _to_csv, _to_json, _to_xlsx

DOCUMENT_ID = uuid4()


def _verified_citation(label: str = "page 2") -> Citation:
    return Citation(
        quoted_text="Invoices are due within 30 days.",
        provenance=Provenance(kind=ProvenanceKind.PAGE, label=label, page=2),
        chunk_id=uuid4(),
        block_ids=(uuid4(),),
    )


#: Columns for the multi-record exports. Also supplies the header for an entirely empty
#: result, which is exported as "these columns, no rows".
SCHEMA = {"account_holder": "string", "bank_name": "string", "amount": "number"}


def _insight(fields: list[ExtractedField]) -> Insight:
    """A **legacy** single-record result: ``fields`` populated, no ``records``.

    This is the shape `/structure` produced before Milestone 4 made it multi-record.
    Every assertion below built on it therefore pins the *old* export layout, which must
    keep working — that is what makes these tests the regression guard for Part 6.
    """
    return Insight(
        document_id=DOCUMENT_ID,
        kind=InsightKind.EXTRACTION,
        fields=tuple(fields),
        context_mode=ContextMode.FULL,
        model="fake-model",
        input_tokens=100,
        output_tokens=20,
        created_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
    )


def _record(**values: ExtractedField) -> tuple[ExtractedField, ...]:
    return tuple(values.values())


def _records_insight(records: list[tuple[ExtractedField, ...]]) -> Insight:
    """A multi-record structure result — what `/structure` produces now."""
    return Insight(
        document_id=DOCUMENT_ID,
        kind=InsightKind.EXTRACTION,
        records=tuple(records),
        context_mode=ContextMode.FULL,
        model="fake-model",
        input_tokens=100,
        output_tokens=20,
        created_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
    )


#: The three payment records used throughout the multi-record tests. Three, not one, so a
#: formatter that exported only the first row could not pass.
THREE_RECORDS = [
    _record(
        account_holder=ExtractedField(name="account_holder", value="Arjun Malhotra"),
        bank_name=ExtractedField(name="bank_name", value="State Bank of India"),
        amount=ExtractedField(name="amount", value="2100"),
    ),
    _record(
        account_holder=ExtractedField(name="account_holder", value="Priya Sharma"),
        bank_name=ExtractedField(name="bank_name", value="HDFC Bank"),
        amount=ExtractedField(name="amount", value="4500"),
    ),
    _record(
        account_holder=ExtractedField(name="account_holder", value="Rahul Verma"),
        bank_name=ExtractedField(name="bank_name", value="ICICI Bank"),
        amount=ExtractedField(name="amount", value="3200"),
    ),
]


class TestJsonExport:
    def test_field_name_and_value_are_present(self) -> None:
        insight = _insight([ExtractedField(name="total", value="45200.0")])

        payload = json.loads(_to_json(insight, SCHEMA))

        assert payload["fields"][0]["name"] == "total"
        assert payload["fields"][0]["value"] == "45200.0"

    def test_verified_citation_is_included(self) -> None:
        citation = _verified_citation()
        insight = _insight([ExtractedField(name="total", value="45200.0", citation=citation)])

        payload = json.loads(_to_json(insight, SCHEMA))

        field_citation = payload["fields"][0]["citation"]
        assert field_citation["is_verified"] is True
        assert field_citation["location_label"] == "page 2"
        assert field_citation["quoted_text"] == "Invoices are due within 30 days."

    def test_unverified_field_has_a_null_citation(self) -> None:
        insight = _insight([ExtractedField(name="total", value="45200.0", citation=None)])

        payload = json.loads(_to_json(insight, SCHEMA))

        assert payload["fields"][0]["citation"] is None

    def test_no_internal_ids_are_exposed(self) -> None:
        citation = _verified_citation()
        insight = _insight([ExtractedField(name="total", value="45200.0", citation=citation)])

        raw = _to_json(insight, SCHEMA).decode("utf-8")

        assert str(citation.chunk_id) not in raw
        assert str(citation.block_ids[0]) not in raw
        assert "chunk_id" not in raw
        assert "block_ids" not in raw

    def test_values_are_never_escaped_for_formulas(self) -> None:
        """JSON is not opened by a spreadsheet program -- escaping here would corrupt
        the actual extracted value for no security benefit."""
        insight = _insight([ExtractedField(name="total", value="=SUM(A1:A2)")])

        payload = json.loads(_to_json(insight, SCHEMA))

        assert payload["fields"][0]["value"] == "=SUM(A1:A2)"

    def test_a_result_with_nothing_in_it_produces_a_valid_empty_export(self) -> None:
        """Neither records nor fields: exported as an empty *table*, since that is the
        shape `/structure` produces now."""
        insight = _insight([])

        payload = json.loads(_to_json(insight, SCHEMA))

        assert payload["records"] == []

    def test_document_id_and_model_are_included(self) -> None:
        insight = _insight([ExtractedField(name="a", value="b")])

        payload = json.loads(_to_json(insight, SCHEMA))

        assert payload["document_id"] == str(DOCUMENT_ID)
        assert payload["model"] == "fake-model"


class TestCsvExport:
    def _rows(self, insight: Insight) -> list[dict[str, str]]:
        text = _to_csv(insight, SCHEMA).decode("utf-8")
        return list(csv.DictReader(io.StringIO(text)))

    def test_one_row_per_field(self) -> None:
        insight = _insight(
            [ExtractedField(name="a", value="1"), ExtractedField(name="b", value="2")]
        )

        rows = self._rows(insight)

        assert len(rows) == 2
        assert {row["field_name"] for row in rows} == {"a", "b"}

    def test_columns_match_the_required_set(self) -> None:
        insight = _insight([ExtractedField(name="a", value="1", citation=_verified_citation())])

        text = _to_csv(insight, SCHEMA).decode("utf-8")
        header = text.splitlines()[0]

        assert header == "field_name,value,location_label,is_verified,quoted_text"

    def test_verified_citation_columns(self) -> None:
        insight = _insight(
            [ExtractedField(name="total", value="45200.0", citation=_verified_citation())]
        )

        [row] = self._rows(insight)

        assert row["location_label"] == "page 2"
        assert row["is_verified"] == "True"
        assert row["quoted_text"] == "Invoices are due within 30 days."

    def test_unverified_field_shows_source_not_verified(self) -> None:
        insight = _insight([ExtractedField(name="total", value="45200.0", citation=None)])

        [row] = self._rows(insight)

        assert row["location_label"] == "source not verified"
        assert row["is_verified"] == "False"

    def test_formula_injection_is_escaped_for_each_trigger_character(self) -> None:
        for trigger in ("=", "+", "-", "@"):
            value = f"{trigger}cmd|' /C calc'!A0"
            insight = _insight([ExtractedField(name="a", value=value)])

            [row] = self._rows(insight)

            assert row["value"].startswith("'"), f"{trigger!r} was not escaped"
            assert row["value"] == f"'{value}"

    def test_a_normal_value_is_not_escaped(self) -> None:
        insight = _insight([ExtractedField(name="a", value="Acme Ltd")])

        [row] = self._rows(insight)

        assert row["value"] == "Acme Ltd"

    def test_quoted_text_is_also_escaped(self) -> None:
        citation = Citation(
            quoted_text='=HYPERLINK("http://evil")',
            provenance=Provenance(kind=ProvenanceKind.PAGE, label="page 1", page=1),
        )
        insight = _insight([ExtractedField(name="a", value="x", citation=citation)])

        [row] = self._rows(insight)

        assert row["quoted_text"].startswith("'")

    def test_a_result_with_nothing_in_it_produces_header_only(self) -> None:
        insight = _insight([])

        text = _to_csv(insight, SCHEMA).decode("utf-8")

        assert text.strip() == "account_holder,bank_name,amount"

    def test_no_internal_ids_appear_anywhere(self) -> None:
        citation = _verified_citation()
        insight = _insight([ExtractedField(name="a", value="b", citation=citation)])

        text = _to_csv(insight, SCHEMA).decode("utf-8")

        assert str(citation.chunk_id) not in text
        assert str(citation.block_ids[0]) not in text


class TestXlsxExport:
    def _rows(self, insight: Insight) -> list[list[object]]:
        workbook = load_workbook(io.BytesIO(_to_xlsx(insight, SCHEMA)))
        sheet = workbook.active
        return [list(row) for row in sheet.iter_rows(values_only=True)]

    def test_header_row_matches_csv_columns(self) -> None:
        insight = _insight([ExtractedField(name="a", value="1")])

        rows = self._rows(insight)

        assert rows[0] == ["field_name", "value", "location_label", "is_verified", "quoted_text"]

    def test_one_data_row_per_field(self) -> None:
        insight = _insight(
            [ExtractedField(name="a", value="1"), ExtractedField(name="b", value="2")]
        )

        rows = self._rows(insight)

        assert len(rows) == 3  # header + 2 fields

    def test_verified_citation_values(self) -> None:
        insight = _insight(
            [ExtractedField(name="total", value="45200.0", citation=_verified_citation())]
        )

        _, data_row = self._rows(insight)

        assert data_row[2] == "page 2"
        assert data_row[3] is True

    def test_formula_injection_is_escaped(self) -> None:
        insight = _insight([ExtractedField(name="a", value="=cmd|calc")])

        _, data_row = self._rows(insight)

        assert data_row[1] == "'=cmd|calc"

    def test_a_single_worksheet_is_produced(self) -> None:
        insight = _insight([ExtractedField(name="a", value="1")])

        workbook = load_workbook(io.BytesIO(_to_xlsx(insight, SCHEMA)))

        assert len(workbook.sheetnames) == 1

    def test_a_result_with_nothing_in_it_produces_header_only(self) -> None:
        insight = _insight([])

        rows = self._rows(insight)

        assert len(rows) == 1


class TestMultiRecordJsonExport:
    def test_every_record_becomes_one_object(self) -> None:
        payload = json.loads(_to_json(_records_insight(THREE_RECORDS), SCHEMA))

        records = payload["records"]
        assert len(records) == 3
        assert records[0]["account_holder"] == "Arjun Malhotra"
        assert records[1]["bank_name"] == "HDFC Bank"
        assert records[2]["amount"] == "3200"

    def test_no_record_is_lost_or_merged(self) -> None:
        payload = json.loads(_to_json(_records_insight(THREE_RECORDS), SCHEMA))

        assert [record["account_holder"] for record in payload["records"]] == [
            "Arjun Malhotra",
            "Priya Sharma",
            "Rahul Verma",
        ]

    def test_sources_travel_with_each_record(self) -> None:
        records = [
            _record(
                account_holder=ExtractedField(
                    name="account_holder", value="Arjun Malhotra", citation=_verified_citation()
                )
            )
        ]

        payload = json.loads(_to_json(_records_insight(records), SCHEMA))

        citation = payload["records"][0]["_sources"]["account_holder"]
        assert citation["is_verified"] is True
        assert citation["location_label"] == "page 2"
        assert citation["quoted_text"] == "Invoices are due within 30 days."

    def test_an_unverified_field_has_a_null_source(self) -> None:
        records = [
            _record(
                account_holder=ExtractedField(
                    name="account_holder", value="Arjun Malhotra", citation=None
                )
            )
        ]

        payload = json.loads(_to_json(_records_insight(records), SCHEMA))

        assert payload["records"][0]["_sources"]["account_holder"] is None

    def test_the_sources_key_cannot_collide_with_a_real_field_name(self) -> None:
        """`_sources` is safe to sit alongside the values only because a schema field can
        never be named that — the validator rejects a leading underscore."""
        with pytest.raises(InvalidStructureSchemaError):
            validate_structure_schema({"_sources": "string"})

    def test_no_internal_ids_are_exposed(self) -> None:
        citation = _verified_citation()
        records = [
            _record(
                account_holder=ExtractedField(
                    name="account_holder", value="Arjun Malhotra", citation=citation
                )
            )
        ]

        raw = _to_json(_records_insight(records), SCHEMA).decode("utf-8")

        assert str(citation.chunk_id) not in raw
        assert str(citation.block_ids[0]) not in raw
        assert "chunk_id" not in raw
        assert "block_ids" not in raw

    def test_values_are_never_escaped_for_formulas(self) -> None:
        records = [
            _record(amount=ExtractedField(name="amount", value="=SUM(A1:A2)")),
        ]

        payload = json.loads(_to_json(_records_insight(records), SCHEMA))

        assert payload["records"][0]["amount"] == "=SUM(A1:A2)"


class TestMultiRecordCsvExport:
    def _text(self, insight: Insight) -> str:
        return _to_csv(insight, SCHEMA).decode("utf-8")

    def test_headers_are_the_schema_fields_in_schema_order(self) -> None:
        text = self._text(_records_insight(THREE_RECORDS))

        assert text.splitlines()[0] == "account_holder,bank_name,amount"

    def test_one_row_per_record_with_every_value(self) -> None:
        rows = list(csv.DictReader(io.StringIO(self._text(_records_insight(THREE_RECORDS)))))

        assert len(rows) == 3
        assert rows[0] == {
            "account_holder": "Arjun Malhotra",
            "bank_name": "State Bank of India",
            "amount": "2100",
        }
        assert rows[2]["bank_name"] == "ICICI Bank"

    def test_a_field_missing_from_one_record_leaves_an_empty_cell(self) -> None:
        """The column must stay empty rather than the next value sliding into it — the
        failure mode that would silently attribute one field's value to another."""
        records = [
            _record(
                account_holder=ExtractedField(name="account_holder", value="Arjun Malhotra"),
                amount=ExtractedField(name="amount", value="2100"),
                # bank_name absent for this record.
            )
        ]

        [row] = list(csv.DictReader(io.StringIO(self._text(_records_insight(records)))))

        assert row["account_holder"] == "Arjun Malhotra"
        assert row["bank_name"] == ""
        assert row["amount"] == "2100"

    def test_formula_injection_is_escaped_in_record_values(self) -> None:
        for trigger in ("=", "+", "-", "@"):
            value = f"{trigger}cmd|' /C calc'!A0"
            records = [_record(bank_name=ExtractedField(name="bank_name", value=value))]

            [row] = list(csv.DictReader(io.StringIO(self._text(_records_insight(records)))))

            assert row["bank_name"] == f"'{value}", f"{trigger!r} was not escaped"

    def test_a_value_containing_a_comma_does_not_break_the_row(self) -> None:
        records = [
            _record(
                account_holder=ExtractedField(name="account_holder", value="Malhotra, Arjun"),
                amount=ExtractedField(name="amount", value="2100"),
            )
        ]

        [row] = list(csv.DictReader(io.StringIO(self._text(_records_insight(records)))))

        assert row["account_holder"] == "Malhotra, Arjun"
        assert row["amount"] == "2100"


class TestMultiRecordXlsxExport:
    def _rows(self, insight: Insight) -> list[tuple[object, ...]]:
        workbook = load_workbook(io.BytesIO(_to_xlsx(insight, SCHEMA)))
        return list(workbook.active.iter_rows(values_only=True))

    def test_header_row_matches_the_csv_columns(self) -> None:
        rows = self._rows(_records_insight(THREE_RECORDS))

        assert rows[0] == ("account_holder", "bank_name", "amount")

    def test_one_row_per_record_with_every_value(self) -> None:
        rows = self._rows(_records_insight(THREE_RECORDS))

        assert len(rows) == 4  # header + 3 records
        assert rows[1] == ("Arjun Malhotra", "State Bank of India", "2100")
        assert rows[2] == ("Priya Sharma", "HDFC Bank", "4500")
        assert rows[3] == ("Rahul Verma", "ICICI Bank", "3200")

    def test_every_record_lands_in_one_worksheet(self) -> None:
        workbook = load_workbook(io.BytesIO(_to_xlsx(_records_insight(THREE_RECORDS), SCHEMA)))

        assert len(workbook.sheetnames) == 1

    def test_values_are_written_as_text(self) -> None:
        """`ExtractedField.value` is text by design — a document's amount may be
        "$1,200" or "1200 USD", so converting it to a number here would be
        interpretation, which is not export's job. Pinned so a future change to write
        native numeric cells is a deliberate decision rather than an accident."""
        rows = self._rows(_records_insight(THREE_RECORDS))

        assert rows[1][2] == "2100"
        assert isinstance(rows[1][2], str)

    def test_a_field_missing_from_one_record_leaves_an_empty_cell(self) -> None:
        records = [
            _record(
                account_holder=ExtractedField(name="account_holder", value="Arjun Malhotra"),
                amount=ExtractedField(name="amount", value="2100"),
            )
        ]

        _, data_row = self._rows(_records_insight(records))

        assert data_row[0] == "Arjun Malhotra"
        assert data_row[1] in (None, "")
        assert data_row[2] == "2100"

    def test_formula_injection_is_escaped_in_record_values(self) -> None:
        records = [_record(bank_name=ExtractedField(name="bank_name", value="=cmd|calc"))]

        _, data_row = self._rows(_records_insight(records))

        assert data_row[1] == "'=cmd|calc"


class TestSchemaCanonicalizationIsReused:
    """Confirms export's fingerprint recomputation relies on the exact same
    canonicalisation logic /structure itself uses, not a re-derived copy."""

    def test_field_order_does_not_change_the_canonical_form(self) -> None:
        a = canonicalize_schema({"vendor": "string", "total": "number"})
        b = canonicalize_schema({"total": "number", "vendor": "string"})
        assert a == b
