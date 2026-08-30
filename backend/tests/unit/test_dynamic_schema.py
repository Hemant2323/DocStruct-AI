"""Tests for runtime-built structure schemas (Milestone 4.5).

Pure logic, no network and no database — schema validation, dynamic model construction,
and fingerprint canonicalisation are all deterministic Python.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.ai.dynamic_schema import (
    MAX_STRUCTURE_FIELDS,
    MAX_STRUCTURE_RECORDS,
    SUPPORTED_FIELD_TYPES,
    InferredStructureField,
    InferredStructureSchema,
    build_dynamic_model,
    build_dynamic_record_list_model,
    canonicalize_schema,
    schema_from_inference,
    validate_structure_schema,
)
from app.core.exceptions import InvalidStructureSchemaError


class TestValidSchemas:
    @pytest.mark.parametrize("field_type", sorted(SUPPORTED_FIELD_TYPES))
    def test_every_supported_type_validates(self, field_type: str) -> None:
        validate_structure_schema({"a_field": field_type})  # must not raise

    def test_all_five_types_together(self) -> None:
        schema = {
            "invoice_number": "string",
            "total": "number",
            "line_count": "integer",
            "is_paid": "boolean",
            "due_date": "date",
        }
        validate_structure_schema(schema)  # must not raise

    def test_maximum_field_count_is_accepted(self) -> None:
        schema = {f"field_{i}": "string" for i in range(MAX_STRUCTURE_FIELDS)}
        validate_structure_schema(schema)  # must not raise


class TestRejection:
    def test_empty_schema_is_rejected(self) -> None:
        with pytest.raises(InvalidStructureSchemaError) as exc_info:
            validate_structure_schema({})
        assert exc_info.value.code == "invalid_structure_schema"

    def test_unsupported_type_is_rejected(self) -> None:
        with pytest.raises(InvalidStructureSchemaError):
            validate_structure_schema({"tags": "array"})

    def test_too_many_fields_is_rejected(self) -> None:
        schema = {f"field_{i}": "string" for i in range(MAX_STRUCTURE_FIELDS + 1)}
        with pytest.raises(InvalidStructureSchemaError) as exc_info:
            validate_structure_schema(schema)
        assert exc_info.value.details["field_count"] == MAX_STRUCTURE_FIELDS + 1

    def test_uppercase_field_name_is_rejected(self) -> None:
        with pytest.raises(InvalidStructureSchemaError):
            validate_structure_schema({"InvoiceNumber": "string"})

    def test_field_name_starting_with_a_digit_is_rejected(self) -> None:
        with pytest.raises(InvalidStructureSchemaError):
            validate_structure_schema({"1invoice": "string"})

    def test_field_name_with_a_hyphen_is_rejected(self) -> None:
        with pytest.raises(InvalidStructureSchemaError):
            validate_structure_schema({"invoice-number": "string"})

    def test_field_name_with_a_space_is_rejected(self) -> None:
        with pytest.raises(InvalidStructureSchemaError):
            validate_structure_schema({"invoice number": "string"})

    def test_field_name_with_special_characters_is_rejected(self) -> None:
        with pytest.raises(InvalidStructureSchemaError):
            validate_structure_schema({"invoice#number!": "string"})

    def test_model_prefixed_field_name_is_rejected(self) -> None:
        with pytest.raises(InvalidStructureSchemaError) as exc_info:
            validate_structure_schema({"model_config": "string"})
        assert exc_info.value.details["field"] == "model_config"

    def test_bare_model_prefix_is_rejected(self) -> None:
        with pytest.raises(InvalidStructureSchemaError):
            validate_structure_schema({"model_x": "string"})

    def test_overly_long_field_name_is_rejected(self) -> None:
        with pytest.raises(InvalidStructureSchemaError):
            validate_structure_schema({"a" * 65: "string"})

    def test_exactly_64_characters_is_accepted(self) -> None:
        validate_structure_schema({"a" * 64: "string"})  # must not raise

    def test_rejection_message_never_leaks_the_full_schema(self) -> None:
        """details() should name only the offending field, not dump the whole request."""
        with pytest.raises(InvalidStructureSchemaError) as exc_info:
            validate_structure_schema({"Bad Name": "string", "good_name": "string"})
        assert "good_name" not in str(exc_info.value.details)


class TestBuildDynamicModel:
    def test_returns_a_pydantic_model_class(self) -> None:
        model = build_dynamic_model({"invoice_number": "string"})
        assert isinstance(model, type)
        assert issubclass(model, BaseModel)

    def test_each_field_is_optional(self) -> None:
        """A required field would pressure the model into inventing a value when the
        document genuinely does not contain one."""
        model = build_dynamic_model({"invoice_number": "string", "total": "number"})
        instance = model()  # must not raise: every field may be omitted
        assert instance.invoice_number is None
        assert instance.total is None

    def test_each_present_field_carries_value_and_source_quote(self) -> None:
        model = build_dynamic_model({"invoice_number": "string"})
        instance = model.model_validate(
            {"invoice_number": {"value": "INV-1001", "source_quote": "Invoice INV-1001"}}
        )
        assert instance.invoice_number.value == "INV-1001"
        assert instance.invoice_number.source_quote == "Invoice INV-1001"

    def test_number_field_accepts_a_float(self) -> None:
        model = build_dynamic_model({"total": "number"})
        instance = model.model_validate({"total": {"value": 45200.50, "source_quote": "q"}})
        assert instance.total.value == 45200.50

    def test_integer_field_rejects_a_non_integer(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        model = build_dynamic_model({"count": "integer"})
        with pytest.raises(PydanticValidationError):
            model.model_validate({"count": {"value": "not a number", "source_quote": "q"}})

    def test_boolean_field_accepts_a_bool(self) -> None:
        model = build_dynamic_model({"is_paid": "boolean"})
        instance = model.model_validate({"is_paid": {"value": True, "source_quote": "q"}})
        assert instance.is_paid.value is True

    def test_date_field_is_represented_as_a_string(self) -> None:
        """Requirement: no strict datetime.date validation yet."""
        model = build_dynamic_model({"due_date": "date"})
        instance = model.model_validate(
            {"due_date": {"value": "2026-09-01", "source_quote": "due 2026-09-01"}}
        )
        assert instance.due_date.value == "2026-09-01"
        assert isinstance(instance.due_date.value, str)

    def test_invalid_schema_is_rejected_even_when_called_directly(self) -> None:
        """build_dynamic_model must be self-contained, not rely on an API-layer check
        having already run — it is exercised directly by these very tests."""
        with pytest.raises(InvalidStructureSchemaError):
            build_dynamic_model({})

        with pytest.raises(InvalidStructureSchemaError):
            build_dynamic_model({"Bad-Name": "string"})


class TestBuildDynamicRecordListModel:
    """Milestone 4: banking/payment documents contain many matching records (one per
    transaction or person), not just one — this is the model that lets a single schema
    describe all of them instead of only the first."""

    def test_returns_a_pydantic_model_class(self) -> None:
        model = build_dynamic_record_list_model({"account_holder": "string"})
        assert isinstance(model, type)
        assert issubclass(model, BaseModel)

    def test_records_defaults_to_an_empty_list(self) -> None:
        model = build_dynamic_record_list_model({"account_holder": "string"})
        instance = model()  # must not raise: no records is a valid (if useless) reply
        assert instance.records == []

    def test_accepts_multiple_records_with_multiple_fields(self) -> None:
        """The example from the M4 spec: two transactions, three fields each."""
        model = build_dynamic_record_list_model(
            {"account_holder": "string", "bank_name": "string", "amount": "number"}
        )
        instance = model.model_validate(
            {
                "records": [
                    {
                        "account_holder": {"value": "Arjun Malhotra", "source_quote": "q1"},
                        "bank_name": {"value": "State Bank of India", "source_quote": "q2"},
                        "amount": {"value": 2100, "source_quote": "q3"},
                    },
                    {
                        "account_holder": {"value": "Person 2", "source_quote": "q4"},
                        "bank_name": {"value": "HDFC Bank", "source_quote": "q5"},
                        "amount": {"value": 4500, "source_quote": "q6"},
                    },
                ]
            }
        )
        assert len(instance.records) == 2
        assert instance.records[0].account_holder.value == "Arjun Malhotra"
        assert instance.records[0].bank_name.value == "State Bank of India"
        assert instance.records[0].amount.value == 2100
        assert instance.records[1].account_holder.value == "Person 2"
        assert instance.records[1].bank_name.value == "HDFC Bank"
        assert instance.records[1].amount.value == 4500

    def test_each_record_field_is_optional(self) -> None:
        """A given record legitimately might not state every requested field."""
        model = build_dynamic_record_list_model({"account_holder": "string", "amount": "number"})
        instance = model.model_validate(
            {"records": [{"account_holder": {"value": "Person 1", "source_quote": "q"}}]}
        )
        assert instance.records[0].account_holder.value == "Person 1"
        assert instance.records[0].amount is None

    @pytest.mark.parametrize("field_type", sorted(SUPPORTED_FIELD_TYPES))
    def test_every_supported_type_works_inside_a_record(self, field_type: str) -> None:
        model = build_dynamic_record_list_model({"a_field": field_type})
        raw_value: object = {
            "string": "text",
            "number": 12.5,
            "integer": 3,
            "boolean": True,
            "date": "2026-09-01",
        }[field_type]

        instance = model.model_validate(
            {"records": [{"a_field": {"value": raw_value, "source_quote": "q"}}]}
        )
        assert instance.records[0].a_field.value == raw_value

    def test_source_quote_is_preserved_per_record(self) -> None:
        model = build_dynamic_record_list_model({"amount": "number"})
        instance = model.model_validate(
            {"records": [{"amount": {"value": 2100, "source_quote": "Amount: 2100"}}]}
        )
        assert instance.records[0].amount.source_quote == "Amount: 2100"

    def test_maximum_record_count_is_accepted(self) -> None:
        model = build_dynamic_record_list_model({"amount": "number"}, max_records=3)
        records = [{"amount": {"value": i, "source_quote": "q"}} for i in range(3)]

        instance = model.model_validate({"records": records})
        assert len(instance.records) == 3

    def test_too_many_records_is_rejected(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        model = build_dynamic_record_list_model({"amount": "number"}, max_records=3)
        records = [{"amount": {"value": i, "source_quote": "q"}} for i in range(4)]

        with pytest.raises(PydanticValidationError):
            model.model_validate({"records": records})

    def test_default_max_records_is_the_module_constant(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        model = build_dynamic_record_list_model({"amount": "number"})
        records = [
            {"amount": {"value": i, "source_quote": "q"}} for i in range(MAX_STRUCTURE_RECORDS + 1)
        ]

        with pytest.raises(PydanticValidationError):
            model.model_validate({"records": records})

    def test_max_records_below_one_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_records"):
            build_dynamic_record_list_model({"amount": "number"}, max_records=0)

    def test_invalid_schema_is_rejected_even_when_called_directly(self) -> None:
        with pytest.raises(InvalidStructureSchemaError):
            build_dynamic_record_list_model({})

        with pytest.raises(InvalidStructureSchemaError):
            build_dynamic_record_list_model({"Bad-Name": "string"})


class TestSchemaFromInference:
    """Milestone 4: turning a schema-inference result (the model's answer to a user's
    natural-language instruction) into the same flat schema dict every other structure
    function accepts, validated exactly the same way a caller-supplied schema is."""

    def test_valid_inferred_schema_produces_the_expected_dict(self) -> None:
        inferred = InferredStructureSchema(
            fields=[InferredStructureField(name="account_holder", type="string")]
        )
        assert schema_from_inference(inferred) == {"account_holder": "string"}

    def test_multiple_fields_preserve_order_and_types(self) -> None:
        """The M4 spec example: account holder, bank, amount, date, in that order."""
        inferred = InferredStructureSchema(
            fields=[
                InferredStructureField(name="account_holder", type="string"),
                InferredStructureField(name="bank_name", type="string"),
                InferredStructureField(name="amount", type="number"),
                InferredStructureField(name="payment_date", type="date"),
            ]
        )
        schema = schema_from_inference(inferred)
        assert schema == {
            "account_holder": "string",
            "bank_name": "string",
            "amount": "number",
            "payment_date": "date",
        }
        assert list(schema) == ["account_holder", "bank_name", "amount", "payment_date"]

    @pytest.mark.parametrize("field_type", sorted(SUPPORTED_FIELD_TYPES))
    def test_every_supported_type_is_accepted(self, field_type: str) -> None:
        inferred = InferredStructureSchema(
            fields=[InferredStructureField(name="a_field", type=field_type)]
        )
        assert schema_from_inference(inferred) == {"a_field": field_type}

    def test_unsupported_type_is_rejected(self) -> None:
        inferred = InferredStructureSchema(
            fields=[InferredStructureField(name="tags", type="array")]
        )
        with pytest.raises(InvalidStructureSchemaError):
            schema_from_inference(inferred)

    def test_invalid_field_name_is_rejected(self) -> None:
        """`InferredStructureField.name` has no pattern constraint at the Pydantic layer
        — this is rejected downstream, by the same `validate_structure_schema` a
        caller-supplied schema goes through."""
        inferred = InferredStructureSchema(
            fields=[InferredStructureField(name="Account Holder", type="string")]
        )
        with pytest.raises(InvalidStructureSchemaError):
            schema_from_inference(inferred)

    def test_empty_field_list_is_rejected(self) -> None:
        inferred = InferredStructureSchema(fields=[])
        with pytest.raises(InvalidStructureSchemaError) as exc_info:
            schema_from_inference(inferred)
        assert exc_info.value.code == "invalid_structure_schema"

    def test_too_many_fields_is_rejected(self) -> None:
        """`InferredStructureSchema.fields` itself caps at MAX_STRUCTURE_FIELDS (the same
        constant `validate_structure_schema` enforces), so an oversized list is rejected
        by Pydantic at construction — before `schema_from_inference` is even reached."""
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            InferredStructureSchema(
                fields=[
                    InferredStructureField(name=f"field_{i}", type="string")
                    for i in range(MAX_STRUCTURE_FIELDS + 1)
                ]
            )

    def test_maximum_field_count_is_accepted(self) -> None:
        inferred = InferredStructureSchema(
            fields=[
                InferredStructureField(name=f"field_{i}", type="string")
                for i in range(MAX_STRUCTURE_FIELDS)
            ]
        )
        schema = schema_from_inference(inferred)
        assert len(schema) == MAX_STRUCTURE_FIELDS


class TestCanonicalization:
    def test_identical_schemas_produce_identical_output(self) -> None:
        a = canonicalize_schema({"invoice_number": "string", "total": "number"})
        b = canonicalize_schema({"invoice_number": "string", "total": "number"})
        assert a == b

    def test_field_order_does_not_affect_the_result(self) -> None:
        a = canonicalize_schema({"invoice_number": "string", "total": "number"})
        b = canonicalize_schema({"total": "number", "invoice_number": "string"})
        assert a == b

    def test_different_field_names_produce_different_output(self) -> None:
        a = canonicalize_schema({"invoice_number": "string"})
        b = canonicalize_schema({"vendor_name": "string"})
        assert a != b

    def test_different_field_types_produce_different_output(self) -> None:
        a = canonicalize_schema({"total": "number"})
        b = canonicalize_schema({"total": "string"})
        assert a != b

    def test_different_field_counts_produce_different_output(self) -> None:
        a = canonicalize_schema({"a": "string"})
        b = canonicalize_schema({"a": "string", "b": "string"})
        assert a != b
