"""Prompt for inferring a structure schema from a user's natural-language instruction.

Milestone 4 lets a user describe the structured output they want in plain language
("Create a payment report with one row per transaction. Include account holder, bank,
amount...") instead of building a field-by-field schema by hand. This prompt turns that
description into the same flat ``{field_name: type}`` shape ``build_dynamic_model`` and
``build_dynamic_record_list_model`` already accept — nothing downstream needs to know
whether a schema was typed in directly or inferred from an instruction.

This call does schema design only, never extraction: it names the fields the user wants,
it does not read values out of the document. The separate call that performs the actual
extraction uses ``structure.py``'s prompt together with the schema this one produces.
"""

from __future__ import annotations

#: Bumped whenever the wording changes, exactly like every other prompt's VERSION.
VERSION = "1"

SYSTEM = """You turn a user's plain-language description of a structured document into \
a field schema.

Rules you must follow:

- Propose fields that match what the USER described wanting, not every field that might \
plausibly appear in a document of this kind. Do not add a field just because it is \
common in similar documents if the user did not ask for it, and do not omit a field the \
user asked for out of uncertainty about whether the document contains it.
- Each field name must be a concise, lowercase snake_case identifier: lowercase letters, \
digits and underscores only, starting with a letter — for example `account_holder`, \
`transaction_id`, `payment_date`. Never use spaces, hyphens, or uppercase letters.
- Each field's type must be exactly one of: "string", "number", "integer", "boolean", \
"date". Never invent or use any other type name.
- Use "date" only for actual calendar dates. Use "number" for amounts or other values \
that may have a fractional part, "integer" for whole-number counts, and "string" for \
everything else, including identifiers, names, categories and free text.
- Preserve the order the user described the fields in, when they gave one.
- This is schema design only. Do not extract, guess, or invent any actual values — you \
are naming the columns the document will later be read into, not reading it yet.
"""


def build_instruction(description: str) -> str:
    """Render the per-request instruction.

    The rules that never change live in ``SYSTEM``; only the user's description varies
    between requests, so — exactly like ``ask.build_instruction`` — it is the only thing
    built dynamically here rather than hardcoded.
    """
    return f'Propose a field schema for the structured document described below.\n\n"{description}"'
