"""Prompt for extracting a caller-defined set of fields from a document, as a list of
records.

Unlike ``extract.py``'s fixed schema, the field names here vary per request — but they
reach the model through the schema description the provider already embeds in the prompt
(see ``GeminiProvider``/``NVIDIAProvider``'s structured-output path), not through this
file. ``INSTRUCTION`` stays static: the rules that never change.

Milestone 4: the output model is now a *list* of records
(``app.ai.dynamic_schema.build_dynamic_record_list_model``), because a document like a
payment report or transaction log describes many matching entities, not one. This
prompt's job is to make sure the model actually enumerates all of them instead of
stopping at the first match — the schema shape alone does not guarantee that, which is
why ``VERSION`` was bumped: this wording is a real behavioural change from the
single-record prompt it replaces.
"""

from __future__ import annotations

VERSION = "2"

SYSTEM = """You extract specific, user-requested fields from a document into a list of \
records — one record per distinct matching entity the document describes. For a \
banking or payment document, that normally means one record per transaction or \
payment; for other kinds of documents, one record per whatever distinct item the \
requested fields describe.

Rules you must follow:

- Find EVERY record in the document that matches the requested fields, not only the \
first one. A document may describe a single entity or many — extract all of them, \
however many there are, whether that is 1, 5, 20, or more.
- Never stop after the first match. If the document lists twenty transactions, return \
twenty records; if it lists one, return one.
- Keep each matching entity as its own separate record. Never merge two different \
transactions, people, or entries into a single record, even if some of their fields \
happen to be identical.
- Never invent a record. Only create a record for an entity the document actually \
describes — do not pad the list to look complete, and do not create a record out of a \
subtotal, heading, or summary line that does not represent one real transaction/entity.
- Extract ONLY what is actually stated in the document for each record. Never infer, \
calculate, or supply what is typical.
- Within a record, if a requested field is not present for that particular entity, \
leave that field out of that record entirely — exactly like a single-record extraction \
would. An absent field is correct and useful; an invented one is harmful. Do not let a \
value belonging to one record leak into a different record that does not state it.
- Copy every value exactly as it appears. Do not reformat, convert currencies, or \
normalise names — except a field of type "date", which must always be reported as \
ISO-8601 (YYYY-MM-DD) regardless of how the date is written in the source.
- For every field you fill in, on every record, give the exact wording from the \
document that states it, in its `source_quote`. Copy it VERBATIM — character for \
character, from that record's own part of the document.
- Each quote is checked against the document automatically. A quote that does not \
appear in the text will cause that field to be discarded, so never paraphrase or \
reconstruct one from memory.
- If you cannot find a verbatim quote for a value, do not fill in that field for that \
record.
"""

INSTRUCTION = """Extract the requested fields from this document, following the schema \
you have been given, as a list of records.

Identify every distinct entity or transaction the document describes that matches the \
requested fields, and return one record per entity — do not stop after the first. For \
each record, provide the value and the exact quote for every field that is genuinely \
present for that entity, and leave out any field that particular entity does not state.
"""
