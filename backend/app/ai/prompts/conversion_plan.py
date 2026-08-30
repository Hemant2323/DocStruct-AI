"""Prompt for planning the shape of a converted document.

The model's job here is **layout, not content**. It decides which sections exist, what
goes in them, how they are grouped and ordered — and then stops. It never writes the
document; ``app.ai.assembly`` does that, deterministically, from values that were
extracted and verified long before this call.

That split is what makes the feature safe. A planner cannot invent a record, because
it never emits one: it emits the word "table" and a list of field names that must
already exist. It is also what keeps the response small, which matters — an earlier
milestone hit the provider's output-token ceiling by asking for too much structured data
in one reply, and a plan is a few dozen lines however large the document is.

The field list is supplied in the instruction rather than baked into ``SYSTEM``, because
it changes per document while the rules never do.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.models.domain.insight import ExtractedField

#: Bumped whenever the wording changes, exactly like every other prompt's VERSION.
VERSION = "2"

#: How many records to show the planner. It needs a feel for the data — how values are
#: written, which fields are actually populated — not the whole document. Three is enough
#: to reveal formatting inconsistency, and small enough that the planning call stays
#: cheap on a document with thousands of rows.
SAMPLE_RECORD_COUNT = 3

#: Values in the sample are clipped to this length. A planner does not need a whole
#: address to decide that "address" belongs in a table.
SAMPLE_VALUE_CHARS = 80

SYSTEM = """You plan the structure of a document. You do not write it.

Another system will build the finished document from data that has already been \
extracted and verified. Your only job is to say how that data should be organised.

The data may come from a document of any kind — an inventory, a set of measurements, a \
register, a catalogue, a schedule, correspondence, survey results, or something with no \
obvious category. The field names you are given are the only reliable guide to what it \
is. Do not assume a subject area, and do not impose a layout borrowed from a document \
type this may not be.

Rules you must follow:

- Use ONLY the field names you are given. Never introduce a field that is not in the \
list, however useful it might sound. If the user asks for something the data does not \
contain, organise what is there and leave the rest out.
- Never write values, records, totals, or prose that states facts about the data. You \
are choosing a layout, not reporting content.
- Choose a section kind that fits: "table" for many records with few fields, "detail" \
for a small number of records with many fields, "note" for a short explanatory remark \
the user asked for.
- Group and filter only on fields that exist. A section that filters on a value the \
data never contains will simply be empty, which helps nobody.
- Prefer few, meaningful sections over many thin ones. If the user asked for a separate \
section for a subset (for example unresolved cases), express that as a filter.
- Where a field's values are written inconsistently but clearly mean the same few \
things, you may declare a vocabulary for it: the small set of canonical values that \
field should use. Derive it from the values you can see and what the user asked for. \
Never invent a category the data does not show.
- Give each field a readable label. Labels are for display only; the field name itself \
is never changed.
"""


def build_instruction(
    instruction: str,
    schema: Mapping[str, str],
    records: Sequence[Sequence[ExtractedField]],
) -> str:
    """Render the per-request instruction.

    The rules that never change live in ``SYSTEM``; the user's request, the available
    fields and a short sample of the data vary per document, so only those are built
    here — the same split ``ask.build_instruction`` uses.

    Args:
        instruction: What the user asked for, in their own words.
        schema: Field name to type. This is the closed set the planner may reference.
        records: Extracted records. Only :data:`SAMPLE_RECORD_COUNT` of them are shown.
    """
    fields = "\n".join(f"- {name} ({field_type})" for name, field_type in schema.items())
    sample = _render_sample(records)

    return (
        "Plan a document that satisfies this request:\n\n"
        f'"{instruction.strip()}"\n\n'
        "These are the only fields available. Use their exact names:\n"
        f"{fields}\n\n"
        f"{sample}\n"
        f"The data has {len(records)} record(s) in total. Plan for all of them — the "
        "sample above is only to show how values are written."
    )


def _render_sample(records: Sequence[Sequence[ExtractedField]]) -> str:
    """Show a few records so the planner can see how values are actually written."""
    if not records:
        return "No sample records are available.\n"

    lines = ["A sample of the data:"]
    for position, record in enumerate(records[:SAMPLE_RECORD_COUNT]):
        values = ", ".join(
            f"{item.name}={_clip(item.value)}" for item in record if item.value.strip()
        )
        lines.append(f"- record {position}: {values}")
    return "\n".join(lines) + "\n"


def _clip(value: str) -> str:
    text = value.strip()
    if len(text) <= SAMPLE_VALUE_CHARS:
        return text
    return text[:SAMPLE_VALUE_CHARS] + "…"
