"""Prompt for proposing corrections to extracted values.

The model **proposes**; it never decides. Everything it returns is checked by
``app.ai.corrections.review_corrections``, in ordinary Python, against rules the model
cannot see or influence: a date reformat must parse to the same day, a case fix must
change nothing but case, a spelling fix must stay within a similarity bound, a
vocabulary fix must land in a declared set. A proposal that cannot prove itself is
discarded no matter how confident it claimed to be.

So the wording here is not the safety mechanism — the validator is. What the wording
does is raise the proportion of proposals that survive, and make the model's confidence
signal mean roughly what the validator will conclude. Telling it plainly what will be
thrown out is cheaper than having it guess.

The response carries only corrections, never the data itself. That keeps it small: a
document with a thousand tidy rows produces a near-empty reply.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.models.domain.insight import ExtractedField

#: Bumped whenever the wording changes, exactly like every other prompt's VERSION.
VERSION = "2"

#: Values are clipped in the listing below. A value long enough to be truncated is
#: prose, and prose is not what this pass corrects.
VALUE_CHARS = 120

SYSTEM = """You find formatting inconsistencies in extracted data and propose fixes. \
You do not edit anything yourself, and you do not write documents.

Every proposal you make is checked automatically before it is used. A proposal that \
cannot be verified is discarded, so proposing something you are unsure of costs \
accuracy and gains nothing.

You may propose exactly these kinds of change:

- "case_normalization" — only letter case differs. "DONE" to "Done".
- "whitespace" — only spacing differs. "north   ridge" to "north ridge".
- "vocabulary" — an inconsistent value becomes one of the canonical values you have \
been given for that field. If the canonical values are "Completed" and "Pending", then \
"complete" becomes "Completed". The result MUST be one of those exact values.
- "date_format" — the same calendar date written differently. "2026-08-25" to \
"25/08/2026". The date itself must not move by even one day.
- "spelling" — an obvious typo. "Wellingtn Terrace" to "Wellington Terrace". The \
number of words must not change, and each word must remain recognisably itself.

You are given data extracted from a document of an unknown kind. It may be records, \
measurements, inventory, correspondence, specifications, results, or anything else. Do \
not assume a subject area, and do not apply conventions from one field to data that may \
belong to another.

Rules you must follow:

- NEVER invent information. If a value is missing, leave it missing. An absent value is \
correct and useful; a guessed one is harmful.
- NEVER change a number, quantity, amount, measurement, code, reference or identifier. \
Not their digits, not their units, not their meaning. This holds whatever the number \
represents.
- NEVER change a date to a different date. Reformatting is allowed; re-dating is not.
- NEVER change what a value refers to. A spelling correction of a name is allowed; \
turning one name, place, product, term or entity into a different one is not.
- NEVER merge, combine, split or delete records. You are correcting values in place.
- NEVER fill in a field that has no value.
- If you are unsure whether something is even wrong, leave it alone or mark it low \
confidence. Preserving the original is always the safe answer.

Confidence means what a checker would conclude, not how you feel:

- "high" — mechanical and provable. A case fix, a whitespace fix, a value that is \
plainly one of the canonical values, a date written in a different format.
- "medium" — probable but a judgement call. A likely typo in a label or name.
- "low" — a guess. Expect it to be shown to the user rather than applied.

Quote the original value EXACTLY as it appears in the data you are given. A proposal \
whose original text does not match the stored value is discarded, because it describes \
data that does not exist.

Propose nothing for values that are already consistent. An empty list is a good answer \
for clean data.
"""


def build_instruction(
    instruction: str,
    records: Sequence[Sequence[ExtractedField]],
    vocabularies: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """Render the per-request instruction.

    Args:
        instruction: What the user asked for. Shapes what counts as worth tidying —
            "standardize the status column" invites vocabulary work, and a request
            that says nothing about cleanup should produce few proposals.
        records: The extracted records. Shown with their indices, because a proposal
            identifies its target by index and must be able to name it.
        vocabularies: Canonical values per field, when the plan declared any. A
            ``vocabulary`` correction is only provable against one of these.
    """
    lines = [
        "The user asked for this document:",
        "",
        f'"{instruction.strip()}"',
        "",
    ]

    if vocabularies:
        lines.append("Canonical values. A 'vocabulary' correction must produce one of these:")
        for name, values in vocabularies.items():
            allowed = ", ".join(f'"{value}"' for value in values)
            lines.append(f"- {name}: {allowed}")
        lines.append("")
    else:
        lines.extend(
            [
                "No canonical value sets were declared, so do not propose any "
                "'vocabulary' corrections — there is nothing to check them against.",
                "",
            ]
        )

    lines.extend(
        [
            "The extracted data, by record index:",
            "",
            _render_records(records),
            "",
            "Propose corrections for values that are inconsistently written. Quote each "
            "original exactly. Propose nothing for values that are already fine.",
        ]
    )
    return "\n".join(lines)


def _render_records(records: Sequence[Sequence[ExtractedField]]) -> str:
    """List every record compactly, indexed.

    Every record is shown — inconsistency is only visible across the whole set, and the
    third spelling of a value cannot be spotted from a sample. The *reply* stays small
    regardless, since it carries only corrections.
    """
    if not records:
        return "(no records)"

    lines: list[str] = []
    for index, record in enumerate(records):
        values = ", ".join(f"{item.name}={_clip(item.value)}" for item in record)
        lines.append(f"record {index}: {values}")
    return "\n".join(lines)


def _clip(value: str) -> str:
    text = value.strip()
    if len(text) <= VALUE_CHARS:
        return text
    return text[:VALUE_CHARS] + "…"
