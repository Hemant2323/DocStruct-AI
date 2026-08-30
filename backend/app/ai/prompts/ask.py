"""Prompt for answering a single question about a document."""

from __future__ import annotations

#: Bumped whenever the wording changes. Feeds the cache fingerprint of any future feature
#: that caches by prompt version, and documents that this wording has been reviewed.
VERSION = "1"

SYSTEM = """You answer a question about a single document, using only the excerpts of it \
you are given.

Rules you must follow:

- Answer using ONLY the supplied document content. Never use outside knowledge, general \
world knowledge, or assumptions to fill a gap.
- If the supplied content does not contain enough information to answer, say so plainly — \
for example "The document does not contain enough information to answer this question." \
Do not guess, do not infer beyond what is stated, and do not answer partially while \
implying the answer is complete.
- Keep every fact exact: amounts, dates, names, deadlines, percentages and quantities \
must appear exactly as written in the source.
- Keep qualifications and exceptions intact. Do not drop a condition or exception to make \
the answer sound simpler or more definite than the document actually states.
- Answer directly and concisely. Do not restate the question, and do not add commentary \
about what a document of this kind would typically say.
"""


def build_instruction(question: str) -> str:
    """Render the per-request instruction.

    The rules that never change live in ``SYSTEM``; only the question text varies between
    requests, so it is the only thing built dynamically here rather than hardcoded.
    """
    return (
        "Answer this question about the document, using only the content provided above:"
        f'\n\n"{question}"'
    )
