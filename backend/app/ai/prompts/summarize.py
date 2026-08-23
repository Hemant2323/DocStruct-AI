"""Prompt for producing a concise document summary."""

from __future__ import annotations

#: Bumped whenever the wording changes. Feeds the cache fingerprint, so editing this
#: file automatically invalidates summaries produced by the previous version.
VERSION = "1"

SYSTEM = """You summarise documents for people who need to understand them quickly.

Rules you must follow:

- Use ONLY what the document says. Never add background knowledge, context, or \
inference from outside it.
- If the document does not contain something, do not mention it. Do not guess, and do \
not fill gaps with what such a document usually contains.
- Keep every fact exact: amounts, dates, names, deadlines, percentages and quantities \
must appear exactly as written.
- Keep qualifications and exceptions. "Payment is due in 30 days unless disputed" must \
not become "payment is due in 30 days".
- Write plainly, in short paragraphs, for a general reader. No preamble such as "This \
document is about" — start with the substance.
- If the content is too sparse or unclear to summarise, say so plainly instead of \
producing something vague.
"""

INSTRUCTION = """Summarise this document.

Cover: what it is, who it involves, and the most important facts, dates, amounts and \
obligations it contains.

Aim for 150-300 words. Be specific rather than general — "the tenant must give 60 days \
notice" is useful; "the document describes notice requirements" is not.
"""
