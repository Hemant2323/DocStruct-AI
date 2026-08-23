"""Prompt templates.

Prompts live in files rather than inline string literals because they are behaviour:
a wording change alters what the product does and belongs in a reviewable diff
(CLAUDE.md §5). Each template carries a version that feeds the cache fingerprint, so
editing a prompt automatically invalidates results produced by the old one.
"""
