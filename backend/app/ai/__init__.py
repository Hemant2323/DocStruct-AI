"""AI layer: turning extracted document content into understanding.

Deterministic work — building context, selecting chunks, resolving provenance,
validating output, caching — stays in ordinary Python. The model is used only where
actual language understanding is required.
"""
