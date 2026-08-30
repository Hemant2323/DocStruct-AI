"""Content hashing used to detect a stale chunk embedding.

An embedding's ``content_hash`` (see ``ChunkEmbeddingRecord``) is compared against the
hash of its chunk's *current* text to decide whether the stored vector still corresponds
to what the chunk actually says. Whatever code computes and saves an embedding must use
this exact function when populating that column — a different hash scheme would make
every embedding look permanently stale to retrieval.
"""

from __future__ import annotations

import hashlib


def compute_content_hash(text: str) -> str:
    """Return a stable hash of chunk text, for staleness comparison."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
