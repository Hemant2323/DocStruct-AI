"""Deterministic validation of proposed corrections.

A model may *propose* that a value is wrong. This module decides whether the proposal is
safe enough to act on, and it decides in ordinary Python — no network, no model, no
judgement call at runtime. That division is the whole point: a confident-sounding
suggestion is not evidence, and the place to stop an invented "fix" is before it reaches
a document somebody will rely on.

Every proposal must clear the same sequence of gates:

1. **Anchoring** — the value the proposer says it is changing must actually be there.
   This alone rejects most fabrications: a model that hallucinates a field, a row, or a
   current value is describing a document we do not have.
2. **Kind-specific proof** — each :class:`CorrectionKind` carries its own obligation.
   A date reformat must parse to the same calendar date. A case normalization must
   change nothing but case. A vocabulary fix must land inside a vocabulary the caller
   declared. A spelling fix must stay within a tight similarity bound.
3. **No growth** — a correction may not introduce material that is in neither the
   original nor the declared vocabulary. Cleaning is allowed; enrichment is not.
4. **Confidence** — only ``HIGH`` is applied. ``MEDIUM`` and ``LOW`` become suggestions
   that leave the document untouched.

A proposal that fails gates 1-3 is *rejected* whatever confidence it claimed — failing
them means the proposal is wrong, not merely uncertain. Only a proposal that is provably
safe but under-confident becomes a suggestion.

Nothing here mutates its inputs. :func:`review_corrections` returns a
:class:`CorrectionReview` describing what may be applied, what should be offered, and
what was thrown out and why.
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime

from app.models.domain.conversion import (
    Confidence,
    Correction,
    CorrectionKind,
    ProposedCorrection,
    RejectedCorrection,
)
from app.models.domain.insight import ExtractedField

#: Date formats a value may be written in. Deliberately explicit rather than a
#: best-effort parser: a format we cannot name is a format we cannot prove equality
#: across, and an unprovable date reformat is rejected instead of guessed at.
_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%m/%d/%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%Y/%m/%d",
)

#: Minimum whole-string similarity for a spelling correction. Tight on purpose — at 0.90
#: a transposed or dropped letter still passes ("Rajesh Kumr Sharma" scores 0.97 against
#: "Rajesh Kumar Sharma") while a different name does not come close ("Arjun Malhotra"
#: against "Priya Nair" scores 0.33). The cost of being strict is a missed typo fix,
#: which is visible and harmless; the cost of being loose is a silently renamed person.
_SPELLING_MIN_RATIO = 0.90

#: Maximum per-token similarity distance for a spelling correction. Applied token by
#: token so a long name cannot hide one entirely replaced word behind a high overall
#: score.
_SPELLING_MIN_TOKEN_RATIO = 0.70


@dataclass(frozen=True, slots=True)
class CorrectionReview:
    """The outcome of validating a batch of proposals.

    Attributes:
        applied: Corrections the document should show. Every one is ``HIGH`` confidence
            and has passed every gate.
        suggestions: Corrections that are provably safe but not confident enough to
            apply. The document keeps the original value; these are shown to the user.
        rejected: Proposals that failed a gate, each with the reason — kept so a
            rejection can be explained and tested rather than silently vanishing.
    """

    applied: tuple[Correction, ...] = ()
    suggestions: tuple[Correction, ...] = ()
    rejected: tuple[RejectedCorrection, ...] = ()

    @property
    def applied_by_target(self) -> dict[tuple[int, str], Correction]:
        """Applied corrections keyed by ``(record_index, field)``, for assembly."""
        return {(item.record_index, item.field): item for item in self.applied}


def review_corrections(
    proposals: Sequence[ProposedCorrection],
    records: Sequence[Sequence[ExtractedField]],
    *,
    vocabularies: Mapping[str, Sequence[str]] | None = None,
) -> CorrectionReview:
    """Validate proposed corrections against the records they claim to correct.

    Args:
        proposals: What the proposer wants to change. Untrusted.
        records: The extracted records, in extraction order. ``records[i]`` is the list
            of fields for record ``i``.
        vocabularies: Closed value sets by field name, e.g.
            ``{"condition": ["Good", "Fair", "Poor"]}``. A ``VOCABULARY`` correction
            for a field with no declared vocabulary cannot be proved and is rejected.

    Returns:
        A :class:`CorrectionReview`. Inputs are never modified.
    """
    vocab = vocabularies or {}

    applied: list[Correction] = []
    suggestions: list[Correction] = []
    rejected: list[RejectedCorrection] = []

    for position, proposal in enumerate(proposals):
        # Derived from the proposal's position, not a counter over accepted items, so
        # that ids stay stable when an earlier proposal is rejected — and so the same
        # input always produces the same ids.
        correction_id = f"c{position + 1}"

        failure = _validate(proposal, records, vocab)
        if failure is not None:
            rejected.append(RejectedCorrection(proposal=proposal, reason=failure))
            continue

        is_applied = proposal.confidence is Confidence.HIGH
        correction = Correction(
            id=correction_id,
            kind=proposal.kind,
            record_index=proposal.record_index,
            field=proposal.field,
            original=proposal.original,
            corrected=proposal.corrected,
            reason=proposal.reason,
            confidence=proposal.confidence,
            applied=is_applied,
        )
        (applied if is_applied else suggestions).append(correction)

    return CorrectionReview(
        applied=tuple(applied),
        suggestions=tuple(suggestions),
        rejected=tuple(rejected),
    )


def _validate(
    proposal: ProposedCorrection,
    records: Sequence[Sequence[ExtractedField]],
    vocabularies: Mapping[str, Sequence[str]],
) -> str | None:
    """Return the reason this proposal must be rejected, or ``None`` if it is safe."""
    anchor_failure = _check_anchoring(proposal, records)
    if anchor_failure is not None:
        return anchor_failure

    if proposal.corrected.strip() == "":
        return "the corrected value is empty; a correction may not delete a value"

    if proposal.original == proposal.corrected:
        return "the corrected value is identical to the original"

    kind_failure = _check_kind(proposal, vocabularies)
    if kind_failure is not None:
        return kind_failure

    return _check_no_growth(proposal, vocabularies)


def _check_anchoring(
    proposal: ProposedCorrection, records: Sequence[Sequence[ExtractedField]]
) -> str | None:
    """Gate 1: the value being corrected must actually exist, exactly as claimed."""
    if proposal.record_index < 0 or proposal.record_index >= len(records):
        return f"record {proposal.record_index} does not exist"

    for item in records[proposal.record_index]:
        if item.name == proposal.field:
            if item.value != proposal.original:
                return (
                    f"anchor mismatch: record {proposal.record_index} field "
                    f"{proposal.field!r} holds a different value than the proposal claims"
                )
            return None

    return f"record {proposal.record_index} has no field {proposal.field!r}"


def _check_kind(
    proposal: ProposedCorrection, vocabularies: Mapping[str, Sequence[str]]
) -> str | None:
    """Gate 2: the correction must satisfy its own kind's proof obligation."""
    if proposal.kind is CorrectionKind.DATE_FORMAT:
        return _check_date_format(proposal)
    if proposal.kind is CorrectionKind.CASE_NORMALIZATION:
        return _check_case_only(proposal)
    if proposal.kind is CorrectionKind.WHITESPACE:
        return _check_whitespace_only(proposal)
    if proposal.kind is CorrectionKind.VOCABULARY:
        return _check_vocabulary(proposal, vocabularies)
    if proposal.kind is CorrectionKind.SPELLING:
        return _check_spelling(proposal)
    # Unreachable while CorrectionKind has no other members; a new kind must add its own
    # proof here rather than defaulting to "allowed".
    return f"unsupported correction kind {proposal.kind!r}"  # pragma: no cover


def _check_date_format(proposal: ProposedCorrection) -> str | None:
    """A reformat must not move the date. Both sides are parsed and compared."""
    original = _parse_date(proposal.original)
    if original is None:
        return f"{proposal.original!r} is not a date in any recognised format"

    corrected = _parse_date(proposal.corrected)
    if corrected is None:
        return f"{proposal.corrected!r} is not a date in any recognised format"

    if original != corrected:
        return (
            f"reformatting would change the date from {original.isoformat()} "
            f"to {corrected.isoformat()}"
        )
    return None


def _parse_date(value: str) -> date | None:
    text = value.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _check_case_only(proposal: ProposedCorrection) -> str | None:
    """Only letter case may differ — no character may be added or removed."""
    if proposal.original.casefold() != proposal.corrected.casefold():
        return "a case normalization may only change letter case, nothing else"
    return None


def _check_whitespace_only(proposal: ProposedCorrection) -> str | None:
    """Only whitespace may differ; the tokens themselves must be untouched."""
    if proposal.original.split() != proposal.corrected.split():
        return "a whitespace normalization may only change spacing, nothing else"
    return None


def _check_vocabulary(
    proposal: ProposedCorrection, vocabularies: Mapping[str, Sequence[str]]
) -> str | None:
    """The result must be a member of a vocabulary the caller declared for this field."""
    allowed = vocabularies.get(proposal.field)
    if not allowed:
        return f"no vocabulary is declared for field {proposal.field!r}"

    if proposal.corrected not in set(allowed):
        return (
            f"{proposal.corrected!r} is not in the declared vocabulary for field {proposal.field!r}"
        )
    return None


def _check_spelling(proposal: ProposedCorrection) -> str | None:
    """A spelling fix must stay recognisably the same text.

    Three constraints together: the token count may not change, each token must remain
    similar to the token it replaces, and the whole string must stay above the overall
    threshold. A single replaced word therefore cannot hide inside an otherwise
    identical sentence.
    """
    original_tokens = proposal.original.split()
    corrected_tokens = proposal.corrected.split()

    if len(original_tokens) != len(corrected_tokens):
        return "a spelling correction may not add or remove words"

    for before, after in zip(original_tokens, corrected_tokens, strict=True):
        if before == after:
            continue
        if before[:1].casefold() != after[:1].casefold():
            return f"{before!r} and {after!r} do not start with the same letter"
        if _ratio(before, after) < _SPELLING_MIN_TOKEN_RATIO:
            return f"{before!r} and {after!r} are too dissimilar to be a spelling fix"

    overall = _ratio(proposal.original, proposal.corrected)
    if overall < _SPELLING_MIN_RATIO:
        return (
            f"{proposal.original!r} and {proposal.corrected!r} are too dissimilar "
            f"(similarity {overall:.2f} < {_SPELLING_MIN_RATIO})"
        )
    return None


def _ratio(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, left.casefold(), right.casefold()).ratio()


def _check_no_growth(
    proposal: ProposedCorrection, vocabularies: Mapping[str, Sequence[str]]
) -> str | None:
    """Gate 3: a correction may tidy what is there; it may not add new material.

    A result drawn from a declared vocabulary is safe by construction — the caller chose
    that set. A date reformat is exempt because its proof is stronger than any token
    comparison: the two sides denote the same day, whatever words they use to say so.
    Everything else may not come out with more tokens than it went in with, which is
    what stops a "cleanup" from appending an address, a title, or a reference number.
    """
    if proposal.kind is CorrectionKind.DATE_FORMAT:
        return None

    allowed = vocabularies.get(proposal.field)
    if allowed and proposal.corrected in set(allowed):
        return None

    if len(proposal.corrected.split()) > len(proposal.original.split()):
        return "the correction adds words that are not in the original or the vocabulary"

    return None
