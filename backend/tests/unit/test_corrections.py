"""Tests for the correction validation engine.

Pure logic, no network and no database — validation is deterministic Python, which is
exactly why it is the part of document conversion worth testing hardest. Every test here
answers one question: *can a plausible-sounding but wrong correction get through?*

The banking data below is synthetic and mirrors the messiness the feature exists to
clean: a misspelled name, three spellings of "savings", two spellings of "done", and two
date formats.
"""

from __future__ import annotations

import pytest

from app.ai.corrections import review_corrections
from app.models.domain.content import Provenance, ProvenanceKind
from app.models.domain.conversion import (
    Confidence,
    CorrectionKind,
    ProposedCorrection,
)
from app.models.domain.insight import Citation, ExtractedField

VOCABULARIES = {
    "account_type": ["Savings", "Current", "Salary"],
    "status": ["Completed", "Pending", "Failed"],
}


def _citation(label: str = "row 2") -> Citation:
    return Citation(
        quoted_text="Account Holder: Rajesh Kumr Sharma",
        provenance=Provenance(kind=ProvenanceKind.ROW_RANGE, label=label, row_start=2, row_end=2),
    )


def _records() -> list[list[ExtractedField]]:
    """Two synthetic bank records, deliberately inconsistent."""
    return [
        [
            ExtractedField(name="name", value="Rajesh Kumr Sharma", citation=_citation()),
            ExtractedField(name="account_type", value="sav", citation=_citation()),
            ExtractedField(name="status", value="DONE", citation=_citation()),
            ExtractedField(name="opening_date", value="2026-08-25", citation=_citation()),
            ExtractedField(name="balance", value="145320", citation=_citation()),
        ],
        [
            ExtractedField(name="name", value="Priya Sharma", citation=_citation("row 3")),
            ExtractedField(name="account_type", value="SAVINGS", citation=_citation("row 3")),
            ExtractedField(name="status", value="done", citation=_citation("row 3")),
            ExtractedField(name="opening_date", value="25/08/2026", citation=_citation("row 3")),
            ExtractedField(name="balance", value="45320", citation=_citation("row 3")),
        ],
    ]


def _propose(
    *,
    kind: CorrectionKind,
    field: str,
    original: str,
    corrected: str,
    confidence: Confidence = Confidence.HIGH,
    record_index: int = 0,
    reason: str = "test",
) -> ProposedCorrection:
    return ProposedCorrection(
        kind=kind,
        record_index=record_index,
        field=field,
        original=original,
        corrected=corrected,
        reason=reason,
        confidence=confidence,
    )


def _review(*proposals: ProposedCorrection):  # type: ignore[no-untyped-def]
    return review_corrections(proposals, _records(), vocabularies=VOCABULARIES)


class TestConfidenceGate:
    def test_high_confidence_correction_is_applied(self) -> None:
        review = _review(
            _propose(
                kind=CorrectionKind.VOCABULARY,
                field="account_type",
                original="sav",
                corrected="Savings",
            )
        )

        assert len(review.applied) == 1
        assert review.applied[0].applied is True
        assert review.applied[0].corrected == "Savings"
        assert review.suggestions == ()

    def test_medium_confidence_becomes_a_suggestion(self) -> None:
        review = _review(
            _propose(
                kind=CorrectionKind.SPELLING,
                field="name",
                original="Rajesh Kumr Sharma",
                corrected="Rajesh Kumar Sharma",
                confidence=Confidence.MEDIUM,
            )
        )

        assert review.applied == ()
        assert len(review.suggestions) == 1
        assert review.suggestions[0].applied is False

    def test_low_confidence_becomes_a_suggestion(self) -> None:
        review = _review(
            _propose(
                kind=CorrectionKind.SPELLING,
                field="name",
                original="Rajesh Kumr Sharma",
                corrected="Rajesh Kumar Sharma",
                confidence=Confidence.LOW,
            )
        )

        assert review.applied == ()
        assert len(review.suggestions) == 1
        assert review.suggestions[0].applied is False


class TestAnchoring:
    def test_a_value_that_does_not_match_is_rejected(self) -> None:
        """The proposer claims the field holds something it does not — so it is
        describing a document we do not have."""
        review = _review(
            _propose(
                kind=CorrectionKind.VOCABULARY,
                field="account_type",
                original="current",  # actual value is "sav"
                corrected="Current",
            )
        )

        assert review.applied == ()
        assert len(review.rejected) == 1
        assert "anchor mismatch" in review.rejected[0].reason

    def test_an_unknown_field_is_rejected(self) -> None:
        review = _review(
            _propose(
                kind=CorrectionKind.VOCABULARY,
                field="branch_code",
                original="x",
                corrected="Savings",
            )
        )

        assert "no field" in review.rejected[0].reason

    def test_an_out_of_range_record_is_rejected(self) -> None:
        review = _review(
            _propose(
                kind=CorrectionKind.VOCABULARY,
                field="account_type",
                original="sav",
                corrected="Savings",
                record_index=99,
            )
        )

        assert "does not exist" in review.rejected[0].reason

    def test_a_negative_record_index_is_rejected(self) -> None:
        """Guards against a negative index silently indexing from the end."""
        review = _review(
            _propose(
                kind=CorrectionKind.VOCABULARY,
                field="account_type",
                original="sav",
                corrected="Savings",
                record_index=-1,
            )
        )

        assert "does not exist" in review.rejected[0].reason


class TestDateFormat:
    def test_same_date_reformatted_is_accepted(self) -> None:
        review = _review(
            _propose(
                kind=CorrectionKind.DATE_FORMAT,
                field="opening_date",
                original="2026-08-25",
                corrected="25/08/2026",
            )
        )

        assert len(review.applied) == 1

    def test_a_reformat_that_changes_the_date_is_rejected(self) -> None:
        """The single most important date test: 25 August must not become 26 August."""
        review = _review(
            _propose(
                kind=CorrectionKind.DATE_FORMAT,
                field="opening_date",
                original="2026-08-25",
                corrected="26/08/2026",
            )
        )

        assert review.applied == ()
        assert "would change the date" in review.rejected[0].reason

    def test_an_unparseable_correction_is_rejected(self) -> None:
        review = _review(
            _propose(
                kind=CorrectionKind.DATE_FORMAT,
                field="opening_date",
                original="2026-08-25",
                corrected="sometime in August",
            )
        )

        assert "not a date" in review.rejected[0].reason

    def test_an_unparseable_original_is_rejected(self) -> None:
        review = _review(
            _propose(
                kind=CorrectionKind.DATE_FORMAT,
                field="balance",
                original="145320",
                corrected="2026-08-25",
            )
        )

        assert "not a date" in review.rejected[0].reason


class TestCaseNormalization:
    def test_a_case_only_change_is_accepted(self) -> None:
        review = _review(
            _propose(
                kind=CorrectionKind.CASE_NORMALIZATION,
                field="status",
                original="DONE",
                corrected="Done",
            )
        )

        assert len(review.applied) == 1

    def test_a_change_beyond_case_is_rejected(self) -> None:
        """ "DONE" -> "Completed" is a real change of content, so it may not travel
        under the case-normalization kind — it must justify itself as a vocabulary fix."""
        review = _review(
            _propose(
                kind=CorrectionKind.CASE_NORMALIZATION,
                field="status",
                original="DONE",
                corrected="Completed",
            )
        )

        assert review.applied == ()
        assert "only change letter case" in review.rejected[0].reason


class TestWhitespace:
    def test_whitespace_normalization_is_accepted(self) -> None:
        records = [[ExtractedField(name="name", value="Rajesh   Kumar  Sharma")]]
        review = review_corrections(
            [
                _propose(
                    kind=CorrectionKind.WHITESPACE,
                    field="name",
                    original="Rajesh   Kumar  Sharma",
                    corrected="Rajesh Kumar Sharma",
                )
            ],
            records,
        )

        assert len(review.applied) == 1

    def test_a_change_beyond_whitespace_is_rejected(self) -> None:
        review = _review(
            _propose(
                kind=CorrectionKind.WHITESPACE,
                field="name",
                original="Rajesh Kumr Sharma",
                corrected="Rajesh Kumar Sharma",
            )
        )

        assert "only change spacing" in review.rejected[0].reason


class TestVocabulary:
    def test_a_value_inside_the_vocabulary_is_accepted(self) -> None:
        review = _review(
            _propose(
                kind=CorrectionKind.VOCABULARY,
                field="account_type",
                original="sav",
                corrected="Savings",
            )
        )

        assert len(review.applied) == 1

    def test_done_to_completed_is_accepted_because_it_is_declared(self) -> None:
        """The spec's example: only legitimate because "Completed" is in the vocabulary."""
        review = _review(
            _propose(
                kind=CorrectionKind.VOCABULARY,
                field="status",
                original="DONE",
                corrected="Completed",
            )
        )

        assert len(review.applied) == 1
        assert review.applied[0].corrected == "Completed"

    def test_a_value_outside_the_vocabulary_is_rejected(self) -> None:
        review = _review(
            _propose(
                kind=CorrectionKind.VOCABULARY,
                field="account_type",
                original="sav",
                corrected="Investment",
            )
        )

        assert review.applied == ()
        assert "not in the declared vocabulary" in review.rejected[0].reason

    def test_a_field_with_no_declared_vocabulary_is_rejected(self) -> None:
        """Without a declared set there is nothing to check the result against, so the
        correction cannot be proved and is refused rather than trusted."""
        review = _review(
            _propose(
                kind=CorrectionKind.VOCABULARY,
                field="name",
                original="Rajesh Kumr Sharma",
                corrected="Rajesh Kumar Sharma",
            )
        )

        assert "no vocabulary is declared" in review.rejected[0].reason


class TestSpelling:
    def test_a_genuine_typo_is_accepted(self) -> None:
        review = _review(
            _propose(
                kind=CorrectionKind.SPELLING,
                field="name",
                original="Rajesh Kumr Sharma",
                corrected="Rajesh Kumar Sharma",
            )
        )

        assert len(review.applied) == 1

    def test_a_different_name_is_rejected(self) -> None:
        """The test this gate exists for: one person must never become another."""
        records = [[ExtractedField(name="name", value="Arjun Malhotra")]]
        review = review_corrections(
            [
                _propose(
                    kind=CorrectionKind.SPELLING,
                    field="name",
                    original="Arjun Malhotra",
                    corrected="Priya Nair",
                )
            ],
            records,
        )

        assert review.applied == ()
        assert review.suggestions == ()
        assert len(review.rejected) == 1

    def test_replacing_one_word_entirely_is_rejected(self) -> None:
        """A single swapped surname must not hide behind an otherwise identical name."""
        review = _review(
            _propose(
                kind=CorrectionKind.SPELLING,
                field="name",
                original="Rajesh Kumr Sharma",
                corrected="Rajesh Kumr Chatterjee",
            )
        )

        assert review.applied == ()
        assert len(review.rejected) == 1

    def test_adding_a_word_is_rejected(self) -> None:
        review = _review(
            _propose(
                kind=CorrectionKind.SPELLING,
                field="name",
                original="Rajesh Kumr Sharma",
                corrected="Rajesh Kumar Sharma Jr",
            )
        )

        assert "may not add or remove words" in review.rejected[0].reason


class TestNoGrowth:
    def test_a_correction_that_appends_information_is_rejected(self) -> None:
        """Cleaning is allowed; enrichment is not. An address the document never stated
        must not arrive under the banner of "standardizing formatting"."""
        review = _review(
            _propose(
                kind=CorrectionKind.WHITESPACE,
                field="name",
                original="Rajesh Kumr Sharma",
                corrected="Rajesh Kumr Sharma Mumbai Branch",
            )
        )

        assert review.applied == ()
        assert len(review.rejected) == 1

    def test_a_vocabulary_value_may_have_more_words_than_the_original(self) -> None:
        """No-growth defers to a declared vocabulary: the caller chose that set, so a
        longer canonical form is legitimate."""
        records = [[ExtractedField(name="status", value="P")]]
        review = review_corrections(
            [
                _propose(
                    kind=CorrectionKind.VOCABULARY,
                    field="status",
                    original="P",
                    corrected="Pending",
                )
            ],
            records,
            vocabularies={"status": ["Pending", "Completed"]},
        )

        assert len(review.applied) == 1


class TestDegenerateProposals:
    def test_an_identical_correction_is_rejected(self) -> None:
        review = _review(
            _propose(
                kind=CorrectionKind.VOCABULARY,
                field="account_type",
                original="sav",
                corrected="sav",
            )
        )

        assert "identical to the original" in review.rejected[0].reason

    def test_an_empty_correction_is_rejected(self) -> None:
        """A correction may fix a value; it may not delete one."""
        review = _review(
            _propose(
                kind=CorrectionKind.VOCABULARY,
                field="account_type",
                original="sav",
                corrected="   ",
            )
        )

        assert "may not delete a value" in review.rejected[0].reason


class TestAuditAndDeterminism:
    def test_every_applied_correction_has_an_audit_id(self) -> None:
        review = _review(
            _propose(
                kind=CorrectionKind.VOCABULARY,
                field="account_type",
                original="sav",
                corrected="Savings",
            ),
            _propose(
                kind=CorrectionKind.VOCABULARY,
                field="status",
                original="DONE",
                corrected="Completed",
            ),
        )

        assert len(review.applied) == 2
        assert all(correction.id for correction in review.applied)
        assert len({correction.id for correction in review.applied}) == 2

    def test_suggestions_also_carry_ids(self) -> None:
        review = _review(
            _propose(
                kind=CorrectionKind.SPELLING,
                field="name",
                original="Rajesh Kumr Sharma",
                corrected="Rajesh Kumar Sharma",
                confidence=Confidence.MEDIUM,
            )
        )

        assert review.suggestions[0].id

    def test_ids_are_stable_when_an_earlier_proposal_is_rejected(self) -> None:
        """Ids come from input position, not a counter over accepted items — so a
        rejection cannot renumber the corrections that follow it."""
        rejected_first = _propose(
            kind=CorrectionKind.VOCABULARY,
            field="account_type",
            original="wrong anchor",
            corrected="Savings",
        )
        accepted_second = _propose(
            kind=CorrectionKind.VOCABULARY,
            field="status",
            original="DONE",
            corrected="Completed",
        )

        review = _review(rejected_first, accepted_second)

        assert len(review.rejected) == 1
        assert review.applied[0].id == "c2"

    def test_rejection_reasons_are_preserved(self) -> None:
        review = _review(
            _propose(
                kind=CorrectionKind.VOCABULARY,
                field="account_type",
                original="sav",
                corrected="Investment",
            )
        )

        assert review.rejected[0].proposal.corrected == "Investment"
        assert review.rejected[0].reason

    def test_identical_input_produces_identical_output(self) -> None:
        proposals = [
            _propose(
                kind=CorrectionKind.VOCABULARY,
                field="account_type",
                original="sav",
                corrected="Savings",
            ),
            _propose(
                kind=CorrectionKind.SPELLING,
                field="name",
                original="Rajesh Kumr Sharma",
                corrected="Rajesh Kumar Sharma",
                confidence=Confidence.MEDIUM,
            ),
        ]

        first = review_corrections(proposals, _records(), vocabularies=VOCABULARIES)
        second = review_corrections(proposals, _records(), vocabularies=VOCABULARIES)

        assert first == second

    def test_inputs_are_never_mutated(self) -> None:
        records = _records()
        before = records[0][1].value

        review_corrections(
            [
                _propose(
                    kind=CorrectionKind.VOCABULARY,
                    field="account_type",
                    original="sav",
                    corrected="Savings",
                )
            ],
            records,
            vocabularies=VOCABULARIES,
        )

        assert records[0][1].value == before == "sav"

    def test_applied_by_target_indexes_corrections_for_assembly(self) -> None:
        review = _review(
            _propose(
                kind=CorrectionKind.VOCABULARY,
                field="account_type",
                original="sav",
                corrected="Savings",
            )
        )

        assert review.applied_by_target[(0, "account_type")].corrected == "Savings"

    def test_an_empty_proposal_list_reviews_cleanly(self) -> None:
        review = review_corrections([], _records())

        assert review.applied == ()
        assert review.suggestions == ()
        assert review.rejected == ()


class TestValidationPrecedesConfidence:
    @pytest.mark.parametrize("confidence", [Confidence.HIGH, Confidence.MEDIUM, Confidence.LOW])
    def test_an_invalid_proposal_is_rejected_at_every_confidence(
        self, confidence: Confidence
    ) -> None:
        """Failing a gate means wrong, not merely uncertain — such a proposal must never
        be offered to the user as a "possible correction" either."""
        review = _review(
            _propose(
                kind=CorrectionKind.DATE_FORMAT,
                field="opening_date",
                original="2026-08-25",
                corrected="26/08/2026",
                confidence=confidence,
            )
        )

        assert review.applied == ()
        assert review.suggestions == ()
        assert len(review.rejected) == 1
