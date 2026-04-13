"""Tests for NegationDetector -- written before implementation (TDD)."""

from __future__ import annotations

from bh_sentinel.core.negation_detector import NegationDetector

DEFAULT_NEG = ["denies", "no", "not", "never", "denied"]


def nd() -> NegationDetector:
    return NegationDetector()


class TestSimpleNegation:
    def test_denies_si(self):
        text = "Denies SI."
        # "SI" is at position 7-9
        assert nd().is_negated(text, 7, 9, DEFAULT_NEG) is True

    def test_no_si(self):
        text = "No SI."
        assert nd().is_negated(text, 3, 5, DEFAULT_NEG) is True

    def test_clinical_shorthand_negative_for(self):
        text = "negative for SI"
        si_start = text.index("SI")
        assert (
            nd().is_negated(text, si_start, si_start + 2, [r"(?:\-|negative) (?:for )?SI"]) is True
        )


class TestScopeTermination:
    def test_comma_terminates(self):
        text = "Denies depression, reports SI."
        # "SI" is at position 27-29
        si_start = text.index("SI")
        assert nd().is_negated(text, si_start, si_start + 2, DEFAULT_NEG) is False

    def test_but_terminates(self):
        text = "Denies SI but reports hopelessness."
        hop_start = text.index("hopelessness")
        assert nd().is_negated(text, hop_start, hop_start + 12, DEFAULT_NEG) is False

    def test_period_terminates(self):
        text = "Denies SI. Reports hopelessness."
        hop_start = text.index("hopelessness")
        assert nd().is_negated(text, hop_start, hop_start + 12, DEFAULT_NEG) is False

    def test_slash_does_not_terminate(self):
        text = "Denies SI/HI."
        hi_start = text.index("HI")
        assert nd().is_negated(text, hi_start, hi_start + 2, DEFAULT_NEG) is True

    def test_and_does_not_terminate(self):
        text = "Denies SI and HI."
        hi_start = text.index("HI")
        assert nd().is_negated(text, hi_start, hi_start + 2, DEFAULT_NEG) is True


class TestPostNegation:
    def test_denied_after_match(self):
        text = "Suicidal ideation: denied."
        assert nd().is_negated(text, 0, 17, DEFAULT_NEG) is True

    def test_negative_after_match(self):
        text = "SI -- negative."
        assert (
            nd().is_negated(text, 0, 2, ["denies", "no", "not", "never", "denied", "negative"])
            is True
        )


class TestPseudoNegation:
    def test_no_longer_denies(self):
        text = "No longer denies SI."
        si_start = text.index("SI")
        assert nd().is_negated(text, si_start, si_start + 2, DEFAULT_NEG) is False

    def test_unable_to(self):
        text = "Unable to sleep for days."
        assert nd().is_negated(text, 10, 15, DEFAULT_NEG) is False

    def test_cannot_stop(self):
        text = "Cannot stop thinking about suicide."
        assert nd().is_negated(text, 13, 34, DEFAULT_NEG) is False

    def test_no_improvement(self):
        text = "No improvement in SI."
        si_start = text.index("SI")
        assert nd().is_negated(text, si_start, si_start + 2, DEFAULT_NEG) is False

    def test_no_reason_to_live(self):
        text = "No reason to live."
        assert nd().is_negated(text, 0, 18, DEFAULT_NEG) is False


class TestOutOfRange:
    def test_negation_cue_too_far_away(self):
        text = "Denies having any issues " + ("x " * 30) + "SI reported."
        si_start = text.index("SI reported")
        assert nd().is_negated(text, si_start, si_start + 2, DEFAULT_NEG) is False


class TestBoundaryConditions:
    def test_empty_text(self):
        assert nd().is_negated("", 0, 0, DEFAULT_NEG) is False

    def test_match_at_position_zero(self):
        text = "SI denied."
        assert nd().is_negated(text, 0, 2, DEFAULT_NEG) is True

    def test_no_negation_phrases(self):
        text = "Denies SI."
        assert nd().is_negated(text, 7, 9, []) is False


class TestOverlappingCues:
    def test_double_negation_both_cues(self):
        """'not denying' — both 'not' and 'denying' are cue-adjacent."""
        text = "Patient is not denying SI."
        si_start = text.index("SI")
        # "not" is a negation cue, and "denying" is near the match.
        # The rightmost cue ("denying") is closest — but pseudo-negation
        # "no longer denies" doesn't apply here. This should still negate.
        assert nd().is_negated(text, si_start, si_start + 2, DEFAULT_NEG) is True

    def test_denies_and_no_both_present(self):
        """Both 'denies' and 'no' in lookback — rightmost wins."""
        text = "Denies any, no SI."
        si_start = text.index("SI")
        assert nd().is_negated(text, si_start, si_start + 2, DEFAULT_NEG) is True
