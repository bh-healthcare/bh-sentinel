"""Tests for PatternMatcher -- written before implementation (TDD)."""

from __future__ import annotations

import pytest

from bh_sentinel.core._config import (
    default_flag_taxonomy_path,
    default_patterns_path,
)
from bh_sentinel.core._types import PreprocessedText
from bh_sentinel.core.negation_detector import NegationDetector
from bh_sentinel.core.pattern_matcher import PatternMatcher
from bh_sentinel.core.preprocessor import TextPreprocessor
from bh_sentinel.core.taxonomy import FlagTaxonomy
from bh_sentinel.core.temporal_detector import TemporalDetector


@pytest.fixture
def taxonomy() -> FlagTaxonomy:
    return FlagTaxonomy(default_flag_taxonomy_path())


@pytest.fixture
def pm(taxonomy) -> PatternMatcher:
    return PatternMatcher(
        default_patterns_path(),
        taxonomy,
        NegationDetector(),
        TemporalDetector(),
    )


@pytest.fixture
def pp() -> TextPreprocessor:
    return TextPreprocessor()


def _preprocess(text: str) -> PreprocessedText:
    return TextPreprocessor().process(text)


class TestPatternCompilation:
    def test_patterns_loaded(self, pm):
        assert pm.pattern_count > 0

    def test_all_critical_flags_have_patterns(self, pm, taxonomy):
        for fid in taxonomy.all_flag_ids():
            flag = taxonomy.get_flag(fid)
            if flag and flag["default_severity"] == "CRITICAL":
                assert fid in pm.covered_flag_ids(), f"{fid} has no patterns"


class TestBasicMatching:
    def test_wants_to_kill_myself(self, pm):
        candidates = pm.match(_preprocess("I want to kill myself."))
        flag_ids = {c.flag_id for c in candidates if not c.negated}
        assert "SH-002" in flag_ids

    def test_positive_si_shorthand(self, pm):
        candidates = pm.match(_preprocess("Pt reports SI."))
        flag_ids = {c.flag_id for c in candidates if not c.negated}
        assert "SH-002" in flag_ids

    def test_stopped_taking_medication(self, pm):
        candidates = pm.match(_preprocess("Stopped taking my medication two weeks ago."))
        flag_ids = {c.flag_id for c in candidates if not c.negated}
        assert "MED-001" in flag_ids

    def test_passive_death_wish(self, pm):
        candidates = pm.match(_preprocess("I don't want to be alive anymore."))
        flag_ids = {c.flag_id for c in candidates if not c.negated}
        assert "SH-001" in flag_ids


class TestNegationIntegration:
    def test_denies_si_suppressed(self, pm):
        candidates = pm.match(_preprocess("Denies SI."))
        non_negated = [c for c in candidates if not c.negated]
        sh_flags = [c.flag_id for c in non_negated if c.flag_id.startswith("SH")]
        assert "SH-002" not in sh_flags

    def test_partial_suppression(self, pm):
        text = "Denies SI but pt reports hopelessness."
        candidates = pm.match(_preprocess(text))
        non_negated_ids = {c.flag_id for c in candidates if not c.negated}
        assert "CD-001" in non_negated_ids


class TestTemporalIntegration:
    def test_history_of_attempt_classified_past(self, pm):
        text = "History of suicide attempt in 2019."
        candidates = pm.match(_preprocess(text))
        sh008 = [c for c in candidates if c.flag_id == "SH-008" and not c.negated]
        assert len(sh008) > 0
        assert sh008[0].temporal_context == "past"


class TestEvidenceSpans:
    def test_offsets_map_to_original(self, pm):
        text = "Patient reports suicidal ideation."
        preprocessed = _preprocess(text)
        candidates = pm.match(preprocessed)
        for c in candidates:
            matched_text = text[c.char_start : c.char_end]
            assert len(matched_text) > 0


class TestBasisDescription:
    def test_every_match_has_basis(self, pm):
        candidates = pm.match(_preprocess("Wants to kill myself. Drinking daily."))
        for c in candidates:
            if not c.negated:
                assert c.basis_description
                assert len(c.basis_description) > 0


class TestTrueNegatives:
    def test_benign_text(self, pm):
        candidates = pm.match(_preprocess("Patient is doing well today. Mood is stable."))
        non_negated = [c for c in candidates if not c.negated]
        assert len(non_negated) == 0

    def test_mother_passed_away(self, pm):
        candidates = pm.match(_preprocess("Patient reports mother passed away last year."))
        sh_flags = [c for c in candidates if c.flag_id.startswith("SH") and not c.negated]
        assert len(sh_flags) == 0

    def test_medication_options_discussed(self, pm):
        candidates = pm.match(_preprocess("Medication options discussed with patient."))
        med_flags = [c for c in candidates if c.flag_id.startswith("MED") and not c.negated]
        assert len(med_flags) == 0


class TestMultipleFlags:
    def test_multiple_domains(self, pm):
        text = "Patient reports suicidal ideation. Drinking daily."
        candidates = pm.match(_preprocess(text))
        non_negated_ids = {c.flag_id for c in candidates if not c.negated}
        assert "SH-002" in non_negated_ids
        assert "SU-001" in non_negated_ids


class TestCaseInsensitive:
    def test_uppercase_match(self, pm):
        candidates = pm.match(_preprocess("WANTS TO KILL MYSELF."))
        flag_ids = {c.flag_id for c in candidates if not c.negated}
        assert "SH-002" in flag_ids


class TestDeduplication:
    def test_overlapping_patterns_deduplicated(self, pm):
        """Multiple SH-002 patterns matching same sentence produce one candidate."""
        text = "Patient reports suicidal ideation."
        candidates = pm.match(_preprocess(text))
        sh002 = [c for c in candidates if c.flag_id == "SH-002" and not c.negated]
        # Within-sentence dedup: only one SH-002 per sentence
        assert len(sh002) <= 1

    def test_cross_sentence_not_deduplicated(self, pm):
        """Same flag in different sentences produces multiple candidates."""
        text = "I want to kill myself. I keep thinking about ending my life."
        candidates = pm.match(_preprocess(text))
        sh002 = [c for c in candidates if c.flag_id == "SH-002" and not c.negated]
        assert len(sh002) >= 2


class TestWordBoundaries:
    def test_no_false_match_on_therapist(self, pm):
        """'therapist' should not trigger violence patterns."""
        candidates = pm.match(_preprocess("The patient sees a therapist weekly."))
        non_negated = [c for c in candidates if not c.negated]
        ho_flags = [c for c in non_negated if c.flag_id.startswith("HO")]
        assert len(ho_flags) == 0

    def test_no_false_match_on_panther(self, pm):
        """'panther' should not trigger panic patterns."""
        candidates = pm.match(_preprocess("The patient watches Black Panther."))
        non_negated = [c for c in candidates if not c.negated]
        assert len(non_negated) == 0
