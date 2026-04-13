"""Tests for internal types (_types.py)."""

from __future__ import annotations

import pytest

from bh_sentinel.core._types import (
    EmotionScores,
    PatternMatchCandidate,
    PreprocessedText,
    SentenceBoundary,
)


class TestSentenceBoundary:
    def test_construction(self):
        sb = SentenceBoundary(text="Hello world.", index=0, char_start=0, char_end=12)
        assert sb.text == "Hello world."
        assert sb.index == 0
        assert sb.char_start == 0
        assert sb.char_end == 12

    def test_frozen(self):
        sb = SentenceBoundary(text="test", index=0, char_start=0, char_end=4)
        with pytest.raises(AttributeError):
            sb.text = "changed"  # type: ignore[misc]


class TestPreprocessedText:
    def test_construction(self):
        s = SentenceBoundary(text="test", index=0, char_start=0, char_end=4)
        pt = PreprocessedText(original="test", sentences=(s,))
        assert pt.original == "test"
        assert len(pt.sentences) == 1

    def test_frozen(self):
        pt = PreprocessedText(original="test", sentences=())
        with pytest.raises(AttributeError):
            pt.original = "changed"  # type: ignore[misc]


class TestPatternMatchCandidate:
    def test_construction_with_defaults(self):
        c = PatternMatchCandidate(
            flag_id="SH-001",
            domain="self_harm",
            name="Passive death wish",
            default_severity="HIGH",
            confidence=0.92,
            sentence_index=0,
            char_start=0,
            char_end=10,
            pattern_text="test pattern",
            basis_description="test basis",
            matched_context_hint="passive death wish",
        )
        assert c.negated is False
        assert c.temporal_context == "present"

    def test_mutable(self):
        c = PatternMatchCandidate(
            flag_id="SH-001",
            domain="self_harm",
            name="test",
            default_severity="HIGH",
            confidence=0.92,
            sentence_index=0,
            char_start=0,
            char_end=10,
            pattern_text="test",
            basis_description="test",
            matched_context_hint="test",
        )
        c.negated = True
        assert c.negated is True
        c.temporal_context = "past"
        assert c.temporal_context == "past"


class TestEmotionScores:
    def test_empty_scores(self):
        es = EmotionScores()
        assert es.scores == {}
        assert es.primary is None
        assert es.secondary is None

    def test_primary_returns_highest(self):
        es = EmotionScores(scores={"hopelessness": 0.5, "agitation": 0.3, "anger": 0.1})
        assert es.primary == "hopelessness"

    def test_secondary_returns_second_highest(self):
        es = EmotionScores(scores={"hopelessness": 0.5, "agitation": 0.3, "anger": 0.1})
        assert es.secondary == "agitation"

    def test_primary_none_when_all_zero(self):
        es = EmotionScores(scores={"hopelessness": 0.0, "agitation": 0.0})
        assert es.primary is None

    def test_secondary_none_when_only_one_nonzero(self):
        es = EmotionScores(scores={"hopelessness": 0.5, "agitation": 0.0})
        assert es.secondary is None

    def test_secondary_none_when_single_category(self):
        es = EmotionScores(scores={"hopelessness": 0.5})
        assert es.secondary is None
