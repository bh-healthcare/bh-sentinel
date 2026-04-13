"""Tests for EmotionLexicon -- written before implementation (TDD)."""

from __future__ import annotations

import pytest

from bh_sentinel.core._config import default_emotion_lexicon_path
from bh_sentinel.core.emotion_lexicon import EmotionLexicon


@pytest.fixture
def lexicon() -> EmotionLexicon:
    return EmotionLexicon(default_emotion_lexicon_path())


class TestLoading:
    def test_loads_default(self, lexicon):
        assert lexicon is not None

    def test_category_count(self, lexicon):
        assert len(lexicon.categories) == 11

    def test_term_count(self, lexicon):
        assert lexicon.term_count == 247


class TestScoring:
    def test_hopelessness_text_scores_positive(self, lexicon):
        scores = lexicon.score("I feel hopeless and defeated, there is no point.")
        assert scores.scores["hopelessness"] > 0.0

    def test_empty_text_all_zeros(self, lexicon):
        scores = lexicon.score("")
        for v in scores.scores.values():
            assert v == 0.0

    def test_score_range(self, lexicon):
        scores = lexicon.score("I feel hopeless and angry and restless.")
        for v in scores.scores.values():
            assert 0.0 <= v <= 1.0

    def test_multi_category_word(self, lexicon):
        scores = lexicon.score("I feel trapped.")
        assert scores.scores["hopelessness"] > 0.0
        assert scores.scores["anxiety"] > 0.0
        assert scores.scores["negative_valence"] > 0.0

    def test_all_categories_present(self, lexicon):
        scores = lexicon.score("test text")
        expected = {
            "hopelessness",
            "agitation",
            "anxiety",
            "anger",
            "sadness",
            "guilt",
            "shame",
            "mania",
            "dissociation",
            "positive_valence",
            "negative_valence",
        }
        assert set(scores.scores.keys()) == expected


class TestTokenization:
    def test_case_insensitive(self, lexicon):
        lower = lexicon.score("hopeless")
        upper = lexicon.score("HOPELESS")
        assert lower.scores["hopelessness"] == upper.scores["hopelessness"]

    def test_punctuation_stripped(self, lexicon):
        scores = lexicon.score("hopeless!")
        assert scores.scores["hopelessness"] > 0.0


class TestEdgeCases:
    def test_single_word_input(self, lexicon):
        scores = lexicon.score("hopeless")
        assert scores.scores["hopelessness"] > 0.0

    def test_multi_word_phrases(self, lexicon):
        scores = lexicon.score("I am giving up on everything.")
        assert scores.scores["hopelessness"] > 0.0
