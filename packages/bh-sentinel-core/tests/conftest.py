"""Shared pytest fixtures for bh-sentinel-core tests."""

from __future__ import annotations

import pytest

from bh_sentinel.core._config import (
    default_emotion_lexicon_path,
    default_flag_taxonomy_path,
    default_patterns_path,
    default_rules_path,
)
from bh_sentinel.core.emotion_lexicon import EmotionLexicon
from bh_sentinel.core.negation_detector import NegationDetector
from bh_sentinel.core.pattern_matcher import PatternMatcher
from bh_sentinel.core.pipeline import Pipeline
from bh_sentinel.core.preprocessor import TextPreprocessor
from bh_sentinel.core.rules_engine import RulesEngine
from bh_sentinel.core.taxonomy import FlagTaxonomy
from bh_sentinel.core.temporal_detector import TemporalDetector


@pytest.fixture
def taxonomy() -> FlagTaxonomy:
    return FlagTaxonomy(default_flag_taxonomy_path())


@pytest.fixture
def preprocessor() -> TextPreprocessor:
    return TextPreprocessor()


@pytest.fixture
def negation_detector() -> NegationDetector:
    return NegationDetector()


@pytest.fixture
def temporal_detector() -> TemporalDetector:
    return TemporalDetector()


@pytest.fixture
def pattern_matcher(taxonomy, negation_detector, temporal_detector) -> PatternMatcher:
    return PatternMatcher(default_patterns_path(), taxonomy, negation_detector, temporal_detector)


@pytest.fixture
def emotion_lexicon() -> EmotionLexicon:
    return EmotionLexicon(default_emotion_lexicon_path())


@pytest.fixture
def rules_engine(taxonomy) -> RulesEngine:
    return RulesEngine(default_rules_path(), taxonomy)


@pytest.fixture
def pipeline() -> Pipeline:
    return Pipeline()
