"""Tests for _config.py default path resolution."""

from __future__ import annotations

from bh_sentinel.core._config import (
    default_emotion_lexicon_path,
    default_flag_taxonomy_path,
    default_patterns_path,
    default_rules_path,
)


def test_default_flag_taxonomy_path_exists():
    assert default_flag_taxonomy_path().exists()


def test_default_patterns_path_exists():
    assert default_patterns_path().exists()


def test_default_rules_path_exists():
    assert default_rules_path().exists()


def test_default_emotion_lexicon_path_exists():
    assert default_emotion_lexicon_path().exists()


def test_all_config_paths_inside_package():
    for path in (
        default_flag_taxonomy_path(),
        default_patterns_path(),
        default_rules_path(),
        default_emotion_lexicon_path(),
    ):
        assert "bh_sentinel/core/_default_config" in str(path)
