"""Default config loading from vendored files bundled in the wheel.

The config/ directory at the repo root is the source of truth for editing.
These vendored copies are bundled into the PyPI wheel so that ``pip install
bh-sentinel-core`` has working defaults without requiring the full monorepo.

At runtime, callers can override the default path by passing an explicit
config directory to FlagTaxonomy, PatternMatcher, RulesEngine, or
EmotionLexicon constructors.
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path


def _default_config_dir() -> Path:
    """Return the path to the vendored default config files."""
    return Path(str(importlib.resources.files("bh_sentinel.core") / "_default_config"))


def default_flag_taxonomy_path() -> Path:
    return _default_config_dir() / "flag_taxonomy.json"


def default_patterns_path() -> Path:
    return _default_config_dir() / "patterns.yaml"


def default_rules_path() -> Path:
    return _default_config_dir() / "rules.json"


def default_emotion_lexicon_path() -> Path:
    return _default_config_dir() / "emotion_lexicon.json"
