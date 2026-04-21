"""Tests for the zero-shot hypothesis loader.

Every flag_id in the core taxonomy must have exactly one NLI hypothesis
template. A gap (taxonomy flag with no hypothesis) or an orphan (hypothesis
without a taxonomy flag) fails loudly at construction -- zero-shot
inference must be defined for every flag it can emit.
"""

from __future__ import annotations

import json

import pytest
from bh_sentinel.core._config import default_flag_taxonomy_path

from bh_sentinel.ml._config import (
    HypothesesError,
    default_hypotheses_path,
    load_hypotheses,
)


def _all_taxonomy_flag_ids() -> set[str]:
    with open(default_flag_taxonomy_path()) as f:
        data = json.load(f)
    ids: set[str] = set()
    for domain in data.get("domains", []):
        for flag in domain.get("flags", []):
            ids.add(flag["flag_id"])
    return ids


def test_default_hypotheses_path_exists() -> None:
    path = default_hypotheses_path()
    assert path.exists(), f"default zero_shot_hypotheses.yaml missing: {path}"
    assert path.name == "zero_shot_hypotheses.yaml"


def test_load_hypotheses_returns_dict() -> None:
    h = load_hypotheses()
    assert isinstance(h, dict)
    assert all(isinstance(k, str) for k in h)
    assert all(isinstance(v, str) and v for v in h.values())


def test_every_taxonomy_flag_has_a_hypothesis() -> None:
    taxonomy_flags = _all_taxonomy_flag_ids()
    hypothesis_flags = set(load_hypotheses().keys())
    missing = taxonomy_flags - hypothesis_flags
    assert not missing, f"flags with no NLI hypothesis: {sorted(missing)}"


def test_no_orphan_hypotheses_without_a_taxonomy_flag() -> None:
    taxonomy_flags = _all_taxonomy_flag_ids()
    hypothesis_flags = set(load_hypotheses().keys())
    orphans = hypothesis_flags - taxonomy_flags
    assert not orphans, f"hypotheses for non-existent flags: {sorted(orphans)}"


def test_hypothesis_count_matches_taxonomy_count() -> None:
    """Defensive: there should be exactly 40 hypotheses today."""
    assert len(load_hypotheses()) == len(_all_taxonomy_flag_ids()) == 40


def test_loader_rejects_empty_hypothesis_string(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("SH-001: \nSH-002: A real hypothesis.\n")
    with pytest.raises(HypothesesError):
        load_hypotheses(bad)
