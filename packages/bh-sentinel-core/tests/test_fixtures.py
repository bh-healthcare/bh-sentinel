"""Parametrized test suite for all 84 test_fixtures.yaml entries."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bh_sentinel.core._config import default_flag_taxonomy_path, default_patterns_path
from bh_sentinel.core.negation_detector import NegationDetector
from bh_sentinel.core.pattern_matcher import PatternMatcher
from bh_sentinel.core.preprocessor import TextPreprocessor
from bh_sentinel.core.taxonomy import FlagTaxonomy
from bh_sentinel.core.temporal_detector import TemporalDetector

# ---------------------------------------------------------------------------
# Load fixtures once at module level for parametrize
# ---------------------------------------------------------------------------
_FIXTURES_PATH = Path(default_flag_taxonomy_path()).parent / "test_fixtures.yaml"

with open(_FIXTURES_PATH) as _f:
    _RAW = yaml.safe_load(_f)

_ALL_FIXTURES: list[tuple[str, int, dict]] = []
for _category, _entries in _RAW.items():
    if _category.startswith("_") or not isinstance(_entries, list):
        continue
    for _i, _entry in enumerate(_entries):
        _ALL_FIXTURES.append((_category, _i, _entry))


def _make_id(val):
    category, idx, entry = val
    short = entry["input"][:50].replace(" ", "_")
    return f"{category}_{idx}_{short}"


# ---------------------------------------------------------------------------
# Shared matcher built once per module
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def matcher() -> PatternMatcher:
    taxonomy = FlagTaxonomy(default_flag_taxonomy_path())
    return PatternMatcher(default_patterns_path(), taxonomy, NegationDetector(), TemporalDetector())


@pytest.fixture(scope="module")
def pp() -> TextPreprocessor:
    return TextPreprocessor()


# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "category,idx,entry",
    _ALL_FIXTURES,
    ids=[_make_id(f) for f in _ALL_FIXTURES],
)
def test_fixture(matcher, pp, category, idx, entry):
    text = entry["input"]
    expect_flags = set(entry.get("expect_flags", []))
    expect_suppressed = set(entry.get("expect_suppressed", []))
    expect_none = entry.get("expect_none", False)
    expect_temporal = entry.get("expect_temporal")

    preprocessed = pp.process(text)
    candidates = matcher.match(preprocessed)
    non_negated = {c.flag_id for c in candidates if not c.negated}
    negated = {c.flag_id for c in candidates if c.negated}

    if expect_none:
        assert len(non_negated) == 0, (
            f"Expected no flags but got {sorted(non_negated)} for: {text[:80]}"
        )

    if expect_flags:
        missing = expect_flags - non_negated
        assert not missing, (
            f"Missing expected flags {sorted(missing)} (got {sorted(non_negated)}) for: {text[:80]}"
        )

    if expect_suppressed:
        missing_suppressed = expect_suppressed - negated
        assert not missing_suppressed, (
            f"Expected suppressed flags {sorted(missing_suppressed)} "
            f"not found in negated set {sorted(negated)} for: {text[:80]}"
        )

    if expect_temporal:
        for c in candidates:
            if c.flag_id in expect_flags and not c.negated:
                assert c.temporal_context == expect_temporal, (
                    f"Flag {c.flag_id} expected temporal={expect_temporal} "
                    f"but got {c.temporal_context} for: {text[:80]}"
                )
