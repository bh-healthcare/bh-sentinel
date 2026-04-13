"""Pattern fixture test runner for bh-sentinel."""

from __future__ import annotations

import yaml

from bh_sentinel.core._config import (
    default_flag_taxonomy_path,
    default_patterns_path,
)
from bh_sentinel.core.negation_detector import NegationDetector
from bh_sentinel.core.pattern_matcher import PatternMatcher
from bh_sentinel.core.preprocessor import TextPreprocessor
from bh_sentinel.core.taxonomy import FlagTaxonomy
from bh_sentinel.core.temporal_detector import TemporalDetector


def run_test_patterns() -> int:
    """Run all pattern test fixtures. Returns 0 on all pass, 1 on any failure."""
    taxonomy = FlagTaxonomy(default_flag_taxonomy_path())
    pm = PatternMatcher(
        default_patterns_path(),
        taxonomy,
        NegationDetector(),
        TemporalDetector(),
    )
    pp = TextPreprocessor()

    # Load fixtures from vendored config.
    import importlib.resources

    config_dir = importlib.resources.files("bh_sentinel.core") / "_default_config"
    test_fixtures_path = config_dir / "test_fixtures.yaml"

    with open(str(test_fixtures_path)) as f:
        data = yaml.safe_load(f)

    total = 0
    passed = 0
    failed = 0

    for category, entries in data.items():
        if category.startswith("_"):
            continue
        if not isinstance(entries, list):
            continue

        for entry in entries:
            total += 1
            text = entry["input"]
            expect_flags = set(entry.get("expect_flags", []))
            expect_suppressed = set(entry.get("expect_suppressed", []))
            expect_none = entry.get("expect_none", False)

            preprocessed = pp.process(text)
            candidates = pm.match(preprocessed)
            non_negated = {c.flag_id for c in candidates if not c.negated}
            negated = {c.flag_id for c in candidates if c.negated}

            ok = True
            notes: list[str] = []

            if expect_none and non_negated:
                ok = False
                notes.append(f"expected none, got {sorted(non_negated)}")

            if expect_flags and not expect_flags.issubset(non_negated):
                missing = expect_flags - non_negated
                ok = False
                notes.append(f"missing flags: {sorted(missing)}")

            if expect_suppressed and not expect_suppressed.issubset(negated):
                missing = expect_suppressed - negated
                ok = False
                notes.append(f"expected suppressed: {sorted(missing)}")

            if ok:
                passed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL"

            note = entry.get("note", "")
            detail = f" -- {'; '.join(notes)}" if notes else ""
            print(f"  [{status}] {category}: {text[:60]}... {note}{detail}")

    print(f"\n{passed}/{total} passed, {failed} failed.")
    return 0 if failed == 0 else 1
