"""Verify vendored config matches source config."""

from __future__ import annotations

import filecmp
from pathlib import Path

from bh_sentinel.core._config import _default_config_dir


def test_vendored_config_matches_source():
    """Vendored _default_config/ must match the source config/ directory."""
    vendored = _default_config_dir()
    # Source config is at repo root: bh-sentinel/config/
    source = Path(__file__).parents[2] / "../../config"

    if not source.exists():
        # Running from installed wheel, skip this test.
        return

    source = source.resolve()
    vendored_resolved = vendored.resolve()

    for vendored_file in vendored_resolved.iterdir():
        if vendored_file.name.startswith("."):
            continue
        source_file = source / vendored_file.name
        assert source_file.exists(), f"Source missing: {vendored_file.name}"
        assert filecmp.cmp(str(vendored_file), str(source_file), shallow=False), (
            f"Drift detected: {vendored_file.name}"
        )
