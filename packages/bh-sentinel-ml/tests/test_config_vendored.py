"""Verify vendored config matches the canonical repo-root config.

Mirrors the core package's test_config_vendored.py discipline: the YAML
shipped inside the wheel at src/bh_sentinel/ml/_default_config/ must be
byte-identical to the canonical source at bh-sentinel/config/ml/.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_DIR = REPO_ROOT / "config" / "ml"
VENDORED_DIR = (
    REPO_ROOT / "packages" / "bh-sentinel-ml" / "src" / "bh_sentinel" / "ml" / "_default_config"
)

_FILES = ["ml_config.yaml", "zero_shot_hypotheses.yaml"]


def test_canonical_ml_config_dir_exists() -> None:
    assert CANONICAL_DIR.exists(), f"canonical config dir missing: {CANONICAL_DIR}"


def test_vendored_ml_config_dir_exists() -> None:
    assert VENDORED_DIR.exists(), f"vendored config dir missing: {VENDORED_DIR}"


def test_ml_config_yaml_is_byte_identical() -> None:
    canonical = (CANONICAL_DIR / "ml_config.yaml").read_bytes()
    vendored = (VENDORED_DIR / "ml_config.yaml").read_bytes()
    assert canonical == vendored, (
        "ml_config.yaml drift: canonical and vendored copies diverge. "
        "Sync from config/ml/ to packages/bh-sentinel-ml/src/bh_sentinel/ml/_default_config/."
    )


def test_hypotheses_yaml_is_byte_identical() -> None:
    canonical = (CANONICAL_DIR / "zero_shot_hypotheses.yaml").read_bytes()
    vendored = (VENDORED_DIR / "zero_shot_hypotheses.yaml").read_bytes()
    assert canonical == vendored, (
        "zero_shot_hypotheses.yaml drift: canonical and vendored copies diverge. "
        "Sync from config/ml/ to packages/bh-sentinel-ml/src/bh_sentinel/ml/_default_config/."
    )


def test_no_extra_files_in_either_dir() -> None:
    canonical_files = {p.name for p in CANONICAL_DIR.iterdir() if p.is_file()}
    vendored_files = {p.name for p in VENDORED_DIR.iterdir() if p.is_file()}
    # Vendored may contain py.typed and __init__ bytes; filter to YAML only.
    vendored_yamls = {n for n in vendored_files if n.endswith((".yaml", ".yml"))}
    assert canonical_files == set(_FILES), (
        f"canonical dir has unexpected files: {canonical_files - set(_FILES)}"
    )
    assert vendored_yamls == set(_FILES), (
        f"vendored dir has unexpected YAML files: {vendored_yamls - set(_FILES)}"
    )
