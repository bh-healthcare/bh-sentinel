"""Tests for the `bh-sentinel-ml calibrate` CLI subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bh_sentinel.ml.cli.__main__ import main


def _write_labels(path: Path, n: int = 300) -> None:
    """Write a JSONL labels file with deliberately overconfident examples."""
    rng = np.random.default_rng(123)
    labels = rng.integers(0, 3, size=n)
    with open(path, "w") as f:
        for label in labels:
            logits = rng.normal(scale=0.1, size=3).tolist()
            # 60% correct at high confidence (classic overconfident classifier)
            if rng.random() < 0.6:
                logits[label] += 5.0
            else:
                wrong = int(rng.choice([c for c in range(3) if c != label]))
                logits[wrong] += 5.0
            f.write(json.dumps({"logits": logits, "label": int(label)}) + "\n")


def _run_cli(argv: list[str]) -> int:
    try:
        return main(argv)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0


def test_calibrate_writes_output_with_expected_keys(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    out_path = tmp_path / "calibration.json"
    _write_labels(labels_path, n=300)

    rc = _run_cli(["calibrate", "--labels", str(labels_path), "--out", str(out_path)])
    assert rc == 0

    with open(out_path) as f:
        data = json.load(f)
    assert data["strategy"] == "temperature_scaling"
    assert "T" in data
    assert "ece" in data
    assert data["n_examples"] == 300
    # Overconfidence constructed above -> T > 1.
    assert data["T"] > 1.0


def test_calibrate_fails_on_malformed_labels(tmp_path: Path) -> None:
    labels_path = tmp_path / "bad.jsonl"
    out_path = tmp_path / "calibration.json"
    labels_path.write_text("not json\n")

    rc = _run_cli(["calibrate", "--labels", str(labels_path), "--out", str(out_path)])
    assert rc != 0
    assert not out_path.exists()


def test_calibrate_requires_labels_flag(tmp_path: Path) -> None:
    rc = _run_cli(["calibrate"])
    assert rc != 0
