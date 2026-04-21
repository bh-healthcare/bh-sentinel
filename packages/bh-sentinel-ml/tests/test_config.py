"""Tests for the ML config loader (ml_config.yaml)."""

from __future__ import annotations

from pathlib import Path

import pytest

from bh_sentinel.ml._config import (
    MLConfig,
    MLConfigError,
    default_ml_config_path,
    load_ml_config,
)


def test_default_ml_config_path_exists() -> None:
    """The vendored default config ships with the wheel."""
    path = default_ml_config_path()
    assert path.exists(), f"default ml_config.yaml missing: {path}"
    assert path.name == "ml_config.yaml"


def test_load_default_ml_config_returns_mlconfig() -> None:
    cfg = load_ml_config()
    assert isinstance(cfg, MLConfig)


def test_ml_config_required_fields_present() -> None:
    cfg = load_ml_config()
    assert isinstance(cfg.model_repo, str) and cfg.model_repo
    assert isinstance(cfg.model_revision, str) and cfg.model_revision
    assert isinstance(cfg.onnx_filename, str) and cfg.onnx_filename
    assert isinstance(cfg.model_sha256, str) and cfg.model_sha256
    assert isinstance(cfg.max_sentence_length, int) and cfg.max_sentence_length > 0
    assert isinstance(cfg.max_batch_size, int) and cfg.max_batch_size > 0
    assert isinstance(cfg.min_emit_confidence, float)
    assert 0.0 <= cfg.min_emit_confidence <= 1.0
    assert isinstance(cfg.calibration, dict)
    assert cfg.calibration.get("strategy") in ("fixed_discount", "temperature_scaling")


def test_malformed_yaml_raises_mlconfigerror(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("::::not valid yaml::::\n  indent: broken\n")
    with pytest.raises(MLConfigError):
        load_ml_config(bad)


def test_missing_required_key_raises_mlconfigerror(tmp_path: Path) -> None:
    """Missing keys must fail loudly with the key name in the error."""
    partial = tmp_path / "partial.yaml"
    partial.write_text("model_repo: foo\n")
    with pytest.raises(MLConfigError) as exc:
        load_ml_config(partial)
    assert "model_revision" in str(exc.value) or "missing" in str(exc.value).lower()


def test_load_ml_config_accepts_custom_path(tmp_path: Path) -> None:
    sha = "a" * 64
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        "model_repo: test/repo\n"
        "model_revision: abc123\n"
        "onnx_filename: model.onnx\n"
        f'model_sha256: "{sha}"\n'
        "max_sentence_length: 256\n"
        "max_batch_size: 32\n"
        "min_emit_confidence: 0.55\n"
        "calibration:\n"
        "  strategy: fixed_discount\n"
        "  discount: 0.85\n"
    )
    cfg = load_ml_config(custom)
    assert cfg.model_repo == "test/repo"
    assert cfg.model_revision == "abc123"


def test_sha256_must_be_64_hex_chars(tmp_path: Path) -> None:
    bad_sha = tmp_path / "badsha.yaml"
    bad_sha.write_text(
        "model_repo: test/repo\n"
        "model_revision: abc123\n"
        "onnx_filename: model.onnx\n"
        "model_sha256: shortsha\n"
        "max_sentence_length: 256\n"
        "max_batch_size: 32\n"
        "min_emit_confidence: 0.55\n"
        "calibration:\n"
        "  strategy: fixed_discount\n"
        "  discount: 0.85\n"
    )
    with pytest.raises(MLConfigError):
        load_ml_config(bad_sha)
