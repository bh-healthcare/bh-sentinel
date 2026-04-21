"""Configuration loaders for bh-sentinel-ml.

Two configs, both YAML, both shipped vendored inside the wheel:

- ml_config.yaml: model identity, inference budgets, calibration strategy
- zero_shot_hypotheses.yaml: one NLI hypothesis per flag_id

Canonical sources live at bh-sentinel/config/ml/. The wheel ships copies
under packages/bh-sentinel-ml/src/bh_sentinel/ml/_default_config/. Drift
is caught by test_config_vendored.py on every CI run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent / "_default_config"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

_REQUIRED_KEYS: tuple[str, ...] = (
    "model_repo",
    "model_revision",
    "onnx_filename",
    "model_sha256",
    "max_sentence_length",
    "max_batch_size",
    "min_emit_confidence",
    "calibration",
)


class MLConfigError(ValueError):
    """Raised when ml_config.yaml is malformed or missing required fields."""


class HypothesesError(ValueError):
    """Raised when zero_shot_hypotheses.yaml is malformed or incomplete."""


@dataclass(frozen=True, slots=True)
class MLConfig:
    """Parsed bh-sentinel-ml configuration.

    Frozen so downstream components cannot mutate shared state. All fields
    are validated at load time -- invalid values never reach inference.
    """

    model_repo: str
    model_revision: str
    onnx_filename: str
    model_sha256: str
    max_sentence_length: int
    max_batch_size: int
    min_emit_confidence: float
    calibration: dict[str, Any]


def default_ml_config_path() -> Path:
    """Path to the vendored ml_config.yaml shipped in the wheel."""
    return _DEFAULT_CONFIG_DIR / "ml_config.yaml"


def default_hypotheses_path() -> Path:
    """Path to the vendored zero_shot_hypotheses.yaml shipped in the wheel."""
    return _DEFAULT_CONFIG_DIR / "zero_shot_hypotheses.yaml"


def load_ml_config(path: Path | None = None) -> MLConfig:
    """Load and validate the ML configuration.

    Raises:
        MLConfigError: if the file is missing, malformed, missing required
            keys, or has invalid field values.
    """
    cfg_path = path or default_ml_config_path()
    raw = _read_yaml(cfg_path, error_cls=MLConfigError)

    if not isinstance(raw, dict):
        raise MLConfigError(f"{cfg_path} must be a YAML mapping, got {type(raw).__name__}")

    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise MLConfigError(f"{cfg_path} is missing required key(s): {', '.join(missing)}")

    model_sha256 = str(raw["model_sha256"])
    if not _SHA256_RE.match(model_sha256):
        raise MLConfigError(
            f"{cfg_path}: model_sha256 must be 64 hex characters, got {len(model_sha256)} chars"
        )

    min_conf = float(raw["min_emit_confidence"])
    if not 0.0 <= min_conf <= 1.0:
        raise MLConfigError(
            f"{cfg_path}: min_emit_confidence must be in [0.0, 1.0], got {min_conf}"
        )

    for int_key in ("max_sentence_length", "max_batch_size"):
        val = raw[int_key]
        if not isinstance(val, int) or val <= 0:
            raise MLConfigError(f"{cfg_path}: {int_key} must be a positive integer, got {val!r}")

    calibration = raw["calibration"]
    if not isinstance(calibration, dict) or "strategy" not in calibration:
        raise MLConfigError(f"{cfg_path}: calibration must be a mapping with a 'strategy' key")
    if calibration["strategy"] not in ("fixed_discount", "temperature_scaling"):
        raise MLConfigError(
            f"{cfg_path}: calibration.strategy must be 'fixed_discount' or "
            f"'temperature_scaling', got {calibration['strategy']!r}"
        )

    return MLConfig(
        model_repo=str(raw["model_repo"]),
        model_revision=str(raw["model_revision"]),
        onnx_filename=str(raw["onnx_filename"]),
        model_sha256=model_sha256,
        max_sentence_length=int(raw["max_sentence_length"]),
        max_batch_size=int(raw["max_batch_size"]),
        min_emit_confidence=min_conf,
        calibration=dict(calibration),
    )


def load_hypotheses(path: Path | None = None) -> dict[str, str]:
    """Load the zero-shot hypothesis mapping.

    Returns a plain dict of flag_id -> hypothesis template. Cross-checking
    against the flag taxonomy (for gaps or orphans) is the caller's job
    -- the loader only validates structural well-formedness.

    Raises:
        HypothesesError: if the file is missing, malformed, or any
            hypothesis is empty or not a string.
    """
    hyp_path = path or default_hypotheses_path()
    raw = _read_yaml(hyp_path, error_cls=HypothesesError)

    if not isinstance(raw, dict):
        raise HypothesesError(f"{hyp_path} must be a YAML mapping, got {type(raw).__name__}")

    result: dict[str, str] = {}
    for flag_id, hypothesis in raw.items():
        if not isinstance(flag_id, str):
            raise HypothesesError(
                f"{hyp_path}: flag_id keys must be strings, got {type(flag_id).__name__}"
            )
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise HypothesesError(
                f"{hyp_path}: hypothesis for {flag_id!r} must be a non-empty string"
            )
        result[flag_id] = hypothesis.strip()

    return result


def _read_yaml(path: Path, *, error_cls: type[Exception]) -> Any:
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError as e:
        raise error_cls(f"config file not found: {path}") from e
    except yaml.YAMLError as e:
        raise error_cls(f"{path}: invalid YAML ({e})") from e
