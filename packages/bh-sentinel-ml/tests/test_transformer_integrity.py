"""Tests for TransformerClassifier's verify-on-load SHA256 check.

The integrity check must run BEFORE any ONNX InferenceSession is created.
This is the guardrail that keeps a stale or tampered container bake from
silently drifting from whatever was validated during the release.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from bh_sentinel.ml.transformer import (
    ModelIntegrityError,
    TransformerClassifier,
)


def _write_bytes_and_sha(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


_MODULE_ORT_SESSION = "bh_sentinel.ml.transformer.ort.InferenceSession"


def test_integrity_check_passes_on_matching_sha(tmp_path: Path) -> None:
    onnx_path = tmp_path / "model.onnx"
    expected = _write_bytes_and_sha(onnx_path, b"fake onnx bytes for test")

    # InferenceSession construction is stubbed so we only exercise the
    # verify path; actual inference is covered in test_transformer_inference.
    with patch(_MODULE_ORT_SESSION) as mock_sess:
        TransformerClassifier(
            model_path=onnx_path,
            tokenizer_path=onnx_path,  # reused for this test; not loaded
            expected_sha256=expected,
            max_length=256,
            max_batch_size=32,
            skip_tokenizer_load=True,
        )
    mock_sess.assert_called_once()


def test_integrity_check_fails_on_mismatched_sha(tmp_path: Path) -> None:
    """A mismatched SHA raises before ANY InferenceSession is created."""
    onnx_path = tmp_path / "model.onnx"
    _write_bytes_and_sha(onnx_path, b"real bytes")
    wrong_sha = "f" * 64  # definitely not the real SHA

    with patch(_MODULE_ORT_SESSION) as mock_sess:
        with pytest.raises(ModelIntegrityError) as exc:
            TransformerClassifier(
                model_path=onnx_path,
                tokenizer_path=onnx_path,
                expected_sha256=wrong_sha,
                max_length=256,
                max_batch_size=32,
                skip_tokenizer_load=True,
            )
    # No session was ever constructed -- this is the whole point.
    mock_sess.assert_not_called()
    msg = str(exc.value)
    assert "SHA" in msg.upper() or "integrity" in msg.lower()


def test_integrity_check_fails_when_file_missing(tmp_path: Path) -> None:
    ghost = tmp_path / "does-not-exist.onnx"
    with patch(_MODULE_ORT_SESSION) as mock_sess:
        with pytest.raises((ModelIntegrityError, FileNotFoundError)):
            TransformerClassifier(
                model_path=ghost,
                tokenizer_path=ghost,
                expected_sha256="0" * 64,
                max_length=256,
                max_batch_size=32,
                skip_tokenizer_load=True,
            )
    mock_sess.assert_not_called()
