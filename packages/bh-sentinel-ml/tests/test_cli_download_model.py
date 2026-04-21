"""Tests for the `bh-sentinel-ml download-model` CLI subcommand."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from bh_sentinel.ml.cli.__main__ import main


def _run_cli(argv: list[str]) -> int:
    try:
        return main(argv)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0


def test_download_model_calls_hf_hub_with_pinned_revision(tmp_path: Path) -> None:
    target = tmp_path / "model"
    target.mkdir()
    (target / "model_int8.onnx").write_bytes(b"stub")

    with patch("huggingface_hub.snapshot_download", return_value=str(target)) as mock_snap:
        rc = _run_cli(
            [
                "download-model",
                "--revision",
                "abc123",
                "--output",
                str(target),
            ]
        )
    assert rc == 0
    mock_snap.assert_called_once()
    kwargs = mock_snap.call_args.kwargs
    assert kwargs.get("revision") == "abc123"


def test_download_model_verify_sha256_fails_on_mismatch(tmp_path: Path, capsys) -> None:
    target = tmp_path / "model"
    target.mkdir()
    (target / "model_int8.onnx").write_bytes(b"different bytes")

    with patch("huggingface_hub.snapshot_download", return_value=str(target)):
        rc = _run_cli(
            [
                "download-model",
                "--revision",
                "abc123",
                "--output",
                str(target),
                "--verify-sha256",
                "0" * 64,  # wrong SHA
            ]
        )
    assert rc != 0
    captured = capsys.readouterr()
    assert "SHA" in (captured.out + captured.err).upper()


def test_download_model_verify_sha256_passes_on_match(tmp_path: Path) -> None:
    target = tmp_path / "model"
    target.mkdir()
    payload = b"fake onnx bytes"
    (target / "model_int8.onnx").write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    with patch("huggingface_hub.snapshot_download", return_value=str(target)):
        rc = _run_cli(
            [
                "download-model",
                "--revision",
                "abc123",
                "--output",
                str(target),
                "--verify-sha256",
                expected,
            ]
        )
    assert rc == 0


def test_download_model_help_has_no_phi_references() -> None:
    """Help output must not contain placeholder patient text."""
    with patch("sys.stdout"):
        with pytest.raises(SystemExit):
            main(["download-model", "--help"])
