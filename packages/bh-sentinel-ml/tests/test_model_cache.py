"""Tests for the model cache path resolution and production safety rails.

Covers:
- explicit path > env var > platformdirs default precedence
- auto-download behavior when cache is empty
- BH_SENTINEL_ML_OFFLINE=1 forces auto_download=False
- huggingface_hub is not imported at all when offline
- ModelNotFoundError carries a PHI-safe static message listing paths checked
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from bh_sentinel.ml.model_cache import (
    ModelNotFoundError,
    resolve_model_path,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure no leaked env vars between tests."""
    monkeypatch.delenv("BH_SENTINEL_ML_OFFLINE", raising=False)
    monkeypatch.delenv("BH_SENTINEL_ML_CACHE", raising=False)
    yield


def test_explicit_model_path_wins(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit_cache"
    explicit.mkdir()
    (explicit / "model_int8.onnx").write_bytes(b"stub")
    resolved = resolve_model_path(
        model_path=explicit, auto_download=False, onnx_filename="model_int8.onnx"
    )
    assert resolved == explicit


def test_env_var_cache_used_when_no_explicit_path(tmp_path: Path, monkeypatch) -> None:
    env_cache = tmp_path / "env_cache"
    env_cache.mkdir()
    (env_cache / "model_int8.onnx").write_bytes(b"stub")
    monkeypatch.setenv("BH_SENTINEL_ML_CACHE", str(env_cache))
    resolved = resolve_model_path(
        model_path=None, auto_download=False, onnx_filename="model_int8.onnx"
    )
    assert resolved == env_cache


def test_no_cache_and_auto_download_false_raises(tmp_path: Path, monkeypatch) -> None:
    """Missing cache + auto_download disabled = fail fast with a clean message."""
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("BH_SENTINEL_ML_CACHE", str(empty))
    with pytest.raises(ModelNotFoundError) as exc:
        resolve_model_path(model_path=None, auto_download=False, onnx_filename="model_int8.onnx")
    msg = str(exc.value)
    # PHI-safe: no input text, just paths and the CLI invocation.
    assert "bh-sentinel-ml download-model" in msg
    assert str(empty) in msg


def test_offline_env_var_forces_auto_download_false(tmp_path: Path, monkeypatch) -> None:
    """BH_SENTINEL_ML_OFFLINE=1 overrides the caller kwarg -- the production rail."""
    empty = tmp_path / "empty_offline"
    empty.mkdir()
    monkeypatch.setenv("BH_SENTINEL_ML_CACHE", str(empty))
    monkeypatch.setenv("BH_SENTINEL_ML_OFFLINE", "1")
    with pytest.raises(ModelNotFoundError):
        # Caller passes auto_download=True, but the env var must override.
        resolve_model_path(
            model_path=None,
            auto_download=True,
            onnx_filename="model_int8.onnx",
        )


def test_offline_env_var_does_not_import_huggingface_hub(tmp_path: Path, monkeypatch) -> None:
    """When offline, huggingface_hub must not be imported at all.

    This guards against accidental cold-start network calls in VPC-isolated
    Lambdas. Verified by removing the module from sys.modules, running the
    resolve, and asserting it was never reloaded.
    """
    empty = tmp_path / "empty_noimport"
    empty.mkdir()
    monkeypatch.setenv("BH_SENTINEL_ML_CACHE", str(empty))
    monkeypatch.setenv("BH_SENTINEL_ML_OFFLINE", "1")

    # Snapshot modules, drop huggingface_hub, run resolve, re-check.
    dropped = [name for name in list(sys.modules) if name.startswith("huggingface_hub")]
    for name in dropped:
        del sys.modules[name]

    with pytest.raises(ModelNotFoundError):
        resolve_model_path(model_path=None, auto_download=True, onnx_filename="model_int8.onnx")

    assert not any(name.startswith("huggingface_hub") for name in sys.modules), (
        "huggingface_hub was imported in the offline path"
    )


def test_auto_download_calls_hf_hub(tmp_path: Path, monkeypatch) -> None:
    """With auto_download=True and no cache, we call snapshot_download once."""
    target_cache = tmp_path / "hf_cache"
    target_cache.mkdir()
    # Pre-populate what HF would have returned so the resolve succeeds after.
    (target_cache / "model_int8.onnx").write_bytes(b"stub")

    monkeypatch.setenv("BH_SENTINEL_ML_CACHE", str(target_cache / "nonexistent"))

    fake_snapshot = patch(
        "huggingface_hub.snapshot_download",
        return_value=str(target_cache),
    )

    with fake_snapshot as mock_snapshot:
        resolved = resolve_model_path(
            model_path=None,
            auto_download=True,
            onnx_filename="model_int8.onnx",
            model_repo="test/repo",
            model_revision="abc123",
        )
    mock_snapshot.assert_called_once()
    kwargs = mock_snapshot.call_args.kwargs
    assert kwargs.get("repo_id") == "test/repo"
    assert kwargs.get("revision") == "abc123"
    assert resolved == target_cache


def test_auto_download_missing_onnx_file_raises(tmp_path: Path, monkeypatch) -> None:
    """If HF snapshot completes but the ONNX file isn't there, we fail cleanly."""
    target_cache = tmp_path / "hf_empty"
    target_cache.mkdir()
    # NOTE: no model_int8.onnx inside target_cache.

    monkeypatch.setenv("BH_SENTINEL_ML_CACHE", str(tmp_path / "nonexistent"))

    with patch("huggingface_hub.snapshot_download", return_value=str(target_cache)):
        with pytest.raises(ModelNotFoundError):
            resolve_model_path(
                model_path=None,
                auto_download=True,
                onnx_filename="model_int8.onnx",
                model_repo="test/repo",
                model_revision="abc123",
            )
