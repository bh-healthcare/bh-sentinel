"""Model cache path resolution.

Hybrid distribution strategy (see docs/architecture.md and the v0.2 plan):

1. Explicit model_path wins -- for container bakes and tests.
2. $BH_SENTINEL_ML_CACHE -- for ops with a shared filesystem location.
3. platformdirs user cache -- for local dev machines.

Auto-download falls back to huggingface_hub when the cache is empty.
The env var $BH_SENTINEL_ML_OFFLINE=1 is the production safety rail: when
set, auto_download is forced to False regardless of the caller kwarg, and
huggingface_hub is never imported -- required for VPC-isolated Lambda
cold starts where any outbound network call is a compliance failure.
"""

from __future__ import annotations

import os
from pathlib import Path

from bh_sentinel.ml.exceptions import ModelNotFoundError

_OFFLINE_ENV_VAR = "BH_SENTINEL_ML_OFFLINE"
_CACHE_ENV_VAR = "BH_SENTINEL_ML_CACHE"


def resolve_model_path(
    model_path: Path | None,
    *,
    auto_download: bool = True,
    onnx_filename: str,
    model_repo: str | None = None,
    model_revision: str | None = None,
) -> Path:
    """Resolve the directory containing the ONNX model file.

    Returns the directory (not the file itself). The transformer loader
    joins onnx_filename to this path.

    Args:
        model_path: explicit cache directory. Wins over env var + defaults.
        auto_download: if True and no cache is populated, call HF Hub.
        onnx_filename: the ONNX file that must exist in the resolved dir.
        model_repo: HF Hub repo_id for auto-download.
        model_revision: pinned HF Hub revision SHA for auto-download.

    Raises:
        ModelNotFoundError: no path is populated and auto_download is
            disabled (either by caller or by $BH_SENTINEL_ML_OFFLINE=1).
            The error message lists the paths that were checked and
            includes a CLI remediation hint.
    """
    offline = os.environ.get(_OFFLINE_ENV_VAR) == "1"
    effective_auto_download = auto_download and not offline

    candidates: list[Path] = []

    if model_path is not None:
        candidates.append(Path(model_path))

    env_cache = os.environ.get(_CACHE_ENV_VAR)
    if env_cache:
        candidates.append(Path(env_cache))

    candidates.append(_platformdirs_default())

    for candidate in candidates:
        if (candidate / onnx_filename).exists():
            return candidate

    if effective_auto_download:
        if not (model_repo and model_revision):
            raise ModelNotFoundError(
                "auto_download requested but model_repo and model_revision "
                "are not set. Pass them explicitly or use a pre-populated cache."
            )
        downloaded = _download_from_hf(repo_id=model_repo, revision=model_revision)
        if (downloaded / onnx_filename).exists():
            return downloaded
        raise ModelNotFoundError(_not_found_message(candidates + [downloaded], onnx_filename))

    raise ModelNotFoundError(_not_found_message(candidates, onnx_filename))


def _platformdirs_default() -> Path:
    """User-level cache directory for bh-sentinel-ml."""
    from platformdirs import user_cache_dir

    return Path(user_cache_dir("bh-sentinel-ml"))


def _download_from_hf(*, repo_id: str, revision: str) -> Path:
    """Fetch a pinned HF Hub revision and return the local snapshot path."""
    from huggingface_hub import snapshot_download

    local = snapshot_download(repo_id=repo_id, revision=revision)
    return Path(local)


def _not_found_message(paths: list[Path], onnx_filename: str) -> str:
    """Build a PHI-safe, static-form error message.

    Lists only paths and the CLI invocation. No input text, no host
    info beyond what the runtime environment already exposes via paths.
    """
    path_list = "\n  - ".join(str(p) for p in paths)
    return (
        f"ONNX model {onnx_filename!r} not found in any of these locations:\n"
        f"  - {path_list}\n"
        f"\n"
        f"Resolve by either:\n"
        f"  1. Pre-bake the model in a container at build time:\n"
        f"       bh-sentinel-ml download-model --output /opt/bh-sentinel-ml/model\n"
        f"     then set transformer_model_path to that directory.\n"
        f"  2. Unset $BH_SENTINEL_ML_OFFLINE and run once online to populate\n"
        f"     the cache via auto_download.\n"
    )
