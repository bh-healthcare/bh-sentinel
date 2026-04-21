"""bh-sentinel-ml: Transformer-based clinical safety signal detection."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

__version__ = "0.2.0"

_MIN_CORE_VERSION: tuple[int, int, int] = (0, 1, 1)
"""Minimum bh-sentinel-core release that exposes the Pipeline kwargs bh-sentinel-ml depends on."""


def _parse_release(raw: str) -> tuple[int, int, int] | None:
    """Parse the leading ``X.Y.Z`` release segment of a PEP 440 version string.

    Returns ``None`` for anything we can't confidently compare (exotic tags,
    non-numeric segments). Callers MUST treat ``None`` as "skip the check" --
    pip's install-time resolver is the strict guard; this runtime check only
    exists to catch the ``--no-deps`` / vendored / editable-monorepo case where
    somebody has paired bh-sentinel-ml with an older bh-sentinel-core by hand.
    """
    head = raw.split("+", 1)[0]
    for sep in (".dev", ".post", "a", "b", "rc", "-"):
        head = head.split(sep, 1)[0]
    parts = head.split(".")
    if len(parts) < 3 or not all(p.isdigit() for p in parts[:3]):
        return None
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def _check_core_compatibility() -> None:
    """Fail fast if the installed bh-sentinel-core predates the kwargs we need."""
    try:
        core_version_str = _pkg_version("bh-sentinel-core")
    except PackageNotFoundError as exc:
        min_str = ".".join(str(n) for n in _MIN_CORE_VERSION)
        raise ImportError(
            f"bh-sentinel-ml {__version__} requires bh-sentinel-core>={min_str}, "
            "but bh-sentinel-core is not installed. "
            "Install it with: pip install bh-sentinel-core"
        ) from exc

    parsed = _parse_release(core_version_str)
    if parsed is not None and parsed < _MIN_CORE_VERSION:
        min_str = ".".join(str(n) for n in _MIN_CORE_VERSION)
        raise ImportError(
            f"bh-sentinel-ml {__version__} requires bh-sentinel-core>={min_str}, "
            f"found bh-sentinel-core=={core_version_str}. "
            "Upgrade with: pip install -U bh-sentinel-core"
        )


_check_core_compatibility()


# Re-exports are deliberately placed AFTER the compatibility check so a version
# mismatch surfaces as a clean ImportError from this module rather than a
# confusing AttributeError from somewhere inside bh_sentinel.core.*.
from bh_sentinel.ml.calibration import (  # noqa: E402
    Calibrator,
    FixedDiscount,
    TemperatureScaling,
    compute_ece,
)
from bh_sentinel.ml.exceptions import (  # noqa: E402
    InferenceError,
    ModelIntegrityError,
    ModelNotFoundError,
)
from bh_sentinel.ml.merge import MergeResult, merge_candidates  # noqa: E402
from bh_sentinel.ml.model_cache import resolve_model_path  # noqa: E402
from bh_sentinel.ml.transformer import TransformerClassifier  # noqa: E402
from bh_sentinel.ml.zero_shot import ZeroShotClassifier  # noqa: E402

__all__ = [
    "__version__",
    "TransformerClassifier",
    "ZeroShotClassifier",
    "Calibrator",
    "FixedDiscount",
    "TemperatureScaling",
    "compute_ece",
    "MergeResult",
    "merge_candidates",
    "resolve_model_path",
    "InferenceError",
    "ModelIntegrityError",
    "ModelNotFoundError",
]
