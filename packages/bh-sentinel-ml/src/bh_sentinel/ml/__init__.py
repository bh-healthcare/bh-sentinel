"""bh-sentinel-ml: Transformer-based clinical safety signal detection."""

__version__ = "0.2.0"

from bh_sentinel.ml.calibration import (
    Calibrator,
    FixedDiscount,
    TemperatureScaling,
    compute_ece,
)
from bh_sentinel.ml.exceptions import (
    InferenceError,
    ModelIntegrityError,
    ModelNotFoundError,
)
from bh_sentinel.ml.merge import MergeResult, merge_candidates
from bh_sentinel.ml.model_cache import resolve_model_path
from bh_sentinel.ml.transformer import TransformerClassifier
from bh_sentinel.ml.zero_shot import ZeroShotClassifier

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
