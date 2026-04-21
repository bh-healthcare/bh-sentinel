"""Confidence calibration for transformer logits.

Architecture reference: docs/architecture.md section 4.8.

Phase A (v0.2, default): FixedDiscount(0.85) -- raw softmax probabilities
multiplied by a conservative factor. Matches the architecture's
"interim discount" guidance until labeled clinical data is available.

Phase B (v0.3): TemperatureScaling(T) -- fitted on a held-out validation
set, validated against ECE < 0.05. Fully implemented here; the fitting
workflow lives in the calibrate CLI.

Any Calibrator is passed raw logits of shape (N, C) and returns
calibrated per-class probabilities of the same shape.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

__all__ = [
    "Calibrator",
    "FixedDiscount",
    "TemperatureScaling",
    "compute_ece",
]

# IdentityCalibrator and softmax are intentionally not in __all__.
# - IdentityCalibrator is a test-only stand-in that skips calibration;
#   production code should use FixedDiscount or TemperatureScaling.
# - softmax is an internal numerical helper. Tests that need it can still
#   `from bh_sentinel.ml.calibration import softmax` explicitly.


class Calibrator(Protocol):
    """Structural protocol for calibrators.

    Implementations take raw logits shaped (N, C) and return calibrated
    per-class probabilities shaped (N, C). Rows must sum to at most 1.0;
    rows may sum to less than 1.0 under conservative calibration (e.g.
    FixedDiscount). Implementations must be deterministic and pure -- no
    state mutation, no network, no file I/O at call time.
    """

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Calibrate raw logits to per-class probabilities. See class docstring."""


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


class IdentityCalibrator:
    """Pass-through calibrator: softmax without any adjustment.

    Used only in tests where the raw softmax values are the point of
    the assertion. Never used in production -- see FixedDiscount."""

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        return softmax(np.asarray(logits, dtype=np.float32))


class FixedDiscount:
    """Multiply softmax probabilities by a fixed factor.

    Phase A default per architecture 4.8. Conservative: a factor of 0.85
    means "treat every transformer-derived probability as 85% of what
    the raw softmax said" -- a deliberate dampening that prevents the
    uncalibrated L2 layer from dominating L1 in the max-merge.
    """

    def __init__(self, factor: float = 0.85) -> None:
        if not 0.0 < factor <= 1.0:
            raise ValueError(f"FixedDiscount factor must be in (0.0, 1.0], got {factor}")
        self._factor = float(factor)

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        probs = softmax(np.asarray(logits, dtype=np.float32))
        return probs * self._factor


class TemperatureScaling:
    """Temperature scaling calibration.

    calibrated = softmax(logits / T)

    At T=1 the output is the raw softmax (identity). T>1 flattens the
    distribution (useful for overconfident classifiers); T<1 sharpens
    it. A single scalar T is fit against a held-out validation set by
    minimizing NLL.

    For v0.2 this is a fully-working implementation, but it is not
    validated against clinical data yet. Phase B (v0.3) runs fit()
    against clinician-labeled examples and asserts ECE < 0.05 before
    switching the default strategy in ml_config.yaml.
    """

    def __init__(self, T: float = 1.0) -> None:
        if T <= 0:
            raise ValueError(f"temperature must be positive, got {T}")
        self._T = float(T)

    @property
    def T(self) -> float:
        return self._T

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        scaled = np.asarray(logits, dtype=np.float32) / self._T
        return softmax(scaled)

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        *,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> float:
        """Fit T by minimizing NLL on labeled data.

        Uses a 1-D ternary search over log(T). Monotonic NLL curve in
        log(T) makes this robust without pulling in scipy. Returns the
        fitted T.
        """
        logits_arr = np.asarray(logits, dtype=np.float64)
        labels_arr = np.asarray(labels, dtype=np.int64)
        if logits_arr.ndim != 2:
            raise ValueError("logits must be 2-D")
        if labels_arr.ndim != 1 or labels_arr.shape[0] != logits_arr.shape[0]:
            raise ValueError("labels must be 1-D with the same number of rows as logits")

        def nll(T: float) -> float:
            probs = softmax(logits_arr / T)
            picked = probs[np.arange(len(labels_arr)), labels_arr]
            eps = 1e-12
            return float(-np.log(np.clip(picked, eps, 1.0)).mean())

        lo, hi = 1e-3, 1e3
        for _ in range(max_iter):
            m1 = lo + (hi - lo) / 3
            m2 = hi - (hi - lo) / 3
            if nll(m1) < nll(m2):
                hi = m2
            else:
                lo = m1
            if hi - lo < tol:
                break

        self._T = (lo + hi) / 2
        return self._T


def compute_ece(probs: np.ndarray, labels: np.ndarray, *, n_bins: int = 10) -> float:
    """Expected Calibration Error (Guo et al. 2017) over n_bins.

    Bins predictions by their max-probability; ECE is the weighted
    absolute gap between each bin's average confidence and accuracy.
    Zero is perfect calibration. Target for production per architecture
    4.8 is < 0.05.
    """
    probs_arr = np.asarray(probs, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    confidence = probs_arr.max(axis=-1)
    predicted = probs_arr.argmax(axis=-1)
    correct = (predicted == labels_arr).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(labels_arr)
    total = 0.0
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        bin_size = int(mask.sum())
        if bin_size == 0:
            continue
        bin_conf = float(confidence[mask].mean())
        bin_acc = float(correct[mask].mean())
        total += (bin_size / n) * abs(bin_conf - bin_acc)
    return float(total)
