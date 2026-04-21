"""Tests for calibration strategies and the ECE helper."""

from __future__ import annotations

import numpy as np
import pytest

from bh_sentinel.ml.calibration import (
    FixedDiscount,
    TemperatureScaling,
    compute_ece,
    softmax,
)


def test_softmax_rows_sum_to_one() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 5.0]])
    probs = softmax(logits)
    np.testing.assert_allclose(probs.sum(axis=-1), [1.0, 1.0], rtol=1e-6)


def test_fixed_discount_scales_probs() -> None:
    cal = FixedDiscount(factor=0.85)
    logits = np.array([[2.0, 1.0, 0.0]])
    calibrated = cal.calibrate(logits)
    raw = softmax(logits)
    np.testing.assert_allclose(calibrated, raw * 0.85, rtol=1e-5)


def test_fixed_discount_preserves_argmax() -> None:
    cal = FixedDiscount(factor=0.85)
    logits = np.array([[0.1, 0.5, 0.2]])
    raw = softmax(logits)
    calibrated = cal.calibrate(logits)
    assert np.argmax(raw, axis=-1)[0] == np.argmax(calibrated, axis=-1)[0]


def test_fixed_discount_rejects_invalid_factor() -> None:
    with pytest.raises(ValueError):
        FixedDiscount(factor=0.0)
    with pytest.raises(ValueError):
        FixedDiscount(factor=1.5)
    with pytest.raises(ValueError):
        FixedDiscount(factor=-0.1)


def test_temperature_scaling_identity_at_T_equal_1() -> None:
    cal = TemperatureScaling(T=1.0)
    logits = np.array([[1.0, 2.0, 0.5], [-1.0, 0.0, 1.0]])
    calibrated = cal.calibrate(logits)
    np.testing.assert_allclose(calibrated, softmax(logits), rtol=1e-6)


def test_temperature_scaling_flattens_with_large_T() -> None:
    """Large T -> near-uniform distribution, lower max-probability."""
    cal_hot = TemperatureScaling(T=10.0)
    cal_cool = TemperatureScaling(T=0.5)
    logits = np.array([[5.0, 1.0, 0.0]])
    hot = cal_hot.calibrate(logits)
    cool = cal_cool.calibrate(logits)
    # cool (T<1) sharpens; hot (T>1) flattens.
    assert hot.max() < softmax(logits).max() < cool.max()


def test_temperature_scaling_preserves_rank_within_row() -> None:
    cal = TemperatureScaling(T=2.0)
    logits = np.array([[0.5, 2.0, 1.0]])
    raw_order = np.argsort(-softmax(logits), axis=-1)
    cal_order = np.argsort(-cal.calibrate(logits), axis=-1)
    np.testing.assert_array_equal(raw_order, cal_order)


def test_temperature_scaling_fit_reduces_nll() -> None:
    """Fitting T to labeled data must improve NLL vs T=1.

    Construction: model is 60% accurate but outputs ~99% confidence on
    every prediction (classic overconfidence). Flattening (T > 1)
    brings the confidence down to match the real accuracy and lowers
    NLL.
    """
    rng = np.random.default_rng(42)
    n = 500
    labels = rng.integers(0, 3, size=n)
    logits = rng.normal(scale=0.1, size=(n, 3))
    for i, label in enumerate(labels):
        if rng.random() < 0.6:
            logits[i, label] += 5.0  # correct and confident
        else:
            wrong = rng.choice([c for c in range(3) if c != label])
            logits[i, wrong] += 5.0  # wrong but confident

    cal = TemperatureScaling()
    initial_nll = _nll(softmax(logits), labels)
    cal.fit(logits, labels)
    fitted_nll = _nll(cal.calibrate(logits), labels)
    assert fitted_nll < initial_nll
    assert cal.T > 1.0  # overconfidence -> flatten


def test_compute_ece_on_hand_constructed_reference() -> None:
    """Manually construct a set where we know the ECE analytically."""
    # All predictions confidence 1.0, all correct -> perfect calibration -> 0.
    probs = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    labels = np.array([1, 1, 1])
    ece = compute_ece(probs, labels, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-6)


def test_compute_ece_detects_miscalibration() -> None:
    """Predictions at 90% confidence that are only 50% accurate -> ECE 0.4."""
    n = 100
    probs = np.zeros((n, 2))
    probs[:, 1] = 0.9  # 90% confident class 1
    probs[:, 0] = 0.1
    labels = np.zeros(n, dtype=int)
    labels[: n // 2] = 1  # only half are correct
    ece = compute_ece(probs, labels, n_bins=10)
    # Confidence 0.9, accuracy 0.5 -> gap of 0.4.
    assert ece == pytest.approx(0.4, abs=0.01)


def test_compute_ece_handles_empty_bins() -> None:
    """Bins with zero samples must not produce NaN/Inf."""
    probs = np.array([[0.1, 0.9], [0.2, 0.8]])
    labels = np.array([1, 1])
    ece = compute_ece(probs, labels, n_bins=20)
    assert np.isfinite(ece)


def _nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Average negative log-likelihood for a labeled sample."""
    eps = 1e-12
    picked = probs[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(picked, eps, 1.0)).mean())
