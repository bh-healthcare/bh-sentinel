"""Tests for TransformerClassifier.infer() batched NLI inference.

Uses the tiny_nli_model fixture (deterministic, checked into fixtures/)
so these tests run without any network access, without torch, and
without a pre-trained model. Tokenization is stubbed via a toy
tokenizer that produces deterministic token ids from string length.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bh_sentinel.ml.exceptions import InferenceError
from bh_sentinel.ml.transformer import TransformerClassifier


class _ToyEncoding:
    def __init__(self, length: int) -> None:
        # Deterministic: token at position i is i+1; mask is all ones.
        self.ids = list(range(1, length + 1))
        self.attention_mask = [1] * length


class _ToyTokenizer:
    """Deterministic stand-in for tokenizers.Tokenizer.

    The mask sum determines the model's output logits, so we vary
    mask length with premise length to produce distinguishable outputs.
    """

    def encode(self, premise: str, hypothesis: str):
        return _ToyEncoding(len(premise) + len(hypothesis) + 2)


def _make_classifier(onnx_path: Path, sha: str) -> TransformerClassifier:
    """Build a TransformerClassifier against the tiny ONNX fixture."""
    clf = TransformerClassifier(
        model_path=onnx_path,
        tokenizer_path=onnx_path,  # not loaded when skip_tokenizer_load=True
        expected_sha256=sha,
        max_length=64,
        max_batch_size=8,
        skip_tokenizer_load=True,
    )
    clf._tokenizer = _ToyTokenizer()
    return clf


def test_infer_returns_shape_n_by_3(tiny_nli_model: Path, tiny_nli_sha256: str) -> None:
    clf = _make_classifier(tiny_nli_model, tiny_nli_sha256)
    premises = ["I feel hopeless.", "I am looking forward to tomorrow."]
    hypotheses = [
        "The speaker expresses hopelessness.",
        "The speaker expresses future orientation.",
    ]
    logits = clf.infer(premises, hypotheses)
    assert logits.shape == (2, 3)
    assert logits.dtype == np.float32


def test_infer_empty_input_returns_empty_array(tiny_nli_model: Path, tiny_nli_sha256: str) -> None:
    clf = _make_classifier(tiny_nli_model, tiny_nli_sha256)
    logits = clf.infer([], [])
    assert logits.shape == (0, 3)


def test_infer_splits_oversize_batches(tiny_nli_model: Path, tiny_nli_sha256: str) -> None:
    """Batches larger than max_batch_size are split into sub-batches
    and their results concatenated. 8 is the fixture's max_batch_size;
    pushing 19 pairs through forces 3 sub-batches (8 + 8 + 3)."""
    clf = _make_classifier(tiny_nli_model, tiny_nli_sha256)
    n = 19
    premises = [f"sentence {i}" for i in range(n)]
    hypotheses = [f"hypothesis {i}" for i in range(n)]
    logits = clf.infer(premises, hypotheses)
    assert logits.shape == (n, 3)


def test_infer_length_mismatch_raises(tiny_nli_model: Path, tiny_nli_sha256: str) -> None:
    clf = _make_classifier(tiny_nli_model, tiny_nli_sha256)
    with pytest.raises(InferenceError):
        clf.infer(["a", "b"], ["only one hypothesis"])


def test_infer_is_deterministic(tiny_nli_model: Path, tiny_nli_sha256: str) -> None:
    clf = _make_classifier(tiny_nli_model, tiny_nli_sha256)
    premises = ["The patient feels isolated."]
    hypotheses = ["The speaker describes isolation."]
    a = clf.infer(premises, hypotheses)
    b = clf.infer(premises, hypotheses)
    np.testing.assert_array_equal(a, b)


def test_infer_wraps_ort_errors_as_inference_error(
    tiny_nli_model: Path, tiny_nli_sha256: str
) -> None:
    """Any error from the ORT session is wrapped as InferenceError.
    No input text leaks to the exception message."""
    clf = _make_classifier(tiny_nli_model, tiny_nli_sha256)

    class _ExplodingSession:
        def run(self, *args, **kwargs):
            raise RuntimeError("simulated ORT failure")

    clf._session = _ExplodingSession()
    with pytest.raises(InferenceError) as exc:
        clf.infer(["premise"], ["hypothesis"])
    assert "premise" not in str(exc.value)
    assert "hypothesis" not in str(exc.value)
