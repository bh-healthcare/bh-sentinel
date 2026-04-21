"""ONNX Runtime transformer inference for clinical text classification.

The runtime wrapper pays the ONNX session load cost once at construction,
then runs batched NLI inference per request. SHA256 verify-on-load is the
integrity guardrail: if the ONNX bytes do not match the pinned digest in
ml_config.yaml, no InferenceSession is created and ModelIntegrityError
bubbles up immediately.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from bh_sentinel.ml.exceptions import (
    InferenceError,
    ModelIntegrityError,
    ModelNotFoundError,  # re-exported for the package's public surface
)

if TYPE_CHECKING:
    pass


__all__ = [
    "TransformerClassifier",
    "ModelIntegrityError",
    "ModelNotFoundError",
    "InferenceError",
]


class TransformerClassifier:
    """ONNX-quantized transformer model running entirely in-process.

    Constructed once per pipeline instance. One InferenceSession, one
    tokenizer. Batched NLI inference: pair sentences with hypothesis
    templates and run them as a single forward pass where possible.

    Integrity guardrail: the ONNX file's SHA256 must match
    expected_sha256 (from ml_config.yaml). On mismatch, no session is
    constructed -- the constructor raises ModelIntegrityError. This
    catches stale container bakes, corrupted downloads, and tampered
    model files before they can serve any predictions.
    """

    def __init__(
        self,
        *,
        model_path: Path,
        tokenizer_path: Path,
        expected_sha256: str,
        max_length: int = 256,
        max_batch_size: int = 32,
        session_options: Any | None = None,
        skip_tokenizer_load: bool = False,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise ModelIntegrityError(f"ONNX model file does not exist: {model_path}")

        actual_sha256 = _sha256_of_file(model_path)
        if actual_sha256.lower() != expected_sha256.lower():
            raise ModelIntegrityError(
                "ONNX model SHA256 does not match the pinned digest. "
                "The model bytes in this environment diverge from the "
                "version that was validated for this release. "
                "Rebuild the container image or re-run the download CLI "
                "with --verify-sha256."
            )

        self._max_length = max_length
        self._max_batch_size = max_batch_size
        self._session = self._build_session(model_path, session_options)
        self._tokenizer = None if skip_tokenizer_load else self._build_tokenizer(tokenizer_path)

    @staticmethod
    def _build_session(model_path: Path, session_options: Any | None) -> Any:
        return ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

    @staticmethod
    def _build_tokenizer(tokenizer_path: Path) -> Any:
        return Tokenizer.from_file(str(tokenizer_path))

    def infer(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        """Run batched NLI inference.

        Returns logits of shape (N, 3) for N (premise, hypothesis) pairs,
        ordered (entailment, neutral, contradiction).

        Splits into sub-batches of at most max_batch_size pairs to bound
        peak memory on long documents.

        Raises:
            InferenceError: wraps any ONNX Runtime exception. Callers
                convert this to LayerStatus.FAILED without propagating
                the underlying exception or any input text.
        """
        if len(premises) != len(hypotheses):
            raise InferenceError("premises and hypotheses must be the same length")
        if not premises:
            return np.empty((0, 3), dtype=float)

        if self._tokenizer is None:
            raise InferenceError("tokenizer was not loaded; inference is not available")

        outputs: list[np.ndarray] = []
        n = len(premises)
        for start in range(0, n, self._max_batch_size):
            end = min(start + self._max_batch_size, n)
            try:
                chunk = self._infer_batch(premises[start:end], hypotheses[start:end])
            except Exception as exc:
                raise InferenceError("ONNX Runtime inference failed") from exc
            outputs.append(chunk)
        return np.concatenate(outputs, axis=0)

    def _infer_batch(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        """One ONNX forward pass for a sub-batch.

        Tokenizes each (premise, hypothesis) pair, pads to a uniform
        length, runs the session, and returns the classification logits.
        """
        encodings = [
            self._tokenizer.encode(p, h) for p, h in zip(premises, hypotheses, strict=True)
        ]
        ids = _pad([e.ids for e in encodings], self._max_length)
        mask = _pad([e.attention_mask for e in encodings], self._max_length)

        feed = {
            "input_ids": np.asarray(ids, dtype=np.int64),
            "attention_mask": np.asarray(mask, dtype=np.int64),
        }
        outputs = self._session.run(None, feed)
        logits = outputs[0]
        if logits.ndim != 2 or logits.shape[1] != 3:
            raise InferenceError(f"expected (N, 3) logits, got shape {logits.shape}")
        return logits


def _pad(seqs: list[list[int]], max_length: int) -> list[list[int]]:
    """Pad or truncate each sequence to exactly max_length with zero padding."""
    out: list[list[int]] = []
    for s in seqs:
        if len(s) >= max_length:
            out.append(list(s[:max_length]))
        else:
            out.append(list(s) + [0] * (max_length - len(s)))
    return out


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
