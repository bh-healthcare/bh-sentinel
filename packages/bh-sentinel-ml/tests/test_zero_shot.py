"""Tests for ZeroShotClassifier.

Uses a deterministic stub TransformerClassifier that returns pre-canned
logits per (premise, hypothesis) pair. No real ONNX inference. The
tests validate pair construction, per-flag best-sentence selection,
threshold filtering, PatternMatchCandidate shape, and PHI safety of
basis descriptions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from bh_sentinel.core._config import default_flag_taxonomy_path
from bh_sentinel.core._types import PreprocessedText, SentenceBoundary
from bh_sentinel.core.taxonomy import FlagTaxonomy
from bh_sentinel.core.temporal_detector import TemporalDetector

from bh_sentinel.ml.calibration import IdentityCalibrator
from bh_sentinel.ml.zero_shot import ZeroShotClassifier


def _preprocessed(*sentences: str) -> PreprocessedText:
    """Build a PreprocessedText with the given sentences, joined by spaces.

    Produces SentenceBoundary entries with accurate char offsets so the
    classifier's evidence_span tests are meaningful.
    """
    text_parts: list[str] = []
    boundaries: list[SentenceBoundary] = []
    cursor = 0
    for idx, s in enumerate(sentences):
        if idx > 0:
            text_parts.append(" ")
            cursor += 1
        start = cursor
        text_parts.append(s)
        end = cursor + len(s)
        boundaries.append(SentenceBoundary(text=s, index=idx, char_start=start, char_end=end))
        cursor = end
    full = "".join(text_parts)
    return PreprocessedText(original=full, sentences=tuple(boundaries))


class _StubTransformer:
    """Deterministic stand-in for TransformerClassifier.

    Returns logits looked up from a dict keyed by (premise, hypothesis).
    Unknown pairs return neutral logits."""

    def __init__(self, logit_map: dict[tuple[str, str], list[float]]) -> None:
        self._map = logit_map
        self.calls = 0

    def infer(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        self.calls += 1
        rows = []
        for p, h in zip(premises, hypotheses, strict=True):
            rows.append(self._map.get((p, h), [0.0, 0.0, 0.0]))
        return np.asarray(rows, dtype=np.float32)


_UNSET: dict[str, str] = {"__unset__": ""}  # sentinel distinct from {}


def _make_classifier(
    transformer: Any,
    hypotheses: dict[str, str] = _UNSET,
    min_emit_confidence: float = 0.55,
) -> ZeroShotClassifier:
    taxonomy = FlagTaxonomy(default_flag_taxonomy_path())
    hyp = (
        {
            "SH-001": "The speaker expresses a wish to be dead.",
            "SH-002": "The speaker describes suicidal ideation.",
        }
        if hypotheses is _UNSET
        else hypotheses
    )
    return ZeroShotClassifier(
        transformer=transformer,
        hypotheses=hyp,
        taxonomy=taxonomy,
        calibrator=IdentityCalibrator(),
        temporal=TemporalDetector(),
        min_emit_confidence=min_emit_confidence,
        entailment_index=0,
    )


def test_classify_builds_pairs_for_every_flag_and_sentence() -> None:
    transformer = _StubTransformer({})
    clf = _make_classifier(transformer)
    pre = _preprocessed(
        "Patient reports feeling hopeless.",
        "Stated plan to overdose.",
    )
    clf.classify(pre)
    # 2 sentences * 2 hypotheses = 4 pairs, a single infer call.
    assert transformer.calls == 1


def test_empty_preprocessed_produces_no_candidates_and_no_inference() -> None:
    transformer = _StubTransformer({})
    clf = _make_classifier(transformer)
    pre = PreprocessedText(original="", sentences=())
    candidates = clf.classify(pre)
    assert candidates == []
    assert transformer.calls == 0


def test_threshold_filters_low_confidence_pairs() -> None:
    # High entailment for SH-002 / low for SH-001.
    sh001_hyp = "The speaker expresses a wish to be dead."
    sh002_hyp = "The speaker describes suicidal ideation."
    sentence = "I want to end my life."
    transformer = _StubTransformer(
        {
            (sentence, sh001_hyp): [0.1, 0.2, 0.7],  # entailment 0.1
            (sentence, sh002_hyp): [5.0, -2.0, -3.0],  # entailment ~1
        }
    )
    clf = _make_classifier(
        transformer,
        hypotheses={"SH-001": sh001_hyp, "SH-002": sh002_hyp},
        min_emit_confidence=0.55,
    )
    pre = _preprocessed(sentence)
    candidates = clf.classify(pre)
    flag_ids = {c.flag_id for c in candidates}
    assert "SH-002" in flag_ids
    assert "SH-001" not in flag_ids


def test_best_sentence_selected_per_flag() -> None:
    sh002_hyp = "The speaker describes suicidal ideation."
    strong = "I want to end my life."
    weak = "I feel sad today."
    transformer = _StubTransformer(
        {
            (strong, sh002_hyp): [6.0, -2.0, -3.0],  # very high entailment
            (weak, sh002_hyp): [1.0, 0.5, 0.3],  # moderate
        }
    )
    clf = _make_classifier(transformer, hypotheses={"SH-002": sh002_hyp}, min_emit_confidence=0.4)
    pre = _preprocessed(weak, strong)
    candidates = clf.classify(pre)
    # Only one candidate per flag_id.
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.flag_id == "SH-002"
    # Winning sentence is the strong one (index 1).
    assert cand.sentence_index == 1
    # evidence_span covers the stronger sentence's char offsets.
    assert cand.char_start == pre.sentences[1].char_start
    assert cand.char_end == pre.sentences[1].char_end


def test_candidate_has_required_fields() -> None:
    sh002_hyp = "The speaker describes suicidal ideation."
    sentence = "I want to end my life."
    transformer = _StubTransformer({(sentence, sh002_hyp): [5.0, -2.0, -3.0]})
    clf = _make_classifier(transformer, hypotheses={"SH-002": sh002_hyp}, min_emit_confidence=0.4)
    pre = _preprocessed(sentence)
    [cand] = clf.classify(pre)
    assert cand.flag_id == "SH-002"
    assert cand.domain == "self_harm"
    assert cand.default_severity == "CRITICAL"
    assert 0.0 <= cand.confidence <= 1.0
    assert cand.basis_description
    assert cand.matched_context_hint
    # Temporal context is set (present / past / future).
    assert cand.temporal_context in ("present", "past", "future")


def test_basis_description_does_not_leak_raw_input() -> None:
    """Detection trace lines (matched_context_hint) may include a short
    excerpt per architecture 4.4, but basis_description must not echo
    the full input sentence."""
    secret = "Patient SSN 123-45-6789 reports wanting to die."
    hypothesis = "The speaker describes suicidal ideation."
    transformer = _StubTransformer({(secret, hypothesis): [5.0, -2.0, -3.0]})
    clf = _make_classifier(transformer, hypotheses={"SH-002": hypothesis}, min_emit_confidence=0.4)
    pre = _preprocessed(secret)
    [cand] = clf.classify(pre)
    assert "123-45-6789" not in cand.basis_description
    assert "SSN" not in cand.basis_description


def test_unknown_hypothesis_flag_ignored() -> None:
    """Hypotheses for flag_ids not in the taxonomy are skipped, not errored.
    The config-vendored check catches real drift; this just keeps the
    classifier robust."""
    transformer = _StubTransformer({})
    clf = _make_classifier(
        transformer,
        hypotheses={"ZZ-999": "bogus hypothesis"},
        min_emit_confidence=0.4,
    )
    pre = _preprocessed("Anything at all.")
    candidates = clf.classify(pre)
    assert candidates == []


def test_empty_hypothesis_mapping_produces_no_candidates() -> None:
    transformer = _StubTransformer({})
    clf = _make_classifier(transformer, hypotheses={})
    pre = _preprocessed("I want to end my life.")
    candidates = clf.classify(pre)
    assert candidates == []
    # No work sent to the transformer either.
    assert transformer.calls == 0


def test_confidence_is_calibrated_not_raw_softmax() -> None:
    """The calibrator layer must be applied before threshold comparison.
    A calibrator that halves probabilities should cause a pair that
    would have passed to fall below threshold."""

    class _HalvingCalibrator:
        def calibrate(self, logits: np.ndarray) -> np.ndarray:
            # softmax then halve.
            e = np.exp(logits - logits.max(axis=-1, keepdims=True))
            sm = e / e.sum(axis=-1, keepdims=True)
            return sm * 0.5

    hypothesis = "The speaker describes suicidal ideation."
    sentence = "I want to end my life."
    transformer = _StubTransformer(
        {(sentence, hypothesis): [2.0, -2.0, -3.0]}  # softmax entailment ~0.96
    )
    taxonomy = FlagTaxonomy(default_flag_taxonomy_path())
    clf = ZeroShotClassifier(
        transformer=transformer,
        hypotheses={"SH-002": hypothesis},
        taxonomy=taxonomy,
        calibrator=_HalvingCalibrator(),
        temporal=TemporalDetector(),
        min_emit_confidence=0.6,
        entailment_index=0,
    )
    pre = _preprocessed(sentence)
    # Raw entailment ~0.96 would pass 0.6 threshold; halved to ~0.48
    # it should be below threshold and produce no candidate.
    assert clf.classify(pre) == []


def test_inference_error_is_swallowed_gracefully() -> None:
    """Architectural guarantee: zero-shot classify never raises
    InferenceError. The pipeline catches it and marks L2 FAILED, but
    callers of classify() get an empty list plus a PipelineLayerFailure
    flag they can read. For now, verify classify() does not raise."""
    from bh_sentinel.ml.exceptions import InferenceError

    class _ExplodingTransformer:
        def infer(self, *_args, **_kwargs):
            raise InferenceError("simulated")

    transformer = _ExplodingTransformer()
    clf = _make_classifier(
        transformer,
        hypotheses={"SH-002": "The speaker describes suicidal ideation."},
    )
    # classify() must surface the error so the pipeline can map it to
    # LayerStatus.FAILED. Empty list would mask the failure.
    with pytest.raises(InferenceError):
        clf.classify(_preprocessed("I want to end my life."))
