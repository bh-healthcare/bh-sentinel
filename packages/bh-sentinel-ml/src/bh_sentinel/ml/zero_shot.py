"""Natural Language Inference zero-shot classifier over sentence boundaries.

Input: PreprocessedText (sentence boundaries with character offsets)
Output: list[PatternMatchCandidate] -- the same internal type the core
rules engine already accepts via RulesEngine.evaluate(l2_candidates=...).

Algorithm (architecture section 4.5, Layer 2):
  1. Build (sentence, hypothesis) pairs for every flag_id with a
     hypothesis, across every sentence.
  2. One batched transformer.infer() call.
  3. Extract entailment probability per pair via the calibrator.
  4. For each flag_id, pick the highest-scoring sentence.
  5. If the calibrated score >= min_emit_confidence, emit a
     PatternMatchCandidate with evidence_span covering the winning
     sentence's char offsets.

PHI safety:
- basis_description is a static template referencing only
  {flag_name, sentence_index, confidence} -- no raw input text.
- matched_context_hint is short and bounded (first ~80 chars of the
  winning sentence) to stay consistent with L1's behavior.
- On InferenceError, classify() propagates; the pipeline catches and
  maps to LayerStatus.FAILED.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from bh_sentinel.core._types import (
    PatternMatchCandidate,
    PreprocessedText,
    SentenceBoundary,
)
from bh_sentinel.core.taxonomy import FlagTaxonomy
from bh_sentinel.core.temporal_detector import TemporalDetector

from bh_sentinel.ml.calibration import softmax

_CONTEXT_HINT_MAX_LEN = 80

__all__ = ["FlagScore", "ZeroShotClassifier"]


@dataclass(frozen=True, slots=True)
class FlagScore:
    """Per-flag Layer 2 score diagnostic for threshold tuning and evaluation.

    ``raw_entailment`` is the softmax entailment probability before calibration.
    ``calibrated_score`` applies the configured calibrator (e.g. FixedDiscount).
    ``would_emit`` is True when ``calibrated_score >= min_emit_confidence``.
    """

    flag_id: str
    best_sentence_index: int
    raw_entailment: float
    calibrated_score: float
    min_emit_confidence: float
    calibration_discount: float
    would_emit: bool
    margin_to_emit: float


class ZeroShotClassifier:
    """Zero-shot NLI over the flag taxonomy.

    Args:
        transformer: any object with `.infer(premises, hypotheses) -> ndarray`.
            In production this is TransformerClassifier; tests use a stub.
        hypotheses: flag_id -> NLI hypothesis template mapping.
        taxonomy: loaded FlagTaxonomy (resolves flag_id -> domain, severity, name).
        calibrator: any Calibrator-compatible object.
        temporal: shared core TemporalDetector. Negation is intentionally
            not run at this layer -- the NLI model reasons about negation
            internally via entailment scoring.
        min_emit_confidence: calibrated-probability floor to emit a candidate.
        entailment_index: index in the logits array that represents
            "entailment". Default 2 matches `FacebookAI/roberta-large-mnli`'s
            class ordering (id2label = {0: CONTRADICTION, 1: NEUTRAL,
            2: ENTAILMENT}), which is the model pinned in `ml_config.yaml`
            for `bh-sentinel-ml >= 0.2.2`. The same default also works for
            other HuggingFace NLI heads in the BART/RoBERTa family (BART-
            large-MNLI uses identical class order). Override only if you
            pin a model whose head uses a different class order, e.g. the
            rejected `valhalla/distilbart-mnli-12-3` used entailment at
            index 0; shipping that source would require entailment_index=0.
    """

    def __init__(
        self,
        *,
        transformer: Any,
        hypotheses: dict[str, str],
        taxonomy: FlagTaxonomy,
        calibrator: Any,
        temporal: TemporalDetector,
        min_emit_confidence: float = 0.55,
        entailment_index: int = 2,
    ) -> None:
        self._transformer = transformer
        self._taxonomy = taxonomy
        self._calibrator = calibrator
        self._temporal = temporal
        self._min_conf = float(min_emit_confidence)
        self._entailment_index = int(entailment_index)

        # Keep only hypotheses whose flag_id exists in the taxonomy.
        self._hypotheses: dict[str, str] = {}
        for flag_id, hyp in hypotheses.items():
            if self._taxonomy.get_flag(flag_id) is not None:
                self._hypotheses[flag_id] = hyp

    def _calibration_discount(self) -> float:
        factor = getattr(self._calibrator, "_factor", None)
        if factor is not None:
            return float(factor)
        return 1.0

    def _infer_entailment_matrices(
        self, preprocessed: PreprocessedText
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[SentenceBoundary]] | None:
        sentences = preprocessed.sentences
        if not sentences or not self._hypotheses:
            return None

        flag_ids = list(self._hypotheses.keys())
        hyp_list = [self._hypotheses[f] for f in flag_ids]

        premises: list[str] = []
        hypotheses: list[str] = []
        for sent in sentences:
            for hyp in hyp_list:
                premises.append(sent.text)
                hypotheses.append(hyp)

        logits = self._transformer.infer(premises, hypotheses)
        logits_arr = np.asarray(logits, dtype=np.float32)
        raw_probs = softmax(logits_arr)
        calibrated_probs = self._calibrator.calibrate(logits_arr)

        n_sent = len(sentences)
        n_flag = len(flag_ids)
        raw_entail = raw_probs[:, self._entailment_index].reshape(n_sent, n_flag)
        calibrated_entail = calibrated_probs[:, self._entailment_index].reshape(n_sent, n_flag)
        return raw_entail, calibrated_entail, flag_ids, sentences

    def score_flags(self, preprocessed: PreprocessedText) -> list[FlagScore]:
        """Return per-flag best-sentence scores, including sub-threshold flags."""
        matrices = self._infer_entailment_matrices(preprocessed)
        if matrices is None:
            return []

        raw_entail, calibrated_entail, flag_ids, sentences = matrices
        discount = self._calibration_discount()
        scores: list[FlagScore] = []

        for flag_col, flag_id in enumerate(flag_ids):
            raw_column = raw_entail[:, flag_col]
            cal_column = calibrated_entail[:, flag_col]
            best_sent_idx = int(np.argmax(cal_column))
            raw_score = float(raw_column[best_sent_idx])
            calibrated_score = float(cal_column[best_sent_idx])
            would_emit = calibrated_score >= self._min_conf
            scores.append(
                FlagScore(
                    flag_id=flag_id,
                    best_sentence_index=sentences[best_sent_idx].index,
                    raw_entailment=raw_score,
                    calibrated_score=calibrated_score,
                    min_emit_confidence=self._min_conf,
                    calibration_discount=discount,
                    would_emit=would_emit,
                    margin_to_emit=calibrated_score - self._min_conf,
                )
            )

        return scores

    def classify(self, preprocessed: PreprocessedText) -> list[PatternMatchCandidate]:
        matrices = self._infer_entailment_matrices(preprocessed)
        if matrices is None:
            return []

        _, calibrated_entail, flag_ids, sentences = matrices
        candidates: list[PatternMatchCandidate] = []
        for flag_col, flag_id in enumerate(flag_ids):
            column = calibrated_entail[:, flag_col]
            best_sent_idx = int(np.argmax(column))
            best_score = float(column[best_sent_idx])
            if best_score < self._min_conf:
                continue
            cand = self._build_candidate(
                flag_id=flag_id,
                confidence=best_score,
                sentence=sentences[best_sent_idx],
            )
            if cand is not None:
                candidates.append(cand)

        return candidates

    def _build_candidate(
        self,
        *,
        flag_id: str,
        confidence: float,
        sentence: SentenceBoundary,
    ) -> PatternMatchCandidate | None:
        flag = self._taxonomy.get_flag(flag_id)
        if flag is None:
            return None

        domain = self._taxonomy.get_domain_for_flag(flag_id)
        if domain is None:
            return None

        temporal_ctx = self._temporal.classify(sentence.text, 0, len(sentence.text))

        hint_text = sentence.text.strip()
        if len(hint_text) > _CONTEXT_HINT_MAX_LEN:
            hint_text = hint_text[: _CONTEXT_HINT_MAX_LEN - 3] + "..."

        basis = (
            f"Transformer zero-shot classification identified "
            f"'{flag['name']}' in sentence {sentence.index} "
            f"(calibrated confidence {confidence:.2f})"
        )

        return PatternMatchCandidate(
            flag_id=flag_id,
            domain=domain,
            name=flag["name"],
            default_severity=flag["default_severity"],
            confidence=float(confidence),
            sentence_index=sentence.index,
            char_start=sentence.char_start,
            char_end=sentence.char_end,
            pattern_text="",  # Layer 2 has no regex pattern
            basis_description=basis,
            matched_context_hint=hint_text,
            negated=False,  # NLI handles negation internally
            temporal_context=temporal_ctx,
        )
