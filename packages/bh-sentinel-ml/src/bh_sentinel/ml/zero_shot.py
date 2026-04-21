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

from typing import Any

import numpy as np
from bh_sentinel.core._types import (
    PatternMatchCandidate,
    PreprocessedText,
    SentenceBoundary,
)
from bh_sentinel.core.taxonomy import FlagTaxonomy
from bh_sentinel.core.temporal_detector import TemporalDetector

_CONTEXT_HINT_MAX_LEN = 80


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
            "entailment" (default 0 matches DistilBART-MNLI).
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
        entailment_index: int = 0,
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

    def classify(self, preprocessed: PreprocessedText) -> list[PatternMatchCandidate]:
        sentences = preprocessed.sentences
        if not sentences or not self._hypotheses:
            return []

        flag_ids = list(self._hypotheses.keys())
        hyp_list = [self._hypotheses[f] for f in flag_ids]

        premises: list[str] = []
        hypotheses: list[str] = []
        for sent in sentences:
            for hyp in hyp_list:
                premises.append(sent.text)
                hypotheses.append(hyp)

        logits = self._transformer.infer(premises, hypotheses)
        probs = self._calibrator.calibrate(np.asarray(logits))

        n_sent = len(sentences)
        n_flag = len(flag_ids)
        # Reshape from (n_sent * n_flag, C) back to per-sentence-per-flag.
        entail = probs[:, self._entailment_index].reshape(n_sent, n_flag)

        candidates: list[PatternMatchCandidate] = []
        for flag_col, flag_id in enumerate(flag_ids):
            column = entail[:, flag_col]
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
