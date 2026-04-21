"""Layer 1 / Layer 2 candidate merge per architecture section 4.7.

Rules (single source of truth here; architecture doc should match):

1. One flag per flag_id in the merged output (dedupe within flag_id).
2. confidence = max(l1.confidence, l2.confidence). Never averaged --
   the stronger signal represents the better detection.
3. When both layers detect the same flag, L1's precise char-level
   evidence_span is preferred over L2's sentence-level span.
4. When only L2 detects, L2's sentence-level span is used.
5. matched_context_hint follows the winning (higher-confidence) layer's
   hint when both detect; falls back to the sole detector's hint
   otherwise.
6. basis_description starts from the winning layer's description. When
   both layers detect, a short static corroboration note is appended
   naming the other layer and its confidence. No raw input text is
   inserted -- the note is a fixed template.
7. Temporal context: present wins over past when the layers disagree.
   Both past -> past. Both present -> present.
8. Severity (default_severity) is identical across layers because both
   pull from flag_taxonomy.json. No merge needed.
9. corroborating_layers is returned as a side channel (flag_id -> list
   of non-primary layer names). The pipeline hydrates this onto the
   final Flag objects after the rules engine runs.

The merge operates entirely on PatternMatchCandidate. Intra-layer
duplicates (for example two pattern matches of the same flag_id in
different sentences) are preserved -- collapsing them is the
PatternMatcher's job, not the merger's.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bh_sentinel.core._types import PatternMatchCandidate

__all__ = ["MergeResult", "merge_candidates"]


@dataclass(slots=True)
class MergeResult:
    """Output of merge_candidates.

    candidates: the merged PatternMatchCandidate list to hand to the
        rules engine.
    corroborating_layers: flag_id -> list of non-primary DetectionLayer
        names that also detected this flag. Populated only when a flag
        was detected by both L1 and L2. The pipeline copies this onto
        Flag.corroborating_layers after rules engine hydration.
    """

    candidates: list[PatternMatchCandidate] = field(default_factory=list)
    corroborating_layers: dict[str, list[str]] = field(default_factory=dict)


def merge_candidates(
    l1_candidates: list[PatternMatchCandidate],
    l2_candidates: list[PatternMatchCandidate],
) -> MergeResult:
    """Merge L1 (pattern_match) and L2 (transformer) candidates.

    See module docstring for the full rule list. Intra-layer dupes are
    preserved; cross-layer dupes for the same flag_id are merged into
    one candidate following architecture 4.7.
    """
    # Keep the single best L2 candidate per flag_id so we can look up by
    # flag_id when merging. Intra-layer dupes within L2 are unexpected
    # (one flag, one sentence winner), but just in case, take the max.
    l2_by_flag: dict[str, PatternMatchCandidate] = {}
    for cand in l2_candidates:
        existing = l2_by_flag.get(cand.flag_id)
        if existing is None or cand.confidence > existing.confidence:
            l2_by_flag[cand.flag_id] = cand

    merged: list[PatternMatchCandidate] = []
    corroborated: dict[str, list[str]] = {}
    flag_ids_with_l1: set[str] = set()

    for l1 in l1_candidates:
        l2 = l2_by_flag.get(l1.flag_id)
        if l2 is None:
            merged.append(l1)
        else:
            merged_cand = _merge_pair(l1, l2)
            merged.append(merged_cand)
            # Non-primary layer is the one that did NOT win confidence.
            if l1.confidence >= l2.confidence:
                corroborated[l1.flag_id] = ["transformer"]
            else:
                corroborated[l1.flag_id] = ["pattern_match"]
        flag_ids_with_l1.add(l1.flag_id)

    # Keep L2-only flags (those not in flag_ids_with_l1).
    for flag_id, cand in l2_by_flag.items():
        if flag_id not in flag_ids_with_l1:
            merged.append(cand)

    return MergeResult(candidates=merged, corroborating_layers=corroborated)


def _merge_pair(l1: PatternMatchCandidate, l2: PatternMatchCandidate) -> PatternMatchCandidate:
    """Merge an L1 and L2 candidate for the same flag_id.

    L1's evidence span is always kept (it's char-level precise). The
    winning layer's hint and basis-description lead; the other layer's
    confidence is mentioned in the corroboration note.
    """
    l1_wins = l1.confidence >= l2.confidence
    confidence = max(l1.confidence, l2.confidence)

    # Winning layer drives hint + basis.
    if l1_wins:
        winner_basis = l1.basis_description
        winner_hint = l1.matched_context_hint
        loser_layer_name = "transformer"
        loser_confidence = l2.confidence
    else:
        winner_basis = l2.basis_description
        winner_hint = l2.matched_context_hint
        loser_layer_name = "pattern_match"
        loser_confidence = l1.confidence

    # Static corroboration note -- no PHI.
    corroboration = (
        f"; also corroborated by {loser_layer_name} layer at confidence {loser_confidence:.2f}"
    )
    merged_basis = winner_basis + corroboration

    # Temporal: present wins when layers disagree.
    if l1.temporal_context == l2.temporal_context:
        temporal = l1.temporal_context
    elif "present" in (l1.temporal_context, l2.temporal_context):
        temporal = "present"
    else:
        temporal = l1.temporal_context  # default to L1 when neither is present

    return PatternMatchCandidate(
        flag_id=l1.flag_id,
        domain=l1.domain,
        name=l1.name,
        default_severity=l1.default_severity,
        confidence=confidence,
        sentence_index=(l1 if l1_wins else l2).sentence_index,
        # L1's char-level span is ALWAYS preferred when both detect.
        char_start=l1.char_start,
        char_end=l1.char_end,
        pattern_text=l1.pattern_text,
        basis_description=merged_basis,
        matched_context_hint=winner_hint,
        negated=l1.negated,  # L1 negation wins; L2 never sets negated.
        temporal_context=temporal,
    )
