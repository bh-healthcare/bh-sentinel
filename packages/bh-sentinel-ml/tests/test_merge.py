"""Tests for the L1/L2 merge algorithm (architecture section 4.7)."""

from __future__ import annotations

from bh_sentinel.core._types import PatternMatchCandidate

from bh_sentinel.ml.merge import MergeResult, merge_candidates


def _c(
    flag_id: str,
    *,
    confidence: float,
    char_start: int,
    char_end: int,
    basis: str = "L1 pattern match",
    hint: str = "hint",
    temporal: str = "present",
    sentence_index: int = 0,
    domain: str = "self_harm",
    name: str = "flag name",
    severity: str = "HIGH",
    pattern: str = "regex",
) -> PatternMatchCandidate:
    return PatternMatchCandidate(
        flag_id=flag_id,
        domain=domain,
        name=name,
        default_severity=severity,
        confidence=confidence,
        sentence_index=sentence_index,
        char_start=char_start,
        char_end=char_end,
        pattern_text=pattern,
        basis_description=basis,
        matched_context_hint=hint,
        temporal_context=temporal,
    )


def test_l1_only_flag_passes_through_unchanged() -> None:
    l1 = [_c("SH-002", confidence=0.9, char_start=5, char_end=15)]
    result = merge_candidates(l1, [])
    assert isinstance(result, MergeResult)
    assert len(result.candidates) == 1
    merged = result.candidates[0]
    assert merged.flag_id == "SH-002"
    assert merged.confidence == 0.9
    # evidence span unchanged
    assert (merged.char_start, merged.char_end) == (5, 15)
    # no corroboration
    assert result.corroborating_layers == {}


def test_l2_only_flag_passes_through_unchanged() -> None:
    l2 = [
        _c(
            "SH-002",
            confidence=0.82,
            char_start=0,
            char_end=20,
            basis="L2 transformer",
        )
    ]
    result = merge_candidates([], l2)
    assert len(result.candidates) == 1
    assert result.candidates[0].confidence == 0.82
    assert result.corroborating_layers == {}


def test_both_detect_l1_wins_on_confidence() -> None:
    """L1 0.95, L2 0.70 -> L1 wins confidence, L1 span, L2 corroborates."""
    l1 = [
        _c(
            "SH-002",
            confidence=0.95,
            char_start=10,
            char_end=18,  # precise char span
            basis="L1: matched pattern foo",
            hint="L1 hint",
        )
    ]
    l2 = [
        _c(
            "SH-002",
            confidence=0.70,
            char_start=0,
            char_end=50,  # coarse sentence span
            basis="L2: classified sentence",
            hint="L2 hint",
        )
    ]
    result = merge_candidates(l1, l2)
    assert len(result.candidates) == 1
    m = result.candidates[0]
    # max(L1, L2)
    assert m.confidence == 0.95
    # L1's precise char span preferred
    assert (m.char_start, m.char_end) == (10, 18)
    # L1's basis_description with L2 corroboration note appended
    assert "L1: matched pattern foo" in m.basis_description
    assert "corroborated" in m.basis_description.lower() or "also" in m.basis_description.lower()
    # L1's hint wins
    assert m.matched_context_hint == "L1 hint"
    # Corroboration metadata
    assert result.corroborating_layers["SH-002"] == ["transformer"]


def test_both_detect_l2_wins_on_confidence() -> None:
    l1 = [
        _c(
            "SH-002",
            confidence=0.60,
            char_start=5,
            char_end=10,
            basis="L1 basis",
        )
    ]
    l2 = [
        _c(
            "SH-002",
            confidence=0.90,
            char_start=0,
            char_end=30,
            basis="L2 basis",
        )
    ]
    result = merge_candidates(l1, l2)
    m = result.candidates[0]
    assert m.confidence == 0.90
    # Architecture 4.7: "If only L2 detected the flag, use L2's
    # sentence-level span." But when both detect, L1's precise span is
    # always preferred, regardless of confidence winner.
    assert (m.char_start, m.char_end) == (5, 10)
    assert "L2 basis" in m.basis_description
    assert result.corroborating_layers["SH-002"] == ["pattern_match"]


def test_temporal_disagreement_present_wins() -> None:
    """L1 past, L2 present -> merged candidate is present."""
    l1 = [
        _c(
            "SH-002",
            confidence=0.90,
            char_start=5,
            char_end=10,
            temporal="past",
        )
    ]
    l2 = [
        _c(
            "SH-002",
            confidence=0.80,
            char_start=0,
            char_end=20,
            temporal="present",
        )
    ]
    result = merge_candidates(l1, l2)
    assert result.candidates[0].temporal_context == "present"


def test_temporal_both_past_stays_past() -> None:
    l1 = [_c("SH-002", confidence=0.9, char_start=0, char_end=5, temporal="past")]
    l2 = [_c("SH-002", confidence=0.8, char_start=0, char_end=5, temporal="past")]
    result = merge_candidates(l1, l2)
    assert result.candidates[0].temporal_context == "past"


def test_disjoint_flags_preserved() -> None:
    """Multiple different flag_ids, one per layer -> both survive merge."""
    l1 = [_c("SH-002", confidence=0.9, char_start=0, char_end=5)]
    l2 = [_c("CD-001", confidence=0.8, char_start=10, char_end=20)]
    result = merge_candidates(l1, l2)
    ids = {c.flag_id for c in result.candidates}
    assert ids == {"SH-002", "CD-001"}
    assert result.corroborating_layers == {}


def test_duplicate_within_single_layer_kept_as_is() -> None:
    """If L1 emits SH-002 twice (different sentences), merge doesn't collapse
    intra-layer duplicates -- that's the core PatternMatcher's job."""
    l1 = [
        _c("SH-002", confidence=0.9, char_start=0, char_end=5),
        _c("SH-002", confidence=0.85, char_start=30, char_end=40),
    ]
    result = merge_candidates(l1, [])
    assert len(result.candidates) == 2


def test_corroboration_note_does_not_leak_raw_input() -> None:
    """PHI-safe: the corroboration note is a static template -- no text
    from the other layer's hint or basis leaks into the message."""
    l1 = [
        _c(
            "SH-002",
            confidence=0.95,
            char_start=0,
            char_end=5,
            basis="L1 basis clean",
            hint="patient SSN 123-45-6789",
        )
    ]
    l2 = [
        _c(
            "SH-002",
            confidence=0.70,
            char_start=0,
            char_end=5,
            hint="more PHI-ish content",
        )
    ]
    result = merge_candidates(l1, l2)
    m = result.candidates[0]
    assert "123-45-6789" not in m.basis_description
    assert "PHI-ish" not in m.basis_description
