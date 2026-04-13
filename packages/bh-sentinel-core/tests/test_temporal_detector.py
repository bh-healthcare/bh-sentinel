"""Tests for TemporalDetector -- written before implementation (TDD)."""

from __future__ import annotations

from bh_sentinel.core.temporal_detector import TemporalDetector


def td() -> TemporalDetector:
    return TemporalDetector()


class TestPastContext:
    def test_history_of(self):
        text = "History of suicide attempt."
        assert td().classify(text, 11, 26) == "past"

    def test_hx_of(self):
        text = "Hx of SI."
        assert td().classify(text, 6, 8) == "past"

    def test_prior(self):
        text = "Prior suicide attempt in records."
        assert td().classify(text, 6, 21) == "past"

    def test_year_reference(self):
        text = "Attempt in 2019."
        assert td().classify(text, 0, 7) == "past"

    def test_used_to(self):
        text = "Used to cut herself."
        assert td().classify(text, 8, 11) == "past"

    def test_as_a_teenager(self):
        text = "As a teenager, attempted suicide."
        assert td().classify(text, 15, 31) == "past"

    def test_resolved(self):
        text = "SI resolved after treatment."
        assert td().classify(text, 0, 2) == "past"


class TestPresentContext:
    def test_currently(self):
        text = "Currently endorses SI."
        assert td().classify(text, 19, 21) == "present"

    def test_recently(self):
        text = "Recently started cutting."
        assert td().classify(text, 17, 24) == "present"

    def test_again(self):
        text = "Started using again."
        assert td().classify(text, 8, 13) == "present"


class TestPresentOverridesPast:
    def test_both_markers_present_wins(self):
        text = "History of attempt, but currently endorses SI."
        si_start = text.index("SI")
        assert td().classify(text, si_start, si_start + 2) == "present"


class TestDefaultPresent:
    def test_no_markers(self):
        text = "Patient reports SI."
        assert td().classify(text, 16, 18) == "present"


class TestBoundaryConditions:
    def test_empty_text(self):
        assert td().classify("", 0, 0) == "present"

    def test_match_at_position_zero(self):
        text = "SI endorsed."
        assert td().classify(text, 0, 2) == "present"


class TestFutureTense:
    def test_plans_to_attempt(self):
        text = "Plans to attempt suicide."
        assert td().classify(text, 9, 24) == "present"
