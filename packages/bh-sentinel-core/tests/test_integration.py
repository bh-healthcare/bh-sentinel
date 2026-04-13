"""End-to-end pipeline integration tests."""

from __future__ import annotations

from bh_sentinel.core.models.flags import Severity
from bh_sentinel.core.models.response import AnalysisResponse
from bh_sentinel.core.pipeline import Pipeline


def _pipeline() -> Pipeline:
    return Pipeline()


class TestEndToEnd:
    def test_crisis_note_escalation(self):
        text = (
            "Patient reports suicidal ideation and has been "
            "drinking daily. Pt reports hopelessness."
        )
        result = _pipeline().analyze_sync(text)
        assert isinstance(result, AnalysisResponse)
        assert len(result.flags) > 0
        assert result.summary.requires_immediate_review is True

    def test_routine_note_no_flags(self):
        text = "Patient is doing well today. Mood is euthymic. Sleep is adequate."
        result = _pipeline().analyze_sync(text)
        assert isinstance(result, AnalysisResponse)
        assert len(result.flags) == 0

    def test_temporal_deescalation(self):
        text = "History of suicide attempt in 2019."
        result = _pipeline().analyze_sync(text)
        assert isinstance(result, AnalysisResponse)
        sh008 = [f for f in result.flags if f.flag_id == "SH-008"]
        if sh008:
            assert sh008[0].severity == Severity.MEDIUM
            assert sh008[0].temporal_context == "past"

    def test_clinical_use_notice_always_present(self):
        result = _pipeline().analyze_sync("Patient reports suicidal ideation.")
        assert isinstance(result, AnalysisResponse)
        assert "not a diagnostic tool" in result.clinical_use_notice.lower()

    def test_mixed_negated_and_affirmed(self):
        """'Denies SI. Pt reports hopelessness.' — SI suppressed, CD-001 present."""
        text = "Denies SI. Pt reports hopelessness."
        result = _pipeline().analyze_sync(text)
        assert isinstance(result, AnalysisResponse)
        flag_ids = {f.flag_id for f in result.flags}
        assert "SH-002" not in flag_ids, "SI should be negated"
        assert "CD-001" in flag_ids, "Hopelessness should be detected"

    def test_compound_rule_fires(self):
        """Substance use + self-harm -> COMP-001 immediate review."""
        text = "Patient reports suicidal ideation. Drinking daily, about a fifth of vodka."
        result = _pipeline().analyze_sync(text)
        assert isinstance(result, AnalysisResponse)
        assert result.summary.requires_immediate_review is True
        flag_ids = {f.flag_id for f in result.flags}
        assert "SH-002" in flag_ids
        assert "SU-001" in flag_ids
