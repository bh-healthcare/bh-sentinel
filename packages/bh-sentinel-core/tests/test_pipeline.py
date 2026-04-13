"""Tests for Pipeline -- written before implementation (TDD)."""

from __future__ import annotations

import pytest

from bh_sentinel.core._disclaimer import CLINICAL_USE_NOTICE
from bh_sentinel.core.models.flags import Domain, LayerStatus, Severity
from bh_sentinel.core.models.response import AnalysisResponse, ErrorResponse
from bh_sentinel.core.pipeline import Pipeline


class TestBasicAnalysis:
    def test_simple_analysis(self):
        result = Pipeline().analyze_sync("Patient reports suicidal ideation for the past two days.")
        assert isinstance(result, AnalysisResponse)

    def test_response_has_required_fields(self):
        result = Pipeline().analyze_sync("Patient reports suicidal ideation for the past two days.")
        assert result.request_id
        assert result.processing_time_ms > 0
        assert result.taxonomy_version == "1.0.0"
        assert result.clinical_use_notice == CLINICAL_USE_NOTICE
        assert result.pipeline_status is not None
        assert result.summary is not None


class TestPipelineStatus:
    def test_layer_statuses(self):
        result = Pipeline().analyze_sync("Patient reports suicidal ideation.")
        ps = result.pipeline_status
        assert ps.layer_1_pattern == LayerStatus.COMPLETED
        assert ps.layer_2_transformer == LayerStatus.SKIPPED
        assert ps.layer_3_emotion_lexicon == LayerStatus.COMPLETED
        assert ps.layer_3_comprehend == LayerStatus.NOT_RUN
        assert ps.layer_4_rules == LayerStatus.COMPLETED

    def test_disabled_lexicon_reports_skipped_not_failed(self):
        p = Pipeline(enable_emotion_lexicon=False)
        result = p.analyze_sync("Patient reports suicidal ideation.")
        assert isinstance(result, AnalysisResponse)
        assert result.pipeline_status.layer_3_emotion_lexicon == LayerStatus.SKIPPED

    def test_bad_lexicon_path_reports_failed(self):
        from pathlib import Path

        p = Pipeline(lexicon_path=Path("/nonexistent/lexicon.json"))
        result = p.analyze_sync("Patient reports suicidal ideation.")
        assert isinstance(result, AnalysisResponse)
        assert result.pipeline_status.layer_3_emotion_lexicon == LayerStatus.FAILED


class TestFiltering:
    def test_domain_filtering(self):
        from bh_sentinel.core.models.request import AnalysisConfig

        config = AnalysisConfig(domains=[Domain.SELF_HARM])
        result = Pipeline().analyze_sync(
            "Patient reports suicidal ideation. Drinking daily vodka.",
            config=config,
        )
        for flag in result.flags:
            assert flag.domain == Domain.SELF_HARM

    def test_min_severity_filtering(self):
        from bh_sentinel.core.models.request import AnalysisConfig

        config = AnalysisConfig(min_severity=Severity.CRITICAL)
        result = Pipeline().analyze_sync(
            "Patient reports suicidal ideation.",
            config=config,
        )
        for flag in result.flags:
            assert flag.severity == Severity.CRITICAL

    def test_protective_factors_separated(self):
        result = Pipeline().analyze_sync(
            "Patient reports suicidal ideation. Attending all appointments."
        )
        assert isinstance(result, AnalysisResponse)
        for pf in result.protective_factors:
            assert pf.domain == Domain.PROTECTIVE_FACTORS
        for f in result.flags:
            assert f.domain != Domain.PROTECTIVE_FACTORS


class TestSummary:
    def test_total_flags_accurate(self):
        result = Pipeline().analyze_sync("Patient reports suicidal ideation for two days.")
        assert result.summary.total_flags == len(result.flags)

    def test_domains_flagged_accurate(self):
        result = Pipeline().analyze_sync("Patient reports suicidal ideation.")
        assert len(result.summary.domains_flagged) > 0

    def test_requires_immediate_review_on_crisis(self):
        result = Pipeline().analyze_sync(
            "Patient reports suicidal ideation and has been drinking daily."
        )
        assert isinstance(result, AnalysisResponse)
        assert result.summary.requires_immediate_review is True

    def test_recommended_action_present_on_critical(self):
        result = Pipeline().analyze_sync(
            "Patient reports suicidal ideation. Pt reports hopelessness."
        )
        assert isinstance(result, AnalysisResponse)
        if any(f.severity == Severity.CRITICAL for f in result.flags):
            assert result.summary.recommended_action is not None


class TestErrorHandling:
    def test_empty_text_returns_error(self):
        result = Pipeline().analyze_sync("")
        assert isinstance(result, ErrorResponse)

    def test_whitespace_only_returns_error(self):
        result = Pipeline().analyze_sync("   ")
        assert isinstance(result, ErrorResponse)

    def test_too_long_returns_error(self):
        result = Pipeline().analyze_sync("a" * 50_001)
        assert isinstance(result, ErrorResponse)


class TestPHISafety:
    def test_no_raw_text_in_flags(self):
        text = "Patient reports suicidal ideation for two days."
        result = Pipeline().analyze_sync(text)
        if isinstance(result, AnalysisResponse):
            for flag in result.flags:
                assert flag.matched_context_hint != text


class TestAsyncContext:
    @pytest.mark.asyncio
    async def test_analyze_async_works(self):
        result = await Pipeline().analyze(
            "Patient reports suicidal ideation for the past two days."
        )
        assert isinstance(result, AnalysisResponse)

    @pytest.mark.asyncio
    async def test_analyze_sync_raises_in_async_context(self):
        with pytest.raises(RuntimeError, match="cannot be called from within a running event loop"):
            Pipeline().analyze_sync("Patient reports SI.")
