"""Public API smoke tests for bh-sentinel-core."""

from __future__ import annotations


def test_version_exists():
    from bh_sentinel.core import __version__

    assert isinstance(__version__, str)
    assert __version__ == "0.1.1"


def test_all_exists():
    from bh_sentinel.core import __all__

    assert isinstance(__all__, list)
    assert "__version__" in __all__


def test_models_importable():
    from bh_sentinel.core import (
        AnalysisConfig,
        AnalysisContext,
        AnalysisRequest,
        AnalysisResponse,
        AnalysisSummary,
        DetectionLayer,
        Domain,
        EmotionResult,
        EvidenceSpan,
        Flag,
        LayerStatus,
        PipelineStatus,
        Severity,
    )

    assert Severity.CRITICAL == "CRITICAL"
    assert Domain.SELF_HARM == "self_harm"
    assert DetectionLayer.PATTERN_MATCH == "pattern_match"
    assert LayerStatus.NOT_RUN == "not_run"

    for cls in (
        AnalysisConfig,
        AnalysisContext,
        AnalysisRequest,
        AnalysisResponse,
        AnalysisSummary,
        EmotionResult,
        EvidenceSpan,
        Flag,
        PipelineStatus,
    ):
        assert callable(cls)


def test_core_classes_importable():
    from bh_sentinel.core import (
        EmotionLexicon,
        FlagTaxonomy,
        NegationDetector,
        PatternMatcher,
        Pipeline,
        RulesEngine,
        TemporalDetector,
        TextPreprocessor,
    )

    for cls in (
        EmotionLexicon,
        FlagTaxonomy,
        NegationDetector,
        PatternMatcher,
        Pipeline,
        RulesEngine,
        TemporalDetector,
        TextPreprocessor,
    ):
        assert callable(cls)


def test_all_exports_are_importable():
    """Every name in __all__ must be importable from the package."""
    import bh_sentinel.core as core

    for name in core.__all__:
        assert hasattr(core, name), f"{name} is in __all__ but not importable"


def test_flag_rejects_invalid_enum_values():
    """Flag model must reject values outside the defined enums."""
    import pytest
    from pydantic import ValidationError

    from bh_sentinel.core import EvidenceSpan, Flag

    with pytest.raises(ValidationError):
        Flag(
            flag_id="TEST-001",
            domain="BANANA",
            name="test",
            severity="BANANA",
            confidence=0.5,
            detection_layer="magic",
            matched_context_hint="test",
            basis_description="test",
            evidence_span=EvidenceSpan(sentence_index=0, char_start=0, char_end=5),
        )


def test_clinical_use_notice_exportable():
    """CLINICAL_USE_NOTICE must be in __all__ and importable."""
    from bh_sentinel.core import CLINICAL_USE_NOTICE, __all__

    assert "CLINICAL_USE_NOTICE" in __all__
    assert isinstance(CLINICAL_USE_NOTICE, str)
    assert "not a diagnostic tool" in CLINICAL_USE_NOTICE.lower()
    assert "not FDA-cleared" in CLINICAL_USE_NOTICE


def test_analysis_response_includes_clinical_notice():
    """AnalysisResponse must include clinical_use_notice field."""
    from bh_sentinel.core import (
        CLINICAL_USE_NOTICE,
        AnalysisResponse,
        AnalysisSummary,
        PipelineStatus,
        Severity,
    )

    response = AnalysisResponse(
        request_id="test-123",
        processing_time_ms=1.0,
        taxonomy_version="1.0.0",
        flags=[],
        summary=AnalysisSummary(
            max_severity=Severity.LOW,
            total_flags=0,
            domains_flagged=[],
            requires_immediate_review=False,
        ),
        pipeline_status=PipelineStatus(),
    )
    assert response.clinical_use_notice == CLINICAL_USE_NOTICE


def test_clinical_notice_in_json_serialization():
    """clinical_use_notice must appear in JSON output."""
    from bh_sentinel.core import (
        AnalysisResponse,
        AnalysisSummary,
        PipelineStatus,
        Severity,
    )

    response = AnalysisResponse(
        request_id="test-123",
        processing_time_ms=1.0,
        taxonomy_version="1.0.0",
        flags=[],
        summary=AnalysisSummary(
            max_severity=Severity.LOW,
            total_flags=0,
            domains_flagged=[],
            requires_immediate_review=False,
        ),
        pipeline_status=PipelineStatus(),
    )
    json_str = response.model_dump_json()
    assert "clinical_use_notice" in json_str


def test_flag_has_temporal_context_field():
    """Flag must have temporal_context field defaulting to 'present'."""
    from bh_sentinel.core import DetectionLayer, Domain, EvidenceSpan, Flag, Severity

    flag = Flag(
        flag_id="SH-001",
        domain=Domain.SELF_HARM,
        name="test",
        severity=Severity.HIGH,
        confidence=0.9,
        detection_layer=DetectionLayer.PATTERN_MATCH,
        matched_context_hint="test",
        basis_description="test",
        evidence_span=EvidenceSpan(sentence_index=0, char_start=0, char_end=5),
    )
    assert flag.temporal_context == "present"

    flag_past = Flag(
        flag_id="SH-008",
        domain=Domain.SELF_HARM,
        name="test",
        severity=Severity.HIGH,
        confidence=0.9,
        detection_layer=DetectionLayer.PATTERN_MATCH,
        matched_context_hint="test",
        basis_description="test",
        evidence_span=EvidenceSpan(sentence_index=0, char_start=0, char_end=5),
        temporal_context="past",
    )
    assert flag_past.temporal_context == "past"


def test_emotion_result_has_category_scores():
    """EmotionResult must have category_scores field."""
    from bh_sentinel.core import EmotionResult

    result = EmotionResult()
    assert result.category_scores == {}

    result_with_scores = EmotionResult(category_scores={"hopelessness": 0.3, "agitation": 0.1})
    assert result_with_scores.category_scores["hopelessness"] == 0.3
