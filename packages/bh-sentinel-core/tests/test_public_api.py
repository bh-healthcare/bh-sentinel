"""Public API smoke tests for bh-sentinel-core."""

from __future__ import annotations


def test_version_exists():
    from bh_sentinel.core import __version__

    assert isinstance(__version__, str)
    assert __version__ == "0.1.0"


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
