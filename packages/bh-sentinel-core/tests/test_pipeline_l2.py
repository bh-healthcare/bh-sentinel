"""Tests for the Layer 2 wiring in core's Pipeline.

These tests live in the core package because Phase 5 edits core's
pipeline.py. They use stub transformer implementations to avoid
requiring onnxruntime/tokenizers at core's test boundary. The real
model is exercised in packages/bh-sentinel-ml/tests/test_integration.py
(Phase 7).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from bh_sentinel.core.models.flags import DetectionLayer, LayerStatus
from bh_sentinel.core.models.response import AnalysisResponse


def _make_ml_stubs(
    candidates_to_emit: list | None = None,
    explode: Exception | None = None,
):
    """Return (ZeroShotClassifier, TransformerClassifier) stubs suitable
    for injection into the pipeline via the lazy-import hook.

    The stubs avoid importing bh_sentinel.ml so tests stay independent
    of onnxruntime/tokenizers availability.
    """
    from bh_sentinel.core._types import PatternMatchCandidate, PreprocessedText

    class StubZeroShot:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def classify(self, preprocessed: PreprocessedText) -> list[PatternMatchCandidate]:
            if explode is not None:
                raise explode
            return candidates_to_emit or []

    class StubTransformer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    return StubZeroShot, StubTransformer


def test_pipeline_enable_transformer_false_does_not_import_ml() -> None:
    """The core package stays zero-dep on ml -- enable_transformer=False
    must never import bh_sentinel.ml."""
    import sys

    for name in [n for n in list(sys.modules) if n.startswith("bh_sentinel.ml")]:
        del sys.modules[name]

    from bh_sentinel.core.pipeline import Pipeline

    _ = Pipeline(enable_transformer=False)

    assert not any(name.startswith("bh_sentinel.ml") for name in sys.modules), (
        "bh_sentinel.ml was imported despite enable_transformer=False"
    )


def test_pipeline_enable_transformer_missing_package_raises_import_error() -> None:
    """When enable_transformer=True but ml isn't installed, raise
    ImportError with a clear install hint -- don't silently degrade."""
    from bh_sentinel.core.pipeline import Pipeline

    # Simulate the ml package being absent by patching the lazy-import hook.
    with patch(
        "bh_sentinel.core.pipeline._load_ml",
        side_effect=ImportError(
            "bh-sentinel-ml is not installed. Run `pip install bh-sentinel-ml`."
        ),
    ):
        with pytest.raises(ImportError) as exc:
            Pipeline(enable_transformer=True)
    assert "bh-sentinel-ml" in str(exc.value)


def test_pipeline_l2_happy_path_populates_corroborating_layers() -> None:
    """L1 and L2 both detect SH-002 -> merged flag carries
    corroborating_layers=[TRANSFORMER] and max confidence."""
    from bh_sentinel.core._types import PatternMatchCandidate
    from bh_sentinel.core.pipeline import Pipeline

    StubZeroShot, StubTransformer = _make_ml_stubs(
        candidates_to_emit=[
            PatternMatchCandidate(
                flag_id="SH-002",
                domain="self_harm",
                name="Active suicidal ideation, nonspecific",
                default_severity="CRITICAL",
                confidence=0.88,
                sentence_index=0,
                char_start=0,
                char_end=30,
                pattern_text="",
                basis_description="L2 basis",
                matched_context_hint="hint",
                temporal_context="present",
            )
        ]
    )

    def loader():
        return StubZeroShot, StubTransformer, object()  # 3rd is MLConfig-like

    with patch("bh_sentinel.core.pipeline._load_ml", side_effect=loader):
        pipeline = Pipeline(
            enable_transformer=True,
            transformer_model_path=None,
            transformer_auto_download=False,
        )

    # Clinical text that L1 patterns definitely flag as SH-002.
    text = "I want to end my life."
    response = asyncio.run(pipeline.analyze(text))
    assert isinstance(response, AnalysisResponse)
    assert response.pipeline_status.layer_2_transformer == LayerStatus.COMPLETED

    sh002_flags = [f for f in response.flags if f.flag_id == "SH-002"]
    assert len(sh002_flags) == 1
    flag = sh002_flags[0]
    # L1's confidence from patterns.yaml is typically >= 0.85; L2 is 0.88.
    # The merge takes max, so the confidence is at least 0.88.
    assert flag.confidence >= 0.88
    # Corroboration: the non-primary layer is listed.
    assert len(flag.corroborating_layers) == 1
    assert flag.corroborating_layers[0] in (
        DetectionLayer.PATTERN_MATCH,
        DetectionLayer.TRANSFORMER,
    )


def test_pipeline_l2_graceful_degradation_on_classify_error() -> None:
    """If L2 classify() raises, the pipeline returns a 200-shaped
    response with L2 FAILED and L1+L3+L4 still populated. No exception
    propagates."""
    from bh_sentinel.ml.exceptions import InferenceError

    from bh_sentinel.core.pipeline import Pipeline

    StubZeroShot, StubTransformer = _make_ml_stubs(explode=InferenceError("simulated"))

    def loader():
        return StubZeroShot, StubTransformer, object()

    with patch("bh_sentinel.core.pipeline._load_ml", side_effect=loader):
        pipeline = Pipeline(
            enable_transformer=True,
            transformer_model_path=None,
            transformer_auto_download=False,
        )

    response = asyncio.run(pipeline.analyze("I want to end my life."))
    assert isinstance(response, AnalysisResponse)
    assert response.pipeline_status.layer_1_pattern == LayerStatus.COMPLETED
    assert response.pipeline_status.layer_2_transformer == LayerStatus.FAILED
    # L1 still detected flags -- L2 failure did NOT blank the output.
    assert len(response.flags) > 0


def test_pipeline_l2_graceful_degradation_on_init_error() -> None:
    """If transformer *construction* fails (model missing, SHA mismatch,
    tokenizer error), the pipeline still constructs and runs -- L2 is
    simply marked FAILED on every request."""
    from bh_sentinel.core.pipeline import Pipeline

    def bad_loader():
        class _BadTransformer:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("simulated transformer init failure")

        def _zs(*args: Any, **kwargs: Any):
            raise RuntimeError("should not be called")

        return _zs, _BadTransformer, object()

    with patch("bh_sentinel.core.pipeline._load_ml", side_effect=bad_loader):
        pipeline = Pipeline(
            enable_transformer=True,
            transformer_model_path=None,
            transformer_auto_download=False,
        )

    # Pipeline constructed successfully; L2 will report FAILED on run.
    response = asyncio.run(pipeline.analyze("I want to end my life."))
    assert response.pipeline_status.layer_2_transformer == LayerStatus.FAILED
    assert response.pipeline_status.layer_1_pattern == LayerStatus.COMPLETED


def test_rules_engine_accepts_l2_candidates_regression() -> None:
    """Regression guard: core's existing RulesEngine API accepts
    l2_candidates per the signature added in v0.1. This test exists to
    catch any accidental break from the Phase 5 edits."""
    from bh_sentinel.core._config import (
        default_flag_taxonomy_path,
        default_rules_path,
    )
    from bh_sentinel.core._types import EmotionScores
    from bh_sentinel.core.rules_engine import RulesEngine
    from bh_sentinel.core.taxonomy import FlagTaxonomy

    taxonomy = FlagTaxonomy(default_flag_taxonomy_path())
    engine = RulesEngine(default_rules_path(), taxonomy)
    result = engine.evaluate(
        candidates=[],
        emotion_scores=EmotionScores(),
        l2_candidates=[],
    )
    assert result.flags == []


def test_pipeline_analyze_sync_works_with_transformer_enabled() -> None:
    """analyze_sync must work when enable_transformer=True."""
    from bh_sentinel.core.pipeline import Pipeline

    StubZeroShot, StubTransformer = _make_ml_stubs()

    def loader():
        return StubZeroShot, StubTransformer, object()

    with patch("bh_sentinel.core.pipeline._load_ml", side_effect=loader):
        pipeline = Pipeline(
            enable_transformer=True,
            transformer_model_path=None,
            transformer_auto_download=False,
        )
    response = pipeline.analyze_sync("I want to end my life.")
    assert isinstance(response, AnalysisResponse)
