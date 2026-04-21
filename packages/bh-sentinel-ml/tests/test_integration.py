"""End-to-end integration: Pipeline(enable_transformer=True) with the
tiny ONNX fixture.

Uses the tests/fixtures/tiny_nli.onnx model built by conftest.py.
Tokenization is stubbed via a toy tokenizer injected after construction;
this lets us exercise the full L1+L2+L3+L4 pipeline path without
requiring a real HF tokenizer or network access.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

from bh_sentinel.core.models.flags import LayerStatus
from bh_sentinel.core.models.response import AnalysisResponse
from bh_sentinel.core.pipeline import Pipeline


class _ToyEncoding:
    def __init__(self, length: int) -> None:
        self.ids = list(range(1, length + 1))
        self.attention_mask = [1] * length


class _ToyTokenizer:
    def encode(self, premise: str, hypothesis: str):
        return _ToyEncoding(len(premise) + len(hypothesis) + 2)


def _pipeline_with_real_l2(tiny_nli_model: Path, tiny_nli_sha256: str) -> Pipeline:
    """Build a Pipeline that uses a REAL TransformerClassifier backed by
    the tiny ONNX fixture. The toy tokenizer is swapped in after init
    so we don't need a HF tokenizer.json file.
    """
    # Patch the ml_config loader so verify-on-load accepts the fixture.
    from bh_sentinel.ml._config import MLConfig

    fake_cfg = MLConfig(
        model_repo="test/local",
        model_revision="HEAD",
        onnx_filename=tiny_nli_model.name,
        model_sha256=tiny_nli_sha256,
        max_sentence_length=64,
        max_batch_size=8,
        min_emit_confidence=0.3,
        calibration={"strategy": "fixed_discount", "discount": 0.85},
    )

    # resolve_model_path returns the fixture directory.
    def fake_resolve(*_args, **_kwargs):
        return tiny_nli_model.parent

    with (
        patch("bh_sentinel.ml._config.load_ml_config", return_value=fake_cfg),
        patch(
            "bh_sentinel.ml.model_cache.resolve_model_path",
            side_effect=fake_resolve,
        ),
        # Skip real tokenizer load; we inject the toy tokenizer below.
        patch(
            "bh_sentinel.ml.transformer.Tokenizer.from_file",
            return_value=_ToyTokenizer(),
        ),
    ):
        pipeline = Pipeline(
            enable_transformer=True,
            transformer_model_path=tiny_nli_model.parent,
            transformer_auto_download=False,
        )
    return pipeline


def test_full_pipeline_l2_completes(tiny_nli_model: Path, tiny_nli_sha256: str) -> None:
    pipeline = _pipeline_with_real_l2(tiny_nli_model, tiny_nli_sha256)
    response = asyncio.run(pipeline.analyze("The patient reports wanting to end her life."))
    assert isinstance(response, AnalysisResponse)
    assert response.pipeline_status.layer_1_pattern == LayerStatus.COMPLETED
    assert response.pipeline_status.layer_2_transformer == LayerStatus.COMPLETED


def test_full_pipeline_graceful_degradation_on_missing_model(
    tmp_path: Path,
) -> None:
    """Model path that doesn't exist -> L2 FAILED, L1+L3 still populated."""
    pipeline = Pipeline(
        enable_transformer=True,
        transformer_model_path=tmp_path / "does-not-exist",
        transformer_auto_download=False,
    )
    response = asyncio.run(pipeline.analyze("I want to end my life."))
    assert isinstance(response, AnalysisResponse)
    assert response.pipeline_status.layer_1_pattern == LayerStatus.COMPLETED
    assert response.pipeline_status.layer_2_transformer == LayerStatus.FAILED
    # L1 still detected flags.
    assert len(response.flags) > 0


def test_full_pipeline_l1_only_regression(tmp_path: Path) -> None:
    """enable_transformer=False must behave exactly like v0.1."""
    pipeline = Pipeline(enable_transformer=False)
    response = asyncio.run(pipeline.analyze("I want to end my life."))
    assert isinstance(response, AnalysisResponse)
    assert response.pipeline_status.layer_2_transformer == LayerStatus.SKIPPED
    # No L2 means no transformer corroboration on any flag.
    for flag in response.flags:
        assert all("transformer" not in str(layer).lower() for layer in flag.corroborating_layers)
