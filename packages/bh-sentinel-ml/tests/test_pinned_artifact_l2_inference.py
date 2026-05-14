"""End-to-end L2 inference smoke test against the actual pinned artifact.

This test exists because v0.2.1 shipped an ONNX with static input axes that
crashed `TransformerClassifier._infer_batch` at runtime. The static-axes
bug was caught by graceful degradation (L2 silently became `FAILED` for
every request), passing all the existing unit + integration tests that
used the tiny synthetic fixture.

This test downloads the canonical pinned artifact from HF Hub via
`auto_download=True`, builds a real `Pipeline(enable_transformer=True)`,
runs analyze_sync against a short clinical text, and asserts:

  1. `pipeline_status.layer_2_transformer == COMPLETED` (NOT `FAILED`).
  2. At least one flag was either emitted by L2 or corroborated by L2,
     proving the ONNX session ran a real forward pass without raising.

The test is marked `real_model` and skipped by default. Run with:

    pytest -m real_model packages/bh-sentinel-ml/tests/test_pinned_artifact_l2_inference.py

Pre-release checklist (docs/release-process.md) MUST include running this
test against the pinned artifact before tagging a new ml-v* release.
"""

from __future__ import annotations

import pytest
from bh_sentinel.core import Pipeline
from bh_sentinel.core.models.flags import DetectionLayer, LayerStatus
from bh_sentinel.core.models.response import AnalysisResponse


@pytest.mark.real_model
def test_real_pinned_artifact_l2_completes() -> None:
    """Pipeline with the pinned artifact must complete L2 inference, not FAIL.

    The exact regression v0.2.1 had: this test asserts the
    `pipeline_status.layer_2_transformer == LayerStatus.COMPLETED` invariant
    that 0.2.1 silently violated for every request.
    """
    pipeline = Pipeline(enable_transformer=True, transformer_auto_download=True)
    assert pipeline._zero_shot is not None, (
        f"L2 failed to construct (init error: {pipeline._l2_init_error}). "
        "The pinned artifact is unreachable, SHA256 mismatch, or the "
        "tokenizer.json is missing from the HF repo root."
    )

    text = (
        "I feel hopeless and just want it to end. I cannot keep going. "
        "Nothing I do seems to matter and the future feels impossible."
    )
    response = pipeline.analyze_sync(text)
    assert isinstance(response, AnalysisResponse), (
        f"Pipeline returned non-AnalysisResponse: {type(response).__name__}"
    )

    status = response.pipeline_status.layer_2_transformer
    assert status == LayerStatus.COMPLETED, (
        f"L2 status was {status.value!r}, expected 'completed'. "
        "This is the v0.2.1 static-axes regression -- check that "
        "config/ml/ml_config.yaml's model_revision and model_sha256 "
        "point at an ONNX with symbolic [batch_size, sequence_length] "
        "input axes (NOT static [N, M] like the v0.2.1 artifact)."
    )


@pytest.mark.real_model
def test_real_pinned_artifact_produces_l2_evidence() -> None:
    """Beyond status=COMPLETED, at least one flag should bear L2 evidence.

    The merge logic emits a flag as `detection_layer=transformer` when L2
    found something L1 missed, or as `pattern_match` with `transformer`
    in `corroborating_layers` when both layers agreed. Either way, on a
    high-acuity clinical text we expect at least one flag to carry L2
    evidence -- otherwise L2 is structurally running but producing no
    candidates, which is itself a bug worth catching.
    """
    pipeline = Pipeline(enable_transformer=True, transformer_auto_download=True)
    text = (
        "I feel hopeless and just want it to end. I cannot keep going. "
        "Nothing I do seems to matter and the future feels impossible."
    )
    response = pipeline.analyze_sync(text)
    assert isinstance(response, AnalysisResponse)

    all_flags = list(response.flags) + list(response.protective_factors)
    l2_evidence = [
        f
        for f in all_flags
        if f.detection_layer == DetectionLayer.TRANSFORMER
        or DetectionLayer.TRANSFORMER in f.corroborating_layers
    ]
    assert l2_evidence, (
        "L2 completed but emitted zero flags / corroborations on a "
        "high-acuity test text. Either the hypothesis templates aren't "
        "wired to the model's class indices, the min_emit_confidence "
        "floor is too high, or the model's logits are all near zero "
        "(an indicator of a quantization or export problem). "
        f"All flags: {[f'{f.flag_id}/{f.detection_layer.value}' for f in all_flags]}"
    )
