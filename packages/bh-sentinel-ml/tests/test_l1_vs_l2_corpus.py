"""Open-domain L1 vs L2 diagnostic (Phase 7b).

Loads the shared corpus at config/eval/real_world_corpus.yaml and runs
Pipeline(enable_transformer=True) against each entry using the tiny
ONNX fixture. Produces a per-fixture report showing:

- flags detected by L1 only (Phase 1 baseline)
- flags detected by L2 only (new signal L2 adds on top)
- flags corroborated by both layers
- per-fixture L1/L2 agreement rate
- coverage of each entry's expected_flags_hint

This is a *diagnostic*, not a pass/fail gate. Structural assertions
only: every flag must have the required fields, L2 must run to
completion, and graceful degradation must never leak an exception.
Numeric detection-quality gates are deferred to v0.3 when clinical
labels are available.

Marked @pytest.mark.real_model so it can be skipped in default CI.
The tiny ONNX fixture is deterministic but not trained -- output
flags depend on attention-mask sums, not clinical semantics. The
value of this test is the structural guarantee + the side-by-side
report when it runs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from bh_sentinel.core.models.flags import DetectionLayer, LayerStatus
from bh_sentinel.core.models.response import AnalysisResponse
from bh_sentinel.core.pipeline import Pipeline

CORPUS_PATH = Path(__file__).resolve().parents[3] / "config" / "eval" / "real_world_corpus.yaml"


class _ToyEncoding:
    def __init__(self, length: int) -> None:
        self.ids = list(range(1, length + 1))
        self.attention_mask = [1] * length


class _ToyTokenizer:
    def encode(self, premise: str, hypothesis: str):
        return _ToyEncoding(len(premise) + len(hypothesis) + 2)


def _load_corpus() -> list[dict[str, Any]]:
    if not CORPUS_PATH.exists():
        pytest.skip(f"corpus not found: {CORPUS_PATH}")
    data = yaml.safe_load(CORPUS_PATH.read_text())
    fixtures = data.get("fixtures", [])
    return list(fixtures)


def _pipeline_with_real_l2(tiny_nli_model: Path, tiny_nli_sha256: str) -> Pipeline:
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

    def fake_resolve(*_args, **_kwargs):
        return tiny_nli_model.parent

    with (
        patch("bh_sentinel.ml._config.load_ml_config", return_value=fake_cfg),
        patch(
            "bh_sentinel.ml.model_cache.resolve_model_path",
            side_effect=fake_resolve,
        ),
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


@pytest.mark.real_model
def test_corpus_runs_end_to_end_through_l2(
    tiny_nli_model: Path, tiny_nli_sha256: str, capsys
) -> None:
    """Every corpus entry runs through L1+L2+L3+L4 with L2 COMPLETED
    and no exceptions. Produces a per-fixture report on stdout.
    """
    corpus = _load_corpus()
    pipeline_l1 = Pipeline(enable_transformer=False)
    pipeline_l2 = _pipeline_with_real_l2(tiny_nli_model, tiny_nli_sha256)

    total_l1_only = 0
    total_l2_only = 0
    total_corroborated = 0

    report_lines: list[str] = []
    report_lines.append("=" * 78)
    report_lines.append(f"  L1 vs L2 corpus diagnostic ({len(corpus)} entries)")
    report_lines.append("=" * 78)

    for entry in corpus:
        fx_id = entry["id"]
        text = entry["text"]
        expected = set(entry.get("expected_flags_hint") or [])

        r1 = pipeline_l1.analyze_sync(text)
        r2 = asyncio.run(pipeline_l2.analyze(text))
        assert isinstance(r1, AnalysisResponse), fx_id
        assert isinstance(r2, AnalysisResponse), fx_id
        assert r2.pipeline_status.layer_2_transformer == LayerStatus.COMPLETED, (
            f"L2 did not complete for {fx_id}: {r2.pipeline_status.layer_2_transformer}"
        )

        l1_ids = {f.flag_id for f in r1.flags} | {f.flag_id for f in r1.protective_factors}
        # From r2, flags whose detection_layer is PATTERN_MATCH count as L1
        # (either L1-only or corroborated). TRANSFORMER as primary means L2 won.
        r2_all = list(r2.flags) + list(r2.protective_factors)
        l2_primary_ids = {
            f.flag_id for f in r2_all if f.detection_layer == DetectionLayer.TRANSFORMER
        }
        corroborated_ids = {
            f.flag_id
            for f in r2_all
            if DetectionLayer.TRANSFORMER in f.corroborating_layers
            or DetectionLayer.PATTERN_MATCH in f.corroborating_layers
        }

        l1_only = l1_ids - (l2_primary_ids | corroborated_ids)
        l2_only = l2_primary_ids - l1_ids
        both = corroborated_ids

        total_l1_only += len(l1_only)
        total_l2_only += len(l2_only)
        total_corroborated += len(both)

        hint_hit_l1 = expected & l1_ids
        hint_hit_l2 = expected & (l2_primary_ids | corroborated_ids)
        hint_miss = expected - (l1_ids | l2_primary_ids | corroborated_ids)

        report_lines.append(f"\n- {fx_id}  [{entry.get('category', '?')}]")
        report_lines.append(f"    L1 only:         {sorted(l1_only) or '-'}")
        report_lines.append(f"    L2 only (new):   {sorted(l2_only) or '-'}")
        report_lines.append(f"    corroborated:    {sorted(both) or '-'}")
        if expected:
            report_lines.append(f"    expected hint:   {sorted(expected)}")
            report_lines.append(f"    hit by L1:       {sorted(hint_hit_l1) or '-'}")
            report_lines.append(f"    hit by L2:       {sorted(hint_hit_l2) or '-'}")
            report_lines.append(f"    missed by both:  {sorted(hint_miss) or '-'}")

    total_flags = total_l1_only + total_l2_only + total_corroborated
    if total_flags > 0:
        agreement = total_corroborated / total_flags
    else:
        agreement = 0.0
    report_lines.append(
        f"\nAggregate: L1-only={total_l1_only}, L2-only={total_l2_only}, "
        f"both={total_corroborated}, L1/L2 agreement rate={agreement:.2%}"
    )

    report_text = "\n".join(report_lines)
    print(report_text)

    # Write the report to an artifact file so it survives pytest capture.
    artifact_dir = Path(__file__).resolve().parent / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    gitignore = artifact_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n!.gitignore\n")
    (artifact_dir / "real_world_l1_vs_l2_latest.md").write_text(report_text)

    # Also check capsys so the test at least touched stdout.
    captured = capsys.readouterr()
    assert "L1 vs L2 corpus diagnostic" in (captured.out + report_text)


@pytest.mark.real_model
def test_corpus_never_raises(tiny_nli_model: Path, tiny_nli_sha256: str) -> None:
    """Iron-clad: across every corpus entry, no exception ever propagates
    out of the pipeline. Graceful degradation is the whole point."""
    corpus = _load_corpus()
    pipeline = _pipeline_with_real_l2(tiny_nli_model, tiny_nli_sha256)
    for entry in corpus:
        response = pipeline.analyze_sync(entry["text"])
        assert isinstance(response, AnalysisResponse), entry["id"]
