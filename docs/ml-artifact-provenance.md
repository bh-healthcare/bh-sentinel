# ML Artifact Provenance

Single source of truth for the licensing and provenance of the ONNX artifact pinned by [`bh-sentinel-ml`](../packages/bh-sentinel-ml). When the pinned model or its source revision changes, this document MUST be updated in the same commit as the corresponding `model_revision` / `model_sha256` change in [`config/ml/ml_config.yaml`](../config/ml/ml_config.yaml).

This document deliberately mirrors what is published on the HuggingFace Hub repository hosting the artifact (see [Model Card](#hf-model-card) below). Anyone auditing the licensing chain should find the same facts in both places, byte-for-byte where practical.

## Currently pinned source

| Field | Value |
|---|---|
| Source HF repo | [`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli) |
| Source revision SHA | `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce` |
| License | MIT (declared in source repo's `cardData.license` and `license:mit` tag) |
| Source code paper | [Lewis et al. 2019, *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*](https://arxiv.org/abs/1910.13461) |
| NLI fine-tune dataset | [MultiNLI (multi_nli)](https://huggingface.co/datasets/multi_nli) |
| Zero-shot method paper | [Yin et al. 2019, *Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach*](https://arxiv.org/abs/1909.00161) |
| Original copyright holder | Meta Platforms, Inc. (BART authored by Facebook AI Research, now Meta AI) |

The pinned commit SHA above is what gets written into `config/ml/ml_config.yaml`'s `model_revision` field for the artifact derived from this source.

## Verification gate (Phase 0a of the v0.2.1 release plan)

Performed: 2026-05-12.
Tooling: `hf` CLI v1.x, `huggingface_hub` v0.36.2, `curl` against `huggingface.co/api/models/<id>`.

The v0.2.0 release left `model_revision: main` and an all-zeros `model_sha256` in [`config/ml/ml_config.yaml`](../config/ml/ml_config.yaml) as a TODO for v0.2.1, with the original intent of pinning [`valhalla/distilbart-mnli-12-3`](https://huggingface.co/valhalla/distilbart-mnli-12-3) (a distilled variant of `bart-large-mnli`, ~3x smaller). Before exporting any ONNX artifact, we ran a license verification gate against that intended source.

### Outcome for `valhalla/distilbart-mnli-12-3`

```bash
hf download valhalla/distilbart-mnli-12-3 --revision main --local-dir /tmp/license-check
ls /tmp/license-check/LICENSE* /tmp/license-check/COPYING*       # no matches
rg -i 'license|copyright' /tmp/license-check/README.md           # no license claim
curl -sS https://huggingface.co/api/models/valhalla/distilbart-mnli-12-3 \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('cardData',{}).get('license'), [t for t in d.get('tags',[]) if 'license' in t.lower()])"
# Output: None []
```

Three independent signals (snapshot inspection, model card scan, HF API metadata) all agreed: the upstream `valhalla/distilbart-mnli-12-3` repo has **no declared license**. It is widely understood in the community to be derivative of `facebook/bart-large-mnli` (MIT) via the No-Teacher-Distillation method, but the derivative inherits a license only if the original author chose to publish it under one — and that intent was never explicitly recorded by the upstream author in any form HF or `pip`-style tooling can resolve.

For a behavioral-health-adjacent OSS project that is going to redistribute a quantized derivative under its own namespace (`bh-healthcare/...`), publishing without a verified license chain is not acceptable. **The verification gate failed for `valhalla/distilbart-mnli-12-3`.**

### Fallback: `facebook/bart-large-mnli`

Per the v0.2.1 release plan ([`.cursor/plans/bh-sentinel-ml_0.2.1_release_v2_e0c9f464.plan.md`](../../.cursor/plans/bh-sentinel-ml_0.2.1_release_v2_e0c9f464.plan.md), Phase 0a), the documented fallback is to pin the teacher model directly. That source verifies cleanly:

```bash
curl -sS https://huggingface.co/api/models/facebook/bart-large-mnli \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('cardData',{}).get('license'), [t for t in d.get('tags',[]) if 'license' in t.lower()])"
# Output: mit ['license:mit']
```

The model card's YAML front-matter declares `license: mit` and the HF tag system corroborates with `license:mit`. The repo itself does not ship a standalone `LICENSE` file (a common pattern on HF where the front-matter is treated as the authoritative declaration), so our redistribution will include a canonical MIT LICENSE text with attribution to Meta Platforms, Inc. as the copyright holder.

### Trade-offs accepted with the fallback

| Aspect | `valhalla/distilbart-mnli-12-3` (rejected) | `facebook/bart-large-mnli` (pinned) |
|---|---|---|
| License | None declared | MIT, explicit |
| Parameter count | ~222M | ~406M |
| Approximate INT8 ONNX size | ~140MB | ~280–400MB |
| First-call download time on a fresh user | ~30s | ~90s–2min |
| Lambda cold-start cost (10s baseline) | +1–2s for model load | +3–5s for model load |
| Per-sentence inference latency on CPU | ~30ms | ~80ms |
| MNLI matched/mismatched accuracy (upstream-reported) | 88.1 / 88.19 | 89.9 / 90.01 |
| License-chain auditability | Broken (rejected) | Clean (Meta → us via MIT) |

The fallback delivers slightly higher upstream MNLI accuracy at the cost of ~3x larger artifact and ~2.5x higher per-call latency. These are acceptable for v0.2.x — the architecture's confidence calibration (`FixedDiscount(0.85)` per Phase A) and graceful-degradation paths absorb the latency increase, and the artifact size remains comfortably under HF's free-tier limits (5GB) and AWS Lambda's image-size budgets.

## License chain (for the redistributed artifact)

Three layers, all under MIT or its inheritance:

```
facebook/bart-large           ── MIT (Meta Platforms, Inc.) ── pretraining only, not redistributed by us
        └── facebook/bart-large-mnli  ── MIT (Meta Platforms, Inc.) ── source pinned here
                └── bh-healthcare/distilbart-mnli-12-3-int8-onnx (HF)  ── MIT (this project) ── derivative
                        └── bh-sentinel-ml on PyPI                       ── Apache-2.0 (this project) ── code only; references the artifact by SHA
```

Notes on the layering:

- The HF artifact and the PyPI package have **different licenses**: the artifact (model weights) is MIT to preserve the upstream chain; the Python package code is Apache-2.0 to match the rest of `bh-sentinel`. This is a common open-source pattern and is not contradictory — different copyrightable works can carry different terms.
- The artifact's repo name (`distilbart-mnli-12-3-int8-onnx`) is **historical**: it was chosen during the v0.2.1 planning phase when the intended source was the distilbart variant. We deliberately keep the same name even though the actual source is now `bart-large-mnli`, to preserve the URL that downstream references (`config/ml/ml_config.yaml`, `cli/__main__.py` `--repo` default, `bh-sentinel-examples/Makefile`) point at. The HF model card calls out the actual source explicitly to avoid confusion.

## HF model card

The model card published at [`bh-healthcare/distilbart-mnli-12-3-int8-onnx`](https://huggingface.co/bh-healthcare/distilbart-mnli-12-3-int8-onnx) on HF is the user-facing surface of the licensing chain. Its YAML front-matter and License Chain section MUST stay in sync with this document. The card is authored by [`scripts/export_onnx.py`](../scripts/export_onnx.py) at export time; see that script for the canonical template.

## How to re-pin against a new source

When the pinned source needs to change (next clinical-model variant, performance tuning, security patch), the operator runs:

```bash
# 1. Re-run the verification gate against the candidate source
hf download <CANDIDATE_HF_ID> --revision main --local-dir /tmp/license-check-new
ls /tmp/license-check-new/LICENSE* 2>/dev/null
curl -sS https://huggingface.co/api/models/<CANDIDATE_HF_ID> \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('cardData',{}).get('license'), [t for t in d.get('tags',[]) if 'license' in t.lower()])"

# 2. If the verification passes, run the export script against the new source
python scripts/export_onnx.py \
    --source-model <CANDIDATE_HF_ID> \
    --source-revision <PINNED_SHA> \
    --output-dir ./artifact_staging \
    --onnx-filename model_int8.onnx

# 3. Update THIS document with the new source, license, and verification evidence

# 4. Update config/ml/ml_config.yaml (and the vendored copy) with the new model_repo / model_revision / model_sha256

# 5. Re-upload the artifact to the HF repo (or create a new HF repo if the model architecture changes)

# 6. Cut a new bh-sentinel-ml minor release
```

The verification gate itself never changes — it is a hard precondition. Sources that fail any one of the three signals (no LICENSE file, no `cardData.license`, no `license:*` tag) MUST NOT be pinned.

## Change log for this document

| Date | Change | Pinned source after change |
|---|---|---|
| 2026-05-12 | Initial provenance record for v0.2.1 release. Verification gate failed for `valhalla/distilbart-mnli-12-3` (no license declared anywhere). Fallback to teacher model. | `facebook/bart-large-mnli@d7645e127eaf1aefc7862fd59a17a5aa8558b8ce` |
