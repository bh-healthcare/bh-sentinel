# ML Artifact Provenance

Single source of truth for the licensing and provenance of the ONNX artifact pinned by [`bh-sentinel-ml`](../packages/bh-sentinel-ml). When the pinned model or its source revision changes, this document MUST be updated in the same commit as the corresponding `model_revision` / `model_sha256` change in [`config/ml/ml_config.yaml`](../config/ml/ml_config.yaml).

This document deliberately mirrors what is published on the HuggingFace Hub repository hosting the artifact (see [Model Card](#hf-model-card) below). Anyone auditing the licensing chain should find the same facts in both places, byte-for-byte where practical.

## Currently pinned source

| Field | Value |
|---|---|
| Source HF repo | [`FacebookAI/roberta-large-mnli`](https://huggingface.co/FacebookAI/roberta-large-mnli) |
| Source revision SHA | `2a8f12d27941090092df78e4ba6f0928eb5eac98` |
| Redistribution HF repo | [`bh-healthcare/roberta-large-mnli-int8-onnx`](https://huggingface.co/bh-healthcare/roberta-large-mnli-int8-onnx) |
| Redistribution commit SHA | `69eb03178c210deceb076f0a8302bc5705179a58` (tag `ml-v0.2.2`) |
| ONNX SHA256 | `49fa5562da7f1422525e88fd1145d2d06ca93b17c335adf3c4696a30908c91de` |
| License | MIT (declared in source repo's `cardData.license` and `license:mit` tag) |
| Source code paper | [Liu et al. 2019, *RoBERTa: A Robustly Optimized BERT Pretraining Approach*](https://arxiv.org/abs/1907.11692) |
| NLI fine-tune dataset | [MultiNLI (multi_nli)](https://huggingface.co/datasets/multi_nli) |
| Zero-shot method paper | [Yin et al. 2019, *Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach*](https://arxiv.org/abs/1909.00161) |
| Original copyright holder | Meta Platforms, Inc. (RoBERTa authored by Facebook AI Research, now Meta AI) |

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

### First fallback: `facebook/bart-large-mnli` (also rejected — INT8 quality collapse)

Per the original v0.2.1 release plan, the documented fallback was to pin the teacher model directly. License-wise that source verifies cleanly:

```bash
curl -sS https://huggingface.co/api/models/facebook/bart-large-mnli \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('cardData',{}).get('license'), [t for t in d.get('tags',[]) if 'license' in t.lower()])"
# Output: mit ['license:mit']
```

`bh-sentinel-ml 0.2.1` shipped using this source. The published wheel however had two compounding bugs that together made L2 silently unusable for every user:

1. **Static input axes:** `scripts/export_onnx.py` passed `no_dynamic_axes=True` to optimum, freezing `[batch=2, seq=16]` into the graph. Runtime tensors had different shapes; ONNX Runtime threw; graceful degradation marked L2 `FAILED` for every request.
2. **Entailment index mismatch:** `ZeroShotClassifier` defaulted `entailment_index=0`. BART-large-MNLI uses `id2label={0: contradiction, 1: neutral, 2: entailment}` — the opposite ordering. The runtime read contradiction scores as if they were entailment.

Both bugs were fixed in the v0.2.2 hotfix. But re-running the export and the new real-model L2 smoke test surfaced a third, deeper problem:

3. **INT8 quantization quality collapse on encoder-decoder transformers.** Per-tensor and per-channel INT8 dynamic quantization both destroyed BART-large-MNLI's classification-head discrimination. Empirical comparison:

| Test case | FP32 BART-large | INT8 BART-large (`per_channel=True`) |
|---|---|---|
| "I feel hopeless..." vs "expresses hopelessness" | 0.998 | 0.24 |
| Same premise vs "forecast calls for sunny weather" (control) | 0.0003 | 0.06 |

The FP32 model had ~3000× discrimination ratio between strong-entailment and unrelated-control hypotheses; the INT8 model had ~4×. The signal was effectively gone — L2 candidates couldn't clear the `min_emit_confidence=0.55` threshold under any reasonable calibration. This is a well-documented limitation of INT8 dynamic quantization on encoder-decoder architectures: decoder cross-attention has wide value ranges, classification heads accumulate signal across decoder layers, INT8 error compounds.

**`facebook/bart-large-mnli` was rejected as the source for v0.2.2** because of this quantization quality collapse. The model is fine in FP32 (which is what the `transformers` library normally runs); INT8 specifically breaks it.

### Second (and current) fallback: `FacebookAI/roberta-large-mnli`

Encoder-only architecture, same MNLI 3-class output, same MIT license, same class ordering (`{0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT}`). Verifies cleanly:

```bash
curl -sS https://huggingface.co/api/models/FacebookAI/roberta-large-mnli \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('cardData',{}).get('license'), [t for t in d.get('tags',[]) if 'license' in t.lower()])"
# Output: mit ['license:mit']
```

INT8 quantization preserves discrimination on this source:

| Test case | FP32 RoBERTa-large | INT8 RoBERTa-large (`per_channel=True`) | Loss |
|---|---|---|---|
| "I feel hopeless..." vs "expresses hopelessness" | 0.9931 | 0.9932 | 0% |
| Same premise vs "sunny weather" (control) | 0.0108 | 0.0083 | within noise |
| "vacation" vs "expresses hopelessness" | 0.0005 | 0.0006 | within noise |

This matches the published understanding that **encoder-only transformers (RoBERTa, BERT, DistilBERT) quantize cleanly under dynamic INT8** whereas encoder-decoder architectures (BART, T5) do not.

### Trade-offs accepted with the RoBERTa-large source

| Aspect | `valhalla/distilbart-mnli-12-3` (rejected: no license) | `facebook/bart-large-mnli` (rejected: INT8 quality) | `FacebookAI/roberta-large-mnli` (pinned) |
|---|---|---|---|
| License | None declared | MIT, explicit | MIT, explicit |
| Architecture | Encoder-decoder (distilled) | Encoder-decoder | **Encoder-only** |
| Parameter count | ~222M | ~406M | ~355M |
| INT8 ONNX size | ~140MB | ~390MB | **~342MB** |
| INT8 discrimination preserved | N/A | ~0.25% of FP32 | **>99% of FP32** |
| Per-sentence inference latency (CPU) | ~30ms | ~80–120ms | **~40–80ms** |
| MNLI matched accuracy (upstream-reported) | 88.1 | 89.9 | 90.2 |
| License-chain auditability | Broken (rejected) | Clean | Clean |
| Suitable for INT8 deployment | Unknown | **No** (encoder-decoder + INT8 incompatible) | Yes |

The current pin (RoBERTa-large-MNLI) is the technically-correct choice: encoder-only architecture that quantizes cleanly, smaller artifact than BART-large despite preserving discrimination, identical license posture. The two upstream-rejected candidates remain documented above so any future audit understands the decision history.

## License chain (for the redistributed artifact)

Three layers, all under MIT or its inheritance:

```
FacebookAI/roberta-large           ── MIT (Meta Platforms, Inc.) ── pretraining only, not redistributed by us
        └── FacebookAI/roberta-large-mnli  ── MIT (Meta Platforms, Inc.) ── source pinned here
                └── bh-healthcare/roberta-large-mnli-int8-onnx (HF)  ── MIT (this project) ── derivative
                        └── bh-sentinel-ml on PyPI                     ── Apache-2.0 (this project) ── code only; references the artifact by SHA
```

Notes on the layering:

- The HF artifact and the PyPI package have **different licenses**: the artifact (model weights) is MIT to preserve the upstream chain; the Python package code is Apache-2.0 to match the rest of `bh-sentinel`. This is a common open-source pattern and is not contradictory — different copyrightable works can carry different terms.
- The HF repo is named `bh-healthcare/roberta-large-mnli-int8-onnx` to honestly reflect the actual source (RoBERTa-large, not BART or DistilBART). The earlier `bh-healthcare/distilbart-mnli-12-3-int8-onnx` HF repo (from the v0.2.1 attempt) still exists for audit history but is no longer referenced by any pinned config; new installs of `bh-sentinel-ml >= 0.2.2` resolve only to the RoBERTa repo.

## HF model card

The model card published at [`bh-healthcare/roberta-large-mnli-int8-onnx`](https://huggingface.co/bh-healthcare/roberta-large-mnli-int8-onnx) on HF is the user-facing surface of the licensing chain. Its YAML front-matter and License Chain section MUST stay in sync with this document. The card is authored by [`scripts/export_onnx.py`](../scripts/export_onnx.py) at export time; see that script for the canonical template.

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

| Date | Change | Pinned source after change | Redistribution commit / SHA256 |
|---|---|---|---|
| 2026-05-12 | Initial provenance record for v0.2.1 release. Verification gate failed for `valhalla/distilbart-mnli-12-3` (no license declared anywhere). Fallback to teacher model. | `facebook/bart-large-mnli@d7645e127eaf1aefc7862fd59a17a5aa8558b8ce` | (bh-healthcare/distilbart-mnli-12-3-int8-onnx) `ef4a3a8e...` / `1536ec8e...` (static-axes bug; ml-v0.2.1 yanked from PyPI) |
| 2026-05-13 (a.m.) | First hotfix attempt: re-export with dynamic axes. Source model unchanged; export tooling fixed. | `facebook/bart-large-mnli@d7645e127eaf1aefc7862fd59a17a5aa8558b8ce` | (bh-healthcare/distilbart-mnli-12-3-int8-onnx) `57e1eaeb...` / `adcfa96c...` (later rejected: INT8 quality) |
| 2026-05-13 (p.m.) | Second hotfix: switched source from BART-large-MNLI to RoBERTa-large-MNLI after the new real-model L2 smoke test surfaced INT8 quantization quality collapse on encoder-decoder architectures. RoBERTa is encoder-only and quantizes cleanly (<1% discrimination loss vs FP32). New HF repo `bh-healthcare/roberta-large-mnli-int8-onnx` created. | `FacebookAI/roberta-large-mnli@2a8f12d27941090092df78e4ba6f0928eb5eac98` | (bh-healthcare/roberta-large-mnli-int8-onnx) `69eb0317...` / `49fa5562...` (current; tag ml-v0.2.2) |
