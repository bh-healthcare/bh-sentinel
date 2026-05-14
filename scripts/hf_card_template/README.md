---
license: mit
license_link: LICENSE
library_name: transformers
pipeline_tag: zero-shot-classification
base_model: facebook/bart-large-mnli
base_model_relation: quantized
tags:
  - bh-sentinel
  - onnx
  - int8
  - clinical-decision-support
  - zero-shot-classification
  - bart
language:
  - en
datasets:
  - multi_nli
inference: false
---

# BART-Large-MNLI INT8 ONNX (bh-sentinel pinned artifact)

Quantized ONNX export of [`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli) at commit `{{SOURCE_REVISION}}`. Hosted as the canonical Layer 2 model for [`bh-sentinel-ml`](https://pypi.org/project/bh-sentinel-ml/) ≥ 0.2.1.

> The repository name `distilbart-mnli-12-3-int8-onnx` is **historical** — chosen during the v0.2.1 planning phase when the intended source was the smaller distilled variant `valhalla/distilbart-mnli-12-3`. During the release's license verification gate (see [provenance doc](https://github.com/bh-healthcare/bh-sentinel/blob/main/docs/ml-artifact-provenance.md)), that source was found to have no declared license and was rejected; the teacher model `facebook/bart-large-mnli` (explicit MIT) was used as the fallback. The repo name is preserved so downstream config references in `bh-sentinel-ml` continue to resolve. The actual source model is BART-Large fine-tuned on MultiNLI — see Provenance below.

## Clinical Use Notice

This is clinical decision support software. It is **not** a diagnostic tool, **not** FDA-cleared, and **not** a substitute for clinical judgment. The `bh-sentinel` pipeline that consumes this artifact is intended only for flagging signals for clinician review — never for autonomous clinical action. All outputs are signals for clinician review. See the [main repository's clinical disclaimer](https://github.com/bh-healthcare/bh-sentinel/blob/main/CLINICAL_DISCLAIMER.md) for the full notice.

## License Chain

| Layer | Model | License | Source |
|---|---|---|---|
| Pretraining base | [`facebook/bart-large`](https://huggingface.co/facebook/bart-large) | MIT | Meta Platforms, Inc. |
| Source (MNLI fine-tune) | [`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli) | MIT | Meta Platforms, Inc. |
| This artifact | `bh-healthcare/distilbart-mnli-12-3-int8-onnx` | MIT | bh-healthcare (this repo) |
| Consuming code | [`bh-sentinel-ml`](https://github.com/bh-healthcare/bh-sentinel) | Apache-2.0 | bh-healthcare |

The redistribution is permitted under the upstream MIT terms. The original copyright notice is preserved verbatim in [LICENSE](./LICENSE). The Python package that loads this artifact is Apache-2.0 — different copyrightable works can carry different terms; this is a common open-source pattern and is not contradictory.

## Provenance

| Field | Value |
|---|---|
| Source model | `facebook/bart-large-mnli` |
| Source revision | `{{SOURCE_REVISION}}` |
| Source paper | [Lewis et al. 2019, *BART*](https://arxiv.org/abs/1910.13461) |
| NLI fine-tune dataset | [MultiNLI](https://huggingface.co/datasets/multi_nli) |
| Zero-shot method paper | [Yin et al. 2019](https://arxiv.org/abs/1909.00161) |
| Export tool | `optimum-cli export onnx --task zero-shot-classification` |
| `optimum` version | `{{OPTIMUM_VERSION}}` |
| `onnxruntime` version | `{{ONNXRUNTIME_VERSION}}` |
| `onnx` version | `{{ONNX_VERSION}}` |
| Quantization | `onnxruntime.quantization.quantize_dynamic(weight_type=QInt8, per_channel=False, reduce_range=False)` |
| Export date (UTC) | `{{EXPORT_DATE}}` |
| Export script | [`scripts/export_onnx.py`](https://github.com/bh-healthcare/bh-sentinel/blob/main/scripts/export_onnx.py) at tag `ml-v0.2.1` |

To reproduce locally, install `optimum[onnxruntime]>=1.16` and `onnx>=1.15` and run the export script against the source revision above. The artifact is fully reproducible from those inputs.

## Size & Performance

Quantization compresses MatMul weights from FP32 to INT8 (~4x for the quantized ops). BART-Large has ~406M parameters; ONNX FP32 export is ~1.6GB, INT8 quantized is `{{FILE_SIZE_MB}}` MB.

| Metric | Value | Notes |
|---|---|---|
| ONNX file size (INT8) | `{{FILE_SIZE_MB}}` MB | Down from ~1.6GB FP32 |
| SHA256 of `model_int8.onnx` | `{{ONNX_SHA256}}` | Pinned in `bh-sentinel-ml`'s `ml_config.yaml`; verified at load time |
| First-call download time (typical broadband) | ~60–120s | One-time; subsequent calls hit the local platformdirs cache |
| Per-sentence inference latency (CPU, single thread) | ~80–120 ms | Per-pair (premise, hypothesis) NLI forward pass |
| Per-flag inference cost | ~80–120 ms × N_hypotheses_per_flag | bh-sentinel-ml batches these; see `max_batch_size` in `ml_config.yaml` |
| Recommended Lambda memory | 2GB+ | INT8 model + onnxruntime CPU provider + tokenizer cache |
| Recommended Lambda image size budget | 1GB+ | Model baked in; `BH_SENTINEL_ML_OFFLINE=1` Dockerfile pattern |

These numbers are **operator-reported approximations** for capacity planning. Real-world latency varies with CPU class, ARM vs x86, ONNX Runtime threading config, and concurrent load. The `bh-sentinel-ml` pipeline supports `BH_SENTINEL_ML_OFFLINE=1` for fully air-gapped VPC deployments where the model is pre-baked into the container image.

> **Why not a smaller model?** The originally-intended source, `valhalla/distilbart-mnli-12-3` (~140MB INT8, ~30ms/pair), was rejected by the license verification gate. We will revisit smaller MIT-licensed NLI alternatives (e.g. fine-tuned DeBERTa-v3-base variants) when validated clinical calibration is in scope (bh-sentinel v0.3+).

## Files

| File | Purpose |
|---|---|
| `model_int8.onnx` | Quantized model weights, **`{{FILE_SIZE_MB}}` MB**. SHA256 `{{ONNX_SHA256}}` (consumed by `TransformerClassifier`'s verify-on-load check) |
| `tokenizer.json` | HF `tokenizers` library serialized BPE tokenizer (required by `Tokenizer.from_file()` in the runtime) |
| `tokenizer_config.json` | Tokenizer config (model_max_length, special tokens) |
| `vocab.json` | BPE vocabulary |
| `merges.txt` | BPE merge rules |
| `special_tokens_map.json` | Special token map (BOS, EOS, PAD, etc.) |
| `manifest.json` | Machine-readable provenance metadata (matches the values in the Provenance section above) |
| `LICENSE` | MIT license text + attribution to Meta Platforms, Inc. |

## Intended Use

Consumed by `bh-sentinel-ml ≥ 0.2.1` via `Pipeline(enable_transformer=True)`. The bh-sentinel pipeline performs zero-shot NLI inference against a curated set of clinical hypothesis templates (see [`config/ml/zero_shot_hypotheses.yaml`](https://github.com/bh-healthcare/bh-sentinel/blob/main/config/ml/zero_shot_hypotheses.yaml)) to surface candidate clinical safety flags.

Use cases this artifact is designed for:

- Layer 2 (semantic) detection in the bh-sentinel multi-layer pipeline
- Catching clinical signals that pattern-matching alone misses: indirect language, implied distress, contextual meaning
- Reproducible local evaluation of bh-sentinel against the shared corpus in [`bh-sentinel-examples`](https://github.com/bh-healthcare/bh-sentinel-examples)

## Out-of-Scope Use

This artifact is **not** appropriate for:

- Direct clinical diagnosis or treatment decisions
- Autonomous clinical action of any kind
- Use as the sole basis for any clinical decision
- Inference tasks other than zero-shot NLI (it has not been retrained for other tasks)
- Production deployments without independent clinical validation against your organization's de-identified data
- Fine-tuning starting points for high-stakes clinical applications without further validation

## Limitations

- **Trained on general-purpose MNLI**, not clinical text. Confidence calibration on clinical inputs is dampened via `FixedDiscount(0.85)` in `bh-sentinel-ml`'s Phase A calibrator, but is **not** clinically validated. Validated calibration against clinician-labeled data is a bh-sentinel v0.3 deliverable.
- **INT8 dynamic quantization introduces small numerical drift** relative to the FP32 source. The bh-sentinel test suite includes a corpus-level round-trip check (see [`bh-sentinel-examples`](https://github.com/bh-healthcare/bh-sentinel-examples)) but does not guarantee bit-exact equivalence to FP32.
- **English only.** The MNLI training data is English-language; behavior on other languages is undefined and unsafe.
- **No clinical labels in training.** The model has never seen clinician-labeled examples of behavioral health safety signals. Its outputs are best understood as *plausibility scores against pre-defined hypothesis templates*, not as clinical probabilities.

## Citation

If you use this artifact, please cite both the upstream source and bh-sentinel:

```bibtex
@misc{bart-large-mnli,
  title  = {bart-large-mnli},
  author = {Facebook AI Research},
  year   = {2019},
  publisher = {Hugging Face},
  url    = {https://huggingface.co/facebook/bart-large-mnli}
}

@article{lewis2019bart,
  title   = {{BART}: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension},
  author  = {Lewis, Mike and Liu, Yinhan and Goyal, Naman and Ghazvininejad, Marjan and Mohamed, Abdelrahman and Levy, Omer and Stoyanov, Veselin and Zettlemoyer, Luke},
  journal = {arXiv preprint arXiv:1910.13461},
  year    = {2019},
  url     = {https://arxiv.org/abs/1910.13461}
}

@article{yin2019benchmarking,
  title   = {Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach},
  author  = {Yin, Wenpeng and Hay, Jamaal and Roth, Dan},
  journal = {arXiv preprint arXiv:1909.00161},
  year    = {2019},
  url     = {https://arxiv.org/abs/1909.00161}
}

@software{bh_sentinel,
  title  = {bh-sentinel: Open-source clinical safety signal detection for behavioral health systems},
  year   = {2026},
  publisher = {bh-healthcare},
  url    = {https://github.com/bh-healthcare/bh-sentinel}
}
```

## Acknowledgments

- The source model [`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli) was authored by Facebook AI Research (now Meta AI) and is redistributed here under its original MIT license.
- The zero-shot NLI classification method was proposed by [Yin et al. 2019](https://arxiv.org/abs/1909.00161); HF's blog post by Joe Davison ([May 2020](https://joeddav.github.io/blog/2020/05/29/ZSL.html)) popularized the pattern this artifact serves.
- The `optimum` library by Hugging Face provides the ONNX export path used here.
