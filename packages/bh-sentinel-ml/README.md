# bh-sentinel-ml

**Transformer-based clinical safety signal detection for behavioral health systems.**

`bh-sentinel-ml` is the Layer 2 add-on for [`bh-sentinel-core`](https://pypi.org/project/bh-sentinel-core/). It runs ONNX-Runtime zero-shot NLI inference over clinical sentences to catch signals that deterministic pattern matching misses: implied distress, indirect language, contextual meaning.

The core package always runs without `bh-sentinel-ml` installed. L2 is opt-in.

## Installation

```bash
pip install bh-sentinel-ml
```

This pulls in `bh-sentinel-core>=0.1.1`, `onnxruntime`, `tokenizers`, `huggingface-hub`, and `platformdirs`.

Optional extras:

```bash
pip install "bh-sentinel-ml[eval]"   # adds numpy + scikit-learn for calibrate/evaluate CLIs
```

## Compatibility

| `bh-sentinel-ml` | Requires `bh-sentinel-core` | Python |
|---|---|---|
| `0.2.x` | `>=0.1.1,<1` | `>=3.11` |

`bh-sentinel-ml 0.2.0` depends on the `Pipeline(transformer_model_path=..., transformer_auto_download=...)` kwargs that were added in `bh-sentinel-core 0.1.1`. Pairing it with `bh-sentinel-core 0.1.0` will break at import/construction time.

Enforcement:

- **Install time:** `pip install bh-sentinel-ml` resolves `bh-sentinel-core>=0.1.1,<1` from the wheel metadata. This is the primary guard.
- **Import time:** `import bh_sentinel.ml` verifies the installed `bh-sentinel-core` version via `importlib.metadata` and raises `ImportError` with an actionable upgrade message if it's too old. This catches the `--no-deps`, vendored, and editable-monorepo cases that bypass the pip resolver.

## Quick Start

```python
from bh_sentinel.core import Pipeline, AnalysisConfig

pipeline = Pipeline(enable_transformer=True)  # auto-downloads the pinned model on first run
result = pipeline.analyze_sync("I just can't see the point anymore.")

for flag in result.flags:
    print(flag.flag_id, flag.severity, flag.confidence, flag.corroborating_layers)
```

## Model distribution

`bh-sentinel-ml` uses a **hybrid** distribution strategy. The ~140MB INT8 ONNX model is not bundled in the wheel.

> **Release note (v0.2.0 → v0.2.1):** v0.2.0 ships the full L2 infrastructure
> (classifier, merge, calibration, CLI) but the default `ml_config.yaml`
> still has a **placeholder** `model_revision: main` and an all-zeros
> `model_sha256`. Neither `valhalla/distilbart-mnli-12-3` nor any of
> the common ONNX mirrors publish a pinned INT8 artifact the SHA of
> which we can commit to. The follow-up `bh-sentinel-ml 0.2.1` will:
>
> 1. Ship a canonical ONNX export (pinned revision + pinned SHA256).
> 2. Provide `scripts/export_onnx.py` for users who want to re-export
>    against their own base model.
> 3. Publish reproducible L1-vs-L2 reports against the shared corpus
>    via the [`bh-sentinel-examples`](https://github.com/bh-healthcare/bh-sentinel-examples) repo.
>
> Until `0.2.1` lands, the default auto-download path will fetch the
> HF repo's current `main` revision (unpinned) and the SHA check will
> fail -- meaning the production `BH_SENTINEL_ML_OFFLINE=1` rail is
> the only currently-supported deployment path, with the operator
> responsible for pinning their own SHA. See the
> [`bh-sentinel-examples` README](https://github.com/bh-healthcare/bh-sentinel-examples#regenerating-the-real-model-report-bh-sentinel-ml-v021-prerequisite)
> for the one-off reproduction flow.

**Dev / CI (unrestricted network):**
`pip install bh-sentinel-ml` → first `analyze()` call fetches the pinned HuggingFace revision into a local cache directory. One-time ~30s, zero config.

**Production / VPC-isolated / Lambda:**
Pre-bake the model into your container image at `docker build` time. Lambda cold starts must never hit HuggingFace Hub.

```dockerfile
FROM python:3.12-slim
RUN pip install bh-sentinel-ml
RUN bh-sentinel-ml download-model \
      --revision <PINNED_SHA> \
      --output /opt/bh-sentinel-ml/model \
      --verify-sha256 <PINNED_ONNX_SHA256>
ENV BH_SENTINEL_ML_OFFLINE=1
```

At runtime the pipeline reads the baked-in model:

```python
from pathlib import Path
from bh_sentinel.core import Pipeline

pipeline = Pipeline(
    enable_transformer=True,
    transformer_model_path=Path("/opt/bh-sentinel-ml/model"),
    transformer_auto_download=False,
)
```

### Production safety rails

- **`BH_SENTINEL_ML_OFFLINE=1`** -- set once in the Dockerfile. When set, `auto_download=True` is forced to `False`; `huggingface_hub` is never even imported. Any accidental future code change that tries to download over the network will fail immediately with a static PHI-safe error.
- **Verify-on-load SHA256.** `TransformerClassifier` computes the SHA256 of the ONNX file at pipeline construction and compares it to the pinned digest in `ml_config.yaml`. Mismatch raises `ModelIntegrityError` before any `InferenceSession` is created -- a stale or tampered container bake fails fast, not silently.
- **Graceful L2 failure.** If the model is missing, the SHA mismatches, or inference throws, the pipeline still returns a 200-shaped response with L1+L3+L4 flags and `PipelineStatus.layer_2_transformer == FAILED`. No exception ever propagates.

## Calibration (Phase A)

Architecture §4.8 prescribes `FixedDiscount(0.85)` for v0.2 -- raw softmax probabilities multiplied by a conservative factor. This is the default in `ml_config.yaml`.

`TemperatureScaling` is fully implemented and wired into the `calibrate` CLI, but **it is not validated against clinical data in v0.2**. ECE numbers produced today reflect the fixture data, not clinical reality; treat them as mechanism tests, not calibration claims. Real calibration ships in v0.3 once clinical labels are available per the roadmap.

```bash
bh-sentinel-ml calibrate --labels labels.jsonl --out calibration.json
```

## Evaluation

Run the pipeline against a fixture file and get a per-entry report (human-readable, matches the style of core's `bh-sentinel test-patterns`).

```bash
bh-sentinel-ml evaluate --fixtures my_fixtures.yaml
bh-sentinel-ml evaluate --corpus config/eval/real_world_corpus.yaml --enable-transformer
```

The shared real-world corpus at [`config/eval/real_world_corpus.yaml`](../../config/eval/real_world_corpus.yaml) (public-domain literature + synthetic clinical vignettes + true negatives) is what the L1 vs L2 diagnostic runs against.

## Clinical Use Notice

This is clinical decision support software. It is not a diagnostic tool, not a substitute for clinical judgment, and not FDA-cleared or approved. Organizations deploying this software in clinical settings are responsible for their own clinical validation, regulatory compliance, and patient safety protocols. See [CLINICAL_DISCLAIMER.md](../../CLINICAL_DISCLAIMER.md) in the main repository.

## Documentation

See [docs/architecture.md](../../docs/architecture.md) for the full Layer 2 design, [docs/release-process.md](../../docs/release-process.md) for release mechanics, and the [main repository](https://github.com/bh-healthcare/bh-sentinel) for everything else.

## License

Apache License 2.0.
