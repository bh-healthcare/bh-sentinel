# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [ml-0.2.1] - 2026-05-13

Closes the v0.2.0 → v0.2.1 gap promised in the prior `[Unreleased] / Planned` block: ships the pinned ONNX artifact, the export script that produced it, and reproducible-report machinery in [`bh-sentinel-examples`](https://github.com/bh-healthcare/bh-sentinel-examples).

### Added

- **Canonical ONNX artifact** published at [`bh-healthcare/distilbart-mnli-12-3-int8-onnx`](https://huggingface.co/bh-healthcare/distilbart-mnli-12-3-int8-onnx) on the HuggingFace Hub. `model_revision` and `model_sha256` in [`config/ml/ml_config.yaml`](config/ml/ml_config.yaml) are now real values (pinned commit `ef4a3a8e5dea3746000aa8a73fe5f1617a9289df`, INT8 ONNX SHA256 `1536ec8e38136b25b4a77286d83cafe95e7d48992d24a2e4c9b0dfb162b25dd0`). Production `auto_download=True` now works end-to-end — the verify-on-load SHA check passes.
- **[`scripts/export_onnx.py`](scripts/export_onnx.py)** — operator tool that invokes optimum's ONNX exporter (via the Python API, not the CLI subprocess, for clean error visibility), INT8-quantizes the result with `onnxruntime.quantization.quantize_dynamic`, validates the input/output shape against the runtime contract, copies tokenizer files, and emits a `manifest.json` with full provenance. Includes a fail-fast precondition check that surfaces torch ↔ optimum version skews in milliseconds instead of after the heavy subprocess fails mid-stream.
- **[`scripts/hf_card_template/`](scripts/hf_card_template/)** — HF model card template (`README.md`) + MIT `LICENSE` template with Meta + bh-healthcare attribution, plus a `HOW_TO_USE.md` with a one-liner substitution script. Operator copies these into `artifact_staging/` after `export_onnx.py` runs.
- **[`docs/ml-artifact-provenance.md`](docs/ml-artifact-provenance.md)** — single source of truth for the licensing chain and verification gate. Records why `valhalla/distilbart-mnli-12-3` was rejected (no declared license anywhere — `cardData.license: None`, no `LICENSE` file in the source snapshot, no `license:*` tag) and why the teacher model `facebook/bart-large-mnli` (explicit `license:mit`) was chosen as the fallback. Includes the re-pinning workflow for future releases.
- **`bh-sentinel-ml`'s `download-model` CLI default `--repo`** now points at `bh-healthcare/distilbart-mnli-12-3-int8-onnx` (was `valhalla/distilbart-mnli-12-3`). Help text already promised "default matches ml_config.yaml default" — that contract is now honored.
- **End-to-end tests for `scripts/export_onnx.py`** at [`packages/bh-sentinel-ml/tests/test_export_onnx_script.py`](packages/bh-sentinel-ml/tests/test_export_onnx_script.py). 11 unit tests (run in default CI) cover ONNX I/O contract validation, tokenizer copy semantics, manifest writing, SHA256 computation. 1 `real_model`-marked end-to-end test (opt-in) exercises the full `main()` path against a small public NLI model.

### Changed

- **Pinned source model:** `valhalla/distilbart-mnli-12-3` → `facebook/bart-large-mnli` (per the v0.2.1 license verification gate). The HF artifact repo name retains the historical `distilbart-mnli-12-3-int8-onnx` URL for stability; the actual source is BART-Large-MNLI per [`docs/ml-artifact-provenance.md`](docs/ml-artifact-provenance.md). Operational trade-offs from this swap: ~390MB INT8 (vs. ~140MB for the rejected source), ~60–120s first-call download (vs. ~30s), ~80–120ms per-pair inference latency on CPU (vs. ~30ms). Upstream MNLI accuracy is slightly higher (89.9/90.01 vs. 88.1/88.19 matched/mismatched).
- **`training/export.py`** stub updated to point operators at `scripts/export_onnx.py` for the baseline export path. The `training/` directory remains the home for the v0.3+ fine-tuned-organizational-model export workflow.

### Fixed

- **PyPI maintainer email** in [`packages/bh-sentinel-ml/pyproject.toml`](packages/bh-sentinel-ml/pyproject.toml) and [`packages/bh-sentinel-core/pyproject.toml`](packages/bh-sentinel-core/pyproject.toml): replaced the unreachable scaffold placeholder `oss@bh-healthcare.github.io` (`.github.io` domains cannot receive email) with the maintainer-controlled forwarder `oss@bh-healthcare.org`. The PyPI project page now links to a deliverable address. No runtime effect; metadata-only.

### Compatibility

- The canonical model repo changed from `valhalla/distilbart-mnli-12-3` to `bh-healthcare/distilbart-mnli-12-3-int8-onnx`. **Existing on-disk HF snapshot caches from 0.2.0 are not reused.** First call after upgrade re-downloads ~390MB into the platformdirs cache. Operators on the `BH_SENTINEL_ML_OFFLINE=1` rail must rebuild their container image with the new `bh-sentinel-ml download-model` invocation (which now defaults to the new repo).
- `bh-sentinel-ml 0.2.1` still requires `bh-sentinel-core>=0.1.1,<1`. No core bump in this release.

## [ml-0.2.0] - 2026-04-17

First release of `bh-sentinel-ml` as a Layer 2 add-on to `bh-sentinel-core`.

### Added

- **`bh-sentinel-ml` package** published to PyPI with the hybrid model-distribution strategy: auto-download from HuggingFace Hub by default, explicit `model_path` override for containers, and the `BH_SENTINEL_ML_OFFLINE=1` production safety rail
- `TransformerClassifier` -- ONNX Runtime wrapper with verify-on-load SHA256 integrity check (mismatched bytes fail pipeline construction before any inference session is created)
- `ZeroShotClassifier` -- NLI-based per-flag zero-shot classifier that emits `PatternMatchCandidate` compatible with core's `RulesEngine`
- `Calibrator` protocol plus `FixedDiscount(0.85)` (Phase A default per architecture §4.8), `TemperatureScaling` with LBFGS-free ternary-search fit, and `compute_ece` helper
- `merge_candidates` implementing architecture §4.7 exactly: max-confidence dedup, L1-preferred evidence span, corroboration metadata, present-wins temporal merge
- `bh-sentinel-ml` CLI with three subcommands: `download-model` (container pre-bake with `--verify-sha256`), `calibrate` (TemperatureScaling fit + ECE), `evaluate` (per-fixture L1/L2 diagnostic)
- Shared real-world corpus at [`config/eval/real_world_corpus.yaml`](config/eval/real_world_corpus.yaml) -- Woolf, Gilman, Tolstoy, Dostoevsky, synthetic vignettes, true negatives. Wired into a diagnostic test that produces a side-by-side L1 vs L2 report
- Two new publish workflows ([`publish-core.yml`](.github/workflows/publish-core.yml), [`publish-ml.yml`](.github/workflows/publish-ml.yml)) with per-package tag prefixes (`core-v*` / `ml-v*`) and CI-enforced tag/pyproject version agreement
- [`docs/release-process.md`](docs/release-process.md) -- full release procedure, PyPI Trusted Publisher setup, rollback guidance
- Runtime `bh-sentinel-core` version check in `bh_sentinel.ml.__init__`: if the installed core is older than `0.1.1` (or missing), `import bh_sentinel.ml` raises `ImportError` with an actionable upgrade message. Catches `--no-deps`, vendored, and editable-monorepo installs that bypass pip's resolver; pip's install-time constraint remains the primary guard.
- Explicit **Compatibility** section in [`packages/bh-sentinel-ml/README.md`](packages/bh-sentinel-ml/README.md) (version matrix + install-time vs import-time enforcement notes), surfaced on the PyPI project page as part of the long description.

### Compatibility

- `bh-sentinel-ml` requires `bh-sentinel-core>=0.1.1`
- Calibration is Phase A only: ECE numbers are not production-meaningful until v0.3 ships clinical labels. The mechanism is complete; validation is deferred

## [0.1.1] - 2026-04-17

### Added

- `Pipeline` gains `transformer_model_path` and `transformer_auto_download` kwargs. Both have safe defaults; existing callers continue to work untouched
- L1/L2 candidate merge path in the pipeline (lazy-imported only when `enable_transformer=True`), including corroboration metadata hydration onto `Flag.corroborating_layers`
- `PipelineStatus.layer_2_transformer` is now set to `COMPLETED` / `FAILED` / `SKIPPED` based on actual L2 execution
- Graceful degradation: L2 failures (model missing, SHA mismatch, inference error) never propagate -- the pipeline returns a 200-shaped response with L1+L3+L4 populated and L2 marked `FAILED`

### Compatibility

- Fully backward-compatible with v0.1.0. New kwargs are opt-in and default to the pre-0.1.1 behavior
- Existing `Pipeline(enable_transformer=False)` path makes zero additional imports -- the `bh-sentinel-core` package remains zero-dep on `onnxruntime` / `tokenizers` / `huggingface_hub`

## [0.1.0] - 2026-04-12

### Added

- **bh-sentinel-core** -- complete Layer 1 + Layer 3 + Layer 4 pipeline (text in, flags out)
- TextPreprocessor with sentence splitting, abbreviation handling, and character offset tracking
- NegationDetector with clause-bounded forward-window negation, pseudo-negation exceptions, and post-negation detection
- TemporalDetector with past/present classification and present-overrides-past resolution
- FlagTaxonomy loader with version checks, domain indexes, and 40 flags across 6 domains
- PatternMatcher (Layer 1) with compiled regex, negation integration, temporal awareness, and within-sentence deduplication
- EmotionLexicon (Layer 3) with 11-category density scoring and multi-word phrase matching (project-owned lexicon, 247 terms, 11 categories)
- RulesEngine (Layer 4) with 10 escalation rules, 1 de-escalation rule, 8 compound rules, 1 action rule, and recursive condition evaluation
- Pipeline orchestrator with async parallel execution (L1 + L3), sync wrapper, and graceful degradation
- Config validation CLI (`bh-sentinel validate-config`) with 7-point consistency checks
- Pattern test runner CLI (`bh-sentinel test-patterns`) for fixture validation
- CLINICAL_DISCLAIMER.md and `clinical_use_notice` field in every AnalysisResponse
- `temporal_context` field on Flag model for past/present classification
- `category_scores` field on EmotionResult for 11-category density scores
- `ge=0` constraints on EvidenceSpan fields (sentence_index, char_start, char_end)
- Flag taxonomy configuration (40 flags across 6 domains, v1.0.0)
- Pattern library (351 patterns across 40 flags, with third-person pronoun, nonbinary, bare abbreviation, brand-name medication, and numeric duration coverage)
- Rules engine configuration (20 rules: 10 escalation, 1 de-escalation, 8 compound, 1 action)
- Emotion lexicon (247 terms, 11 categories)
- Test fixtures (84 clinical text cases, all passing)
- Real-world validation suite (public domain literature, clinical vignettes, true negatives)
- 315 tests covering all components, rules, fixtures, integration, PHI safety, and real-world validation
- `bh-sentinel-ml` package skeleton (transformer, zero-shot, export -- implementation in v0.2)
- AWS Lambda deployment stub with Dockerfile
- Terraform directory structure for VPC + API Gateway
- Training pipeline stubs (prepare_data, train, evaluate, export)
- GitHub Actions CI workflow (lint + test, Python 3.11 + 3.12)
- GitHub Actions publish workflow (build -> PyPI -> GitHub Release)
- Documentation: architecture, flag taxonomy, pattern library, FDA CDS analysis, deployment guide
- Contributing guidelines

### Changed

- Split CD-005 "Psychotic symptoms" into four granular sub-flags (40 total flags):
  CD-005a (auditory hallucinations), CD-005b (visual hallucinations),
  CD-005c (paranoid ideation), CD-005d (delusional thinking)

[Unreleased]: https://github.com/bh-healthcare/bh-sentinel/compare/ml-v0.2.1...HEAD
[ml-0.2.1]: https://github.com/bh-healthcare/bh-sentinel/compare/ml-v0.2.0...ml-v0.2.1
[ml-0.2.0]: https://github.com/bh-healthcare/bh-sentinel/releases/tag/ml-v0.2.0
[0.1.1]: https://github.com/bh-healthcare/bh-sentinel/compare/v0.1.0...core-v0.1.1
[0.1.0]: https://github.com/bh-healthcare/bh-sentinel/releases/tag/v0.1.0
