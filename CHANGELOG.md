# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for `bh-sentinel-ml 0.2.1`

- Pin a canonical ONNX export of the zero-shot baseline model with a real `model_revision` SHA and matching `model_sha256` in `config/ml/ml_config.yaml` (the v0.2.0 values are placeholders; production `auto_download=True` currently fails the verify-on-load SHA check as a result).
- Ship a `scripts/export_onnx.py` helper for users who want to re-export locally against a different base model.
- Publish the first reproducible L1-vs-L2 evaluation report against the shared corpus via the [bh-sentinel-examples](https://github.com/bh-healthcare/bh-sentinel-examples) repo.

See the Release note in [`packages/bh-sentinel-ml/README.md`](packages/bh-sentinel-ml/README.md) for the current operator workaround (use the `BH_SENTINEL_ML_OFFLINE=1` rail with a locally-exported model).

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

[Unreleased]: https://github.com/bh-healthcare/bh-sentinel/compare/ml-v0.2.0...HEAD
[ml-0.2.0]: https://github.com/bh-healthcare/bh-sentinel/releases/tag/ml-v0.2.0
[0.1.1]: https://github.com/bh-healthcare/bh-sentinel/compare/v0.1.0...core-v0.1.1
[0.1.0]: https://github.com/bh-healthcare/bh-sentinel/releases/tag/v0.1.0
