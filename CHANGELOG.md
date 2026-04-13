# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/bh-healthcare/bh-sentinel/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/bh-healthcare/bh-sentinel/releases/tag/v0.1.0
