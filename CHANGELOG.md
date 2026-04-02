# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-01

### Added

- Initial repository scaffold with monorepo structure
- `bh-sentinel-core` package skeleton (pattern matcher, rules engine, taxonomy,
  preprocessor, negation detector, temporal detector, emotion lexicon, pipeline)
- `bh-sentinel-ml` package skeleton (transformer, zero-shot, export)
- Flag taxonomy configuration (6 domains, 37 flags)
- Pattern library YAML format with example patterns
- Rules engine JSON configuration format
- Emotion lexicon JSON format
- Documentation placeholders (architecture, flag taxonomy, deployment guide,
  FDA CDS analysis, training guide, pattern library)
- AWS Lambda deployment stub with Dockerfile
- Terraform directory structure for VPC + API Gateway
- Training pipeline stubs (prepare_data, train, evaluate, export)
- GitHub Actions CI workflow for lint + test
- Contributing guidelines

[Unreleased]: https://github.com/bh-healthcare/bh-sentinel/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/bh-healthcare/bh-sentinel/releases/tag/v0.1.0
