# Contributing to bh-sentinel

Thank you for your interest in contributing to bh-sentinel. This document covers
the process for submitting changes and the standards we follow.

## Getting Started

1. Fork the repository
2. Clone your fork and create a feature branch from `main`
3. Install both packages in editable mode:

```bash
make install-all
```

## Development Workflow

### Code Standards

- Python 3.11+ compatibility required
- All code must pass `ruff check` and `ruff format`
- Type annotations are expected on all public APIs
- Run the full lint and test suite before submitting:

```bash
make lint
make test
```

### Tests

- Every new public API must have corresponding test coverage
- Run tests with: `make test-core` or `make test-ml`
- Do not modify test fixtures to make failing tests pass -- fix the code

### Commit Messages

- Use clear, descriptive commit messages
- Reference issue numbers where applicable

## Clinical Detection Logic Changes

**All contributions that modify clinical detection logic require clinician review
before merging.** This includes changes to:

- Pattern library (`config/patterns.yaml`)
- Flag taxonomy (`config/flag_taxonomy.json`)
- Rules engine configuration (`config/rules.json`)
- Negation or temporal detection logic
- Severity escalation / de-escalation rules

If you are a clinician willing to review PRs, please note that in your
contributor profile or PR comment.

## Areas Where Contributions Are Valuable

- **Pattern library expansion:** Additional patterns for existing flag
  categories, especially colloquial and culturally diverse expressions of
  distress
- **Clinical review:** Clinicians who can review and validate pattern libraries
  and flag taxonomy
- **Language support:** Pattern libraries for languages other than English
- **Evaluation datasets:** Ground truth test fixtures for model evaluation

## Namespace Package Constraint

`bh-sentinel-core` and `bh-sentinel-ml` share the `bh_sentinel` namespace using
PEP 420 implicit namespace packages. **Never add a `bh_sentinel/__init__.py` file
in either package.** Adding one will break imports for the other package. The
namespace directory must remain empty (no `__init__.py`) so both sub-packages
can coexist when installed independently or together.

## PHI Safety

- Never include real patient data in tests, examples, or documentation
- All test fixtures must use synthetic, non-identifiable text
- Error messages must be sanitized -- no PHI in logs

## License

By contributing, you agree that your contributions will be licensed under the
Apache License 2.0, consistent with the project license.
