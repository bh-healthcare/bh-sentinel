# Pattern Library

Guide for authoring and extending the bh-sentinel pattern library.

## Overview

The pattern library (`config/patterns.yaml`) contains regex patterns mapped to clinical safety flags. Patterns are compiled at load time and matched against preprocessed clinical text with negation and temporal awareness.

## Pattern Format

Each pattern entry includes:

- `flag_id` -- The flag this pattern detects (must exist in `config/flag_taxonomy.json`)
- `domain` -- The clinical domain
- `patterns` -- List of regex patterns (case-insensitive by default)
- `context_hint` -- Human-readable description of the match category (never contains PHI)

## Authoring Guidelines

- Patterns should target clinical language, not social media vernacular
- Test patterns against both positive and negative examples
- Include negation-aware test cases ("denies SI" should not trigger SH flags)
- Include temporal test cases ("history of attempt" vs. "planning an attempt")

<!-- TODO: Expand with detailed examples, testing methodology, and contribution workflow for pattern additions. -->
