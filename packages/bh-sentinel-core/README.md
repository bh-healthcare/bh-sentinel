# bh-sentinel-core

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/bh-sentinel-core?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/bh-sentinel-core)

**Clinical decision support: pattern-based safety signal detection for behavioral health. Not a diagnostic tool.**

> **Clinical Use Notice:** bh-sentinel is clinical decision support software. It is not a diagnostic tool, not FDA-cleared, and not a substitute for clinical judgment. All outputs are signals for clinician review. See [CLINICAL_DISCLAIMER.md](../../CLINICAL_DISCLAIMER.md).

bh-sentinel-core is the foundational library of the [bh-sentinel](https://github.com/bh-healthcare/bh-sentinel) project. It provides deterministic, regex-based safety signal detection with configurable rules, negation handling, temporal awareness, and an emotion lexicon.

## Installation

```bash
pip install bh-sentinel-core
```

## Dependencies

Minimal: `pydantic`, `pyyaml`. No ML libraries required.

## What's Included

- **Pattern Matcher** -- compiled regex engine with negation and temporal awareness
- **Rules Engine** -- configurable severity escalation, de-escalation, and compound risk detection
- **Flag Taxonomy** -- 40 flags across 6 clinical domains (self-harm, harm to others, medication, substance use, clinical deterioration, protective factors)
- **Text Preprocessor** -- sentence splitting, normalization, and character offset tracking
- **Negation Detector** -- "denies SI", "no suicidal ideation" handling
- **Temporal Detector** -- past vs. present tense detection ("used to cut" vs. "is cutting")
- **Emotion Lexicon** -- project-owned behavioral health lexicon with 11-category density scoring
- **Pipeline** -- orchestrator that runs all layers and produces structured results

## Quick Start

```python
from bh_sentinel.core import Pipeline

pipeline = Pipeline()
result = pipeline.analyze_sync(
    "Patient reports suicidal ideation for the past two days. "
    "Stopped taking my medication last week."
)

for flag in result.flags:
    print(f"[{flag.severity}] {flag.name} (confidence: {flag.confidence})")

print(f"Immediate review: {result.summary.requires_immediate_review}")
print(f"Recommended: {result.summary.recommended_action}")
```

## Adding the ML Layer

For semantic classification on top of pattern matching, install the
companion [`bh-sentinel-ml`](https://pypi.org/project/bh-sentinel-ml/)
package and enable Layer 2:

```bash
pip install bh-sentinel-ml
```

```python
from bh_sentinel.core import Pipeline

pipeline = Pipeline(enable_transformer=True)  # auto-downloads pinned model on first run
```

`bh-sentinel-core` remains installable and usable on its own -- it has no
runtime dependency on `onnxruntime`, `tokenizers`, or `huggingface_hub`.
The ml imports are lazy-loaded only when `enable_transformer=True`.

## Documentation

See the [main repository](https://github.com/bh-healthcare/bh-sentinel)
for full documentation, architecture details, flag taxonomy reference,
and the [release process](../../docs/release-process.md).

## License

Apache License 2.0. See [LICENSE](../../LICENSE) for details.
