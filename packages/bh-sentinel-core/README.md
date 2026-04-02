# bh-sentinel-core

**Pattern-based clinical safety signal detection for behavioral health systems.**

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
- **Flag Taxonomy** -- 37 flags across 6 clinical domains (self-harm, harm to others, medication, substance use, clinical deterioration, protective factors)
- **Text Preprocessor** -- sentence splitting, normalization, and character offset tracking
- **Negation Detector** -- "denies SI", "no suicidal ideation" handling
- **Temporal Detector** -- past vs. present tense detection ("used to cut" vs. "is cutting")
- **Emotion Lexicon** -- NRC Emotion Lexicon for word-level emotion scoring
- **Pipeline** -- orchestrator that runs all layers and produces structured results

## Quick Start

```python
from bh_sentinel.core import PatternMatcher, FlagTaxonomy
from bh_sentinel.core.preprocessor import TextPreprocessor

taxonomy = FlagTaxonomy.load_default()
matcher = PatternMatcher.from_default_config()
preprocessor = TextPreprocessor()

text = "Patient reports not sleeping for 3 days and states she stopped taking her Lexapro."
preprocessed = preprocessor.process(text)
results = matcher.analyze(preprocessed)

for flag in results.flags:
    print(f"[{flag.severity}] {flag.name} (confidence: {flag.confidence})")
```

## Documentation

See the [main repository](https://github.com/bh-healthcare/bh-sentinel) for full documentation, architecture details, and the flag taxonomy reference.

## License

Apache License 2.0. See [LICENSE](../../LICENSE) for details.
