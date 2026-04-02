# bh-sentinel-ml

**Transformer-based clinical safety signal detection for behavioral health systems.**

bh-sentinel-ml is the machine learning layer of the [bh-sentinel](https://github.com/bh-healthcare/bh-sentinel) project. It provides ONNX-based transformer inference and zero-shot classification to catch what pattern matching misses: implied distress, subtle language shifts, and contextual meaning.

## Installation

```bash
pip install bh-sentinel-ml
```

This automatically installs `bh-sentinel-core` as a dependency.

## Dependencies

- `bh-sentinel-core` (auto-installed)
- `onnxruntime` -- ONNX Runtime for in-process transformer inference
- `tokenizers` -- HuggingFace tokenizers for text encoding

## What's Included

- **Transformer Classifier** -- ONNX-quantized model running entirely in-process (no network hop)
- **Zero-Shot Classifier** -- NLI-based zero-shot classification (no training data needed)
- **Model Export** -- Tooling to export and quantize models to ONNX format

## Quick Start

```python
from bh_sentinel.core import Pipeline, AnalysisConfig
from bh_sentinel.ml import TransformerClassifier

pipeline = Pipeline(
    enable_patterns=True,
    enable_transformer=True,
    transformer=TransformerClassifier.from_default_model(),
)

result = pipeline.analyze(
    text="Patient expresses feeling like a burden to her family.",
    config=AnalysisConfig(domains=["self_harm", "clinical_deterioration"]),
)
```

## Documentation

See the [main repository](https://github.com/bh-healthcare/bh-sentinel) for full documentation, architecture details, and the training guide.

## License

Apache License 2.0. See [LICENSE](../../LICENSE) for details.
