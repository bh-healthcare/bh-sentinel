# bh-sentinel

**Open-source clinical safety signal detection for behavioral health systems.**

> Text in. Flags out. Clinician decides.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Status:** Architecture complete. Core implementation in progress. Not yet production-ready.

---

## What This Is

bh-sentinel is a multi-layer NLP pipeline that analyzes unstructured clinical text (intake notes, journal entries, session notes, patient communications, etc.) and returns structured safety flags with severity, confidence, and evidence context.

It detects clinical concern categories across six domains: self-harm and suicidal ideation, harm to others, medication non-adherence, substance use, clinical deterioration, and protective factors. It is built specifically for behavioral health workflows, not adapted from general-purpose sentiment tools.

bh-sentinel is a clinical decision support tool. It flags possible signals and prompts clinician review. It does not diagnose, prescribe, or take autonomous clinical action.

## What This Is Not

bh-sentinel is not a diagnostic tool. It is not a replacement for clinical judgment, validated screening instruments, or the therapeutic relationship. It does not process images, audio, or video. It does not store patient data. It is not FDA-cleared or approved. It is not intended for use as the sole basis for any clinical decision.

## Why This Exists

Behavioral health is one of the most operationally complex and technologically underserved sectors of US healthcare. Clinical safety monitoring in PHP/IOP (Partial Hospitalization and Intensive Outpatient) programs relies almost entirely on clinician observation and manual review. Critical signals in intake notes, patient communications, and journal entries can be missed, especially at scale.

There is no production-ready, open-source tool that does multi-domain behavioral health safety signal detection on clinical text. What exists in the research community is either binary (suicidal / not suicidal), built entirely on social media ML models with no deterministic safety layer, or locked inside proprietary systems. None of it is packaged for deployment in a HIPAA-regulated healthcare environment. bh-sentinel combines clinically-informed pattern matching with ML classification and ships with a fine-tuning pipeline so organizations can train on their own clinical notes for higher accuracy.


## Design Principles

- **Text in, flags out.** Stateless, ephemeral processing. No text is stored. No PHI is persisted.
- **Clinician-in-the-loop.** Every output is a signal for clinical review, never an autonomous action. The system prompts validated assessments (C-SSRS, PHQ-9), not its own.
- **Evidence, not black boxes.** Every flag includes a confidence score, a human-readable basis description, and evidence span offsets so the clinician can locate and review the relevant text.
- **HIPAA-safe by architecture.** Designed for deployment inside a VPC with zero data egress. All inference runs in-process. No clinical text leaves the processing boundary.
- **Extensible.** Flag taxonomy, pattern library, and rules engine are all configuration-driven. Clinical teams can add flags and modify detection logic without code changes.
- **Gracefully degradable.** Optional layers (like AWS Comprehend sentiment) can fail or be disabled without breaking the pipeline.
- **FDA CDS-aligned.** Designed to satisfy the four criteria for Non-Device Clinical Decision Support under the 21st Century Cures Act. Every flag includes a basis description and evidence context so the clinician can independently review the recommendation. See `docs/fda-cds-analysis.md` for the full analysis.

## Packages

bh-sentinel is a monorepo containing two independently installable Python packages.

### bh-sentinel-core

The foundational library. Pattern matching, flag taxonomy, rules engine, text preprocessing, negation detection, temporal detection, and emotion lexicon.

```bash
pip install bh-sentinel-core
```

**Dependencies:** Minimal. pydantic, pyyaml. No ML libraries required.

**Use this when:** You want deterministic, regex-based safety signal detection with configurable rules. This is the fast, lightweight, high-precision layer.

### bh-sentinel-ml

The machine learning layer. ONNX-based transformer inference, zero-shot classification, model export tooling. Catches what patterns miss: implied distress, subtle language shifts, contextual meaning.

```bash
pip install bh-sentinel-ml
```

**Dependencies:** bh-sentinel-core (auto-installed), onnxruntime, tokenizers.

**Use this when:** You want contextual, transformer-based detection in addition to pattern matching.

## Quick Start

> The following examples show the target API. Implementation is in progress.

### Pattern Matching Only (bh-sentinel-core)

```python
from bh_sentinel.core import PatternMatcher, FlagTaxonomy, TextPreprocessor

# Load default patterns and taxonomy
taxonomy = FlagTaxonomy.load_default()
matcher = PatternMatcher.from_default_config()
preprocessor = TextPreprocessor()

text = "Patient reports not sleeping for 3 days and states she stopped taking her Lexapro two weeks ago."
preprocessed = preprocessor.process(text)

# Detect flags
results = matcher.analyze(preprocessed)

for flag in results.flags:
    print(f"[{flag.severity}] {flag.name} (confidence: {flag.confidence})")
    print(f"  Basis: {flag.basis_description}")
    print(f"  Span: sentence {flag.evidence_span.sentence_index}, "
          f"chars {flag.evidence_span.char_start}-{flag.evidence_span.char_end}")
```

### Full Pipeline (bh-sentinel-core + bh-sentinel-ml)

```python
from bh_sentinel.core import Pipeline, AnalysisConfig
from bh_sentinel.ml import TransformerClassifier

# Build pipeline with all layers
pipeline = Pipeline(
    enable_patterns=True,
    enable_transformer=True,
    enable_emotion_lexicon=True,
    transformer=TransformerClassifier.from_default_model(),
)

result = pipeline.analyze(
    text="Patient expresses feeling like a burden to her family and questions the point of continuing treatment.",
    config=AnalysisConfig(
        domains=["self_harm", "clinical_deterioration", "protective_factors"],
        min_severity="MEDIUM",
    ),
)

print(f"Max severity: {result.summary.max_severity}")
print(f"Requires immediate review: {result.summary.requires_immediate_review}")
for flag in result.flags:
    print(f"  [{flag.severity}] {flag.flag_id}: {flag.name}")
    print(f"    {flag.basis_description}")
```

## Flag Taxonomy

bh-sentinel organizes clinical safety signals into six domains, each with specific flags and severity levels.

### Domains

| Domain | Flags | Description |
|---|---|---|
| `self_harm` | SH-001 through SH-008 | Passive death wish, active suicidal ideation (nonspecific, with method, with plan, with intent), preparatory behavior, self-injury, history of attempt |
| `harm_to_others` | HO-001 through HO-006 | Homicidal ideation (nonspecific, specific target, with plan), violent urges, abuse disclosure (perpetrating, experiencing) |
| `medication` | MED-001 through MED-005 | Non-adherence (stated, implied), medication misuse, adverse reaction, desire to discontinue |
| `substance_use` | SU-001 through SU-005 | Active use, relapse, escalation, withdrawal, risky behavior |
| `clinical_deterioration` | CD-001 through CD-008 | Hopelessness, isolation, sleep disruption, appetite disruption, psychotic symptoms, dissociation, panic/anxiety, mania indicators |
| `protective_factors` | PF-001 through PF-005 | Treatment engagement, social support, future orientation, coping skills usage, medication adherence |

The self_harm flag taxonomy is informed by the Columbia Suicide Severity Rating Scale (C-SSRS). The full taxonomy is defined in `config/flag_taxonomy.json` and is versioned for backward compatibility.

### Severity Levels

| Level | Meaning | Expected Response |
|---|---|---|
| CRITICAL | Imminent risk to self or others | Immediate provider notification |
| HIGH | Significant clinical concern | Same-day provider review |
| MEDIUM | Clinically relevant, not urgent | Flag for next session review |
| LOW | Informational | Log for trend analysis |
| POSITIVE | Protective factor detected | Log for treatment progress |

### Validated Instruments

bh-sentinel does not replace validated clinical instruments. When CRITICAL self_harm flags are detected, the recommended workflow is to prompt the clinician to administer the C-SSRS. PHQ-9 and GAD-7 remain part of measurement-based care. bh-sentinel supplements these instruments with free-text signal detection. It does not replace them.

## Architecture

bh-sentinel uses a multi-layer pipeline where layers run in parallel for sub-second latency.

```
[Input Text]
     │
     ▼
[Text Preprocessor]  ── sentence splitting, normalization, offset tracking
     │
     ├──────────────────────────────┐
     │         asyncio.gather       │
     │         (parallel exec)      │
     ▼              ▼               ▼
 [Layer 1]     [Layer 2]       [Layer 3]
 Pattern       Transformer     Emotion
 Matching      (ONNX,          Lexicon
 (~5ms)        in-process)     (~2ms)
               (~50ms)
     │              │               │
     └──────────────┼───────────────┘
                    ▼
              [Layer 4]
              Rules Engine (~3ms)
                    │
                    ▼
            [Structured Response]
            flags + severity + confidence +
            evidence_spans + basis_descriptions
```

**Layer 1 (Pattern Matching):** Deterministic regex/keyword detection. High precision for explicit clinical language. Handles negation ("denies SI") and temporal qualifiers ("used to cut"). Returns character-level evidence spans. This is the safety net for CRITICAL flags.

**Layer 2 (Transformer Classification):** ONNX-quantized transformer model running entirely in-process. No network hop, no external API. Catches implied distress and contextual signals that patterns miss. Supports zero-shot classification (no training data needed) and fine-tuned models.

**Layer 3 (Emotion Lexicon):** In-process NRC Emotion Lexicon for word-level emotion scoring. Maps text to emotion categories: sadness, anger, fear, anxiety, hopelessness, guilt, shame, and others. Fast (~2ms), no ML dependency.

**Layer 4 (Rules Engine):** Combines signals from Layers 1-3 into final flag determinations. Handles severity escalation (e.g., passive SI + strong hopelessness = CRITICAL), de-escalation (historical references), compound risk detection (co-occurring substance use + self-harm), and recommended actions (e.g., "Administer C-SSRS").

## API Response

Every analysis returns a structured response with full evidence context.

```json
{
  "request_id": "uuid-v4",
  "processing_time_ms": 142,
  "taxonomy_version": "1.0.0",
  "flags": [
    {
      "flag_id": "SH-001",
      "domain": "self_harm",
      "name": "Passive death wish",
      "severity": "HIGH",
      "confidence": 0.92,
      "detection_layer": "pattern_match",
      "matched_context_hint": "passive death wish language",
      "basis_description": "Pattern match detected language consistent with wish to be dead or not alive.",
      "evidence_span": {
        "sentence_index": 2,
        "char_start": 45,
        "char_end": 89
      }
    }
  ],
  "protective_factors": [],
  "summary": {
    "max_severity": "HIGH",
    "total_flags": 1,
    "domains_flagged": ["self_harm"],
    "requires_immediate_review": true,
    "recommended_action": "Administer C-SSRS"
  },
  "pipeline_status": {
    "layer_1_pattern": "completed",
    "layer_2_transformer": "completed",
    "layer_3_comprehend": "not_run",
    "layer_3_emotion_lexicon": "completed",
    "layer_4_rules": "completed"
  }
}
```

Key design decisions:

- **matched_context_hint** describes the category of concern, never a verbatim quote from the input text.
- **basis_description** explains why the flag was raised in human-readable language, supporting clinician independent review (FDA CDS Criterion 4).
- **evidence_span** provides integer offsets (sentence index, character start/end) so upstream systems can highlight the relevant region without bh-sentinel exposing raw text.
- **recommended_action** suggests validated clinical workflows (e.g., C-SSRS), never autonomous directives.

## Repository Structure

```
bh-sentinel/
├── packages/
│   ├── bh-sentinel-core/              # PyPI package
│   │   ├── src/bh_sentinel/core/
│   │   │   ├── pattern_matcher.py     # Layer 1: compiled regex, negation, temporal
│   │   │   ├── rules_engine.py        # Layer 4: business logic rules
│   │   │   ├── taxonomy.py            # Flag taxonomy definitions
│   │   │   ├── preprocessor.py        # Text normalization, sentence splitting
│   │   │   ├── negation_detector.py   # "NOT suicidal" handling
│   │   │   ├── temporal_detector.py   # Past vs present tense detection
│   │   │   ├── emotion_lexicon.py     # NRC emotion lexicon (in-process)
│   │   │   ├── pipeline.py            # Orchestrator
│   │   │   └── models/                # Pydantic request/response models
│   │   ├── pyproject.toml
│   │   └── README.md
│   └── bh-sentinel-ml/                # PyPI package
│       ├── src/bh_sentinel/ml/
│       │   ├── transformer.py         # ONNX Runtime inference
│       │   ├── zero_shot.py           # Zero-shot NLI classification
│       │   └── export.py              # Model export tooling
│       ├── pyproject.toml
│       └── README.md
├── config/
│   ├── patterns.yaml                  # Default pattern library (~150-200 patterns)
│   ├── rules.json                     # Default rules engine configuration
│   ├── flag_taxonomy.json             # Versioned flag taxonomy
│   └── emotion_lexicon.json           # NRC Emotion Lexicon data
├── deployment/
│   ├── aws-lambda/                    # Reference: Lambda container deployment
│   │   ├── handler.py
│   │   ├── Dockerfile
│   │   └── README.md
│   └── terraform/                     # Reference: AWS infrastructure (VPC, API GW, etc.)
│       ├── modules/
│       └── environments/
├── training/
│   ├── prepare_data.py                # Dataset preparation scripts
│   ├── train.py                       # Fine-tuning pipeline
│   ├── evaluate.py                    # Model evaluation harness
│   └── export.py                      # Export to ONNX + INT8 quantization
├── docs/
│   ├── architecture.md                # Full architecture document
│   ├── flag-taxonomy.md               # Detailed flag definitions
│   ├── deployment-guide.md            # AWS Lambda deployment walkthrough
│   ├── fda-cds-analysis.md            # FDA CDS compliance analysis
│   ├── training-guide.md              # Model fine-tuning instructions
│   └── pattern-library.md             # Pattern authoring guide
├── LICENSE                            # Apache 2.0
├── CONTRIBUTORS.md
├── CHANGELOG.md
└── README.md                          # This file
```

## Deployment

bh-sentinel is designed to run anywhere Python runs. The `packages/` are pure Python and work on any platform.

For production healthcare deployments, the `deployment/` directory provides a reference architecture for AWS Lambda with the following properties:

- **Zero data egress:** Lambda runs in a private VPC subnet with no internet access. All AWS service calls route through VPC endpoints.
- **Sub-second latency:** Pattern matching + ONNX transformer + rules engine complete in ~60-120ms on warm invocations.
- **No PHI persistence:** Text is processed in-memory and discarded. Only flag metadata is logged.
- **Cost-effective:** ~$50-115/month for the full stack (Lambda + API Gateway + VPC endpoints + S3 config). No SageMaker. No GPU instances.

See `docs/deployment-guide.md` for the full walkthrough.

## Training Your Own Model

The default transformer model uses zero-shot classification (no training data needed). For higher accuracy on your organization's clinical notes, bh-sentinel supports fine-tuning.

### Public Datasets for Pre-Training

| Dataset | Source | Labels | Size | Access |
|---|---|---|---|---|
| Reddit C-SSRS Suicide Dataset | Zenodo (10.5281/zenodo.2667859) | C-SSRS categories (Supportive, Ideation, Behavior, Attempt) | 500 users, psychiatrist-annotated | Open access |
| UMD Reddit Suicidality Dataset v2 | University of Maryland | No/Low/Moderate/Severe risk | ~1,200 users, expert-annotated | Application + DUA |
| Mental Health Sentiment (Kaggle) | Reddit, Twitter | 7 categories (Normal, Depression, Suicidal, Anxiety, Stress, Bipolar, Personality Disorder) | 52,681 texts | Open access |
| SMHD | Georgetown University | 9 conditions (ADHD, Anxiety, Autism, Bipolar, Depression, Eating Disorder, OCD, PTSD, Schizophrenia) | Large-scale | Application + DUA |

**Important:** These datasets are social-media-sourced, not clinical notes. They are useful for teaching the model behavioral health vocabulary and concept space. Accuracy on clinical documentation improves substantially with fine-tuning on your own de-identified clinical notes (even 200-300 annotated notes make a meaningful difference).

See `docs/training-guide.md` and the `training/` directory for the full pipeline.

## FDA CDS Considerations

bh-sentinel is designed to satisfy the four criteria for Non-Device Clinical Decision Support under Section 520(o)(1)(E) of the FD&C Act (21st Century Cures Act). The architecture specifically addresses Criterion 4 (clinician independent review) through the `basis_description`, `evidence_span`, and `detection_layer` fields in every flag response.

This is a clinical decision support tool, not a diagnostic device. It flags signals and prompts clinician action. It does not diagnose, prescribe, or make autonomous clinical decisions.

See `docs/fda-cds-analysis.md` for the full regulatory analysis.

**Note:** Organizations deploying bh-sentinel should review the FDA analysis with their own legal counsel, particularly if extending the tool to patient-facing applications.

## Relationship to bh-healthcare

bh-sentinel is part of the [bh-healthcare](https://github.com/bh-healthcare) open-source organization, which publishes infrastructure, standards, and tooling for behavioral health technology systems. Related projects:

| Project | Description |
|---|---|
| [bh-audit-schema](https://github.com/bh-healthcare/bh-audit-schema) | Canonical audit event standard for behavioral health systems, with HIPAA/SOC 2/42 CFR Part 2 compliance mappings |
| [bh-fastapi-audit](https://github.com/bh-healthcare/bh-fastapi-audit) | FastAPI middleware for compliant audit event emission |
| [bh-audit-logger](https://github.com/bh-healthcare/bh-audit-logger) | Generic audit event emitter |

bh-sentinel can emit audit events through bh-audit-logger for compliance tracking of safety signal detection activities.

## Roadmap

### Current (v0.1)
- [ ] Core pattern matching engine with negation and temporal detection
- [ ] Flag taxonomy v1.0 (37 flags across 6 domains)
- [ ] Rules engine with configurable severity escalation
- [ ] Text preprocessor with sentence splitting and offset tracking
- [ ] NRC Emotion Lexicon integration
- [ ] Default pattern library (~150-200 patterns)

### Next (v0.2)
- [ ] ONNX transformer inference layer (bh-sentinel-ml)
- [ ] Zero-shot classification with DistilBART-MNLI
- [ ] Model evaluation harness with ground truth fixtures
- [ ] AWS Lambda reference deployment
- [ ] Terraform modules for VPC + API Gateway

### Future
- [ ] Fine-tuning pipeline with public dataset preparation
- [ ] Spanish language pattern support
- [ ] Flag trend analysis utilities
- [ ] Clinician feedback integration (dismissed flags feed back into training)
- [ ] FHIR-compatible output format
- [ ] Group session note analysis patterns

## Contributing

Contributions are welcome. See `CONTRIBUTING.md` for guidelines.

Areas where contributions are particularly valuable:
- **Pattern library expansion:** Additional patterns for existing flag categories, especially colloquial and culturally diverse expressions of distress
- **Clinical review:** Clinicians who can review and validate pattern libraries and flag taxonomy
- **Language support:** Pattern libraries for languages other than English
- **Evaluation datasets:** Ground truth test fixtures for model evaluation

All contributions that modify clinical detection logic (patterns, flag taxonomy, rules) should be reviewed by a qualified behavioral health clinician before merging.

## Citation

If you use bh-sentinel in research or clinical implementations, please cite:

```bibtex
@software{bh_sentinel,
  title = {bh-sentinel: Open-Source Clinical Safety Signal Detection for Behavioral Health Systems},
  author = {Kumar, Tanmaya},
  year = {2026},
  url = {https://github.com/bh-healthcare/bh-sentinel},
  license = {Apache-2.0}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

---

**bh-sentinel is clinical decision support software. It is not a diagnostic tool, not a substitute for clinical judgment, and not FDA-cleared or approved. Organizations deploying this software in clinical settings are responsible for their own clinical validation, regulatory compliance, and patient safety protocols.**
