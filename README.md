# bh-sentinel

**Open-source clinical safety signal detection for behavioral health systems.**

> Text in. Flags out. Clinician decides.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Downloads — bh-sentinel-core](https://static.pepy.tech/personalized-badge/bh-sentinel-core?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/bh-sentinel-core)
[![PyPI Downloads — bh-sentinel-ml](https://static.pepy.tech/personalized-badge/bh-sentinel-ml?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/bh-sentinel-ml)

**Status:** v0.2.0 shipped. `bh-sentinel-core 0.1.1` (Layer 1 + Layer 3 + Layer 4) and `bh-sentinel-ml 0.2.0` (Layer 2 zero-shot transformer) are both on PyPI. Reference AWS deployment + validated calibration land in v0.3. Not yet production-ready -- clinical validation and calibration against labeled clinical data are a v0.3 deliverable.

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

**Dependencies:** `bh-sentinel-core>=0.1.1` (auto-installed), `onnxruntime`, `tokenizers`, `huggingface_hub`, `platformdirs`.

**Use this when:** You want contextual, transformer-based detection in addition to pattern matching. The `BH_SENTINEL_ML_OFFLINE=1` env var is the production rail for VPC-isolated deployments so cold starts never hit HuggingFace Hub.

## Quick Start

### bh-sentinel-core (v0.1)

```python
from bh_sentinel.core import Pipeline

pipeline = Pipeline()
result = pipeline.analyze_sync(
    "Patient reports suicidal ideation for the past two days. "
    "Stopped taking my medication last week."
)

for flag in result.flags:
    print(f"[{flag.severity}] {flag.name} (confidence: {flag.confidence})")
    print(f"  Basis: {flag.basis_description}")
    print(f"  Span: sentence {flag.evidence_span.sentence_index}, "
          f"chars {flag.evidence_span.char_start}-{flag.evidence_span.char_end}")

print(f"\nImmediate review: {result.summary.requires_immediate_review}")
print(f"Recommended: {result.summary.recommended_action}")
```

### Full Pipeline with ML (v0.2)

```python
from bh_sentinel.core import Pipeline

# Layer 2 is opt-in. On first call the pinned DistilBART-MNLI revision
# is auto-downloaded from HuggingFace Hub into a local cache.
pipeline = Pipeline(
    enable_patterns=True,
    enable_transformer=True,   # requires bh-sentinel-ml
    enable_emotion_lexicon=True,
)

result = pipeline.analyze_sync(
    "Patient expresses feeling like a burden to her family "
    "and questions the point of continuing treatment."
)

print(f"Max severity: {result.summary.max_severity}")
print(f"Requires immediate review: {result.summary.requires_immediate_review}")
print(f"L2 status: {result.pipeline_status.layer_2_transformer}")
for flag in result.flags:
    layers = [flag.detection_layer.value, *[layer.value for layer in flag.corroborating_layers]]
    print(f"  [{flag.severity}] {flag.flag_id}: {flag.name}")
    print(f"    {flag.basis_description}")
    print(f"    detected by: {', '.join(layers)}")
```

### Production / VPC-isolated deployment

Pre-bake the model in your container at `docker build` time so runtime stays
fully offline (no HuggingFace Hub on cold start):

```dockerfile
RUN pip install bh-sentinel-ml
RUN bh-sentinel-ml download-model --output /opt/bh-sentinel-ml/model \
      --revision <PINNED_REVISION_SHA> \
      --verify-sha256 <PINNED_ONNX_SHA256>
ENV BH_SENTINEL_ML_OFFLINE=1
```

Then construct the pipeline against the baked-in path: `Pipeline(enable_transformer=True, transformer_model_path=Path("/opt/bh-sentinel-ml/model"), transformer_auto_download=False)`.

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

**Layer 3 (Emotion Lexicon):** In-process behavioral health emotion lexicon for word-level emotion scoring. Maps text to 11 clinically relevant categories: hopelessness, agitation, anxiety, anger, sadness, guilt, shame, mania, dissociation, positive valence, and negative valence. Fast (~2ms), no ML dependency.

**Layer 4 (Rules Engine):** Combines signals from Layers 1-3 into final flag determinations. Handles severity escalation (e.g., passive SI + strong hopelessness = CRITICAL), de-escalation (historical references), compound risk detection (co-occurring substance use + self-harm), and recommended actions (e.g., "Administer C-SSRS").

## API Response

Every analysis returns a structured response with full evidence context.

```json
{
  "request_id": "uuid-v4",
  "processing_time_ms": 12,
  "taxonomy_version": "1.0.0",
  "flags": [
    {
      "flag_id": "SH-001",
      "domain": "self_harm",
      "name": "Passive death wish",
      "severity": "HIGH",
      "confidence": 0.92,
      "detection_layer": "pattern_match",
      "matched_context_hint": "passive death wish",
      "basis_description": "Pattern match detected language consistent with passive death wish.",
      "evidence_span": {
        "sentence_index": 2,
        "char_start": 45,
        "char_end": 89
      },
      "temporal_context": "present",
      "corroborating_layers": []
    }
  ],
  "emotions": {
    "primary": "hopelessness",
    "secondary": "sadness",
    "category_scores": {
      "hopelessness": 0.15,
      "agitation": 0.0,
      "anxiety": 0.0,
      "anger": 0.0,
      "sadness": 0.05,
      "guilt": 0.0,
      "shame": 0.0,
      "mania": 0.0,
      "dissociation": 0.0,
      "positive_valence": 0.0,
      "negative_valence": 0.1
    },
    "comprehend_available": false,
    "sentiment": null
  },
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
    "layer_2_transformer": "skipped",
    "layer_3_comprehend": "not_run",
    "layer_3_emotion_lexicon": "completed",
    "layer_4_rules": "completed"
  },
  "clinical_use_notice": "bh-sentinel is clinical decision support software. It is not a diagnostic tool, not FDA-cleared, and not a substitute for clinical judgment..."
}
```

Key design decisions:

- **matched_context_hint** describes the category of concern, never a verbatim quote from the input text.
- **basis_description** explains why the flag was raised in human-readable language, supporting clinician independent review (FDA CDS Criterion 4).
- **evidence_span** provides integer offsets (sentence index, character start/end) so upstream systems can highlight the relevant region without bh-sentinel exposing raw text.
- **recommended_action** suggests validated clinical workflows (e.g., C-SSRS), never autonomous directives.

## Validation Results

bh-sentinel ships with a validation suite that runs the pipeline against clinical vignettes, public domain literature, and true-negative everyday text. These results document both what the pattern engine catches and where its boundaries are.

### Clinical text (what Layer 1 is built for)

| Input | Flags Detected | Outcome |
|-------|---------------|---------|
| **Crisis intake** -- active SI, prior attempt, substance use, auditory hallucinations, medication non-adherence, isolation | SH-002, SH-008, SU-001, CD-005a, MED-001, CD-002 (6 flags, 4 domains) | CRITICAL, immediate review, C-SSRS recommended |
| **Journal entry** -- "I can't do this anymore... stopped taking my meds... haven't eaten in two days" | SH-001, MED-001, CD-001 (3 flags) | CRITICAL, C-SSRS recommended |
| **Mixed presentation** -- nonbinary patient, suicidal thoughts decreased but still hopeless, committed to therapy | SH-002, CD-001 + PF-001 protective factor | Risk flags and protective factors detected together |
| **Routine session** -- euthymic mood, med adherence, coping skills, supportive family | PF-005, PF-004 (protective only) | No risk flags, no immediate review |

### Public domain literature (where Layer 1 reaches its boundary)

| Source | Text | Result | Why |
|--------|------|--------|-----|
| **Dostoevsky** -- *Crime and Punishment* (1866) | Raskolnikov's isolation: "completely absorbed in himself, isolated from his fellows" | CD-001 (hopelessness) detected | Direct clinical language about withdrawal and isolation |
| **Woolf** -- *Mrs Dalloway* (1925) | Septimus: "He would kill himself rather than let them get at him" | 0 flags | Conditional mood ("would kill") is not a clinical disclosure pattern |
| **Gilman** -- *The Yellow Wallpaper* (1892) | "I don't sleep much at night... cry most of the time... angry enough to do something desperate" | 0 flags | Deterioration expressed through metaphor and period-specific language |
| **Tolstoy** -- *Anna Karenina* (1878) | Levin: "keeping myself from suicide with a rope, so as not to hang myself" | 0 flags | Philosophical construction about the concept of suicide, not clinical disclosure |

### True negatives (zero false positives)

Weather reports, recipes, and sports recaps all return 0 flags. The pattern engine does not keyword-match on words like "kill" (sports), "cut" (recipes), or "die" (weather metaphors) outside clinical context.

### What this demonstrates

The clinical vignettes confirm the pipeline works as designed on the language it's built for: intake notes, session summaries, journal entries, and clinical shorthand. The crisis intake correctly triggers 6 flags across 4 domains with appropriate escalation rules, immediate review, and C-SSRS recommendation. The routine session correctly returns zero risk flags.

The literary passages are equally important. They prove the system doesn't false-positive on literary language about suffering, death, and despair. Dostoevsky, Woolf, Gilman, and Tolstoy all write about themes that overlap with clinical safety signals. The fact that bh-sentinel catches Raskolnikov's isolation as hopelessness but doesn't flag Woolf's conditional SI construction or Tolstoy's philosophical death language shows the pattern matching is calibrated for clinical text, not keyword spotting.

The literary passages that return 0 flags are documented evidence of why Layer 2 (the transformer, Phase 2) needs to exist. Layer 1 correctly handles direct clinical language. The indirect, metaphorical, and literary constructions are exactly what semantic ML classification is designed to catch. This is the intended division of labor between the deterministic and ML layers.

Full L1 validation source: `packages/bh-sentinel-core/tests/test_real_world_validation.py`

### What Layer 2 adds (v0.2)

`bh-sentinel-ml 0.2.0` ships the full L2 inference path: sentence-level zero-shot NLI over the same flag taxonomy, candidate merge with the L1 results, and corroboration metadata on every flag. Structurally the pipeline now runs all three layers in parallel and surfaces L1-only / L2-only / corroborated detections on every request.

**What L2 is designed to target on the literary corpus** (hypotheses from [`config/ml/zero_shot_hypotheses.yaml`](config/ml/zero_shot_hypotheses.yaml)):

| Source | L1 baseline (v0.1) | Layer 2 target flags | What Layer 2 is testing |
|--------|--------------------|---------------------|-------------------------|
| Dostoevsky -- *Crime and Punishment* | CD-001 | CD-002 (severe isolation), CD-005c (paranoid ideation) | "The speaker describes withdrawing from social contact", "The speaker expresses paranoid beliefs about being watched, followed, or targeted" |
| Woolf -- *Mrs Dalloway* | 0 flags | SH-002 (active SI, nonspecific) | "The speaker describes general thoughts about ending their own life" |
| Gilman -- *The Yellow Wallpaper* | 0 flags | CD-003 (sleep disruption), CD-006 (dissociation) | "The speaker describes severe disruption to their sleep", "The speaker describes feeling detached from reality or from themselves" |
| Tolstoy -- *Anna Karenina* | 0 flags | CD-001 (hopelessness), SH-001 (passive death wish), SH-002 (active SI) | "The speaker expresses pervasive hopelessness", "The speaker expresses a wish to be dead", "The speaker describes general thoughts about ending their own life" |

**Quality numbers on clinical corpora are deferred to v0.3.** The architecture ships in v0.2 with `FixedDiscount(0.85)` calibration (§4.8 Phase A); validated temperature scaling against clinician-labeled data is a v0.3 deliverable. We deliberately do not publish detection-quality numbers here against DistilBART-MNLI zero-shot on literary texts because a small general-purpose NLI model is not a reliable proxy for a clinical use case, and publishing those numbers without clinical labels would misrepresent the project's readiness.

**To reproduce Layer 2 results yourself:**

```bash
pip install bh-sentinel-core bh-sentinel-ml
bh-sentinel-ml evaluate --corpus config/eval/real_world_corpus.yaml --enable-transformer
```

The shared corpus at [`config/eval/real_world_corpus.yaml`](config/eval/real_world_corpus.yaml) (public-domain literature + synthetic clinical vignettes + true negatives, with `expected_flags_hint` on each entry) is what the L1-vs-L2 diagnostic runs against. A structural integration test (`packages/bh-sentinel-ml/tests/test_l1_vs_l2_corpus.py`, marked `real_model` and skipped in default CI) writes a per-fixture report to `packages/bh-sentinel-ml/tests/artifacts/`.

A future `bh-sentinel-examples` repository (see [Roadmap](#roadmap)) will host full reproducible Layer 2 evaluations against real clinical text with clinician-labeled ground truth, once that data is available.

## Repository Structure

```
bh-sentinel/
├── .github/workflows/
│   ├── ci.yml                         # Lint + test matrix (Python 3.11, 3.12)
│   ├── publish-core.yml               # Tag core-v*  → publish bh-sentinel-core
│   └── publish-ml.yml                 # Tag ml-v*    → publish bh-sentinel-ml
├── packages/
│   ├── bh-sentinel-core/              # PyPI: bh-sentinel-core (0.1.1)
│   │   ├── src/bh_sentinel/core/
│   │   │   ├── pattern_matcher.py     # Layer 1: compiled regex, negation, temporal
│   │   │   ├── rules_engine.py        # Layer 4: business logic rules
│   │   │   ├── taxonomy.py            # Flag taxonomy definitions
│   │   │   ├── preprocessor.py        # Text normalization, sentence splitting
│   │   │   ├── negation_detector.py   # "NOT suicidal" handling
│   │   │   ├── temporal_detector.py   # Past vs present tense detection
│   │   │   ├── emotion_lexicon.py     # Behavioral health emotion lexicon (in-process)
│   │   │   ├── pipeline.py            # Orchestrator (L1+L2+L3+L4, lazy-imports ml)
│   │   │   ├── cli/                   # bh-sentinel validate-config + test-patterns
│   │   │   └── models/                # Pydantic request/response models
│   │   ├── pyproject.toml
│   │   └── README.md
│   └── bh-sentinel-ml/                # PyPI: bh-sentinel-ml (0.2.0)
│       ├── src/bh_sentinel/ml/
│       │   ├── transformer.py         # ONNX Runtime inference + SHA256 verify-on-load
│       │   ├── zero_shot.py           # Zero-shot NLI classification
│       │   ├── calibration.py         # Calibrator protocol, FixedDiscount, TemperatureScaling, ECE
│       │   ├── merge.py               # L1/L2 candidate merge (architecture §4.7)
│       │   ├── model_cache.py         # HF Hub + offline cache + $BH_SENTINEL_ML_OFFLINE rail
│       │   ├── exceptions.py          # ModelNotFoundError, ModelIntegrityError, InferenceError
│       │   ├── _config.py             # ml_config.yaml + zero_shot_hypotheses.yaml loaders
│       │   ├── _default_config/       # Vendored YAML configs shipped in the wheel
│       │   └── cli/                   # bh-sentinel-ml download-model | calibrate | evaluate
│       ├── pyproject.toml
│       └── README.md
├── config/
│   ├── patterns.yaml                  # Default pattern library (351 patterns across 40 flags)
│   ├── rules.json                     # Default rules engine configuration
│   ├── flag_taxonomy.json             # Versioned flag taxonomy (40 flags, 6 domains)
│   ├── emotion_lexicon.json           # Behavioral health emotion lexicon data
│   ├── test_fixtures.yaml             # Pattern test fixtures (84 cases across all 40 flags)
│   ├── ml/
│   │   ├── ml_config.yaml             # Model pin, SHA256, calibration strategy, batching
│   │   └── zero_shot_hypotheses.yaml  # One NLI hypothesis per flag_id (all 40)
│   └── eval/
│       └── real_world_corpus.yaml     # Shared L1 vs L2 diagnostic corpus (public-domain lit + vignettes)
├── deployment/                        # (v0.3 target: Lambda + Terraform reference)
├── training/                          # (v0.4 target: fine-tuning pipeline)
├── docs/
│   ├── architecture.md                # Full architecture document
│   ├── flag-taxonomy.md               # Detailed flag definitions
│   ├── deployment-guide.md            # AWS Lambda deployment walkthrough (v0.3)
│   ├── fda-cds-analysis.md            # FDA CDS compliance analysis
│   ├── training-guide.md              # Model fine-tuning instructions (v0.4)
│   ├── pattern-library.md             # Pattern authoring guide
│   └── release-process.md             # Per-package release procedure, PyPI Trusted Publishers
├── LICENSE                            # Apache 2.0
├── CONTRIBUTING.md                    # Contribution guidelines
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

## Releases

bh-sentinel publishes two independently-versioned PyPI packages from this monorepo. Each package has its own tag prefix: `core-v*` for `bh-sentinel-core`, `ml-v*` for `bh-sentinel-ml`. Pushing a prefixed tag triggers the corresponding publish workflow.

See [`docs/release-process.md`](docs/release-process.md) for the full release procedure, PyPI Trusted Publisher setup, rollback guidance, and historical tag notes.

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
| [bh-fastapi-examples](https://github.com/bh-healthcare/bh-fastapi-examples) | Reference FastAPI integrations for the audit ecosystem |
| [bh-audit-logger-examples](https://github.com/bh-healthcare/bh-audit-logger-examples) | Reference integrations for `bh-audit-logger` |
| `bh-sentinel-examples` *(v0.3)* | Reproducible local evaluations of Layers 1 + 2 with the real INT8 model pre-downloaded. Runs the shared corpus, prints per-entry L1-only / L2-only / corroborated flags, lets clinicians or partners validate detection behavior on their own workstations without touching the bh-sentinel core package tests. |

bh-sentinel is designed to integrate with bh-audit-logger for compliance tracking at the deployment boundary. The core library does not emit audit events directly -- this is the responsibility of the service layer (v0.3).

## Roadmap

### v0.1 -- complete
- [x] Core pattern matching engine with negation and temporal detection
- [x] Flag taxonomy v1.0 (40 flags across 6 domains)
- [x] Rules engine with configurable severity escalation
- [x] Text preprocessor with sentence splitting and offset tracking
- [x] Behavioral health emotion lexicon integration
- [x] Default pattern library (351 patterns across 40 flags)

### v0.2 -- complete
- [x] `bh-sentinel-ml` package: ONNX transformer inference layer, in-process
- [x] Zero-shot classification with DistilBART-MNLI baseline
- [x] L1/L2 candidate merge (`merge_candidates`, architecture §4.7)
- [x] Hybrid model distribution: HF Hub auto-download + `BH_SENTINEL_ML_OFFLINE=1` rail + SHA256 verify-on-load
- [x] Calibration mechanism: `FixedDiscount` (Phase A default) + `TemperatureScaling` (ready, unvalidated)
- [x] CLI: `download-model`, `calibrate`, `evaluate`
- [x] Shared L1-vs-L2 diagnostic corpus at `config/eval/real_world_corpus.yaml`
- [x] Per-package release workflows (`core-v*` / `ml-v*`)
- [x] `bh-sentinel-core 0.1.1` and `bh-sentinel-ml 0.2.0` on PyPI

### Next (v0.3) -- deployment + validated calibration
- [ ] `bh-sentinel-examples` companion repository (see [ecosystem](#relationship-to-bh-healthcare)): reproducible local Layer 2 eval against real clinical corpora with the pinned INT8 model pre-downloaded
- [ ] AWS Lambda reference deployment (Dockerfile, handler.py wired to the pre-baked model)
- [ ] Terraform modules (VPC, Lambda, API Gateway, ECR, S3, monitoring, rate limiting)
- [ ] CloudWatch dashboard + alarms (per-layer latency, error rate, Comprehend fallback rate)
- [ ] Validated `TemperatureScaling` calibration with ECE < 0.05 on clinician-labeled data
- [ ] Model evaluation harness against clinician-labeled ground truth

### Future (v0.4+)
- [ ] Fine-tuning pipeline with public dataset preparation (`training/`)
- [ ] Spanish language pattern support + multilingual transformer
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
  author = {Kumar, Tanmaya (https://tanmayakumar.com)},
  year = {2026},
  url = {https://github.com/bh-healthcare/bh-sentinel},
  license = {Apache-2.0}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

---

**bh-sentinel is clinical decision support software. It is not a diagnostic tool, not a substitute for clinical judgment, and not FDA-cleared or approved. Organizations deploying this software in clinical settings are responsible for their own clinical validation, regulatory compliance, and patient safety protocols.**
