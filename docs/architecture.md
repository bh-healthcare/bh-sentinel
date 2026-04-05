# bh-sentinel: Architecture & Research Document

**Author:** Tanmaya Kumar  
**Date:** April 01, 2026  
**Version:** 1.0  
**Status:** Architecture complete. Core implementation in progress.

---

## 1. Executive Summary

bh-sentinel is a stateless, platform-agnostic text analysis system designed to detect clinical safety flags in unstructured behavioral health text. It operates as a pure "text in, flags out" processor, accepting free-form clinical or patient text and returning a structured set of risk flags, severity scores, evidence spans, and emotion signals.

The system is designed to serve multiple input sources (EMR intake notes, mobile app journal entries, group session notes) without knowledge of or coupling to any specific upstream system. It processes text ephemerally: no text is stored, no PHI is persisted. This makes it a HIPAA-safe intermediary for sensitive clinical content.

bh-sentinel is a clinical decision support tool. It detects signals and prompts clinician review. It does not diagnose, prescribe, or take autonomous clinical action. It is not intended for use as the sole basis for any clinical decision. All outputs are intended to support the clinician's independent judgment, not replace it.

### 1.1 Hard Constraints

- **Zero data egress** - All PHI processing must run inside the deploying organization's AWS account and VPC. No clinical text is ever sent to a third-party system. AWS-native services (Comprehend) are accessed via VPC endpoints so traffic never traverses the public internet.
- **Sub-second latency** - The full pipeline (text in, flags out) must complete in under 1 second on warm invocations. Achieved through parallel layer execution and in-process model inference.
- **HIPAA-safe by design** - No PHI is persisted, logged, cached, or transmitted outside the processing boundary. Only flag metadata (IDs, severity, confidence) is ever stored or logged.
- **Clinician-in-the-loop** - bh-sentinel flags possible signals. Clinicians assess using validated instruments (C-SSRS, PHQ-9). The system never makes autonomous clinical decisions.

### 1.2 Design Philosophy

bh-sentinel is built to do three things well:

1. **Detect concern categories** across a clinically-informed flag taxonomy
2. **Return explicit evidence context and confidence** so the clinician understands the basis of each flag
3. **Trigger workflow, never autonomous clinical action** (escalation tasks, provider notifications, case queue entries)

This philosophy is informed by FDA's Clinical Decision Support guidance (see Section 8) and by the clinical literature on suicide-related CDS, which consistently supports using model output to prompt assessment rather than replace it.

---

## 2. Open-Source Landscape Assessment

### 2.1 What Exists

| Tool / Model | Type | Trained On | Limitations |
|---|---|---|---|
| **MentalBERT** (mental/mental-bert-base-uncased) | Pre-trained LM | Reddit mental health subreddits | Base model only, needs fine-tuning for classification tasks. Not trained on clinical notes. |
| **KevSun/mentalhealth_LM** | BERT classifier (0-5 severity) | Psychiatric diagnoses | 78% accuracy. Generic severity, not flag-specific. Social media text, not clinical. |
| **DunnBC22/canine-c-Mental_Health_Classification** | Binary classifier | Kaggle mental health corpus | Binary only (concern / no concern). 92% accuracy. No flag granularity. |
| **mindpadi/emotion_model** | Emotion classifier | GoEmotions + proprietary | English only. Not clinical, trained on general conversation. |
| **sentinet/suicidality** | BERT suicidality classifier | Reddit, Twitter, Kaggle suicide datasets | Social media only. Binary classification. Not validated on clinical documentation. |
| **Various LSTM/BiLSTM models** | Suicidal ideation detectors | Reddit SuicideWatch, Twitter | Research-grade. Not packaged for production. Binary classification only. |
| **BiomedBERT** (microsoft/BiomedNLP-BiomedBERT) | Pre-trained LM | PubMed abstracts + PMC full text | Strong biomedical language understanding. Not mental-health-specific but useful as a fine-tuning base for clinical note classification. |

### 2.2 Recent Research (2025)

A January 2025 study by Vance et al. (BMC Medical Informatics and Decision Making) trained a transformer-based NLP model on actual clinical notes from the NeuroBlu behavioral health database to detect suicidal ideation and anhedonia. The model achieved F1 scores of 0.89-0.99 across categories (suicidal ideation with intent/plan: PPV 0.95, F1 0.95 on validation). This demonstrates that transformer models can achieve high accuracy on clinical behavioral health notes when trained on appropriate data. However, the model is proprietary to their organization, covers only SI and anhedonia (not the multi-domain taxonomy bh-sentinel provides), and is not available as open-source.

Key takeaway: the approach works. Clinical-note-trained transformers outperform social-media-trained models on clinical text. This validates the Phase B strategy of fine-tuning on de-identified clinical notes from your own organization.

### 2.3 What Doesn't Exist

There is **no production-ready, open-source tool** that provides:
- Multi-flag behavioral health detection (not just binary suicidal/not-suicidal)
- Validation against clinical intake notes (not only social media)
- A deployable service/API with HIPAA-appropriate architecture
- An extensible flag taxonomy
- A deterministic safety layer (pattern matching) that does not depend on ML model accuracy

### 2.4 AWS Native Options

**Amazon Comprehend** provides sentiment analysis (positive/negative/neutral/mixed) and entity extraction. **Comprehend Medical** adds PHI detection and medical entity extraction (medications, conditions, dosages). These are useful layers but neither provides behavioral health-specific flag detection. They are incorporated as supplementary (and optional) signals within the bh-sentinel pipeline. Comprehend is HIPAA-eligible, coverable under an AWS BAA, and accessible via VPC endpoint so traffic never leaves the VPC.

**Important limitation:** Comprehend sentiment is a blunt instrument for behavioral health. A patient describing a suicide attempt in clinical, detached language could score as "neutral" sentiment. Sentiment is a supporting signal, not a primary safety detector. See Layer 3 design for degradation strategy.

### 2.5 Recommended Approach

**Build a custom multi-layer pipeline** combining:
1. **Layer 1 - Pattern matching** (keyword/phrase/regex) for high-confidence flag phrases. In-process, ~5ms
2. **Layer 2 - Transformer classification** via ONNX-quantized model running in-process. No network hop, ~50ms
3. **Layer 3 - Sentiment & emotion analysis** via AWS Comprehend (VPC endpoint, optional/degradable) + in-process emotion lexicon. ~100ms with Comprehend, ~5ms without
4. **Layer 4 - Configurable rules engine** for combining signals into actionable flags. In-process, ~3ms

Layers 1-3 run in **parallel** via asyncio.gather. Total wall clock time is the max of the three (~100ms with Comprehend, ~50ms without), not the sum. Layer 4 runs after all three complete.

This hybrid approach provides high recall on critical safety flags (Layer 1 catches explicit statements), nuanced detection of implicit signals (Layer 2), emotional context (Layer 3), and business-logic flexibility (Layer 4), all within the sub-second latency target.

**Do not rely on emotion models alone.** Emotion classification is useful as a secondary feature, not a primary safety detector. GoEmotions-style models were built from Reddit comments, not behavioral health workflows. The primary safety detection comes from Layers 1 and 2. Layer 3 provides supporting context only.

---

## 3. Clinical Flag Taxonomy

The flag taxonomy is informed by the Columbia Suicide Severity Rating Scale (C-SSRS), standard behavioral health screening instruments (PHQ-9, GAD-7), and common PHP/IOP clinical workflows. Flags are organized into **domains** with **severity levels**.

### 3.1 Flag Domains

#### DOMAIN: self_harm
| Flag ID | Flag Name | Severity | Description |
|---|---|---|---|
| SH-001 | Passive death wish | HIGH | Wish to be dead, wish to not wake up |
| SH-002 | Active suicidal ideation, nonspecific | CRITICAL | General thoughts about ending life |
| SH-003 | Active suicidal ideation, with method | CRITICAL | Thoughts about ending life with a specific method |
| SH-004 | Active suicidal ideation, with plan | CRITICAL | Specific plan for suicide |
| SH-005 | Active suicidal ideation, with intent | CRITICAL | Intent to act on suicidal thoughts |
| SH-006 | Preparatory behavior | CRITICAL | Actions taken to prepare for suicide |
| SH-007 | Self-injury without suicidal intent | HIGH | Cutting, burning, hitting self, etc. |
| SH-008 | History of attempt referenced | HIGH | Reference to past suicide attempt |

#### DOMAIN: harm_to_others
| Flag ID | Flag Name | Severity | Description |
|---|---|---|---|
| HO-001 | Homicidal ideation, nonspecific | CRITICAL | General thoughts of harming others |
| HO-002 | Homicidal ideation, specific target | CRITICAL | Named or identifiable target |
| HO-003 | Homicidal ideation, with plan | CRITICAL | Specific plan to harm another person |
| HO-004 | Violent urges | HIGH | Urges to be physically violent |
| HO-005 | Abuse disclosure, perpetrating | HIGH | Admission of perpetrating abuse |
| HO-006 | Abuse disclosure, experiencing | HIGH | Disclosure of being abused |

#### DOMAIN: medication
| Flag ID | Flag Name | Severity | Description |
|---|---|---|---|
| MED-001 | Non-adherence, stated | MEDIUM | Explicit statement of not taking medication |
| MED-002 | Non-adherence, implied | MEDIUM | Implied non-adherence (ran out, can't afford, don't like, stopped) |
| MED-003 | Medication misuse | HIGH | Taking more than prescribed, stockpiling |
| MED-004 | Adverse reaction reported | MEDIUM | Reporting side effects or bad reactions |
| MED-005 | Desire to discontinue | MEDIUM | Wanting to stop medication |

#### DOMAIN: substance_use
| Flag ID | Flag Name | Severity | Description |
|---|---|---|---|
| SU-001 | Active substance use disclosed | MEDIUM | Current use of substances |
| SU-002 | Relapse indicator | HIGH | Return to use after period of sobriety |
| SU-003 | Substance use escalation | HIGH | Increase in frequency or amount |
| SU-004 | Withdrawal symptoms | HIGH | Reporting withdrawal effects |
| SU-005 | Risky substance behavior | HIGH | Mixing substances, using alone, IV use |

#### DOMAIN: clinical_deterioration
| Flag ID | Flag Name | Severity | Description |
|---|---|---|---|
| CD-001 | Hopelessness | HIGH | Pervasive hopelessness, nothing will get better |
| CD-002 | Severe isolation | MEDIUM | Withdrawal from all social contact |
| CD-003 | Sleep disruption, severe | MEDIUM | Not sleeping for days, or sleeping 16+ hours |
| CD-004 | Appetite disruption, severe | MEDIUM | Not eating for days, or binge/purge |
| CD-005a | Auditory hallucinations | HIGH | Hearing voices, auditory command hallucinations, running commentary |
| CD-005b | Visual hallucinations | HIGH | Seeing things others cannot see, visual distortions perceived as real |
| CD-005c | Paranoid ideation | HIGH | Belief of being watched, followed, plotted against, or targeted without evidence |
| CD-005d | Delusional thinking | HIGH | Fixed false beliefs (grandiose, somatic, referential, persecutory) resistant to counter-evidence |
| CD-006 | Dissociation | MEDIUM | Feeling detached from reality, depersonalization |
| CD-007 | Panic/acute anxiety | MEDIUM | Severe panic, inability to function |
| CD-008 | Mania indicators | HIGH | Grandiosity, decreased need for sleep, racing thoughts, risky behavior |

#### DOMAIN: protective_factors
| Flag ID | Flag Name | Severity | Description |
|---|---|---|---|
| PF-001 | Treatment engagement | POSITIVE | Expressing commitment to treatment |
| PF-002 | Social support | POSITIVE | Mentioning supportive relationships |
| PF-003 | Future orientation | POSITIVE | Making plans for the future, setting goals |
| PF-004 | Coping skills usage | POSITIVE | Actively using taught coping strategies |
| PF-005 | Medication adherence | POSITIVE | Consistently taking medication as prescribed |

### 3.2 Severity Levels

| Level | Meaning | Expected Response |
|---|---|---|
| CRITICAL | Imminent risk to self or others | Immediate provider notification, clinical intervention |
| HIGH | Significant clinical concern | Same-day provider review |
| MEDIUM | Clinically relevant, not urgent | Flag in chart for next session review |
| LOW | Informational | Log for trend analysis |
| POSITIVE | Protective factor detected | Log for treatment progress tracking |

### 3.3 Taxonomy Versioning

The flag taxonomy will evolve as clinical needs change: new flags will be added, severity defaults may shift, and domains may expand (e.g., adding an `access_barriers` domain). To manage this cleanly:

- **Taxonomy version** is tracked in `config/flag_taxonomy.json` with a semantic version (e.g., `1.0.0`, `1.1.0`, `2.0.0`).
- **API responses include `taxonomy_version`** so downstream consumers know which flag set was active at analysis time.
- **Additive changes** (new flags, new domains) bump the minor version (1.0 to 1.1). Existing callers continue to work; they simply won't receive flags they don't know about.
- **Breaking changes** (removing flags, changing flag IDs, restructuring domains) bump the major version and require a coordinated rollout with all consumers.
- **Historical queries** reference the taxonomy version in the audit log, so a flag from 6 months ago can be interpreted against the taxonomy that was active at that time.
- **Clinical review gate:** All taxonomy changes should be reviewed and signed off by qualified clinical leadership before deployment.

### 3.4 Validated Instruments

bh-sentinel does not replace validated clinical instruments. For actual suicide assessment, the system prompts use of validated workflows rather than inventing its own:

- **C-SSRS (Columbia Suicide Severity Rating Scale):** When bh-sentinel detects self_harm domain flags at HIGH or CRITICAL severity, the recommended workflow is to prompt the clinician to administer C-SSRS.
- **PHQ-9:** Remains part of measurement-based care for depression screening. bh-sentinel's free-text analysis supplements PHQ-9 scores, it does not replace them.
- **GAD-7:** Similarly supplemented by bh-sentinel's anxiety-related flags (CD-007).

The tool's job is to detect possible signals in free text, prompt the clinician to use the appropriate validated assessment, and create the task/escalation/audit record.

---

## 4. System Architecture

### 4.1 Design Principles

- **Zero Egress** - All PHI processing stays inside the deploying organization's AWS account and VPC. No clinical text is ever transmitted to third-party systems. AWS services are accessed via VPC endpoints.
- **Stateless & Ephemeral** - No text is stored. Process in-memory, return flags, discard.
- **Sub-Second** - Total pipeline latency under 1 second (warm). Achieved via parallel execution and in-process inference.
- **Platform Agnostic** - Text in, flags out. No awareness of upstream system.
- **HIPAA-Safe by Design** - No PHI persistence. No logging of input text. Only flag IDs, severity, and confidence scores are logged.
- **Extensible** - Flag taxonomy is configuration-driven, not hardcoded.
- **Self-Contained** - All NLP models, tokenizers, and dependencies are bundled together. No external model hosting. No network hops for inference.
- **Gracefully Degradable** - If optional layers (Comprehend) fail or timeout, the pipeline still returns results from the remaining layers rather than failing entirely.

### 4.2 High-Level Architecture

```
┌─────────────────────────────────────────┐
│       API Gateway (REST, Private)       │
│    POST /analyze  -  mTLS / API Key     │
│    Rate limit: 100 req/s burst,         │
│    50 req/s sustained per API key       │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│ bh-sentinel Lambda (Container Image)    │
│ Python 3.11 + ONNX Runtime             │
│ 2048 MB / Provisioned Concurrency x3   │
│                                         │
│   ┌─────────────────────────────────┐   │
│   │  Text Preprocessor (~2ms)       │   │
│   │  + Sentence Splitter            │   │
│   └──────────────┬──────────────────┘   │
│                  │                      │
│        ┌─────────┼─────────┐            │
│        │  asyncio.gather   │            │
│        │  (parallel exec)  │            │
│    ┌───▼───┐┌───▼───┐┌────▼──────┐     │
│    │ L1    ││ L2    ││ L3        │     │
│    │Regex  ││ONNX   ││Comprehend │     │
│    │~5ms   ││~50ms  ││~100ms     │     │
│    │in-proc││in-proc││VPC endpt  │     │
│    │       ││       ││OPTIONAL   │     │
│    └───┬───┘└───┬───┘└─────┬─────┘     │
│        └────────┼──────────┘            │
│                 │                       │
│           ┌─────▼──────┐                │
│           │ L4 Rules   │                │
│           │ Engine~3ms │                │
│           └─────┬──────┘                │
│                 │                       │
│           ┌─────▼──────────────────┐    │
│           │ Response Builder       │    │
│           │ (flags + evidence +    │    │
│           │  spans + severity)     │    │
│           └────────────────────────┘    │
└──────────────┬──────────────────────────┘
               │
  ┌────────────┼────────────┐
  │            │            │
┌─▼──────┐ ┌──▼───────┐ ┌──▼────────┐
│Clinical│ │Mobile App│ │EMR System │
│Ops     │ │Backend   │ │Webhook    │
│Platform│ │          │ │           │
└────────┘ └──────────┘ └───────────┘
```

### 4.3 Latency Budget

| Stage | Execution | Latency |
|---|---|---|
| API Gateway to Lambda | Network | ~5ms |
| Text preprocessor + sentence split | In-process | ~2ms |
| Layer 1: Pattern matching | In-process (compiled regex) | ~5ms |
| Layer 2: ONNX transformer | In-process (ONNX Runtime, DistilBERT INT8) | ~50ms |
| Layer 3: Sentiment (if enabled) | VPC endpoint to Comprehend | ~100ms |
| Layer 3: Emotion lexicon only (fallback) | In-process | ~2ms |
| **Layers 1-3 total (parallel, Comprehend enabled)** | **max(5, 50, 100)** | **~100ms** |
| **Layers 1-3 total (parallel, Comprehend skipped)** | **max(5, 50, 2)** | **~50ms** |
| Layer 4: Rules engine | In-process | ~3ms |
| Response builder + return | In-process + network | ~5ms |
| **Total (warm, Comprehend enabled)** | | **~120-200ms** |
| **Total (warm, Comprehend skipped/failed)** | | **~65-120ms** |
| Cold start (mitigated by provisioned concurrency) | Container init + model load | ~2-3s |

### 4.4 API Contract

**Request:**

```json
POST /analyze
{
  "text": "Patient reports not sleeping for 3 days and states she stopped taking her Lexapro two weeks ago. Expresses commitment to continuing group therapy.",
  "source": "intake_note",
  "context": {
    "program": "php",
    "session_type": "individual",
    "language": "en"
  },
  "config": {
    "domains": ["self_harm", "medication", "clinical_deterioration"],
    "min_severity": "MEDIUM",
    "include_protective": true,
    "include_emotions": true,
    "include_comprehend": true
  }
}
```

**Response:**

```json
{
  "request_id": "uuid-v4",
  "analysis_timestamp": "2026-04-01T14:22:00Z",
  "processing_time_ms": 142,
  "taxonomy_version": "1.0.0",
  "flags": [
    {
      "flag_id": "CD-003",
      "domain": "clinical_deterioration",
      "name": "Sleep disruption, severe",
      "severity": "MEDIUM",
      "confidence": 0.91,
      "detection_layer": "pattern_match",
      "matched_context_hint": "sleep disruption, 3 days",
      "basis_description": "Pattern match detected language consistent with severe sleep disruption (duration specified).",
      "evidence_span": {
        "sentence_index": 0,
        "char_start": 24,
        "char_end": 53
      }
    },
    {
      "flag_id": "MED-001",
      "domain": "medication",
      "name": "Non-adherence, stated",
      "severity": "MEDIUM",
      "confidence": 0.87,
      "detection_layer": "pattern_match",
      "matched_context_hint": "medication non-adherence, explicit statement",
      "basis_description": "Pattern match detected explicit statement of medication discontinuation.",
      "evidence_span": {
        "sentence_index": 0,
        "char_start": 58,
        "char_end": 106
      }
    }
  ],
  "emotions": {
    "primary": "anxiety",
    "secondary": "hopelessness",
    "comprehend_available": true,
    "sentiment": {
      "overall": "NEGATIVE",
      "scores": {
        "positive": 0.05,
        "negative": 0.72,
        "neutral": 0.18,
        "mixed": 0.05
      }
    }
  },
  "protective_factors": [
    {
      "flag_id": "PF-001",
      "name": "Treatment engagement",
      "confidence": 0.65,
      "matched_context_hint": "expressed willingness to continue treatment",
      "basis_description": "Pattern match detected language indicating active commitment to treatment program.",
      "evidence_span": {
        "sentence_index": 1,
        "char_start": 0,
        "char_end": 52
      }
    }
  ],
  "summary": {
    "max_severity": "MEDIUM",
    "total_flags": 2,
    "domains_flagged": ["clinical_deterioration", "medication"],
    "requires_immediate_review": false,
    "recommended_action": null
  },
  "pipeline_status": {
    "layer_1_pattern": "completed",
    "layer_2_transformer": "completed",
    "layer_3_comprehend": "completed",
    "layer_3_emotion_lexicon": "completed",
    "layer_4_rules": "completed"
  }
}
```

**Key design decisions in the contract:**

- **matched_context_hint** provides a category description of what triggered the flag, NOT a verbatim quote. This allows downstream systems to show providers what the concern area is without exposing raw journal text (critical for mobile app use cases where journal privacy must be preserved).
- **basis_description** gives a human-readable explanation of the detection logic. This is critical for FDA CDS Criterion 4: the clinician must be able to independently review the basis for the recommendation (see Section 8).
- **evidence_span** provides sentence_index plus character offsets (char_start/char_end) so upstream UIs can highlight the relevant region of the note without bh-sentinel exposing verbatim text. For mobile app use cases, this can be translated to "concern detected in paragraph 2 of journal entry" without passing raw text through.
- **config.domains** allows callers to scope analysis. A mobile app might only want self_harm and harm_to_others, while intake notes want the full taxonomy.
- **config.include_comprehend** allows callers to skip the Comprehend layer entirely for faster responses (~50ms instead of ~100ms). Useful for high-throughput batch analysis or when emotion data is not needed.
- **detection_layer** tells you which pipeline layer caught the flag, useful for tuning.
- **taxonomy_version** indicates which version of the flag taxonomy was active at analysis time.
- **pipeline_status** reports the state of each layer, making degradation visible. If Comprehend timed out, it shows "timeout" instead of "completed".
- **processing_time_ms** returned for latency monitoring and SLA verification.
- **recommended_action** is populated by the rules engine for CRITICAL flags (e.g., "Administer C-SSRS", "Notify assigned provider"). Always a prompt for clinician action, never an autonomous directive.

### 4.5 NLP Pipeline Layers: Detail

#### Layer 1: Pattern Matching (bh-sentinel-core)

This is a deterministic, regex/keyword layer that catches explicit clinical language. It runs in-process with compiled regex and returns character-level evidence spans. No network hop, no external dependency. This layer is the primary safety net for CRITICAL flags.

**Implementation approach:**
- Compiled regex patterns loaded from `config/patterns.yaml` (loaded once at init, cached in memory)
- Patterns organized by flag_id
- Each pattern has a confidence_score (typically 0.85-0.99 for explicit matches)
- Supports negation detection ("I am NOT having suicidal thoughts" should not flag)
- Supports temporal qualifiers ("I USED TO cut myself" vs "I am cutting myself")
- Returns match positions (char_start, char_end, sentence_index) for evidence span tracking

**Example pattern entries (of ~150-200 total in production):**

```yaml
self_harm:
  SH-001:  # Passive death wish
    patterns:
      - "wish I (was|were) dead"
      - "wish I (could|would) (just )?die"
      - "don't want to (be alive|live|wake up)"
      - "wish I (could|would) (go to )?sleep and (never|not) wake up"
      - "better off dead"
      - "everyone would be better off without me"
      - "no reason to (live|go on|keep going)"
      - "what's the point (anymore|of (living|going on|any of this))"
      - "can't (do this|take it|go on) anymore"
      - "just want (it|everything) to (stop|end|be over)"
    negation_phrases: ["don't", "no longer", "used to", "not anymore", "denies"]
    confidence: 0.92

  SH-002:  # Active SI - nonspecific
    patterns:
      - "want(s|ed|ing)? to (kill|end) (myself|my life|it all)"
      - "thinking (about|of) (killing|ending) (myself|my life)"
      - "suicidal (thoughts|ideation|feelings)"
      - "can't (stop|keep from) thinking about (suicide|dying|ending it)"
    clinical_shorthand:
      - "pt (c/o|reports|endorses) SI"
      - "(+|positive) (for )?SI"
      - "SI (present|endorsed|reported|active)"
    negation_phrases: ["denies", "no", "not", "never", "(-|negative) (for )?SI"]
    confidence: 0.95

medication:
  MED-001:  # Non-adherence - stated
    patterns:
      - "stopped taking (my |the )?(medication|meds|pills|prescription)"
      - "haven't (taken|been taking) (my |the )?(medication|meds)"
      - "ran out of (my |the )?(medication|meds|prescription)"
      - "can't afford (my |the )?(medication|meds|prescription)"
      - "not taking (my |the )?(medication|meds) (anymore|as prescribed)"
      - "skipping (my |the )?(medication|meds|doses)"
    clinical_shorthand:
      - "(med|medication) non-?adherence"
      - "non-?compliant (with|re:?) (meds|medication)"
    confidence: 0.88

harm_to_others:
  HO-001:  # Homicidal ideation - nonspecific
    patterns:
      - "want(s|ed|ing)? to (hurt|harm|kill) (someone|people|them|him|her)"
      - "thinking (about|of) (hurting|harming|killing)"
      - "going to (hurt|harm|kill)"
      - "fantasiz(e|es|ing) about (violence|hurting|killing)"
    clinical_shorthand:
      - "pt (c/o|reports|endorses) HI"
      - "(+|positive) (for )?HI"
      - "HI (present|endorsed|reported|active)"
    negation_phrases: ["don't", "no", "not", "never", "denies", "(-|negative) (for )?HI"]
    confidence: 0.93
```

**Pattern library sizing:** The 37 flags across 6 domains, combined with negation variants, temporal variants, colloquial language variants ("what's the point anymore" vs "suicidal ideation"), and clinical shorthand (e.g., "pt c/o SI", "denies HI"), will require approximately 150-200 patterns in production.

Clinical teams should be able to add/modify patterns without code deployment (patterns.yaml is a configuration file, not code).

#### Layer 2: Transformer Classification (bh-sentinel-ml)

This layer catches what patterns miss: implied distress, subtle language shifts, contextual meaning. The model runs entirely in-process. No network hop, no external service, no PHI leaving the function.

**Model strategy - ONNX in-process:**
- Base model: DistilBERT (or MentalBERT distilled) fine-tuned on flag categories
- Format: Exported to ONNX, quantized to INT8 (~65MB model file)
- Runtime: ONNX Runtime (onnxruntime Python package) runs inference in-process
- Tokenizer: HuggingFace tokenizers library (Rust-based, fast) loaded at init
- Inference time: ~30-60ms for a 512-token input
- No network hop: Model weights are bundled with the package/container. Inference is a function call, not an API call.

**Why ONNX in-process instead of a hosted model service:**
- Hosted model endpoints (e.g., SageMaker) have cold starts of 30+ seconds for serverless, or $50-100/month minimum for always-on
- Hosted endpoints require a network hop, adding latency and creating a PHI transit path
- ONNX in-process: zero cold start risk, zero network hop, model runs as a library call

**Container init (for Lambda deployment, runs once, cached across warm invocations):**

```python
# handler.py - container init (outside the handler function)
import onnxruntime as ort
from tokenizers import Tokenizer

# Loaded once at container start, reused across all invocations
MODEL_SESSION = ort.InferenceSession(
    "/opt/models/bh-sentinel-distilbert-int8.onnx",
    providers=["CPUExecutionProvider"]
)
TOKENIZER = Tokenizer.from_file("/opt/models/tokenizer.json")

# Pattern config loaded at init
PATTERNS = load_patterns()
RULES = load_rules()
```

**Multi-label classification targets:**

```
self_harm_risk:      [none, passive, active_nonspecific, active_with_plan]
harm_others_risk:    [none, ideation, specific_target, with_plan]
medication_concern:  [none, non_adherence, misuse, adverse_reaction]
substance_concern:   [none, active_use, relapse, escalation]
clinical_status:     [stable, mild_concern, moderate_concern, severe_concern]
```

**Phased model development:**
- **Phase A (immediate):** Zero-shot classification using an ONNX-exported NLI model (e.g., DistilBART-MNLI). No training data needed. Expected ~60-75% accuracy on clinical notes (lower than the 70-80% general benchmark due to clinical language complexity, negation patterns, and hedging). Consider also testing BiomedBERT as the NLI base for better medical language comprehension.
- **Phase B (after 500+ annotated notes):** Fine-tune DistilBERT (or distilled MentalBERT) with a multi-label classification head on de-identified clinical notes from your organization. Export to ONNX, quantize to INT8. Target: precision > 0.85, recall > 0.80 for CRITICAL flags.
- **Phase C (ongoing):** Retraining pipeline. Clinician-reviewed flag corrections feed back into training set. New model versions are deployed as new package releases or container images.

**Phase A accuracy note:** During Phase A (zero-shot only), Layer 1 pattern matching must carry the primary safety detection burden for CRITICAL flags. The transformer is the nuance layer, not the safety net. Prioritize getting Layer 1 to very high recall on self_harm and harm_to_others domains before spending time on zero-shot tuning.

#### Layer 3: Sentiment & Emotion Analysis

**Design: Optional and Degradable.** Layer 3 is architecturally optional. If the caller sets `include_comprehend: false`, or if Comprehend times out (200ms timeout), or if the VPC endpoint is unreachable, the pipeline continues with emotion lexicon results only. The pipeline never fails because Comprehend is unavailable.

**AWS Comprehend** for sentiment analysis (when deployed on AWS), accessed exclusively through a VPC endpoint. Traffic never leaves the VPC, never touches the public internet. Comprehend is HIPAA-eligible and coverable under an AWS BAA.

```python
async def run_comprehend_with_fallback(text: str) -> SentimentResult:
    try:
        response = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None,
                lambda: comprehend.detect_sentiment(Text=text, LanguageCode='en')
            ),
            timeout=0.2  # 200ms hard timeout
        )
        return SentimentResult.from_comprehend(response)
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning(f"Comprehend unavailable: {type(e).__name__}. Falling back to lexicon-only.")
        return SentimentResult.empty(reason="comprehend_unavailable")
```

**Emotion detection** via an in-process lexicon-based approach (NRC Emotion Lexicon). No ML dependency, no model to load, ~2ms. Maps words to emotion categories: happiness, sadness, anger, fear, disgust, surprise, anxiety, hopelessness, guilt, shame. This always runs regardless of Comprehend availability.

#### Layer 4: Rules Engine (bh-sentinel-core)

Combines signals from Layers 1-3 into final flag determinations. Runs in-process after all parallel layers complete. Rules are loaded from `config/rules.json` and cached. Rules gracefully handle missing Layer 3 Comprehend data.

```json
[
  {
    "name": "Escalate passive SI with hopelessness",
    "condition": "SH-001.detected AND emotion.hopelessness > 0.6",
    "action": "upgrade SH-001 severity to CRITICAL",
    "rationale": "Passive SI combined with strong hopelessness is a significant risk indicator",
    "recommended_action": "Administer C-SSRS"
  },
  {
    "name": "De-escalate historical reference",
    "condition": "any_flag.detected AND temporal_qualifier == 'past'",
    "action": "downgrade severity by one level, add note 'historical reference'",
    "rationale": "Past-tense references are informational, not immediate risk"
  },
  {
    "name": "Cross-domain compound risk",
    "condition": "substance_use.detected AND self_harm.detected",
    "action": "add compound_risk flag, ensure max severity = CRITICAL",
    "rationale": "Co-occurring substance use and self-harm ideation significantly elevates risk",
    "recommended_action": "Immediate provider review. Administer C-SSRS."
  },
  {
    "name": "Prompt C-SSRS for any self_harm CRITICAL",
    "condition": "self_harm.any.severity == CRITICAL",
    "action": "set recommended_action",
    "recommended_action": "Administer C-SSRS per organization protocol"
  }
]
```

Rules are configuration, not code. Clinical teams can modify detection logic without engineering involvement.

### 4.6 Orchestrator: Parallel Execution

The orchestrator uses asyncio.gather to run Layers 1, 2, and 3 concurrently. Wall clock time is the max of the three layers, not the sum.

```python
async def analyze(text: str, config: AnalysisConfig) -> AnalysisResponse:
    # Preprocess (synchronous, ~2ms)
    preprocessed = preprocess(text)

    # Build layer tasks based on config
    tasks = [
        run_pattern_matching(preprocessed, PATTERNS),
        run_transformer_inference(preprocessed, MODEL_SESSION, TOKENIZER),
    ]

    if config.include_comprehend:
        tasks.append(run_comprehend_with_fallback(preprocessed.raw_text))
    
    # Always run emotion lexicon (in-process, ~2ms)
    tasks.append(run_emotion_lexicon(preprocessed, EMOTION_LEXICON))

    # Run all in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Unpack results, handling any layer failures gracefully
    l1_result, l2_result = results[0], results[1]
    comprehend_result = results[2] if config.include_comprehend else None
    emotion_result = results[-1]

    # Run L4 sequentially (needs results from L1-L3)
    l4_result = apply_rules(l1_result, l2_result, comprehend_result, emotion_result, RULES)

    return build_response(l1_result, l2_result, comprehend_result, emotion_result, l4_result, config)
```

**Why this works for parallelism:**
- L1 (regex) is pure CPU, no I/O wait
- L2 (ONNX) is pure CPU, no I/O wait
- L3 (Comprehend) is async I/O, waiting on VPC endpoint response
- CPU work (L1 + L2) and I/O wait (L3) overlap naturally with asyncio

---

## 5. Reference Deployment Architecture (AWS)

The `deployment/` directory in the bh-sentinel repository provides a complete reference architecture for deploying bh-sentinel on AWS Lambda. This is a reference implementation; organizations may adapt it to their own infrastructure.

### 5.1 AWS Resources

```
┌─────────────────────────────────────────────────────────┐
│  VPC (bh-sentinel-vpc)                                   │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Private Subnet                                   │  │
│  │                                                   │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  Lambda (Container Image, 2048MB)           │  │  │
│  │  │  ┌──────────────────────────────────────┐   │  │  │
│  │  │  │ ONNX Runtime + DistilBERT INT8       │   │  │  │
│  │  │  │ Pattern matcher + Rules engine        │   │  │  │
│  │  │  │ Emotion lexicon                       │   │  │  │
│  │  │  │ Provisioned Concurrency: 3            │   │  │  │
│  │  │  └──────────────────────────────────────┘   │  │  │
│  │  └──────────────┬──────────────┬───────────────┘  │  │
│  │                 │              │                   │  │
│  │    ┌────────────▼────┐  ┌──────▼───────────────┐  │  │
│  │    │  VPC Endpoint   │  │  VPC Endpoint        │  │  │
│  │    │  (S3 Gateway)   │  │  (Comprehend Iface)  │  │  │
│  │    └─────────────────┘  └──────────────────────┘  │  │
│  │                                                   │  │
│  │    ┌─────────────────┐  ┌──────────────────────┐  │  │
│  │    │  S3 Bucket      │  │  CloudWatch Logs     │  │  │
│  │    │  (Config only)  │  │  (flags only no PHI) │  │  │
│  │    └─────────────────┘  └──────────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  API Gateway (REST, Private Endpoint)             │  │
│  │  mTLS + API Key authentication                    │  │
│  │  Rate limit: 100 req/s burst, 50 req/s sustained  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘

              ALL TRAFFIC STAYS INSIDE VPC
     No public internet. No third-party API calls.
 Comprehend accessed via VPC endpoint (AWS backbone only).
```

### 5.2 Terraform Module Structure

```
deployment/terraform/
├── environments/
│   ├── dev/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── terraform.tfvars
│   ├── staging/
│   └── prod/
├── modules/
│   ├── api-gateway/        # REST API, stage, deployment, VPC endpoint, rate limiting
│   ├── lambda/             # Container image function, provisioned concurrency, IAM
│   ├── ecr/                # ECR repository for container images
│   ├── s3-config/          # Config bucket with versioning, KMS encryption
│   ├── monitoring/         # CloudWatch dashboards, alarms, log groups, retention
│   └── networking/         # VPC, subnets, SGs, VPC endpoints (S3/Comprehend/CW)
└── shared/
    ├── backend.tf
    └── providers.tf
```

### 5.3 Key Terraform Resources

```hcl
# ECR Repository
resource "aws_ecr_repository" "bh_sentinel" {
  name                 = "bh-sentinel-${var.environment}"
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration { scan_on_push = true }
  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.bh_sentinel.arn
  }
}

# Lambda - Container Image, all processing in-process
resource "aws_lambda_function" "bh_sentinel" {
  function_name = "bh-sentinel-${var.environment}"
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.bh_sentinel.repository_url}:${var.image_tag}"
  memory_size   = 2048
  timeout       = 10

  vpc_config {
    subnet_ids         = var.private_subnet_ids
    security_group_ids = [aws_security_group.lambda_sg.id]
  }

  environment {
    variables = {
      CONFIG_BUCKET   = aws_s3_bucket.config.id
      PATTERNS_KEY    = "config/patterns.yaml"
      RULES_KEY       = "config/rules.json"
      AWS_REGION_NAME = var.aws_region
      LOG_LEVEL       = "INFO"
    }
  }
}

# Provisioned Concurrency - eliminates cold starts
resource "aws_lambda_provisioned_concurrency_config" "bh_sentinel" {
  function_name                  = aws_lambda_function.bh_sentinel.function_name
  qualifier                      = aws_lambda_alias.live.name
  provisioned_concurrent_executions = var.provisioned_concurrency  # default: 3
}

# API Gateway - Rate Limiting
resource "aws_api_gateway_usage_plan" "bh_sentinel" {
  name = "bh-sentinel-${var.environment}"
  api_stages {
    api_id = aws_api_gateway_rest_api.bh_sentinel.id
    stage  = aws_api_gateway_stage.live.stage_name
  }
  throttle_settings {
    burst_limit = 100
    rate_limit  = 50
  }
}

# Security Group - Lambda can ONLY reach VPC endpoints
resource "aws_security_group" "lambda_sg" {
  name_prefix = "bh-sentinel-lambda-"
  vpc_id      = aws_vpc.main.id

  egress {
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    security_groups = [aws_security_group.vpc_endpoints_sg.id]
    description     = "HTTPS to VPC endpoints only"
  }

  egress {
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    prefix_list_ids = [aws_vpc_endpoint.s3.prefix_list_id]
    description     = "HTTPS to S3 via gateway endpoint"
  }

  # NO other egress - Lambda cannot reach the internet
}
```

### 5.4 Dockerfile

```dockerfile
FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ONNX model baked into image - no runtime download
COPY models/ /opt/models/
COPY config/emotion_lexicon.json /opt/config/emotion_lexicon.json

COPY src/ ${LAMBDA_TASK_ROOT}/

CMD ["handler.lambda_handler"]
```

**Container image size budget:**

| Component | Size |
|---|---|
| Lambda Python 3.11 base | ~200MB |
| onnxruntime | ~40MB |
| tokenizers (Rust) | ~15MB |
| pyyaml, boto3, pydantic | ~10MB |
| DistilBERT INT8 ONNX model | ~65MB |
| Tokenizer JSON + lexicon + app code | ~3MB |
| **Total** | **~333MB** |

---

## 6. Repository Structure

```
bh-sentinel/
├── packages/
│   ├── bh-sentinel-core/
│   │   ├── src/bh_sentinel/core/
│   │   │   ├── __init__.py
│   │   │   ├── pattern_matcher.py     # Layer 1: compiled regex, negation, temporal, evidence spans
│   │   │   ├── rules_engine.py        # Layer 4: business logic rules
│   │   │   ├── taxonomy.py            # Flag taxonomy definitions + versioning
│   │   │   ├── preprocessor.py        # Text normalization, sentence splitting, offset tracking
│   │   │   ├── negation_detector.py   # "NOT suicidal" handling
│   │   │   ├── temporal_detector.py   # Past vs present tense detection
│   │   │   ├── emotion_lexicon.py     # NRC emotion lexicon (in-process)
│   │   │   ├── pipeline.py            # Orchestrator with asyncio.gather
│   │   │   └── models/                # Pydantic request/response models
│   │   │       ├── request.py
│   │   │       ├── response.py        # incl. evidence_span, basis_description
│   │   │       └── flags.py           # Flag taxonomy enums
│   │   ├── tests/
│   │   ├── pyproject.toml
│   │   └── README.md
│   └── bh-sentinel-ml/
│       ├── src/bh_sentinel/ml/
│       │   ├── __init__.py
│       │   ├── transformer.py         # ONNX Runtime inference (in-process)
│       │   ├── zero_shot.py           # Zero-shot NLI classification
│       │   └── export.py              # Model export tooling (PyTorch to ONNX INT8)
│       ├── tests/
│       ├── pyproject.toml
│       └── README.md
├── config/
│   ├── patterns.yaml                  # Default pattern library (~150-200 patterns)
│   ├── rules.json                     # Default rules engine configuration
│   ├── flag_taxonomy.json             # Versioned flag taxonomy (v1.0.0)
│   └── emotion_lexicon.json           # NRC Emotion Lexicon data
├── deployment/
│   ├── aws-lambda/
│   │   ├── handler.py                 # Lambda entry point + container init
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── README.md
│   └── terraform/
│       ├── modules/                   # VPC, Lambda, API GW, ECR, S3, monitoring
│       └── environments/              # dev / staging / prod configs
├── training/
│   ├── prepare_data.py                # Dataset preparation (C-SSRS Reddit, SMHD, etc.)
│   ├── train.py                       # Fine-tuning pipeline
│   ├── evaluate.py                    # Model evaluation harness
│   └── export.py                      # Export to ONNX + INT8 quantization
├── docs/
│   ├── architecture.md                # This document
│   ├── flag-taxonomy.md               # Detailed flag definitions and clinical rationale
│   ├── deployment-guide.md            # AWS Lambda deployment walkthrough
│   ├── fda-cds-analysis.md            # FDA CDS compliance analysis
│   ├── training-guide.md              # Model fine-tuning instructions
│   └── pattern-library.md             # Pattern authoring guide for clinical teams
├── LICENSE                            # Apache 2.0
├── CONTRIBUTORS.md
├── CONTRIBUTING.md
├── CHANGELOG.md
└── README.md
```

---

## 7. HIPAA Compliance & Data Containment

### 7.1 PHI Handling Rules

| Data Point | Touches PHI? | Storage | Logging | Leaves VPC? |
|---|---|---|---|---|
| Input text | YES | In-memory only, never persisted | NEVER logged | NEVER |
| Flag results | NO (metadata) | Returned to caller | Logged (flag_id, severity, confidence) | No |
| Evidence spans | NO (offsets only) | Returned to caller | Logged (sentence_index, char offsets) | No |
| Confidence scores | NO | Returned to caller | Logged | No |
| Request metadata | NO | Not stored | Logged (source, timestamp, request_id) | No |
| Pattern config | NO | S3 or local (encrypted) | Not logged | No |
| Model artifacts | NO | Baked into container/package | Not logged | No |
| Comprehend API call | YES (text sent) | Transient (Comprehend does not store) | Not logged | No (VPC endpoint) |

### 7.2 Architectural Safeguards

- **Zero egress** - Lambda security group permits outbound only to VPC endpoints. No NAT gateway, no internet gateway attached to private subnet. Clinical text physically cannot leave the VPC.
- **VPC endpoints for all AWS services** - S3 (gateway), Comprehend (interface), CloudWatch Logs (interface). Private DNS enabled so service calls resolve to VPC endpoint IPs.
- **No text logging** - CloudWatch log group explicitly filters input text. Application code never passes input text to logger.
- **No text in error messages** - Exception handlers strip text content before logging.
- **Encryption in transit** - API Gateway enforces TLS 1.2+. VPC endpoint traffic is encrypted.
- **Encryption at rest** - S3 bucket uses KMS encryption. Container images in ECR use KMS encryption.
- **No text caching** - Lambda does not cache input text between invocations. Config (patterns, rules) is cached; text is not.
- **In-process model inference** - Transformer model runs as a library call. No model hosting service, no network hop for inference, no PHI in transit to a model endpoint.
- **Evidence spans are offsets, not text** - The `evidence_span` field contains integer offsets (sentence_index, char_start, char_end), not verbatim text. The upstream system (which already has the text) uses offsets to locate the relevant region.
- **Audit trail** - Every request gets a UUID. Logs capture: request_id, source, timestamp, flags returned, processing_time_ms, taxonomy_version. Never text.
- **Network isolation** - Lambda runs in private subnet with no internet access. API Gateway uses private endpoint.

### 7.3 BAA Considerations

AWS services used in the reference deployment are HIPAA-eligible: Lambda, API Gateway, ECR, S3, Comprehend, CloudWatch, KMS, VPC. All should be covered under the deploying organization's AWS BAA. No third-party services are used.

### 7.4 PHI Containment Verification

Automated tests in CI verify that no PHI can leak:
- **test_no_phi_in_logs:** Asserts that no input text appears in any log output during a full pipeline run
- **test_no_phi_in_errors:** Triggers every error path and verifies exception messages contain no text content
- **test_no_phi_in_response:** Verifies response contains only flag metadata, never verbatim input text
- **test_evidence_spans_are_offsets:** Verifies evidence_span fields contain only integers, never text content
- **test_security_group_no_internet:** Terraform plan validation that Lambda SG has no 0.0.0.0/0 egress rules

### 7.5 Audit Log Retention (Reference)

Recommended audit log retention policy for CloudWatch Logs:

| Log Type | Retention | Content | Purpose |
|---|---|---|---|
| Request audit log | 1 year (365 days) | request_id, source, timestamp, flags returned, severity, confidence, taxonomy_version, processing_time_ms | Clinical review, quality assurance, regulatory inquiry |
| Error log | 90 days | request_id, error type, layer that failed (NO text content) | Debugging, reliability monitoring |
| Performance metrics | 90 days | Latency percentiles, cold start counts, Comprehend fallback rate | Capacity planning, SLA monitoring |

Retention periods are configurable per environment via Terraform variables. Deploying organizations should review retention requirements against their own state-specific regulations and organizational policies.

---

## 8. FDA CDS Alignment Analysis

### 8.1 Regulatory Framework

bh-sentinel is designed to operate as Non-Device Clinical Decision Support software under Section 520(o)(1)(E) of the FD&C Act, as amended by the 21st Century Cures Act and interpreted by FDA's updated CDS guidance (January 2026).

The FDA's four-part test for Non-Device CDS exclusion from device regulation:

**Criterion 1: Not intended to acquire, process, or analyze a medical image or signal.**
bh-sentinel processes free-text clinical notes. It does not acquire, process, or analyze medical images, signals from IVDs, or physiological signals. Pass.

**Criterion 2: Intended to display, analyze, or print medical information about a patient.**
bh-sentinel analyzes textual medical information (clinical notes, journal entries) to identify clinically relevant signals. The output is structured flag metadata presented to health care professionals. Pass.

**Criterion 3: Intended to support or provide recommendations to an HCP about prevention, diagnosis, or treatment.**
bh-sentinel supports clinical decision-making by flagging potential safety concerns and recommending validated assessment workflows (e.g., "Administer C-SSRS"). It does not diagnose or prescribe treatment. The updated 2026 guidance takes a more flexible approach to single-recommendation outputs. Pass.

**Criterion 4: Intended to enable the HCP to independently review the basis for the recommendation.**
This is the critical criterion. bh-sentinel is designed to satisfy it through:
- **basis_description**: Each flag includes a human-readable explanation of why it was raised
- **detection_layer**: Indicates which analysis method produced the flag (pattern_match, transformer, rules_engine)
- **confidence score**: Quantifies the system's certainty
- **matched_context_hint**: Describes the category of concern without exposing verbatim text
- **evidence_span**: Provides character offsets so the clinician can locate and review the relevant portion of the original note in context
- **recommended_action**: When present, suggests a validated clinical workflow rather than a definitive clinical directive

The clinician always has access to the full original note and can independently assess whether the flag is clinically meaningful.

### 8.2 Design Guardrails

To maintain Non-Device CDS status:
- bh-sentinel **never makes autonomous clinical decisions**. It flags, it recommends assessment, it creates tasks. A clinician always reviews.
- bh-sentinel **does not replace validated instruments**. For suicide risk, it prompts C-SSRS administration. For depression, it supplements (not replaces) PHQ-9.
- bh-sentinel **does not direct treatment**. It does not recommend medications, dosage changes, or hospitalization.
- All outputs include sufficient information for the clinician to understand and independently evaluate the basis of the flag.

### 8.3 Documentation Requirements

For ongoing compliance, deploying organizations should maintain:
- Records of the intended use statement
- Evidence that clinicians can independently review the basis of each flag
- Documentation that the system supports, rather than replaces, clinical judgment
- Records of clinical team review and validation of flag taxonomy and rules

### 8.4 Note on Patient-Facing Use

The FDA's CDS guidance focuses primarily on HCP-directed software. If bh-sentinel is ever extended to provide output directly to patients (e.g., through a patient-facing mobile app), the regulatory analysis would need to be revisited. Patient-directed CDS software is subject to different FDA policies. The current architecture routes all outputs to providers, not patients.

---

## 9. Integration Patterns

bh-sentinel is platform-agnostic. The following patterns illustrate common integration approaches.

### 9.1 Clinical Operations Platform (Intake Notes)

```
Provider writes intake note in clinical ops platform
        │
        ▼
Platform backend calls bh-sentinel API
(Private API Gateway, same VPC)
        │
        ▼
bh-sentinel returns flags + evidence spans (~150ms)
        │
        ▼
Platform displays flag badges on note card
Uses evidence_span offsets to highlight relevant
regions of note for provider review
        │
        ▼
CRITICAL flags trigger real-time
notification to assigned provider
via organization's notification system
Notification includes recommended_action
(e.g., "Administer C-SSRS")
```

### 9.2 Mobile App (Journal Entries)

```
Patient writes journal entry in app
        │
        ▼
App backend sends text to bh-sentinel API
(text is NOT stored in app backend DB)
(call goes through Private API Gateway)
(config: domains = [self_harm, harm_to_others])
        │
        ▼
bh-sentinel returns flags (~150ms)
        │
        ▼
App backend stores ONLY flags
(flag_id, severity, timestamp, patient_id)
        │
        ▼
If CRITICAL or HIGH flag:
  → Notification to assigned provider
  → Provider sees flag summary in clinical platform
  → Provider sees "concern detected in journal entry"
  → Provider does NOT see journal text
  → Provider follows up using approved clinical workflow
```

This is a key privacy design: the mobile app journal is private to the patient. Providers see that a safety concern was detected and its category, but never the raw text. This preserves therapeutic trust while ensuring safety monitoring.

### 9.3 EMR Integration (Webhooks)

```
EMR webhook fires on note completion
        │
        ▼
Webhook handler calls bh-sentinel API
(Private API Gateway)
        │
        ▼
bh-sentinel returns flags
        │
        ▼
Flags stored back in EMR via API
or surfaced in clinical ops dashboard
```

### 9.4 Audit Event Integration

bh-sentinel can emit audit events through [bh-audit-logger](https://github.com/bh-healthcare/bh-audit-logger) for compliance tracking. Each analysis request generates an audit event recording the request_id, source, flags returned (metadata only), and processing outcome. No clinical text is included in audit events.

---

## 10. Model Evaluation Framework

### 10.1 Ground Truth Development

Before transitioning from Phase A (zero-shot) to Phase B (fine-tuned), and for ongoing quality assurance, a structured evaluation framework is required.

**Ground truth corpus:**
- 30+ sample clinical notes (minimum) annotated by qualified behavioral health clinicians
- Each note annotated with expected flags, severity, and the text region that justifies each flag
- Annotations done by two independent clinicians with disagreements resolved by clinical leadership
- Notes should represent the full range of flag domains, severity levels, and tricky edge cases (negation, temporal references, clinical shorthand)

**Evaluation harness:**

```
# training/evaluate.py

For each note in ground truth:
  1. Run bh-sentinel pipeline
  2. Compare detected flags against expected flags
  3. Compute per-flag metrics:
     - Precision: of flags bh-sentinel raised, how many were correct?
     - Recall: of flags the clinician annotated, how many did bh-sentinel find?
     - F1: harmonic mean of precision and recall
  4. Compute per-severity metrics:
     - CRITICAL recall must be > 0.90 (missing a CRITICAL flag is the worst failure mode)
     - HIGH recall must be > 0.80
     - MEDIUM precision must be > 0.70 (too many false MEDIUM flags = alert fatigue)
  5. Log false negatives and false positives for clinical review
```

### 10.2 A/B Comparison (Phase A vs Phase B)

When the fine-tuned model (Phase B) is ready:
1. Run the full ground truth corpus through both models
2. Compare precision/recall/F1 at each severity level
3. Pay special attention to CRITICAL flag recall: the fine-tuned model must not regress
4. Review false negatives from both models with clinical team
5. Document the comparison results as part of the model deployment decision
6. Only deploy the fine-tuned model if it demonstrably improves on zero-shot for CRITICAL and HIGH flags

### 10.3 Public Datasets for Pre-Training

| Dataset | Source | Labels | Size | Access |
|---|---|---|---|---|
| Reddit C-SSRS Suicide Dataset | Zenodo (10.5281/zenodo.2667859) | C-SSRS categories (Supportive, Ideation, Behavior, Attempt) | 500 users, psychiatrist-annotated | Open access |
| UMD Reddit Suicidality Dataset v2 | University of Maryland | No/Low/Moderate/Severe risk | ~1,200 users, expert-annotated | Application + DUA |
| Mental Health Sentiment (Kaggle) | Reddit, Twitter | 7 categories (Normal, Depression, Suicidal, Anxiety, Stress, Bipolar, Personality Disorder) | 52,681 texts | Open access |
| SMHD | Georgetown University | 9 conditions (ADHD, Anxiety, Autism, Bipolar, Depression, Eating Disorder, OCD, PTSD, Schizophrenia) | Large-scale | Application + DUA |

**Important:** These datasets are social-media-sourced, not clinical notes. They are useful for teaching the model behavioral health vocabulary and concept space. Accuracy on clinical documentation improves substantially with fine-tuning on de-identified clinical notes from the deploying organization (even 200-300 annotated notes make a meaningful difference).

### 10.4 Ongoing Monitoring

After production deployment:
- **Clinician feedback loop:** Flag badges in the clinical UI should include a "dismiss / not relevant" action. Dismissed flags feed back into the evaluation dataset.
- **Monthly metrics review:** Precision and recall computed from clinician feedback. Thresholds adjusted based on alert fatigue signals.
- **Quarterly clinical review:** Clinical leadership reviews flag taxonomy, pattern library, and rules engine configuration.

---

## 11. Cost Estimation (AWS Reference Deployment)

| Component | Monthly (Dev) | Monthly (Prod) |
|---|---|---|
| Lambda (container, 2048MB, provisioned concurrency x3) | ~$15 | ~$45 |
| API Gateway (private) | ~$3 | ~$10 |
| AWS Comprehend (VPC endpoint, 5K texts/day) | ~$8 | ~$25 |
| ECR (container image storage) | < $1 | < $1 |
| VPC Endpoints (Comprehend + CloudWatch interface) | ~$15 | ~$15 |
| S3 + CloudWatch Logs + KMS | ~$7 | ~$17 |
| **Total** | **~$49/month** | **~$113/month** |

No SageMaker costs. Transformer model runs in-process within the Lambda container.

**Note:** If Comprehend is disabled for most callers (e.g., mobile app only sends self_harm/harm_to_others without emotion data), Comprehend costs drop proportionally. With Comprehend disabled entirely, total drops to ~$40/month dev, ~$65/month prod.

---

## 12. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| False negatives on CRITICAL flags | Patient safety | Medium | Layer 1 high recall for explicit language. Layer 2 adds contextual detection. Regular clinical review. Prioritize Layer 1 pattern coverage for self_harm and harm_to_others. |
| False positives causing alert fatigue | Provider burnout | High | Confidence thresholds tuned with clinical team. Rules engine allows adjustment. Start conservative. Clinician "dismiss" action feeds back into tuning. |
| Zero-shot accuracy too low in Phase A | Missed implicit signals | Medium-High | Layer 1 carries CRITICAL detection burden during Phase A. Zero-shot is the nuance layer, not the safety net. Accelerate Phase B data collection. |
| ONNX model too large for container | Rebuild needed | Very Low | DistilBERT INT8 is ~65MB. Container limit is 10GB. Total image ~333MB. |
| Cold start exceeds 1s | First request slow | Low | Provisioned concurrency (3 instances) keeps Lambdas warm. |
| PHI leak in logs/errors | HIPAA violation | Low | Zero-egress SGs, code review, log filters, exception handlers, automated PHI audit in CI, VPC Flow Logs. |
| Pattern library gaps | Missed flags | Medium | Clinical review quarterly. Transformer layer catches what patterns miss. Clinician feedback loop fills gaps. |
| Comprehend VPC endpoint latency spike | Exceeds target | Low | Runs in parallel. Optional/degradable with 200ms timeout. Fallback to lexicon-only. Pipeline never fails due to Comprehend. |
| FDA regulatory reclassification | Design changes needed | Low | Architecture designed to satisfy all 4 Non-Device CDS criteria. basis_description and evidence_span fields specifically address Criterion 4. Monitor FDA guidance updates. |
| Taxonomy version drift across consumers | Incompatible flag interpretations | Low | taxonomy_version in every response. Additive changes only at minor version. Breaking changes require coordinated major version rollout. |

---

## 13. Development Roadmap

### v0.1: Core Pattern Engine

- [ ] bh-sentinel-core package scaffolding (pyproject.toml, namespace packages)
- [ ] Pydantic request/response models (including evidence_span, basis_description, pipeline_status, taxonomy_version)
- [ ] Flag taxonomy v1.0 definitions with version tracking
- [ ] Text preprocessor (normalization, sentence splitting, character offset tracking)
- [ ] Negation detector (window-based negation scope)
- [ ] Temporal detector (past vs present tense markers)
- [ ] Pattern matcher (compiled regex from YAML config, evidence span generation, basis_description generation)
- [ ] NRC Emotion Lexicon integration (in-process, ~2ms)
- [ ] Rules engine (condition DSL, severity escalation/de-escalation, compound risk, recommended_action)
- [ ] Pipeline orchestrator with asyncio.gather and graceful degradation
- [ ] Draft patterns.yaml: ~150-200 patterns across all 6 domains including clinical shorthand
- [ ] Unit tests for all components
- [ ] Performance tests (pattern matching < 10ms for 500-word text)
- [ ] PHI containment tests
- [ ] CI pipeline (lint, type-check, test)
- [ ] Publish bh-sentinel-core v0.1.0 to PyPI

### v0.2: ML Layer + Reference Deployment

- [ ] bh-sentinel-ml package scaffolding
- [ ] ONNX transformer inference layer (in-process, no network hop)
- [ ] Zero-shot classification with DistilBART-MNLI (also test BiomedBERT-NLI)
- [ ] Model export tooling (PyTorch to ONNX INT8, scripts/export_onnx.py)
- [ ] basis_description generation for transformer flags
- [ ] AWS Comprehend integration (VPC endpoint, optional, 200ms timeout, graceful fallback)
- [ ] Model evaluation harness with ground truth fixtures
- [ ] AWS Lambda reference deployment (Dockerfile, handler.py)
- [ ] Terraform modules (VPC, Lambda, API Gateway, ECR, S3, monitoring, rate limiting)
- [ ] End-to-end integration tests
- [ ] Latency tests (p99 < 500ms warm)
- [ ] Load tests (100 concurrent requests)
- [ ] VPC Flow Log audit (no outbound traffic leaves VPC)
- [ ] Publish bh-sentinel-ml v0.1.0 to PyPI

### v0.3: Fine-Tuning + Evaluation

- [ ] Public dataset preparation scripts (C-SSRS Reddit, SMHD, Kaggle Mental Health)
- [ ] Domain adaptation pre-training on Reddit mental health corpora
- [ ] Fine-tuning pipeline (DistilBERT multi-label classification head)
- [ ] A/B evaluation framework (zero-shot vs fine-tuned, per-severity metrics)
- [ ] Clinician feedback integration design (dismiss action feeds back into evaluation)
- [ ] Staging and production Terraform configs
- [ ] CloudWatch alarms and dashboard
- [ ] Runbook documentation

### Future

- [ ] Spanish language pattern support (Layer 1 Spanish patterns, multilingual transformer)
- [ ] EMR webhook integration examples
- [ ] Flag trend analysis utilities
- [ ] FHIR-compatible output format
- [ ] Group session note analysis patterns
- [ ] Model retraining pipeline (new container image per model version)
- [ ] Clinician feedback dashboard (precision/recall from dismiss actions)
- [ ] bh-audit-logger integration for compliance tracking

---

*This document should be reviewed with qualified behavioral health clinical leadership before any production deployment. The flag taxonomy in particular needs clinical validation and may require additions or modifications based on program-specific needs.*

*FDA CDS compliance analysis (Section 8) reflects the January 2026 updated guidance. Deploying organizations should review this analysis with their own legal counsel, particularly if extending the tool to patient-facing applications.*

*bh-sentinel is clinical decision support software. It is not a diagnostic tool, not a substitute for clinical judgment, and not FDA-cleared or approved. Organizations deploying this software in clinical settings are responsible for their own clinical validation, regulatory compliance, and patient safety protocols.*
