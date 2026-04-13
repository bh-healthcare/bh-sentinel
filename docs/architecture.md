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
| SH-004 | Active suicidal ideation, with some intent | CRITICAL | Some intent or expectation of acting, without a specific plan |
| SH-005 | Active suicidal ideation, with plan and intent | CRITICAL | Specific plan for suicide with stated intent to act |
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

### 3.3 Config Versioning and Validation

The pipeline depends on three configuration files that must stay in sync. Each has its own version and a declared dependency on the taxonomy version:

| File | Version field | Current | Depends on |
|---|---|---|---|
| `config/flag_taxonomy.json` | `taxonomy_version` | `1.0.0` | — (source of truth) |
| `config/patterns.yaml` | `_meta.patterns_version` | `1.0.0` | `_meta.requires_taxonomy_version` |
| `config/rules.json` | `rules_version` | `1.0.0` | `requires_taxonomy_version` |

**Taxonomy version** is the primary version. The API response includes `taxonomy_version` so downstream consumers know which flag set was active at analysis time.

**Version coupling:** `patterns.yaml` and `rules.json` each declare a `requires_taxonomy_version` using semver range syntax (e.g., `"1.0.x"` means any 1.0.* patch). At init time the pipeline validates that the loaded taxonomy version satisfies the requirement. If it does not, the pipeline refuses to start — a version mismatch means flag_ids in patterns or rules may reference flags that don't exist (or have been renamed/removed).

**Semantic versioning rules:**

- **Patch** (1.0.0 → 1.0.1): description or metadata changes only. No new flag_ids, no removed flag_ids, no severity changes. Patterns and rules remain compatible.
- **Minor** (1.0.0 → 1.1.0): additive changes — new flags, new domains, new optional fields. Existing flag_ids are unchanged. Patterns and rules remain compatible but may not cover new flags. `requires_taxonomy_version: "1.0.x"` is NOT satisfied by 1.1.0 — patterns/rules must be updated and their own versions bumped to declare compatibility with the new taxonomy.
- **Major** (1.0.0 → 2.0.0): breaking changes — removed flags, renamed flag_ids, restructured domains. All config files must be updated and re-versioned together.

**Config validation (runtime init and CI):**

The pipeline runs these checks at init before accepting any requests. CI runs the same checks on every commit that touches `config/`:

1. **Taxonomy loads cleanly.** `flag_taxonomy.json` parses as valid JSON and contains a `taxonomy_version` string.
2. **Version compatibility.** `patterns.yaml._meta.requires_taxonomy_version` and `rules.json.requires_taxonomy_version` are satisfied by the loaded `taxonomy_version`.
3. **Flag ID referential integrity.** Every `flag_id` key in `patterns.yaml` exists in `flag_taxonomy.json`. Every `flag_id` referenced in `rules.json` conditions (`flag_present`, `any_flag_present`, `escalate_flag`, `escalate_flags`) exists in `flag_taxonomy.json`. Orphan flag_ids fail the check.
4. **Domain consistency.** Every domain key in `patterns.yaml` matches a domain `id` in `flag_taxonomy.json`. Every `domain_present` value in `rules.json` matches a domain `id`.
5. **No missing CRITICAL coverage.** Every flag_id with `default_severity: "CRITICAL"` in the taxonomy has at least one pattern entry in `patterns.yaml`. This is a safety gate — CRITICAL flags must have a deterministic Layer 1 fallback.
6. **Regex compilation.** Every pattern and negation phrase in `patterns.yaml` compiles as a valid `re.IGNORECASE` regex.
7. **Vendored copy sync.** The `config/` files are identical to the vendored copies in `packages/bh-sentinel-core/src/bh_sentinel/core/_default_config/`.

If any check fails at runtime init, the pipeline logs the specific failure (without PHI — these are config errors, not request errors) and refuses to start. In CI, the check fails the build.

**Historical queries** reference the taxonomy version in the audit log, so a flag from 6 months ago can be interpreted against the taxonomy that was active at that time.

**Clinical review gate:** All taxonomy changes should be reviewed and signed off by qualified clinical leadership before deployment.

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

**Input validation and boundary behavior:**

| Constraint | Value | Behavior |
|---|---|---|
| Minimum text length | 3 characters (after normalization) | Rejected with 422: text too short to analyze meaningfully. |
| Maximum text length | 50,000 characters | Rejected with 422: exceeds processing budget. Group session transcripts or concatenated notes should be split by the caller. |
| Empty / whitespace-only | — | Rejected with 422 after stripping. |
| Encoding | UTF-8 required | Input is normalized to NFC Unicode form. Null bytes are stripped. Callers receiving text from legacy clinical systems (Latin-1, Windows-1252) must transcode to UTF-8 before submission. |
| Language | `en` only (v0.1–v0.2) | `context.language` accepts only `"en"`. Non-English text is not rejected but will produce low-confidence or zero flags since patterns and the emotion lexicon are English-only. Spanish support is planned for v0.3. |
| Long document handling | Up to 50,000 characters | Layer 1 processes the full text (pattern matching scales linearly, ~5ms for typical notes, ~15ms at 50K chars). Layer 2 uses sentence-level batch inference (see long document strategy in Section 4.5). No truncation of the input occurs at the API layer. |

These constraints are enforced by the `AnalysisRequest` Pydantic model in `bh_sentinel.core.models.request`. The `text` field validator normalizes Unicode to NFC, strips null bytes, and rejects whitespace-only input.

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

**Error responses:**

All error responses use a consistent `ErrorResponse` model. Error messages never contain input text, PHI, or any content derived from the request body. This is enforced by the error model itself (no text field) and validated by `test_no_phi_in_errors` in CI.

```json
{
  "request_id": "uuid-v4",
  "error_code": "VALIDATION_TEXT_TOO_LONG",
  "message": "Input text exceeds maximum length of 50000 characters.",
  "http_status": 422
}
```

| HTTP status | Error code | Trigger | Message |
|---|---|---|---|
| 422 | `VALIDATION_TEXT_TOO_SHORT` | `text` shorter than 3 characters after normalization | "Input text is too short to analyze (minimum 3 characters)." |
| 422 | `VALIDATION_TEXT_TOO_LONG` | `text` exceeds 50,000 characters | "Input text exceeds maximum length of 50000 characters." |
| 422 | `VALIDATION_TEXT_EMPTY` | `text` is empty or whitespace-only after normalization | "Input text must contain non-whitespace content." |
| 422 | `VALIDATION_LANGUAGE_UNSUPPORTED` | `context.language` is not a supported value | "Unsupported language. Supported: en." |
| 422 | `VALIDATION_DOMAIN_UNKNOWN` | `config.domains` contains an unrecognized domain | "Unknown domain in config. Valid domains: self_harm, harm_to_others, medication, substance_use, clinical_deterioration, protective_factors." |
| 422 | `VALIDATION_SEVERITY_UNKNOWN` | `config.min_severity` is not a valid severity level | "Unknown severity level. Valid levels: CRITICAL, HIGH, MEDIUM, LOW, POSITIVE." |
| 200 | (not an error) | All layers fail but input is valid | Returns a normal `AnalysisResponse` with zero flags and `pipeline_status` showing which layers failed. The pipeline is designed to degrade, not error. |
| 500 | `INTERNAL_PIPELINE_ERROR` | Unrecoverable error in the pipeline (e.g., pattern config failed to load) | "Internal analysis error. Request ID: {request_id}." |
| 503 | `SERVICE_UNAVAILABLE` | Lambda cold start timeout or container init failure | "Service temporarily unavailable. Retry after a brief delay." |

**Graceful degradation vs. error:** When individual layers fail (Comprehend timeout, transformer exception), the pipeline continues with remaining layers and returns a 200 with partial results. The `pipeline_status` field shows which layers completed, failed, timed out, or were skipped. Callers should check `pipeline_status` to understand result completeness. A 500 is reserved for cases where the pipeline cannot produce any meaningful output at all (e.g., config load failure, preprocessor crash).

**PHI safety in errors:** The `ErrorResponse` model has no field that could contain request body content. The `message` field is always a static string template with at most a `request_id` interpolated. Exception handlers in the pipeline catch all errors and return the safe model — they never propagate raw Python exceptions or tracebacks to the caller.

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

**Negation scope algorithm (clause-bounded forward window):**

Negation in clinical text is scoped — "Patient denies suicidal ideation but reports feeling hopeless and that there's no point in going on" negates "suicidal ideation" but not "no point in going on." A naive substring check for "denies" anywhere in the sentence would suppress both. The negation detector uses a clause-bounded forward window inspired by [NegEx](https://doi.org/10.1016/S1532-0464(01)00029-6):

1. For each pattern match, look backward from `match_start` by up to `MAX_LOOKBACK` characters (default: 60).
2. Find the nearest negation cue from the flag's `negation_phrases` list within that lookback window.
3. If a negation cue is found, check whether a **scope terminator** appears between the negation cue's end and the pattern match's start.
4. If no terminator intervenes → the match is within negation scope → **suppress** (drop match, or reduce confidence to below threshold).
5. If a terminator is present → negation does not reach the match → **keep**.

Scope terminators (close the negation window):
- Clause-boundary punctuation: `,` `;` `:` `.`
- Conjunctive shifts: `but`, `however`, `although`, `yet`, `except`, `though`, `only`
- Newlines and paragraph breaks

The algorithm is purely positional — no dependency parser, no syntax tree. This keeps it deterministic, fast (~0.1ms per match), and auditable. It handles the vast majority of clinical negation patterns because clinicians and patients overwhelmingly use clause structure to scope their statements.

**Post-negation cues:** A small set of cues appear after the target (e.g., "suicidal ideation: denied", "SI — negative"). These are handled by a separate forward-scan from `match_end` by up to `MAX_FORWARD` characters (default: 30), checking for post-negation cues like `denied`, `negative`, `absent`, `ruled out`. The same terminator logic applies in reverse.

**Clinical shorthand interactions:**

Clinical notes frequently combine negated and affirmed findings in a single line. The clause-boundary termination handles these:

| Input | Negation behavior |
|---|---|
| `denies SI, +HI` | `,` terminates scope after "SI" → SI suppressed, HI not |
| `denies SI/HI` | `/` treated as a list continuation (not a terminator) → both suppressed |
| `-SI, +HI` | `-` is a cue scoped to the immediately adjacent term → SI suppressed, HI not |
| `no SI or HI` | `or` is not a terminator → both suppressed (correct: "no" scopes over a conjunctive list) |
| `no SI but has HI` | `but` terminates scope → SI suppressed, HI not |
| `pt denies SI. Reports HI.` | `.` terminates scope → SI suppressed, HI not |

**Pseudo-negation (false negation cues):** Some phrases look like negation but are not:
- "no improvement" — "no" negates "improvement," not the clinical finding
- "not able to sleep" — "not" negates "able," which means the symptom IS present
- "no longer denies" — double negation, the finding IS present

These are handled by a pseudo-negation exclusion list compiled at init. If the lookback window contains a pseudo-negation phrase, the negation check is skipped. The exclusion list includes: `no longer denies`, `does not deny`, `no improvement`, `not able to`, `unable to`, `cannot`, `no longer`.

**When negation is not applied:** Some flags intentionally omit `negation_phrases` because negation of the concept is a different clinical signal. For example, MED-001 (non-adherence) patterns like "stopped taking medication" should not be suppressed by "no longer stopped taking medication" — that would indicate adherence (PF-005), a different flag. Flags without `negation_phrases` skip the negation check entirely.

**Example pattern entries (simplified for readability):** These examples use capturing groups `()` for clarity. The canonical `config/patterns.yaml` uses non-capturing groups `(?:)`, apostrophe handling `'?`, and pronoun flexibility groups per the conventions in `docs/pattern-library.md`. Always refer to `config/patterns.yaml` for the actual production patterns.

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

**Pattern library sizing:** The 40 flags across 6 domains, combined with negation variants, temporal variants, colloquial language variants ("what's the point anymore" vs "suicidal ideation"), and clinical shorthand (e.g., "pt c/o SI", "denies HI"), require approximately 150-200 patterns in production. The current library contains 231 patterns + 98 clinical shorthand entries = 329 total.

Clinical teams should be able to add/modify patterns without code deployment (patterns.yaml is a configuration file, not code).

**Temporal detection algorithm (per-match keyword classification):**

Clinical notes routinely mix past and present references in one document: "I used to cut myself in high school. Last night I started again." The first sentence is historical. The second is active relapse, and clinically meaningful precisely because of the historical context. A naive past-tense heuristic that blankets the whole input would misclassify the second sentence.

The temporal detector classifies each pattern match independently using keyword proximity, not the entire document or sentence:

1. For each pattern match, scan a window of `TEMPORAL_LOOKBACK` characters (default: 80) before `match_start` and `TEMPORAL_LOOKAHEAD` characters (default: 40) after `match_end`.
2. Check for **past-context markers** and **present-context markers** within the window.
3. If a present-context marker is found, the match is classified `present` regardless of any co-occurring past markers (present overrides past).
4. If only past-context markers are found, the match is classified `past`.
5. If neither is found, the match defaults to `present` (current until proven otherwise — the safe default for a safety system).

**Past-context markers** (indicate historical reference):
- Temporal phrases: `used to`, `in the past`, `previously`, `years ago`, `months ago`, `when I was`, `as a (child|teenager|kid|adolescent)`, `back in`, `history of`, `hx of`, `prior`, `former`, `remote history`
- Past-tense verbs near clinical terms: `had been`, `was (diagnosed|hospitalized|treated)`, `attempted (in|back|years|months)`, `tried to .* (ago|before|in \d{4})`
- Clinical shorthand: `hx`, `PMH`, `past medical history`, `prior episode`

**Present-context markers** (override past, indicate current/recent):
- Recency phrases: `currently`, `right now`, `today`, `this week`, `last night`, `yesterday`, `recently`, `again`, `started again`, `still`, `now`, `at this time`
- Active-state phrases: `is (cutting|using|drinking|hearing)`, `has been`, `reports`, `endorses`, `ongoing`, `continues to`, `active`
- Escalation phrases: `started again`, `relapsed`, `returned to`, `picked back up`, `worse`, `increasing`

**Why present overrides past:** In "I used to cut myself. Last night I started again," the second sentence contains both a past reference ("used to" in the prior sentence, if the window is wide enough) and a present marker ("last night", "started again"). Present override ensures the relapse is not de-escalated. This is the clinically safe default — a false-positive current flag is better than a false-negative historical classification on an active safety concern.

**Per-match, not per-sentence:** Temporal context is assigned per pattern match, not per sentence. A single sentence can contain multiple matches with different temporal classifications: "She has a history of SI but is currently endorsing active suicidal thoughts" — "history of SI" matches SH-008 (historical attempt) with `past` context, "active suicidal thoughts" matches SH-002 with `present` context. Per-match classification handles this correctly.

**Interaction with DE-001 (severity de-escalation):**

The rules engine DE-001 rule reduces severity by one level when `temporal_context` is `past`. Critical constraints:

- DE-001 applies per-flag, not per-session. If SH-001 is detected twice — once with `past` context and once with `present` — the `present` match takes precedence. The flag is emitted with `present` context and full severity.
- **Present always wins for a given flag_id.** If any match for a flag_id is classified `present`, the flag's temporal context is `present`. De-escalation only applies when ALL matches for that flag_id are classified `past`.
- DE-001 never de-escalates below LOW. A CRITICAL flag with `past` context becomes HIGH, not dropped.
- The `basis_description` for a de-escalated flag includes the temporal context: "Severity reduced: historical reference (past context detected)."

**What the temporal detector does NOT do:**

- It does not use a grammar parser or POS tagger for tense detection. Verb tense in clinical notes is unreliable ("Patient reports cutting" could be present or habitual; "Patient cut himself" could be yesterday or ten years ago). Keyword proximity is more robust for clinical text than syntactic tense.
- It does not propagate context across sentences. Each match's temporal window is independent. Cross-sentence reasoning ("She talked about her childhood trauma. That's when the cutting started.") requires understanding that "that's when" refers to childhood, which is a discourse-level inference beyond the scope of a deterministic keyword detector. Layer 2 (transformer) may capture these patterns; Layer 1 does not attempt to.

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

**Evidence span strategy for transformer flags:** Zero-shot and fine-tuned classifiers produce document-level or sequence-level scores, not character-level spans. To satisfy the Criterion 4 requirement that every flag must include a locatable `evidence_span`, the transformer layer must use **sentence-level classification**: run inference per sentence (using sentence boundaries from the preprocessor), and use the highest-scoring sentence's boundaries as the evidence span. This produces a span that maps to a readable unit a clinician can review in context, and the attribution story is straightforward — "this sentence scored highest for this flag category." If Layer 1 also detected the same flag, prefer the pattern matcher's more precise character-level span. The transformer's sentence-level span is the fallback for transformer-only detections.

**`basis_description` and `evidence_span` are both Criterion 4 requirements.** The `Flag` model requires both fields (non-optional). The transformer layer cannot emit a flag without them. Both `basis_description` generation and `evidence_span` generation for transformer flags must be solved before Layer 2 ships — they are not independent of the CDS alignment that the rest of the architecture is built around.

**Long document strategy:** DistilBERT's token window is 512 tokens (~300-400 words). Clinical intake notes can run 1,000-3,000 words. Sentence-level classification solves both the evidence_span problem and the token window problem simultaneously:

1. **Sentence splitting** (preprocessor, already required for Layer 1). The preprocessor produces sentence boundaries with character offsets tracked against the original document. Individual clinical sentences almost always fall well under 512 tokens.
2. **Batch inference.** All sentences from the note are tokenized, padded to uniform length, and stacked into a single input tensor. ONNX Runtime runs one forward pass on the batch. This avoids N separate inference calls and keeps latency within the ~50ms budget for typical notes (10-30 sentences). For very long notes (50+ sentences), the batch can be split into sub-batches to limit memory pressure, with results concatenated.
3. **Per-flag aggregation.** For each flag category in the multi-label classification head, take the maximum score across all sentences. The sentence that produced the max score provides the evidence_span for that flag.
4. **Offset mapping.** Since the preprocessor tracks `(sentence_index, char_start, char_end)` for every sentence boundary, the winning sentence's offsets map directly back to the original document. No additional offset translation is needed — the preprocessor already maintains the mapping.

**Edge cases:**
- If a single sentence exceeds 512 tokens (rare in clinical notes), truncate the token sequence with a right-truncation strategy. The sentence boundaries still provide the evidence_span.
- Cross-sentence context is lost in sentence-level classification. A patient may say "She said she wants to die" followed by "She was talking about her sick pet." The model classifies each sentence independently and cannot use the second sentence to contextualize the first. This is mitigated by: (a) Layer 1 patterns catching unambiguous phrasing, (b) confidence scores reflecting the model's uncertainty on ambiguous single-sentence inputs, and (c) the rules engine combining cross-domain signals that may resolve ambiguity. For Phase B fine-tuning, sentence-pair or small-window strategies can be explored if isolated sentence classification proves insufficient.

**Latency impact:** The ~50ms latency estimate in the latency budget (Section 4.3) assumes batch inference over a typical clinical note. A 20-sentence intake note batched into a single ONNX call is comparable in wall-clock time to a single 512-token pass because the batch runs as a single matrix operation on the CPU. Notes over ~50 sentences may push Layer 2 toward ~80-100ms, which is still within the parallel execution budget (Layer 3 Comprehend timeout is 200ms).

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

**Emotion detection** via an in-process behavioral health emotion lexicon (`config/emotion_lexicon.json`). No ML dependency, no model to load, ~2ms. This always runs regardless of Comprehend availability.

The lexicon is project-owned, curated from standard behavioral health clinical vocabulary, DSM-5 descriptive language, and common patient-reported emotional terms. It is not derived from the NRC Emotion Lexicon or any other proprietary dataset. The NRC EmoLex was evaluated but its categories (anger, anticipation, disgust, fear, joy, sadness, surprise, trust) are generic — the clinically important categories for behavioral health (hopelessness, agitation, guilt, shame, dissociation, mania) are not covered by NRC. NRC also requires a separate commercial license for non-research use.

**Emotion categories (11, designed for behavioral health clinical utility):**

| Category | Clinical relevance | Example terms |
|---|---|---|
| hopelessness | Proximal correlate of suicide risk; supports ESC-001 (SI + hopelessness) | hopeless, trapped, pointless, no way out, giving up |
| agitation | Increases impulsivity and risk of acting on ideation | agitated, restless, pacing, can't sit still, explosive |
| anxiety | Co-occurring anxiety intensifies suicidal crises (COMP-002) | anxious, panicking, terrified, overwhelmed, dreading |
| anger | Elevates harm-to-others risk | furious, enraged, hostile, seething, rage |
| sadness | General depressive context | crying, depressed, miserable, empty, grieving |
| guilt | Common in self-harm presentations and trauma | my fault, blame, regret, should have, remorse |
| shame | Predicts concealment and reduced help-seeking | worthless, pathetic, disgusting, burden, unlovable |
| mania | Signals manic episode context for CD-008 | invincible, unstoppable, don't need sleep, racing thoughts |
| dissociation | Signals dissociative context for CD-006 | detached, numb, foggy, unreal, watching myself |
| positive_valence | Baseline positive sentiment; supports protective factor context | hopeful, grateful, calm, improving, supported |
| negative_valence | Baseline negative sentiment | suffering, struggling, unbearable, deteriorating |

The lexicon contains 247 terms with binary category associations (a word either belongs to a category or doesn't). Scoring produces a 0.0-1.0 density score per category (count of matched words / total word count). Clinical teams can review and extend the lexicon the same way they review patterns — it is configuration, not code.

**Sentiment vs. emotion:** Comprehend (Layer 3, optional) provides sentiment analysis (positive/negative/neutral/mixed with confidence scores). The emotion lexicon provides emotion category analysis. These are complementary: Comprehend says "the overall tone is negative"; the lexicon says "the negativity is predominantly hopelessness and shame." When Comprehend is unavailable, the lexicon's positive_valence and negative_valence categories provide a basic sentiment fallback.

#### Layer 4: Rules Engine (bh-sentinel-core)

Combines signals from Layers 1-3 into final flag determinations. Runs in-process after all parallel layers complete. Rules are loaded from `config/rules.json` and cached. Rules gracefully handle missing Layer 3 Comprehend data.

Rules are configuration, not code. Clinical teams can modify detection logic without engineering involvement. The canonical rule set is in `config/rules.json`; the examples below are representative, not exhaustive.

**Rules DSL:** The rules engine uses a structured JSON format with typed conditions and actions. No string parsing or expression evaluation is needed — conditions are evaluated by matching against the pipeline's flag results, temporal context, and emotion scores.

**Condition types:**

| Type | Description |
|---|---|
| `flag_present` | A specific flag was detected. Optional `min_confidence` threshold. |
| `any_flag_present` | Any of the listed flags was detected. Optional `min_confidence` threshold. |
| `domain_present` | Any flag in the specified domain was detected. |
| `domain_severity` | Any flag in the domain reached at least the given severity (evaluated after escalation rules). |
| `temporal_context` | The temporal qualifier of the matched text (`past` or `present`). |
| `emotion_above` | A Layer 3 emotion score exceeds the given threshold. Use sparingly; flag co-occurrence is preferred over raw emotion scores. |
| `domain_flag_count` | Total count of detected flags and L2 candidate detections in a domain. L1 counts are always included; set `include_l2_candidates: true` to also count L2 candidates above 0.4 but below emit threshold. See Section 4.9. |

Conditions combine with `all_of` (all conditions must be true) or `any_of` (at least one must be true).

**Rule categories and evaluation order:** `escalation_rules` → `de_escalation_rules` → `compound_rules` → `action_rules`. Within each category, rules are evaluated in array order.

**Example: Escalation rule (ESC-001)**

```json
{
  "id": "ESC-001",
  "description": "Passive SI combined with hopelessness escalates to CRITICAL",
  "condition": {
    "all_of": [
      {"flag_present": "SH-001", "min_confidence": 0.7},
      {"flag_present": "CD-001", "min_confidence": 0.7}
    ]
  },
  "action": {
    "escalate_flag": "SH-001",
    "new_severity": "CRITICAL",
    "recommended_action": "Administer C-SSRS"
  }
}
```

This uses flag co-occurrence rather than raw emotion scores. The hopelessness signal is already captured by CD-001 through Layers 1-2, which is more reliable than conditioning on an emotion lexicon threshold. The `emotion_above` condition type is available for cases where emotion context adds signal beyond what flags capture, but flag-based conditions should be the default.

**Example: Compound rule (COMP-002)**

```json
{
  "id": "COMP-002",
  "description": "Anxiety-to-psychosis escalation: co-occurring anxiety and psychosis flags indicate possible decompensation",
  "condition": {
    "all_of": [
      {"flag_present": "CD-007", "min_confidence": 0.7},
      {"any_flag_present": ["CD-005a", "CD-005b", "CD-005c", "CD-005d"], "min_confidence": 0.7}
    ]
  },
  "action": {
    "escalate_flags": ["CD-005a", "CD-005b", "CD-005c", "CD-005d"],
    "new_severity": "CRITICAL",
    "set_requires_immediate_review": true,
    "recommended_action": "Evaluate for psychotic decompensation with anxiety prodrome"
  }
}
```

A single compound rule can both escalate flag severity and set session-level annotations (`requires_immediate_review`).

**Full rule set:** See `config/rules.json` for all 20 rules. Summary:

| Category | Rules | Coverage |
|---|---|---|
| Escalation (10) | ESC-001 through ESC-010 | Passive SI + hopelessness, NSSI + substance use, active SI + command AH, HI + substance use, mania + risky substance behavior, prior attempt + current SI, hopelessness + isolation, active SI + any destabilizing factor (`any_of`), active SI + agitation (`emotion_above`), HI + anger (`emotion_above`) |
| De-escalation (1) | DE-001 | Historical references reduce severity by one level |
| Compound (8) | COMP-001 through COMP-008 | Substance use + self-harm, anxiety + psychosis, signal density (self-harm), signal density (harm-to-others), medication misuse + self-harm, HI + paranoid ideation, NSSI + passive SI, multi-domain high-acuity (`any_of`) |
| Action (1) | ACT-001 | C-SSRS prompt for any CRITICAL self-harm flag |

**Clinical review gate:** This rule set is an initial matrix covering the most clinically established high-risk combinations. It is not exhaustive. Before production deployment, the full escalation matrix should be reviewed and signed off by qualified behavioral health clinical leadership. Additional combinations to evaluate include but are not limited to: SI + severe sleep disruption (CD-003), substance withdrawal (SU-004) + any self-harm signal, mania (CD-008) + HI, and multi-domain presentations with 3+ domains flagged simultaneously. The rules engine is configuration — clinical teams can extend the matrix without engineering involvement.

**Evidence span preservation:** When a rule escalates or de-escalates an existing flag, the original `evidence_span` and `basis_description` carry through. The rule appends its own rationale to `basis_description` (e.g., "Severity escalated: co-occurring anxiety and psychosis") but does not replace the source span. This ensures that Criterion 4 traceability is maintained through compound rules — the clinician can still locate the original text that triggered the flag.

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

### 4.7 Flag Deduplication and Layer Merge

Layer 1 and Layer 2 can independently detect the same flag_id from the same input text. The response emits **one flag per flag_id**, not one per layer. Deduplication runs after L1 and L2 complete but before the rules engine (L4) evaluates.

**Merge algorithm:**

1. Collect all candidate flags from L1 and L2 into a map keyed by `flag_id`.
2. If a flag_id was detected by only one layer, use that detection as-is.
3. If a flag_id was detected by both layers:
   - **`detection_layer`**: set to the layer with the higher confidence score.
   - **`confidence`**: take `max(l1_confidence, l2_confidence)`. Do not average — the stronger signal represents the better detection, not a noisy mean.
   - **`evidence_span`**: prefer Layer 1's character-level span over Layer 2's sentence-level span. Pattern matching produces exact `(char_start, char_end)` from the regex match; the transformer's sentence-boundary span is coarser. If only L2 detected the flag, use L2's sentence-level span.
   - **`basis_description`**: use the winning layer's description, with a corroboration note appended (e.g., "Also detected by pattern_match layer at confidence 0.92").
   - **`matched_context_hint`**: use the winning layer's hint.
   - **`corroborating_layers`**: populated with the list of all layers that detected this flag (see below).

**Corroboration metadata:**

The `Flag` model includes a `corroborating_layers` field:

```python
corroborating_layers: list[DetectionLayer] = []
```

When only one layer detects a flag, this list is empty (the primary `detection_layer` is the sole source). When both L1 and L2 detect the same flag_id, `corroborating_layers` contains the non-primary layer. Example: if L1 wins on confidence, `detection_layer` is `pattern_match` and `corroborating_layers` is `["transformer"]`.

This field serves three purposes:
- **Rules engine**: corroborated flags can be treated as higher-confidence inputs. A rule condition like `{"flag_present": "SH-002", "min_confidence": 0.7, "corroborated": true}` is more restrictive than confidence alone.
- **Downstream consumers**: clinical UIs can display a "confirmed by multiple detection methods" indicator, which supports Criterion 4 (independent review of the basis).
- **Evaluation**: the model evaluation harness can track L1/L2 agreement rates to measure whether the transformer is adding signal or just echoing patterns.

**Why max, not average or boost:**

Averaging would penalize a high-confidence L1 match (0.95) when L2 produces a lower score (0.70), yielding 0.825 — worse than L1 alone. A confidence boost formula (e.g., `min(1.0, max + 0.05)`) was considered but adds a tunable parameter without clear clinical justification. Taking the max is simple, predictable, and never degrades confidence relative to the best single-layer detection. The `corroborating_layers` field provides the agreement signal without distorting the confidence score.

**When layers disagree on severity or temporal context:**

- **Severity**: both layers use the same `default_severity` from `flag_taxonomy.json`, so severity does not diverge between layers. Severity changes come from the rules engine (L4) after deduplication.
- **Temporal context**: each layer's matches are independently classified by the temporal detector. If L1 classifies SH-002 as `past` and L2 classifies it as `present`, the present-wins rule applies (see temporal detection algorithm). The merged flag gets `present` context.

### 4.8 Confidence Score Calibration

Pattern matches and transformer outputs both produce a `confidence` field on the `Flag` model, but they originate from fundamentally different processes and mean different things unless explicitly calibrated.

**What confidence means (definition):** The confidence score on a flag represents the **estimated probability that the flag is correctly identified** — that the flagged clinical concept is actually present in the input text. A confidence of 0.90 means "we estimate a 90% chance this flag is a true positive." This definition must hold regardless of which layer produced the score, so that `min_confidence` thresholds in the rules engine and downstream UIs behave consistently.

**Layer 1 (pattern matching) — fixed precision estimates:**

Pattern confidence scores in `patterns.yaml` are manually set per flag_id (not per match). They represent the pattern author's estimate of precision for that flag's pattern set: how often a match from these patterns is a true positive, accounting for the negation and temporal detectors. A confidence of 0.92 on SH-001 means "when these patterns fire (and survive negation/temporal checks), we estimate 92% are true positives."

These scores are:
- Static: they do not vary per input. Every SH-001 pattern match gets 0.92.
- Author-estimated: set during pattern authoring based on pattern specificity and expected false-positive rate.
- Validated empirically: during evaluation (v0.3), actual precision is measured against clinician-labeled data. If SH-001's measured precision is 0.88, the confidence score should be adjusted from 0.92 to 0.88 to reflect reality.

This is the correct approach for Layer 1 because deterministic regex matches are binary (match or no match) — there is no per-input probability to calibrate. The confidence is a property of the pattern set, not of the individual match.

**Layer 2 (transformer) — calibrated softmax probabilities:**

Transformer outputs are raw softmax scores from the classification head. These are notoriously miscalibrated: a softmax output of 0.85 does not mean "85% chance of being correct." Neural networks, especially small ones, tend to be overconfident — producing high scores even on ambiguous inputs.

To align transformer confidence with the same "probability of correctness" scale as Layer 1, transformer scores must be calibrated:

- **Phase A (zero-shot):** Apply **temperature scaling** on a held-out validation set of clinician-labeled examples. Temperature scaling learns a single scalar T that adjusts the softmax logits: `calibrated = softmax(logits / T)`. This is the simplest post-hoc calibration method and requires only ~100-200 labeled examples per flag category. Until calibration data is available, raw softmax scores should be discounted by a fixed factor (e.g., multiply by 0.85) as a conservative interim measure, documented in the pipeline config.
- **Phase B (fine-tuned):** Apply **Platt scaling** (logistic regression on logits) or temperature scaling on the validation split of the fine-tuning dataset. Recalibrate after every model retrain.
- **Calibration validation:** After calibration, the expected calibration error (ECE) should be below 0.05 on the validation set. This means that among all flags emitted with confidence ~0.80, approximately 80% are true positives.

**Why this matters for the merge and rules engine:**

The deduplication merge (Section 4.7) uses `max(l1_confidence, l2_confidence)`. If L2 scores are uncalibrated and systematically higher than L1, the transformer will always win the merge — even when L1's pattern match is more reliable. Calibration ensures the max operation selects the genuinely stronger signal.

The rules engine's `min_confidence` thresholds (e.g., `0.7` in ESC-001) assume a uniform scale. A threshold of 0.7 should mean "flag with at least 70% estimated probability of being correct" from either layer. Without calibration, a 0.7 threshold might be conservative for L1 (which rarely produces false positives) and permissive for L2 (which may produce overconfident scores on ambiguous inputs).

**Clinician-facing display:**

Raw confidence scores should not be shown to clinicians as floating-point numbers. They are not interpretable without context. Instead, map to a discrete confidence tier for UI display:

| Score range | Display tier | Meaning |
|---|---|---|
| 0.90–1.00 | High confidence | Strong match — review for clinical action |
| 0.70–0.89 | Moderate confidence | Probable match — review recommended |
| 0.50–0.69 | Low confidence | Possible match — use clinical judgment |
| Below 0.50 | Not displayed | Below threshold — flag suppressed |

The raw score remains available in the API response for programmatic consumers, evaluation, and audit. The tier labels are a UI concern, not a pipeline concern — they are defined here for consistency across integrating systems.

**Calibration is a v0.2 deliverable.** During v0.1 (Layer 1 only), confidence scores are the static pattern-author estimates. During v0.2 (Layer 2 ships), transformer calibration must be implemented before the scores are meaningful. The Phase A interim discount factor provides a conservative safety margin until proper calibration is possible.

### 4.9 Cross-Sentence Signal Detection

**Known limitation:** Some clinical signals only emerge from the accumulation of individually ambiguous statements across multiple sentences. Example:

> "I gave away my dog last week. Cleaned out my closet and gave things to my sister. Updated my will."

No single sentence says "suicidal preparatory behavior" (SH-006), but the pattern across sentences is a textbook description. Layer 1 (pattern matching) processes each sentence independently — it cannot reason across sentence boundaries. Layer 2 (sentence-level transformer classification) has the same limitation in the current architecture.

**Decision: True cross-sentence semantic detection is deferred to Phase B (v0.3). Two partial mitigations are in scope for v0.1.**

**Mitigation 1 — Broad preparatory behavior patterns (Layer 1):**

SH-006 patterns are authored to catch individual preparatory actions in isolation, not just as part of a multi-sentence narrative. "Giving away belongings", "updating will", "putting affairs in order", "writing goodbye letter" each match independently. Not every ambiguous sentence will match (giving away a dog doesn't match "giving away belongings" because it's a single specific item), but the pattern library should be as inclusive as clinically defensible for individual preparatory actions.

**Mitigation 2 — Signal density rules (Layer 4):**

The rules engine can detect when multiple flags from the same domain co-occur in the same input, which suggests accumulated risk even when each individual flag is routine on its own.

Signal density works differently for each layer:

- **Layer 1 (pattern matching):** Every match is binary — a pattern either fires or it doesn't, and every match gets the flag's fixed confidence score. There is no "sub-threshold" for L1. The density signal is **match count**: how many distinct pattern matches fired for flags within a domain. If SH-006 fires twice in the same note (once for "gave things to my sister", once for "updated my will"), that's two matches. If SH-001 also fires, that's three self_harm domain matches total. Three or more matches from the same domain suggests an accumulation pattern.

- **Layer 2 (transformer):** Softmax scores vary per sentence. A sentence might score 0.55 for SH-006 — too low to emit as a flag, but meaningful in aggregate. Layer 2 reports these as **candidate detections** with the raw score. The density rule can count them alongside L1's full-confidence matches.

The `domain_flag_count` condition counts both:

```json
{
  "id": "COMP-003",
  "description": "Signal density: multiple self-harm indicators suggest accumulated risk",
  "condition": {
    "domain_flag_count": {
      "domain": "self_harm",
      "min_count": 3,
      "include_l2_candidates": true
    }
  },
  "action": {
    "set_requires_immediate_review": true,
    "recommended_action": "Multiple self-harm indicators detected across the text. Review the full note for accumulated risk signals."
  }
}
```

**Counting rules:**
- Each distinct L1 pattern match (post-negation, post-temporal) counts as 1, regardless of which specific pattern within the flag matched.
- Each L2 candidate detection (score above a `candidate_threshold` of 0.4 but below the `emit_threshold` of 0.7) counts as 1 when `include_l2_candidates` is true.
- The same flag_id can contribute multiple counts if it matched in multiple locations (e.g., SH-006 matching twice in different sentences = 2 counts).
- Flags that were already emitted (above `emit_threshold`) also count — density is total signal, not just marginal signal.

**No fuzzy matching.** Layer 1 remains fully deterministic — binary match with negation and temporal checks. The density signal comes from quantity of matches across the text, not from weakening the match criteria.

| Type | Description |
|---|---|
| `domain_flag_count` | Total count of detected flags and L2 candidate detections in a domain meets or exceeds `min_count`. L1 counts are always included. Set `include_l2_candidates: true` to also count Layer 2 detections that scored above `candidate_threshold` (0.4) but below `emit_threshold` (0.7). |

**Phase B solution — Sentence-window transformer inference:**

The proper solution for cross-sentence detection is sliding-window inference in Layer 2. Instead of classifying each sentence independently, classify overlapping windows of 2-3 sentences. A 3-sentence window over the preparatory behavior example would feed "I gave away my dog last week. Cleaned out my closet and gave things to my sister. Updated my will." to the transformer as a single input — well within the 512-token limit — and the model can learn that this accumulation pattern maps to SH-006.

This was noted as an option in the long-document strategy (Section 4.5): "For Phase B fine-tuning, sentence-pair or small-window strategies can be explored if isolated sentence classification proves insufficient." Cross-sentence signal detection is the primary use case that makes this exploration necessary.

**Evidence span for density-triggered flags:** When COMP-003 fires, the evidence span should point to the full range from the first match to the last (i.e., `char_start` of the earliest match through `char_end` of the latest). The `basis_description` should enumerate the individual signals: "3 self-harm indicators detected across the text: giving away possessions (sentence 1, SH-006), distributing belongings (sentence 2, SH-006), updating legal documents (sentence 3, SH-006)."

**What this does NOT solve:** Signals that require world knowledge or deep discourse reasoning ("She talked about putting everything in order. Then she asked about her life insurance policy.") are beyond the scope of pattern matching and sub-threshold counting. These require either a fine-tuned transformer with appropriate training examples or human clinical judgment. The architecture is explicit that bh-sentinel supplements, never replaces, clinical assessment.

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
      TAXONOMY_PATH   = "/opt/config/flag_taxonomy.json"
      LEXICON_PATH    = "/opt/config/emotion_lexicon.json"
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

**Config loading strategy:** Config files are split between baked-into-image and loaded-from-S3 based on change frequency and operational coupling:

| Config | Location | Rationale |
|---|---|---|
| `flag_taxonomy.json` | Baked into image | Changes require coordinated version bumps across patterns, rules, and consumers. Rebuilding the image enforces this coordination. |
| `emotion_lexicon.json` | Baked into image | Changes rarely. Lexicon version is coupled to taxonomy version. |
| `patterns.yaml` | S3 (loaded at init) | Clinical teams extend patterns frequently. S3 allows updates without image rebuild or deployment. |
| `rules.json` | S3 (loaded at init) | Clinical teams tune rules frequently. Same rationale as patterns. |
| ONNX model | Baked into image | Model is ~65MB and changes only on retrain (v0.3+). |

The init-time config validator (Section 3.3) runs after S3 configs are loaded, verifying that `patterns.yaml` and `rules.json` are compatible with the baked-in taxonomy version. If a clinical team updates patterns in S3 that reference a flag_id from a newer taxonomy, the validator rejects the config at init and the Lambda reports `SERVICE_UNAVAILABLE` until the image is rebuilt with the new taxonomy.

```dockerfile
FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ONNX model baked into image - no runtime download
COPY models/ /opt/models/

# Configs baked into image (rarely change, version-coupled)
COPY config/flag_taxonomy.json /opt/config/flag_taxonomy.json
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
│   │   │   ├── emotion_lexicon.py     # Behavioral health emotion lexicon (in-process)
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
│   ├── patterns.yaml                  # Default pattern library (351 patterns across 40 flags)
│   ├── rules.json                     # Default rules engine configuration
│   ├── flag_taxonomy.json             # Versioned flag taxonomy (v1.0.0)
│   ├── emotion_lexicon.json           # Behavioral health emotion lexicon data (project-owned)
│   └── test_fixtures.yaml             # Pattern test fixtures (84 cases across all 40 flags)
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

### 7.6 Observability and Metrics

The pipeline is designed to emit structured metrics through two channels: **bh-audit-logger** for compliance-grade event tracking (integration planned for the deployment layer in Phase 3), and **structured JSON log lines** for operational metrics queryable via CloudWatch Logs Insights or any log aggregation tool. In v0.1, the core library does not emit audit events directly — audit integration is the responsibility of the deployment boundary (Lambda handler, FastAPI middleware, etc.).

**Channel 1: bh-audit-logger integration**

Each analysis request produces a single audit event using the `bh-audit-schema` v1.1 format (see `bh-audit-schema/schema/audit_event.schema.json`). This provides tamper-evident, compliance-grade records of every recommendation the system produced and every action the clinician took in response.

```python
audit_logger.emit(
    action=Action(type="CREATE", name="analyze_clinical_text", phi_touched=True, data_classification="PHI"),
    resource=Resource(type="AnalysisResult", id=request_id),
    outcome=Outcome(status="SUCCESS"),
    correlation=Correlation(request_id=request_id),
    metadata={
        "taxonomy_version": "1.0.0",
        "total_flags": 2,
        "max_severity": "MEDIUM",
        "domains_flagged": "clinical_deterioration,medication",
        "requires_immediate_review": False,
        "processing_time_ms": 142,
        "detection_layers_used": "pattern_match,emotion_lexicon",
        "comprehend_available": True,
    },
)
```

The `metadata` field carries only scalar values (per bh-audit-schema constraints), never flag details, text content, or evidence spans. This is enough for compliance queries ("what recommendations were presented for this patient?") and aggregate analytics ("how many CRITICAL flags per week?") without any PHI leakage.

For clinician dismiss/accept actions, the downstream UI emits its own audit event linking the original `request_id`:

```python
audit_logger.emit(
    action=Action(type="UPDATE", name="clinician_flag_review", phi_touched=False, data_classification="NONE"),
    resource=Resource(type="FlagReview", id=review_id),
    outcome=Outcome(status="SUCCESS"),
    correlation=Correlation(request_id=original_request_id),
    metadata={
        "flag_id": "SH-002",
        "clinician_action": "dismiss",
        "dismiss_reason": "historical_reference",
    },
)
```

This creates the feedback loop for the v0.3 clinician feedback integration without bh-sentinel itself needing to know about dismiss actions — the audit trail provides the join key (`request_id`).

**Channel 2: Structured operational metrics**

Each request emits a single JSON log line to a dedicated metrics log group (separate from the audit log). These lines are not audit events — they are operational telemetry for monitoring, alerting, and tuning.

```json
{
  "metric_type": "pipeline_request",
  "request_id": "uuid-v4",
  "timestamp": "2026-04-01T14:22:00Z",
  "processing_time_ms": 142,
  "layers": {
    "pattern_match": {"status": "completed", "latency_ms": 4, "flags_produced": 2},
    "transformer": {"status": "completed", "latency_ms": 48, "flags_produced": 1},
    "comprehend": {"status": "timeout", "latency_ms": 200, "flags_produced": 0},
    "emotion_lexicon": {"status": "completed", "latency_ms": 2},
    "rules_engine": {"status": "completed", "latency_ms": 3, "escalations": 0, "de_escalations": 0}
  },
  "flags_emitted": [
    {"flag_id": "CD-003", "severity": "MEDIUM", "confidence": 0.91, "detection_layer": "pattern_match", "corroborated": false},
    {"flag_id": "MED-001", "severity": "MEDIUM", "confidence": 0.87, "detection_layer": "pattern_match", "corroborated": false}
  ],
  "summary": {
    "total_flags": 2,
    "max_severity": "MEDIUM",
    "requires_immediate_review": false,
    "domains_flagged": ["clinical_deterioration", "medication"],
    "comprehend_fallback": true,
    "deduplication_merges": 0
  }
}
```

No input text, no evidence spans, no patient identifiers. Only flag metadata and operational counters.

**Key metrics derived from structured logs:**

| Metric | Source | Alert threshold | Purpose |
|---|---|---|---|
| Per-flag detection rate | `flags_emitted[].flag_id` count over time | Flag fires >3x or <0.3x its 30-day baseline | Pattern too broad (false positives) or too narrow (regression) |
| Per-layer latency (p50/p95/p99) | `layers.*.latency_ms` | p99 > 200ms for L1, p99 > 150ms for L2 | Performance regression |
| Comprehend fallback rate | `summary.comprehend_fallback` count / total | >10% of requests | VPC endpoint issue or Comprehend service degradation |
| Layer error rate | `layers.*.status == "failed"` count / total | >1% of requests for any layer | Code or config error |
| CRITICAL flag rate | `flags_emitted` where `severity == "CRITICAL"` / total | Sudden spike or drop vs baseline | Clinical workflow impact; may indicate pattern or taxonomy change |
| Deduplication merge rate | `summary.deduplication_merges` / total | Informational (no alert) | Measures L1/L2 agreement; high rate means transformer echoes patterns |
| Corroboration rate | `flags_emitted` where `corroborated == true` / total flags | Informational (no alert) | Measures multi-layer detection reliability |
| Dismiss rate (from audit trail) | `clinician_flag_review` events where `clinician_action == "dismiss"` / total flags | >30% for any flag_id over 30 days | Alert fatigue; pattern may need tightening |
| Processing time (p50/p95/p99) | `processing_time_ms` | p99 > 500ms warm | SLA breach |

**CloudWatch dashboard (reference deployment):**

The Terraform monitoring module creates a dashboard with:
- Request volume (requests/min)
- Processing time percentiles (p50, p95, p99)
- Per-layer latency breakdown
- Comprehend fallback rate
- CRITICAL flag volume
- Error rate by layer
- Cold start count

**CloudWatch alarms:**
- p99 latency > 500ms sustained 5 min → SNS notification
- Layer error rate > 1% sustained 5 min → SNS notification
- Comprehend fallback rate > 10% sustained 15 min → SNS notification
- Zero requests for 30 min during business hours → SNS notification (pipeline may be down)

**PHI safety in metrics:** The metrics log line contains only flag_id, severity, confidence, detection_layer, and operational counters. It never contains text, evidence spans, patient identifiers, or any content derived from the input. This is enforced by the structured log emitter (which takes a `MetricsLogEntry` model, not raw dicts) and validated by `test_no_phi_in_logs` in CI.

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

bh-sentinel is designed to integrate with [bh-audit-logger](https://github.com/bh-healthcare/bh-audit-logger) for compliance tracking at the deployment boundary. The core library (`bh-sentinel-core`) does not emit audit events directly — this is the responsibility of the service layer (e.g., the Lambda handler in Phase 3). Each analysis request should generate an audit event recording the request_id, source, flags returned (metadata only), and processing outcome. No clinical text should be included in audit events.

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

## 10.5 Local Development and Testing for Clinical Teams

The architecture positions patterns, rules, and taxonomy as configuration that clinical teams can modify without engineering involvement. This requires tooling that lets non-engineers validate their changes locally before they reach CI.

**CLI tool: `bh-sentinel validate`**

A single command validates all config files and runs pattern test fixtures. No AWS account, no deployed infrastructure, no Python expertise beyond running a command.

```bash
# Full validation: config integrity + pattern test fixtures
make validate

# Individual targets
make validate-config     # Config integrity checks only
make test-patterns       # Pattern test fixtures only
```

**`make validate-config`** runs the 7-point config validation from Section 3.3:

```
$ make validate-config

Config Validation
=================
✓ flag_taxonomy.json: valid JSON, taxonomy_version 1.0.0
✓ patterns.yaml: valid YAML, patterns_version 1.0.0, requires taxonomy 1.0.x → satisfied
✓ rules.json: valid JSON, rules_version 1.0.0, requires taxonomy 1.0.x → satisfied
✓ Flag ID integrity: 40/40 pattern flag_ids found in taxonomy
✓ Flag ID integrity: 5/5 rule flag_ids found in taxonomy
✓ Domain consistency: 6/6 pattern domains match taxonomy
✓ CRITICAL coverage: 7/7 CRITICAL flags have patterns
✓ Regex compilation: 329/329 patterns compile successfully
✓ Vendored copy sync: config/ matches _default_config/

All checks passed.
```

**`make test-patterns`** loads `config/test_fixtures.yaml`, compiles patterns, runs each fixture through the pattern matcher with negation and temporal detection, and reports results in a clinician-readable format:

```
$ make test-patterns

Pattern Test Results
====================

direct_positives (17 fixtures)
  ✓ "Patient reports suicidal ideation for the past two days." → SH-002
  ✓ "She wants to kill herself." → SH-002
  ✓ "I don't want to be alive anymore." → SH-001
  ...
  17/17 passed

negation_suppressed (5 fixtures)
  ✓ "Patient denies suicidal ideation." → suppressed [SH-002]
  ✓ "No SI." → suppressed [SH-002]
  ...
  5/5 passed

negation_scope (5 fixtures)
  ✓ "Denies SI, but reports feeling hopeless..." → suppressed [SH-002], detected [CD-001, SH-001]
  ...
  5/5 passed

temporal_past (3 fixtures)
  ✓ "History of suicide attempt in 2019." → SH-008 (temporal: past)
  ...
  3/3 passed

longer_notes (2 fixtures)
  ✓ "43 yo female presents for individual therapy..." → CD-003, CD-001, PF-005, PF-001, PF-002 (suppressed: SH-002, HO-001)
  ...
  2/2 passed

true_negatives (5 fixtures)
  ✓ "Patient is doing well today." → no flags
  ...
  5/5 passed

==============================
84/84 fixtures passed (0 failed)
```

On failure, the output shows exactly what went wrong:

```
negation_scope
  ✗ "Denies SI, but reports feeling hopeless..."
    Expected flags: [CD-001, SH-001]
    Actual flags:   [CD-001]
    Missing:        [SH-001]
    Note: Comma+but terminates negation scope. 'No point in going on' is an affirmative SH-001 pattern.
```

**Test fixture format:** Fixtures live in `config/test_fixtures.yaml`. Each entry has:

| Field | Required | Description |
|---|---|---|
| `input` | Yes | Clinical text fragment. Short and realistic, no real PHI. |
| `expect_flags` | No | Flag IDs that should be detected. |
| `expect_suppressed` | No | Flag IDs that match a pattern but should be suppressed by negation/temporal. |
| `expect_none` | No | `true` if no flags should fire. |
| `expect_temporal` | No | Expected temporal classification (`past` or `present`) for the primary flag. |
| `note` | No | Brief explanation for clinical reviewers. |

**Workflow for clinical teams:**

1. Edit `config/patterns.yaml` (add or modify patterns for a flag).
2. Add test fixtures to `config/test_fixtures.yaml` covering positive matches, negation, and temporal edge cases.
3. Run `make validate` locally.
4. If all checks pass, submit the change for review. CI runs the same checks.
5. A clinical reviewer and an engineer both approve. The engineer verifies regex quality; the clinician verifies clinical accuracy.

**Why this works without Python expertise:** `make validate` is a single command. The output uses plain English, flag IDs the clinical team already knows, and ✓/✗ symbols. No tracebacks, no test framework jargon, no assertion errors. The fixture format (YAML with `input`, `expect_flags`, `note`) is readable by anyone who can read the pattern library.

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
| Cross-sentence signals missed | Missed preparatory behavior or accumulated risk | Medium | Signal density rules (COMP-003/004) catch sub-threshold accumulation. Layer 2 sentence-window inference planned for Phase B (v0.3). See Section 4.9. |
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
- [ ] Negation detector (clause-bounded forward window; see Layer 1 negation scope algorithm in Section 4.5)
- [ ] Temporal detector (per-match keyword classification, present-overrides-past; see Layer 1 temporal detection algorithm in Section 4.5)
- [ ] Pattern matcher (compiled regex from YAML config, evidence span generation, basis_description generation)
- [ ] Behavioral health emotion lexicon integration (in-process, ~2ms) — see Section 4.5 Layer 3
- [ ] Rules engine (condition DSL, severity escalation/de-escalation, compound risk, recommended_action)
- [ ] Pipeline orchestrator with asyncio.gather and graceful degradation
- [ ] Draft patterns.yaml: 351 patterns across all 40 flags and 6 domains including clinical shorthand
- [ ] Config validation CLI (`bh_sentinel.cli validate-config`) — see Section 3.3
- [ ] Pattern test runner CLI (`bh_sentinel.cli test-patterns`) with clinician-readable output — see Section 10.5
- [ ] Test fixtures (`config/test_fixtures.yaml`) covering positive, negative, negation, temporal, and multi-flag scenarios — see Section 10.5
- [ ] Unit tests for all components
- [ ] Performance tests (pattern matching < 10ms for 500-word text)
- [ ] PHI containment tests
- [ ] Structured metrics log emitter (MetricsLogEntry model, PHI-safe, one JSON line per request) — see Section 7.6
- [ ] bh-audit-logger integration for compliance-grade audit trail (analysis events with metadata) — see Section 7.6
- [ ] CI pipeline (lint, type-check, test, config validation)
- [ ] Publish bh-sentinel-core v0.1.0 to PyPI

### v0.2: ML Layer + Reference Deployment

- [ ] bh-sentinel-ml package scaffolding
- [ ] ONNX transformer inference layer (in-process, no network hop)
- [ ] Zero-shot classification with DistilBART-MNLI (also test BiomedBERT-NLI)
- [ ] Model export tooling (PyTorch to ONNX INT8, scripts/export_onnx.py)
- [ ] Transformer evidence_span generation (sentence-level classification → sentence boundaries as span) — required for Criterion 4
- [ ] Transformer basis_description generation — required for Criterion 4
- [ ] Transformer confidence calibration (temperature scaling on validation set, interim discount factor for Phase A) — see Section 4.8
- [ ] Flag deduplication and layer merge implementation — see Section 4.7
- [ ] AWS Comprehend integration (VPC endpoint, optional, 200ms timeout, graceful fallback)
- [ ] Model evaluation harness with ground truth fixtures (also used for calibration validation, ECE < 0.05)
- [ ] AWS Lambda reference deployment (Dockerfile, handler.py)
- [ ] Terraform modules (VPC, Lambda, API Gateway, ECR, S3, monitoring, rate limiting)
- [ ] CloudWatch dashboard (request volume, latency percentiles, per-layer breakdown, CRITICAL flag volume, error rates) — see Section 7.6
- [ ] CloudWatch alarms (p99 latency, layer error rate, Comprehend fallback rate, zero-traffic) — see Section 7.6
- [ ] End-to-end integration tests
- [ ] Latency tests (p99 < 500ms warm)
- [ ] Load tests (100 concurrent requests)
- [ ] VPC Flow Log audit (no outbound traffic leaves VPC)
- [ ] Publish bh-sentinel-ml v0.1.0 to PyPI

### v0.3: Fine-Tuning, Evaluation, and Spanish Language Support

- [ ] Public dataset preparation scripts (C-SSRS Reddit, SMHD, Kaggle Mental Health)
- [ ] Domain adaptation pre-training on Reddit mental health corpora
- [ ] Fine-tuning pipeline (DistilBERT multi-label classification head)
- [ ] A/B evaluation framework (zero-shot vs fine-tuned, per-severity metrics)
- [ ] Clinician feedback integration design (dismiss action feeds back into evaluation via bh-audit-logger trail; join on request_id)
- [ ] Per-flag dismiss rate dashboard (alert fatigue monitoring, >30% dismiss rate triggers review)
- [ ] Staging and production Terraform configs
- [ ] CloudWatch alarms and dashboard
- [ ] Runbook documentation
- [ ] Spanish language pattern library (Layer 1 Spanish patterns across all 40 flags)
- [ ] Multilingual transformer support (multilingual DistilBERT or language-specific model)
- [ ] Extend `context.language` to accept `"es"`, route to Spanish patterns and model
- [ ] Spanish negation and temporal marker lists
- [ ] Spanish-language test suite (positive, negative, negated, temporal examples)

### Future

- [ ] EMR webhook integration examples
- [ ] Flag trend analysis utilities
- [ ] FHIR-compatible output format
- [ ] Group session note analysis patterns
- [ ] Model retraining pipeline (new container image per model version)
- [ ] Clinician feedback dashboard (precision/recall from dismiss actions)

---

*This document should be reviewed with qualified behavioral health clinical leadership before any production deployment. The flag taxonomy in particular needs clinical validation and may require additions or modifications based on program-specific needs.*

*FDA CDS compliance analysis (Section 8) reflects the January 2026 updated guidance. Deploying organizations should review this analysis with their own legal counsel, particularly if extending the tool to patient-facing applications.*

*bh-sentinel is clinical decision support software. It is not a diagnostic tool, not a substitute for clinical judgment, and not FDA-cleared or approved. Organizations deploying this software in clinical settings are responsible for their own clinical validation, regulatory compliance, and patient safety protocols.*
