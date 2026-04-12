# Pattern Library

Guide for authoring and extending the bh-sentinel pattern library.

## Overview

The pattern library (`config/patterns.yaml`) contains regex patterns mapped to clinical safety flags. Patterns are compiled at load time and matched against preprocessed clinical text. The pattern matcher applies negation detection and temporal awareness after an initial regex match, so patterns themselves target affirmative language while `negation_phrases` handle the denial/resolved cases separately.

A vendored copy lives at `packages/bh-sentinel-core/src/bh_sentinel/core/_default_config/patterns.yaml`. Both copies must stay in sync.

## YAML Structure

The file is organized as nested YAML: **domain → flag_id → fields**. Each flag_id must exist in `config/flag_taxonomy.json`.

```yaml
<domain>:
  <FLAG-ID>:  # Short name from taxonomy
    patterns:
      - "regex pattern targeting patient or narrative language"
    clinical_shorthand:        # optional
      - "regex pattern targeting clinician charting notation"
    negation_phrases:          # optional
      - "literal or regex phrases that negate a match"
    confidence: 0.90           # default confidence score (0.0–1.0)
```

### Field Reference

| Field | Required | Description |
|---|---|---|
| `patterns` | Yes | List of regex strings targeting patient language, narrative notes, and general clinical text. Compiled with `re.IGNORECASE`. |
| `clinical_shorthand` | No | Separate pattern list for clinician charting abbreviations (`+SI`, `pt endorses HI`, `AH present`). Matched under the same flag_id but kept separate because the language register is different and the patterns are structurally distinct. |
| `negation_phrases` | No | Strings or regex patterns that, when detected near a match, cause the match to be suppressed or downgraded. Plain strings like `"denies"` are valid regex. Use regex syntax when matching needs to be more specific (e.g., `"(?:\\-|negative) (?:for )?SI"`). |
| `confidence` | Yes | Default confidence score for matches from this flag. Higher values mean the pattern set is more specific. Range: 0.0–1.0. |

Severity is **not** set here. Default severity comes from `flag_taxonomy.json` and can be escalated by compound rules in Layer 4.

## Regex Conventions

All patterns are compiled with `re.IGNORECASE`. Follow these conventions to keep the library consistent and reviewable:

- **Always quote patterns.** YAML can consume unquoted special characters. Use double quotes for all regex strings.
- **Use `(?:)` non-capturing groups for alternation.** Prefer `"(?:kill|end|hang)(?:ing)? myself"` over `"(kill|end|hang)(ing)? myself"`. Capturing groups are unnecessary and can interfere with match extraction.
- **Handle apostrophes with `'?`.** Clinical notes inconsistently include apostrophes: `"can'?t"`, `"haven'?t"`, `"don'?t"`, `"what'?s"`.
- **Use space-based boundaries, not `\b`.** The library convention uses literal spaces and `(?:)` groups for word separation rather than `\b`. This avoids edge cases with punctuation-adjacent matches in clinical text.
- **Build in pronoun and determiner flexibility.** Use optional groups for possessives and articles: `"(?:my |the )?(?:medication|meds)"`, `"(?:I |he |she |they )?"`.
- **Prefer multiple narrow patterns over one broad pattern.** Each pattern should target a recognizable phrasing. This makes review, testing, and regression tracking easier.

## Detailed Examples

### Patient/Narrative Language

Active suicidal ideation (SH-002) with clinical shorthand and negation:

```yaml
self_harm:
  SH-002:  # Active SI - nonspecific
    patterns:
      - "want(?:s|ed|ing)? to (?:kill|end) (?:myself|my life|it all)"
      - "thinking (?:about|of) (?:killing|ending) (?:myself|my life)"
      - "suicidal (?:thoughts?|ideation|feelings)"
      - "can'?t (?:stop|keep from) thinking about (?:suicide|dying|ending it)"
    clinical_shorthand:
      - "pt (?:c/o|reports|endorses) SI"
      - "(?:\\+|positive) (?:for )?SI"
      - "SI (?:present|endorsed|reported|active)"
    negation_phrases: ["denies", "no", "not", "never", "(?:\\-|negative) (?:for )?SI"]
    confidence: 0.95
```

The `patterns` list captures how patients describe suicidal thinking in their own words or in narrative notes. The `clinical_shorthand` list captures how clinicians abbreviate the same concept in structured charting. Both contribute to the same flag and confidence score.

### Medication Non-Adherence

Stated non-adherence (MED-001) showing the medication/determiner flexibility pattern:

```yaml
medication:
  MED-001:  # Non-adherence - stated
    patterns:
      - "stopped taking (?:my |the )?(?:medication|meds|pills|prescription)"
      - "haven'?t (?:taken|been taking) (?:my |the )?(?:medication|meds)"
      - "ran out of (?:my |the )?(?:medication|meds|prescription)"
      - "can'?t afford (?:my |the )?(?:medication|meds|prescription)"
      - "not taking (?:my |the )?(?:medication|meds) (?:anymore|as prescribed)"
      - "skipping (?:my |the )?(?:medication|meds|doses)"
    clinical_shorthand:
      - "(?:med|medication) non-?adherence"
      - "non-?compliant (?:with|re:?) (?:meds|medication)"
    confidence: 0.88
```

This entry omits `negation_phrases` because the patterns themselves describe non-adherence. A negation of non-adherence ("patient is not skipping meds") would indicate adherence, which is a different flag (PF-005).

### Minimal Entry

Not every flag needs clinical shorthand or negation phrases. A minimal entry requires only `patterns` and `confidence`:

```yaml
substance_use:
  SU-001:  # Active substance use disclosed
    patterns:
      - "(?:using|used|smoking|drinking) (?:again|daily|every day)"
      - "relapsed"
      - "started (?:using|drinking|smoking) again"
    confidence: 0.85
```

## Test Examples

Pair every pattern addition with test phrases across these categories:

| Category | Example | Expected |
|---|---|---|
| Direct positive | `Patient reports suicidal ideation for the past two days.` | SH-002 match |
| Clinical shorthand positive | `+SI, pt endorses active thoughts.` | SH-002 match |
| Clear negative | `Medication options were discussed with the patient.` | No match |
| Negated statement | `Patient denies suicidal ideation.` | Suppressed by negation |
| Historical/resolved | `History of suicidal ideation as a teenager, none currently.` | Suppressed by temporal detection |
| Spelling variant | `Pt hasn't taken meds in two weeks.` | MED-001 match |

Keep each example short, realistic, and limited to the language needed to prove the behavior. Prefer clinician-style note fragments over artificial keyword lists.

If a pattern is broad enough to match many phrasings, add multiple test examples for the same concept. This helps catch regressions when preprocessing, negation handling, or taxonomy mappings change.

## Negation Test Cases

The negation detector uses clause-bounded forward-window scoping (see architecture.md, Layer 1). Every pattern with `negation_phrases` must be tested against these negation categories. The examples below use SH-002 (active SI) and HO-001 (homicidal ideation) but the same patterns apply to all flags with negation phrases.

### Simple negation (should suppress)

| Input | Expected | Why |
|---|---|---|
| `Patient denies suicidal ideation.` | Suppressed | "denies" is a negation cue within lookback window, no terminator before "suicidal ideation" |
| `No SI.` | Suppressed | "No" negates "SI" directly |
| `Not currently suicidal.` | Suppressed | "Not" within lookback, no terminator |
| `-SI` | Suppressed | "-" is a shorthand negation cue |
| `Negative for SI.` | Suppressed | "Negative for" is a clinical negation cue |

### Negation with scope termination (should NOT suppress the second finding)

| Input | Expected | Why |
|---|---|---|
| `Denies SI, but reports feeling hopeless and that there's no point in going on.` | SI suppressed, CD-001 match on "no point in going on" | `,` before "but" terminates negation scope |
| `Denies SI, +HI.` | SI suppressed, HO-001 match on "+HI" | `,` terminates scope |
| `-SI, +HI, reports hearing voices.` | SI suppressed, HI and CD-005a match | `,` terminates scope after each item |
| `No suicidal ideation. Reports wanting to hurt others.` | SI suppressed, HO-001 match | `.` terminates scope |
| `Denies SI; however, endorses HI with specific target.` | SI suppressed, HO-002 match | `;` and "however" both terminate scope |
| `No SI but has been thinking about hurting people.` | SI suppressed, HO-001 match | "but" terminates scope |

### Slash-delimited lists (negation carries through)

| Input | Expected | Why |
|---|---|---|
| `Denies SI/HI.` | Both suppressed | `/` is not a scope terminator; negation carries through conjunctive list |
| `No SI or HI.` | Both suppressed | `or` is not a scope terminator |
| `Denies SI and HI.` | Both suppressed | `and` is not a scope terminator |

### Pseudo-negation (should NOT suppress — these are not real negations)

| Input | Expected | Why |
|---|---|---|
| `Patient no longer denies suicidal ideation.` | SH-002 match | "no longer denies" is a pseudo-negation (double negative = affirmation) |
| `Unable to sleep for three days.` | CD-003 match | "Unable to" negates ability, not the symptom; the symptom IS present |
| `Cannot stop thinking about suicide.` | SH-002 match | "Cannot stop" means the thinking IS happening |
| `No improvement in suicidal ideation.` | SH-002 match | "No improvement" negates progress, not the finding |

### Post-negation cues (negation cue appears after the match)

| Input | Expected | Why |
|---|---|---|
| `Suicidal ideation: denied.` | Suppressed | "denied" is a post-negation cue within forward window |
| `SI — negative.` | Suppressed | "negative" is a post-negation cue |
| `Suicidal ideation absent.` | Suppressed | "absent" is a post-negation cue |
| `SI reported, HI denied.` | SI match, HI suppressed | "denied" post-negates "HI" only; "reported" is not a negation cue |

### Edge cases requiring careful handling

| Input | Expected | Why |
|---|---|---|
| `Denies suicidal ideation but reports she doesn't want to live anymore.` | SI suppressed, SH-001 match on "doesn't want to live anymore" | "but" terminates scope; "doesn't want to live" is an affirmative match for SH-001 (the "don't" is part of the pattern, not a negation cue) |
| `Patient reports no reason to live. Denies plan or intent.` | SH-001 match on "no reason to live", SH-004/SH-005 suppressed | "no reason to live" is an affirmative pattern (the "no" is part of the passive death wish expression); "Denies" in the second sentence negates plan/intent |
| `She says she doesn't have suicidal thoughts but can't stop thinking about dying.` | First clause suppressed, SH-002 match on "can't stop thinking about dying" | "doesn't" suppresses "suicidal thoughts", "but" terminates scope, second clause is affirmative |

## Temporal Detection Test Cases

The temporal detector classifies each pattern match as `present` or `past` using keyword proximity (see architecture.md, Layer 1). Present overrides past. Default is `present` (safe default for a safety system). DE-001 in the rules engine de-escalates severity by one level when a flag's temporal context is `past`.

### Historical references (should be classified `past`, DE-001 applies)

| Input | Matched flag | Temporal | Why |
|---|---|---|---|
| `History of suicide attempt in 2019.` | SH-008 | past | "History of" is a past-context marker |
| `Used to cut himself in high school.` | SH-007 | past | "Used to" and "in high school" are past markers |
| `PMH significant for prior suicide attempt.` | SH-008 | past | "PMH" and "prior" are past markers |
| `Had been hearing voices years ago, resolved with medication.` | CD-005a | past | "Had been" and "years ago" are past markers |
| `Remote history of substance abuse.` | SU-001 | past | "Remote history" is a past marker |

### Current references (should be classified `present`, no de-escalation)

| Input | Matched flag | Temporal | Why |
|---|---|---|---|
| `Currently endorsing suicidal ideation.` | SH-002 | present | "Currently" is a present marker |
| `Reports hearing voices right now.` | CD-005a | present | "right now" is a present marker |
| `Started cutting again last night.` | SH-007 | present | "again" and "last night" are present markers |
| `Patient is actively suicidal.` | SH-002 | present | "actively" is a present marker |
| `Relapsed this week after two years sober.` | SU-002 | present | "this week" overrides "two years" (past reference to sobriety duration) |

### Present overrides past (mixed references, present wins)

| Input | Matched flag | Temporal | Why |
|---|---|---|---|
| `I used to cut myself in high school. Last night I started again.` | SH-007 (x2) | 1st match: past, 2nd match: present → flag emitted as present | "used to" classifies first match as past; "Last night" and "again" classify second as present; present wins for the flag |
| `History of SI, currently endorsing active thoughts.` | SH-008 (past), SH-002 (present) | Two different flags, each with its own temporal context | "History of" → SH-008 past; "currently" → SH-002 present; no conflict because they are different flags |
| `Had previous suicide attempt. Is now having thoughts again.` | SH-008 (past), SH-002 (present) | Separate flags | "previous" → SH-008 past; "now" and "again" → SH-002 present |
| `Used to hear voices as a teenager but they came back recently.` | CD-005a (x2) | 1st match: past, 2nd match: present → flag emitted as present | "Used to" and "as a teenager" classify first match; "came back" and "recently" classify second; present wins |

### Default to present (no temporal markers found)

| Input | Matched flag | Temporal | Why |
|---|---|---|---|
| `Patient wants to kill herself.` | SH-002 | present | No temporal markers found; default is present |
| `Hearing voices telling him to hurt himself.` | CD-005a | present | No temporal markers; default present |

### Edge cases

| Input | Matched flag | Temporal | Why |
|---|---|---|---|
| `She attempted suicide in 2020 and is thinking about it again.` | SH-008 (past), SH-002 (present) | Two flags, each correctly classified | "in 2020" → SH-008 past; "again" → SH-002 present |
| `No prior attempts. Currently suicidal with plan.` | SH-005 (present) | Negation suppresses SH-008 ("No prior attempts"); SH-005 is present | Negation and temporal interact: negation fires first, then temporal on surviving matches |
| `Patient stopped cutting three months ago.` | SH-007 | past | "three months ago" and "stopped" are past markers; the cessation is historical |
| `Reports he stopped cutting three months ago but started burning himself last week.` | SH-007 (x2) | 1st: past, 2nd: present → flag emitted as present | "three months ago" and "stopped" → past; "last week" and "started" → present; present wins |

## Authoring Guidelines

- Patterns should target clinical language, not social media vernacular.
- One flag_id should map to one clinical concept. If a pattern captures something materially different from the flag's definition in `flag-taxonomy.md`, it belongs under a different flag_id.
- When a flag has both patient-language patterns and clinician-charting patterns, separate them into `patterns` and `clinical_shorthand`. Do not mix the two registers in a single list.
- Include negation phrases when the affirmative pattern could plausibly appear in a negated context. Omit them when negation of the pattern implies a different clinical concept (e.g., MED-001 non-adherence patterns).
- Set confidence higher (0.90+) for specific, low-ambiguity patterns and lower (0.80–0.89) for patterns that cover broader or fuzzier language.
- Test patterns against positive, negative, negated, and temporal examples before merging.

## Contribution Workflow

When adding or changing patterns:

1. Update `config/patterns.yaml` with the smallest pattern set that captures the intended clinical concept.
2. Confirm the `flag_id` exists in `config/flag_taxonomy.json` and that the domain key matches.
3. Add or update tests with positive, negative, negated, temporal, and spelling-variant examples.
4. Review the regexes for unnecessary breadth, duplicate coverage with existing patterns, and consistency with the conventions above.
5. Copy the updated file to `packages/bh-sentinel-core/src/bh_sentinel/core/_default_config/patterns.yaml` to keep the vendored copy in sync.
6. Include a short note in the change description explaining why the new pattern is needed and which test examples were used to validate it.

Keep contributions narrowly scoped. If a change introduces a new clinical concept, add it in a separate reviewable block rather than combining it with unrelated cleanup or refactors.
