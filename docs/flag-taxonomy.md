# Flag Taxonomy

Detailed definitions for all 40 clinical safety flags across 6 domains.

## Domains

| Domain | Flag Range | Count |
|---|---|---|
| Self-Harm and Suicidal Ideation | SH-001 through SH-008 | 8 |
| Harm to Others | HO-001 through HO-006 | 6 |
| Medication Non-Adherence | MED-001 through MED-005 | 5 |
| Substance Use | SU-001 through SU-005 | 5 |
| Clinical Deterioration | CD-001 through CD-008 (CD-005a–d) | 11 |
| Protective Factors | PF-001 through PF-005 | 5 |

## Severity Levels

| Level | Meaning | Expected Response |
|---|---|---|
| CRITICAL | Imminent risk to self or others | Immediate provider notification |
| HIGH | Significant clinical concern | Same-day provider review |
| MEDIUM | Clinically relevant, not urgent | Flag for next session review |
| LOW | Informational | Log for trend analysis |
| POSITIVE | Protective factor detected | Log for treatment progress |

The self-harm taxonomy is informed by the Columbia Suicide Severity Rating Scale (C-SSRS).

The canonical machine-readable taxonomy is in `config/flag_taxonomy.json`.

## Psychosis Flag Detail (CD-005a–d)

The original CD-005 "Psychotic symptoms" flag has been decomposed into four
granular sub-flags for more precise detection and clinical routing:

| Flag | Name | Description | Severity |
|---|---|---|---|
| CD-005a | Auditory hallucinations | Hearing voices, auditory command hallucinations, running commentary. | HIGH |
| CD-005b | Visual hallucinations | Seeing things others cannot see, visual distortions perceived as real. | HIGH |
| CD-005c | Paranoid ideation | Belief of being watched, followed, plotted against, or targeted without evidence. | HIGH |
| CD-005d | Delusional thinking | Fixed false beliefs (grandiose, somatic, referential, persecutory) resistant to counter-evidence. | HIGH |

### Anxiety-to-Psychosis Escalation (Layer 4)

When any psychosis flag (CD-005a–d) co-occurs with acute anxiety (CD-007),
the compound rule COMP-002 triggers:

- All detected psychosis flags escalate to **CRITICAL** severity
- The session is marked `requires_immediate_review`
- Recommended action: evaluate for psychotic decompensation with anxiety prodrome

This captures the clinically significant pattern where anxiety is prodromal to,
or reactive to, emerging psychotic experiences — requiring immediate psychiatric
assessment rather than standard anxiety-track routing.

<!-- TODO: Expand with full flag definitions, clinical rationale, and C-SSRS mappings for each flag. -->
