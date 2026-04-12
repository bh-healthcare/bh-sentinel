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

## Full Flag Definitions

Severity shown is the default from `config/flag_taxonomy.json`. Compound rules (Layer 4) can escalate severity at analysis time; see the Psychosis Flag Detail section above for an example.

C-SSRS mapping is shown for the Self-Harm domain only. Other domains are outside C-SSRS scope. CD-001 (hopelessness) and CD-002 (isolation) are indirectly related to C-SSRS suicide risk formulation but are not mapped to specific C-SSRS categories.

### Self-Harm and Suicidal Ideation

| Flag | Severity | Definition | Clinical rationale | C-SSRS mapping |
|---|---|---|---|---|
| SH-001 | HIGH | Statements reflecting a wish to be dead, not wake up, or disappear without describing self-harm action. | Passive death wish is often the earliest detectable suicide-spectrum marker and raises concern even when intent is denied. | Wish to be Dead |
| SH-002 | CRITICAL | General suicidal thoughts without a stated method, plan, or preparatory step. | Active suicidal thinking marks a meaningful rise above passive ideation and warrants urgent clinical review. | Non-Specific Active Suicidal Thoughts |
| SH-003 | CRITICAL | Suicidal thinking that includes a described method, such as overdose, hanging, or firearm use, without a complete plan. | Access to or rehearsal of method increases feasibility and shortens the path from ideation to action. | Active Suicidal Ideation with Any Methods (Not Plan) without Intent to Act |
| SH-004 | CRITICAL | Suicidal thinking accompanied by some expressed intent or expectation of acting, without a specific plan for when, where, or how. | Intent to act, even without a formed plan, marks a significant escalation in near-term risk and typically warrants same-day safety assessment. | Active Suicidal Ideation with Some Intent to Act, without Specific Plan |
| SH-005 | CRITICAL | Suicidal thinking that includes both a specific plan and stated intent, expectation of acting, or inability to stay safe. | The combination of a specific plan with expressed intent represents the highest ideation-level severity on the C-SSRS and requires immediate intervention. | Active Suicidal Ideation with Specific Plan and Intent |
| SH-006 | CRITICAL | Behaviors such as gathering pills, writing notes, giving away belongings, visiting a location, or rehearsing steps toward suicide. | Preparatory acts demonstrate transition from thought to behavior and are treated as near-attempt level acuity. | Preparatory Acts or Behavior |
| SH-007 | HIGH | Deliberate self-injury such as cutting, burning, or hitting oneself without expressed desire to die. | Non-suicidal self-injury predicts emotional dysregulation and elevates future suicide risk even when intent is absent. | No direct C-SSRS equivalent; adjacent to suicidal behavior history review |
| SH-008 | HIGH | Mention of a prior suicide attempt, aborted attempt, or serious self-harm event in the past. | Prior attempt history is among the strongest predictors of future suicidal behavior and increases baseline risk. | Actual Attempt (historical) |

### Harm to Others

| Flag | Severity | Definition | Clinical rationale |
|---|---|---|---|
| HO-001 | CRITICAL | General thoughts of killing, assaulting, or seriously harming another person without naming a target. | Even nonspecific homicidal ideation can escalate quickly and requires immediate violence-risk assessment. |
| HO-002 | CRITICAL | Homicidal or violent thoughts directed toward an identifiable person or group. | A specific target materially increases plausibility, duty-to-protect considerations, and urgency of intervention. |
| HO-003 | CRITICAL | Violent or homicidal thoughts that include a plan, method, timing, or preparation. | Planning substantially elevates imminent risk and may require emergency action and protective steps. |
| HO-004 | HIGH | Strong urges to hit, attack, or lose control physically, even without articulated intent. | Intense violent urges can precede impulsive assaultive behavior, especially under agitation or intoxication. |
| HO-005 | HIGH | Disclosure of abusing, assaulting, coercing, or otherwise perpetrating harm toward another person. | Perpetration signals active risk to others and may trigger mandated reporting or safeguarding workflows. |
| HO-006 | HIGH | Disclosure of currently experiencing physical, sexual, emotional, or domestic abuse. | Active victimization is a major safety concern and can increase risk of suicide, retaliation, or acute destabilization. |

### Medication Non-Adherence

| Flag | Severity | Definition | Clinical rationale |
|---|---|---|---|
| MED-001 | MEDIUM | Explicit statement that prescribed medication is not being taken as directed. | Stated non-adherence can directly worsen symptoms, destabilize treatment response, and increase relapse risk. |
| MED-002 | MEDIUM | Indirect evidence of non-adherence, such as running out, cost barriers, avoidance, or self-stopping. | Implied adherence problems are common early indicators of treatment drift and merit follow-up before decompensation. |
| MED-003 | HIGH | Taking more than prescribed, hoarding doses, mixing unsafely, or otherwise misusing medication. | Misuse raises overdose risk, toxicity risk, and in some cases creates suicide means concern. |
| MED-004 | MEDIUM | Reports of side effects, adverse reactions, or feeling harmed by medication. | Unaddressed adverse effects are a common driver of discontinuation and worsening engagement. |
| MED-005 | MEDIUM | Expressed desire or plan to stop medication despite an active prescription. | Desire to discontinue may signal poor insight, emerging side effects, or impending non-adherence. |

### Substance Use

| Flag | Severity | Definition | Clinical rationale |
|---|---|---|---|
| SU-001 | MEDIUM | Current use of alcohol, cannabis, illicit drugs, or misuse of prescribed substances. | Active use can impair judgment, worsen mood instability, and amplify both suicide and violence risk. |
| SU-002 | HIGH | Return to substance use after a period of sobriety or recovery. | Relapse often coincides with demoralization, reduced inhibition, and increased psychiatric instability. |
| SU-003 | HIGH | Increasing quantity, frequency, potency, or loss of control over use. | Escalation suggests worsening disorder severity and a higher likelihood of medical or behavioral consequences. |
| SU-004 | HIGH | Symptoms of withdrawal such as tremor, sweats, nausea, agitation, or seizure risk. | Withdrawal can create acute medical danger and severe distress that compounds psychiatric risk. |
| SU-005 | HIGH | High-risk patterns such as mixing substances, using alone, IV use, or intoxicated hazardous behavior. | Risky use patterns increase overdose, accidental death, and impulsive self-harm potential. |

### Clinical Deterioration

| Flag | Severity | Definition | Clinical rationale |
|---|---|---|---|
| CD-001 | HIGH | Pervasive belief that nothing will improve or that the future is irredeemably bleak. | Hopelessness is a well-established proximal correlate of suicide risk and severe depressive burden. |
| CD-002 | MEDIUM | Marked withdrawal from family, friends, treatment, or normal social contact. | Severe isolation reduces buffering support and can allow risk to escalate unnoticed. |
| CD-003 | MEDIUM | Major insomnia, near-total sleep loss, or extreme hypersomnia. | Severe sleep disruption worsens mood regulation, impulse control, psychosis risk, and suicide vulnerability. |
| CD-004 | MEDIUM | Major reduction in intake, bingeing, purging, or other severe appetite disturbance. | Significant appetite disruption can reflect worsening depression, eating pathology, or medical compromise. |
| CD-005a | HIGH | Hearing voices, commands, commentary, or other auditory perceptual experiences others do not hear. | Auditory hallucinations, especially command content, can directly elevate risk of self-harm or aggression. |
| CD-005b | HIGH | Seeing people, figures, shadows, or other visual experiences perceived as real but not externally present. | Visual hallucinations indicate psychotic decompensation or substance/medical causes requiring urgent evaluation. |
| CD-005c | HIGH | Unfounded belief of being watched, followed, targeted, or conspired against. | Paranoid ideation can drive defensive violence, elopement, or rapid disengagement from care. |
| CD-005d | HIGH | Fixed false beliefs resistant to evidence, including persecutory, grandiose, somatic, or referential delusions. | Delusional thinking signals significant loss of reality testing and can impair judgment around safety. |
| CD-006 | MEDIUM | Depersonalization, derealization, or feeling detached from self, body, or surroundings. | Dissociation can impair self-protection, increase distress, and complicate suicide risk assessment reliability. |
| CD-007 | MEDIUM | Acute panic, extreme anxiety, or physiologic overwhelm causing loss of function. | Severe anxiety can rapidly intensify suicidal crises, impulsivity, and psychotic distress. |
| CD-008 | HIGH | Indicators of mania such as decreased need for sleep, racing thoughts, grandiosity, pressured speech, or risky behavior. | Mania elevates impulsivity, impaired judgment, and exposure to high-risk behaviors requiring urgent treatment adjustment. |

### Protective Factors

| Flag | Severity | Definition | Clinical rationale |
|---|---|---|---|
| PF-001 | POSITIVE | Clear commitment to therapy, psychiatry, safety planning, or follow-up care. | Engagement improves monitoring, responsiveness to intervention, and willingness to use crisis supports. |
| PF-002 | POSITIVE | Presence of supportive family, friends, partner, community, or other reliable interpersonal support. | Social support can reduce isolation, improve means restriction, and strengthen crisis containment. |
| PF-003 | POSITIVE | Statements about future plans, goals, responsibilities, or reasons to keep going. | Future orientation is protective because it counters entrapment and supports reasons for living. |
| PF-004 | POSITIVE | Active use of coping skills such as grounding, distraction, journaling, reaching out, or distress-tolerance tools. | Demonstrated coping capacity increases the likelihood of surviving acute distress without acting impulsively. |
| PF-005 | POSITIVE | Consistent medication adherence with prescribed psychiatric treatment. | Reliable adherence supports symptom stabilization and lowers relapse and decompensation risk. |
