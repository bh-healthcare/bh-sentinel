# FDA Clinical Decision Support Analysis

Regulatory analysis of bh-sentinel under Section 520(o)(1)(E) of the FD&C Act (21st Century Cures Act).

## Overview

bh-sentinel is designed to satisfy the four criteria for Non-Device Clinical Decision Support:

1. **Not intended to acquire, process, or analyze a medical image or signal.**
2. **Intended for the purpose of displaying, analyzing, or printing medical information.**
3. **Intended for the purpose of supporting or providing recommendations to a healthcare professional about prevention, diagnosis, or treatment of a disease or condition.**
4. **Intended for the purpose of enabling such healthcare professional to independently review the basis for such recommendations that such software presents so that it is not the intent that such healthcare professional rely primarily on any of such recommendations to make a clinical diagnosis or treatment decision regarding an individual patient without independently reviewing the basis for such recommendations.**

Criterion 4 is addressed through the `basis_description`, `evidence_span`, and `detection_layer` fields in every flag response.

**Note:** Organizations deploying bh-sentinel should review this analysis with their own legal counsel.

## Detailed Criterion Analysis

### Criterion 1: Does not acquire, process, or analyze a medical image or signal

bh-sentinel is designed to operate on already-generated clinical documentation and structured medical information rather than on raw device outputs. It does not ingest radiology images, waveform data, or physiologic signal streams for diagnostic interpretation. The product's core function is to identify documentation gaps, coding inconsistencies, and compliance-relevant omissions within text and record-derived artifacts.

This distinction matters because software that directly analyzes an image, electrocardiogram waveform, continuous physiologic stream, or similarly device-like signal can fall outside the Non-Device CDS exclusion. bh-sentinel should therefore be deployed so that:

- Inputs are limited to textual records, coded data, and other human-interpretable clinical information.
- Any upstream conversion of signals or images into narrative reports occurs outside bh-sentinel.
- Product messaging avoids implying that bh-sentinel interprets primary diagnostic data.

The principal concern under this criterion is analysis of primary diagnostic imaging or signal data rather than review of resulting narrative documentation. If a future implementation introduces analysis of DICOM images, bedside monitoring streams, or algorithmic feature extraction from physiologic data, this criterion would require re-evaluation.

### Criterion 2: Displays, analyzes, or prints medical information

bh-sentinel fits this criterion because it organizes and analyzes existing medical information supplied in the clinical record. Its outputs are tied to identifiable source material and are intended to highlight what is present, missing, inconsistent, or potentially relevant for downstream human review.

Criterion 2 is satisfied most directly because the software displays and analyzes medical information that the healthcare professional can already review in the patient's record, such as notes, problem lists, medication history, laboratory results, and related documentation artifacts. In practice, bh-sentinel should maintain:

- Clear linkage between each flag and the underlying source content.
- Output phrasing that frames results as record-based findings or prompts for review.
- User experience patterns that emphasize inspection of existing medical information rather than passive acceptance of a conclusion.

### Criterion 3: Supports or provides recommendations to a healthcare professional

bh-sentinel is intended to support professional decision-making by surfacing documentation issues and possible follow-up considerations. That is consistent with the CDS exclusion so long as the product remains assistive and is directed to healthcare professionals rather than autonomous patient-facing decision-making.

In this context, documentation and coding integrity support prevention, diagnosis, and treatment by helping ensure that the patient's conditions, clinical reasoning, and care activities are accurately represented in the record that healthcare professionals use to make and communicate care decisions.

The compliance position is strongest when the product:

- Presents flags, prompts, or recommendations as decision support for qualified users.
- Assumes clinical judgment remains necessary before any diagnosis, treatment, or coding action is taken.
- Avoids language suggesting that the output is final, determinative, or self-executing.

This criterion becomes harder to satisfy if the software is marketed as automatically determining what diagnosis should be entered, what treatment must occur, or whether a patient definitively has a condition.

### Criterion 4: Enables independent review of the basis for the recommendation

Criterion 4 is the most important and usually the highest-risk element of the Non-Device CDS analysis. bh-sentinel addresses it through the `basis_description`, `evidence_span`, and `detection_layer` fields in every flag response. Together, these fields are intended to explain what source information triggered the flag, where that information appears, and what type of detection logic produced it. Because `evidence_span` is span metadata rather than echoed source text, the integrating interface must resolve those offsets back into visible source context so the user can review the basis in the chart itself.

To remain within this criterion, the product should ensure that a healthcare professional can reasonably answer the following questions from the interface and supporting documentation:

- What record content caused this flag?
- Where in the chart or source material can that content be reviewed?
- Is the output a direct extraction, rule-based inference, or model-assisted pattern match?
- What uncertainty, limitation, or ambiguity should the user understand before acting?

Independent review is weakened if the system presents opaque risk scores, unexplained classifications, or recommendations that cannot be traced back to source material in a way a clinician can assess. For that reason, future product changes should preserve explainability as a release-blocking requirement, not a secondary usability feature.

## Risk Assessment

The primary regulatory risk is not that bh-sentinel performs support functions, but that users or deployers may treat it as if it were an authoritative clinical decision engine. The highest-risk scenarios are:

- Outputs that appear conclusive without sufficient source attribution.
- User workflows that allow bulk acceptance of recommendations without chart review.
- Marketing or training materials that overstate accuracy or imply autonomous clinical judgment.
- Expansion of inputs into image, signal, or other device-like data sources.
- Product evolution toward closed-model scoring without transparent rationale.

Operational risk also exists if false positives create alert fatigue or if false negatives cause teams to overestimate record completeness. These concerns do not automatically change regulatory status, but they do affect whether real-world use remains consistent with the assumptions supporting the Non-Device CDS position.

## Deployment Considerations

Organizations deploying bh-sentinel should align product configuration, clinical workflow, and governance with the above criteria. Recommended controls include:

- Limit use to qualified healthcare professionals or trained clinical documentation personnel operating under professional supervision.
- Preserve display of `basis_description`, `evidence_span`, and related rationale fields in the production interface, and render span-based evidence in visible source context rather than as offsets alone.
- Require human review before documentation, coding, diagnosis, or treatment changes are finalized.
- Maintain audit logs showing what recommendation was presented, what evidence was shown, and what action the user ultimately took.
- Train users that bh-sentinel is a documentation and decision-support aid, not a substitute for clinical judgment.
- Review new features for regulatory impact before release, especially features involving automation, summarization, prioritization, or new data modalities.

From a governance perspective, deployment teams should treat the Non-Device CDS rationale as a living compliance position. Material product changes, new integrations, or new marketing claims should trigger renewed legal and regulatory review.
