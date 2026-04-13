# Clinical Use Disclaimer

**bh-sentinel** is clinical decision support software designed to assist qualified
healthcare professionals in identifying potential behavioral health safety signals
in unstructured clinical text.

## This Software Is NOT

- A diagnostic tool
- FDA-cleared or FDA-approved
- A substitute for clinical judgment
- A substitute for validated screening instruments (C-SSRS, PHQ-9, GAD-7, etc.)
- A substitute for the therapeutic relationship
- Intended for use as the sole basis for any clinical decision
- A replacement for direct clinical assessment

## This Software IS

- A signal detection tool that flags text for clinician review
- A clinical decision support aid under the 21st Century Cures Act (Non-Device CDS)
- Designed to prompt validated assessments, not replace them
- Built to surface potential concerns that may otherwise be missed at scale

## Intended Use

bh-sentinel analyzes unstructured clinical text (intake notes, journal entries,
session summaries, patient communications) and returns structured safety flags
with severity levels, confidence scores, and evidence context. Every flag is a
signal for clinician review, not a clinical recommendation.

## Limitations

- Pattern-based detection has inherent false positive and false negative rates
- The system cannot understand context, tone, or therapeutic intent
- Clinical text varies significantly across providers, settings, and populations
- The system has not been validated in a clinical trial
- Performance may vary across clinical settings and patient populations

## Responsibility

The treating clinician retains full responsibility for all clinical decisions.
bh-sentinel outputs do not constitute medical advice. Organizations deploying
this software must establish clinical review workflows and ensure that no
autonomous clinical action is taken based solely on system output.

## Regulatory Status

bh-sentinel is designed to satisfy the four criteria for Non-Device Clinical
Decision Support under Section 3060 of the 21st Century Cures Act. It is not
intended to meet the definition of a device under Section 201(h) of the FD&C
Act. See `docs/fda-cds-analysis.md` for the full regulatory analysis.
