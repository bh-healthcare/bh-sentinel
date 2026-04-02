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

<!-- TODO: Expand with detailed criterion-by-criterion analysis, risk assessment, and deployment considerations. -->
