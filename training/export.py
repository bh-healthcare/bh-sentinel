"""Stub for v0.3+ fine-tuned model export.

This file is deliberately a stub. There are two distinct ONNX export paths
in bh-sentinel:

1. **Baseline zero-shot model export (v0.2.1+)** -- see `scripts/export_onnx.py`
   at the repo root. Exports the pinned upstream NLI model (currently
   `facebook/bart-large-mnli`, see `docs/ml-artifact-provenance.md`) to ONNX
   with INT8 quantization, then stages everything `hf upload` needs to
   publish the canonical artifact under `bh-healthcare/...` on HF Hub. This
   is the path used for every public bh-sentinel-ml release artifact.

2. **Fine-tuned organizational model export (v0.3+, not yet implemented)** --
   this file. Future home for the workflow that exports a model an
   organization fine-tuned on their own de-identified clinical text via
   the `training/train.py` pipeline. Different inputs (organizational
   checkpoint, not an HF Hub id), different governance (organizational
   model card, not a public HF repo), and different runtime hookup
   (organizational `model_path`, not the pinned shared SHA).

Both paths converge on the same downstream contract: the resulting ONNX
must satisfy the input/output shape requirements documented in
`scripts/export_onnx.py` so `bh_sentinel.ml.transformer.TransformerClassifier`
can consume it without code changes.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "Fine-tuned model export is a v0.3+ deliverable. "
        "For the baseline zero-shot model export used by the public "
        "bh-sentinel-ml releases, run `python scripts/export_onnx.py` "
        "from the repo root. See docs/training-guide.md and "
        "docs/ml-artifact-provenance.md for context."
    )


if __name__ == "__main__":
    main()
