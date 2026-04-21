"""Domain-specific exceptions for bh-sentinel-ml.

All error messages produced by this package are static templates that do
not include any input text. Raw clinical text must never appear in an
exception message -- callers may log exceptions unfiltered, and the PHI
safety posture depends on every exception path being static-by-design.
"""

from __future__ import annotations


class ModelNotFoundError(FileNotFoundError):
    """Raised when no ONNX model can be located.

    Carries a static remediation message listing the paths that were
    checked and the CLI invocation to resolve the situation. Does not
    include any input text or runtime-derived strings beyond the paths.
    """


class ModelIntegrityError(ValueError):
    """Raised when an ONNX model's SHA256 does not match the pinned digest.

    Critical guardrail: when this fires, no InferenceSession has been
    constructed yet. A stale or tampered container bake cannot silently
    serve predictions -- the pipeline fails fast at construction.
    """


class InferenceError(RuntimeError):
    """Raised when ONNX Runtime inference fails.

    Pipeline callers catch this and mark the L2 layer FAILED while still
    returning L1+L3+L4 results. No input text is included in the message.
    """
