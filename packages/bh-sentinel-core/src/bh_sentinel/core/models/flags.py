"""Flag taxonomy enums and flag definition models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Severity(StrEnum):
    """Clinical severity levels."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    POSITIVE = "POSITIVE"


class Domain(StrEnum):
    """Clinical flag domains."""

    SELF_HARM = "self_harm"
    HARM_TO_OTHERS = "harm_to_others"
    MEDICATION = "medication"
    SUBSTANCE_USE = "substance_use"
    CLINICAL_DETERIORATION = "clinical_deterioration"
    PROTECTIVE_FACTORS = "protective_factors"


class DetectionLayer(StrEnum):
    """Pipeline layer that produced a flag."""

    PATTERN_MATCH = "pattern_match"
    TRANSFORMER = "transformer"
    RULES_ENGINE = "rules_engine"


class LayerStatus(StrEnum):
    """Execution status of a pipeline layer."""

    NOT_RUN = "not_run"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class EvidenceSpan(BaseModel):
    """Character-level evidence span pointing back to the source text."""

    sentence_index: int
    char_start: int
    char_end: int


class Flag(BaseModel):
    """A single clinical safety flag detected by the pipeline."""

    flag_id: str
    domain: Domain
    name: str
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    detection_layer: DetectionLayer
    matched_context_hint: str
    basis_description: str
    evidence_span: EvidenceSpan
