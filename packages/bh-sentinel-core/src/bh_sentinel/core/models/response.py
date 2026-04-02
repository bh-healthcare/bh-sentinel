"""Pydantic models for analysis response, including evidence spans and basis descriptions."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from bh_sentinel.core.models.flags import Domain, Flag, LayerStatus, Severity


class EmotionResult(BaseModel):
    """Emotion analysis result from Layer 3."""

    primary: str | None = None
    secondary: str | None = None
    comprehend_available: bool = False
    sentiment: dict | None = None


class AnalysisSummary(BaseModel):
    """Summary of an analysis run."""

    max_severity: Severity
    total_flags: int
    domains_flagged: list[Domain]
    requires_immediate_review: bool
    recommended_action: str | None = None


class PipelineStatus(BaseModel):
    """Status of each pipeline layer."""

    layer_1_pattern: LayerStatus = LayerStatus.NOT_RUN
    layer_2_transformer: LayerStatus = LayerStatus.NOT_RUN
    layer_3_comprehend: LayerStatus = LayerStatus.NOT_RUN
    layer_3_emotion_lexicon: LayerStatus = LayerStatus.NOT_RUN
    layer_4_rules: LayerStatus = LayerStatus.NOT_RUN


class AnalysisResponse(BaseModel):
    """Complete response from a pipeline analysis run."""

    request_id: str
    analysis_timestamp: datetime | None = None
    processing_time_ms: float
    taxonomy_version: str
    flags: list[Flag]
    emotions: EmotionResult | None = None
    protective_factors: list[Flag] = []
    summary: AnalysisSummary
    pipeline_status: PipelineStatus
