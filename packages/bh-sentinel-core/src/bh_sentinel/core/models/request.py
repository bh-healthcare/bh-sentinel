"""Pydantic models for analysis request configuration."""

from __future__ import annotations

from pydantic import BaseModel

from bh_sentinel.core.models.flags import Domain, Severity


class AnalysisContext(BaseModel):
    """Optional context about the source of the text being analyzed."""

    program: str | None = None
    session_type: str | None = None
    language: str = "en"


class AnalysisConfig(BaseModel):
    """Configuration for a single analysis run."""

    domains: list[Domain] | None = None
    min_severity: Severity = Severity.LOW
    include_protective: bool = True
    include_emotions: bool = True
    include_comprehend: bool = False


class AnalysisRequest(BaseModel):
    """Complete analysis request."""

    text: str
    source: str | None = None
    context: AnalysisContext | None = None
    config: AnalysisConfig | None = None
