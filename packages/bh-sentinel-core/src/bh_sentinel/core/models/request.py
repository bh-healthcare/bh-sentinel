"""Pydantic models for analysis request configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from bh_sentinel.core.models.flags import Domain, Severity

TEXT_MAX_LENGTH = 50_000
TEXT_MIN_LENGTH = 3
SUPPORTED_LANGUAGES = ("en",)


class AnalysisContext(BaseModel):
    """Optional context about the source of the text being analyzed."""

    program: str | None = None
    session_type: str | None = None
    language: Literal["en"] = "en"


class AnalysisConfig(BaseModel):
    """Configuration for a single analysis run."""

    domains: list[Domain] | None = None
    min_severity: Severity = Severity.LOW
    include_protective: bool = True
    include_emotions: bool = True
    include_comprehend: bool = False


class AnalysisRequest(BaseModel):
    """Complete analysis request.

    Input constraints:
    - text must be between TEXT_MIN_LENGTH and TEXT_MAX_LENGTH characters
      after whitespace normalization.
    - Whitespace-only or empty input is rejected.
    - Input is normalized to NFC Unicode form and stripped of null bytes.
    - Only 'en' is accepted for context.language (Spanish support planned
      for v0.3).
    """

    text: str = Field(min_length=TEXT_MIN_LENGTH, max_length=TEXT_MAX_LENGTH)
    source: str | None = None
    context: AnalysisContext | None = None
    config: AnalysisConfig | None = None

    @field_validator("text")
    @classmethod
    def normalize_text(cls, v: str) -> str:
        import unicodedata

        v = unicodedata.normalize("NFC", v)
        v = v.replace("\x00", "")
        v = v.strip()
        if not v:
            msg = "text must contain non-whitespace content"
            raise ValueError(msg)
        return v
