"""Internal types shared between pipeline components. Not part of the public API."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SentenceBoundary:
    """A single sentence with character offsets back to the original text."""

    text: str
    index: int
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class PreprocessedText:
    """Result of text preprocessing: original text plus sentence boundaries."""

    original: str
    sentences: tuple[SentenceBoundary, ...]


@dataclass(slots=True)
class PatternMatchCandidate:
    """A candidate pattern match before rules engine processing."""

    flag_id: str
    domain: str
    name: str
    default_severity: str
    confidence: float
    sentence_index: int
    char_start: int
    char_end: int
    pattern_text: str
    basis_description: str
    matched_context_hint: str
    negated: bool = False
    temporal_context: str = "present"


@dataclass(slots=True)
class EmotionScores:
    """Emotion category density scores from the lexicon."""

    scores: dict[str, float] = field(default_factory=dict)

    @property
    def primary(self) -> str | None:
        """Category with highest score > 0, or None if all zero/empty."""
        if not self.scores:
            return None
        best = max(self.scores, key=self.scores.get)  # type: ignore[arg-type]
        return best if self.scores[best] > 0.0 else None

    @property
    def secondary(self) -> str | None:
        """Category with second-highest score > 0, or None."""
        if len(self.scores) < 2:
            return None
        sorted_cats = sorted(self.scores, key=self.scores.get, reverse=True)  # type: ignore[arg-type]
        return sorted_cats[1] if self.scores[sorted_cats[1]] > 0.0 else None
