"""Behavioral health emotion lexicon for word-level emotion category scoring."""

from __future__ import annotations

import json
import re
from pathlib import Path

from bh_sentinel.core._types import EmotionScores


class EmotionLexicon:
    """In-process behavioral health emotion lexicon.

    Maps text to clinically-relevant emotion categories (hopelessness,
    agitation, anxiety, anger, sadness, guilt, shame, mania, dissociation,
    positive_valence, negative_valence) using word-level lookup from
    config/emotion_lexicon.json. No ML dependency, no external service.

    This is a project-owned lexicon curated from standard behavioral health
    clinical vocabulary.

    Scoring: for each category, count input words that map to that category,
    divide by total word count, producing a 0.0-1.0 density score.
    """

    def __init__(self, path: Path) -> None:
        with open(path) as f:
            data = json.load(f)

        self._categories: list[str] = data["categories"]
        self._words: dict[str, list[str]] = data["words"]

        # Separate single-word and multi-word entries for efficient matching.
        self._single_words: dict[str, list[str]] = {}
        self._multi_word_phrases: list[tuple[str, list[str]]] = []

        for term, cats in self._words.items():
            lower_term = term.lower()
            if " " in lower_term:
                self._multi_word_phrases.append((lower_term, cats))
            else:
                self._single_words[lower_term] = cats

    @property
    def categories(self) -> list[str]:
        return list(self._categories)

    @property
    def term_count(self) -> int:
        return len(self._words)

    def score(self, text: str) -> EmotionScores:
        """Score text across all emotion categories using density scoring."""
        scores: dict[str, float] = {cat: 0.0 for cat in self._categories}

        if not text or not text.strip():
            return EmotionScores(scores=scores)

        # Tokenize: lowercase, split on whitespace, strip punctuation.
        tokens = self._tokenize(text)
        if not tokens:
            return EmotionScores(scores=scores)

        total_tokens = len(tokens)
        category_counts: dict[str, int] = {cat: 0 for cat in self._categories}

        # Single-word matching.
        for token in tokens:
            if token in self._single_words:
                for cat in self._single_words[token]:
                    category_counts[cat] += 1

        # Multi-word phrase matching via sliding window on the lowered text.
        lower_text = text.lower()
        for phrase, cats in self._multi_word_phrases:
            if phrase in lower_text:
                for cat in cats:
                    category_counts[cat] += 1

        # Density scoring: matches / total tokens.
        for cat in self._categories:
            if category_counts[cat] > 0:
                scores[cat] = min(1.0, category_counts[cat] / total_tokens)

        return EmotionScores(scores=scores)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text: lowercase, split whitespace, strip punctuation."""
        tokens = []
        for word in text.lower().split():
            cleaned = re.sub(r"[^\w'-]", "", word)
            if cleaned:
                tokens.append(cleaned)
        return tokens
