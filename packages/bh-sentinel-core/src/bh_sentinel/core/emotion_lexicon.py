"""Behavioral health emotion lexicon for word-level emotion category scoring."""

from __future__ import annotations


class EmotionLexicon:
    """In-process behavioral health emotion lexicon.

    Maps text to clinically-relevant emotion categories (hopelessness,
    agitation, anxiety, anger, sadness, guilt, shame, mania, dissociation,
    positive_valence, negative_valence) using word-level lookup from
    config/emotion_lexicon.json. No ML dependency, no external service,
    ~2ms per analysis.

    This is a project-owned lexicon curated from standard behavioral health
    clinical vocabulary. It is not derived from NRC EmoLex or any other
    proprietary dataset. See the lexicon JSON for sourcing methodology.

    Scoring: for each category, count input words that map to that category,
    divide by total word count, producing a 0.0-1.0 density score.
    """

    pass
