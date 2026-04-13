"""Negation detection for clinical text (e.g., 'denies SI', 'no suicidal ideation')."""

from __future__ import annotations

import re

# Pseudo-negation phrases that should NOT trigger negation suppression.
# These are patterns where a negation word is part of a clinically significant phrase.
_PSEUDO_NEGATIONS = [
    re.compile(r"no longer denies?", re.IGNORECASE),
    re.compile(r"does(?:n'?t| not) deny", re.IGNORECASE),
    re.compile(r"no longer", re.IGNORECASE),
    re.compile(r"unable to", re.IGNORECASE),
    re.compile(r"can(?:'?t|not) (?:stop|keep from)", re.IGNORECASE),
    re.compile(r"no improvement", re.IGNORECASE),
    re.compile(r"no reason to live", re.IGNORECASE),
    re.compile(r"no reason to go on", re.IGNORECASE),
    re.compile(r"no way out", re.IGNORECASE),
]

# Scope terminators: negation does not cross these.
_SCOPE_TERMINATORS = frozenset({".", ",", ";", ":"})
_SCOPE_TERMINATOR_WORDS = frozenset(
    {
        "but",
        "however",
        "although",
        "yet",
        "except",
        "though",
        "only",
    }
)

# Conjunctive connectors: negation DOES cross these.
_CONJUNCTIVE_CONNECTORS = frozenset({"/", "and", "or"})

# Post-negation phrases that appear after the match.
_POST_NEGATION = [
    re.compile(r"\bdenied\b", re.IGNORECASE),
    re.compile(r"\bnegative\b", re.IGNORECASE),
    re.compile(r"\babsent\b", re.IGNORECASE),
    re.compile(r"\bruled out\b", re.IGNORECASE),
]

# Lookback/forward window sizes in characters.
_LOOKBACK_WINDOW = 60
_FORWARD_WINDOW = 30


class NegationDetector:
    """Identifies negation cues in clinical text and marks affected spans
    so pattern matches within negated regions can be suppressed or downgraded.
    """

    def is_negated(
        self,
        text: str,
        match_start: int,
        match_end: int,
        negation_phrases: list[str],
    ) -> bool:
        if not text or not negation_phrases:
            return False

        # Check for pseudo-negation first in the lookback window.
        lookback_start = max(0, match_start - _LOOKBACK_WINDOW)
        lookback_text = text[lookback_start:match_start]

        for pseudo in _PSEUDO_NEGATIONS:
            if pseudo.search(lookback_text):
                return False

        # Check lookback window for negation cues.
        if self._check_lookback(text, match_start, negation_phrases):
            return True

        # Check forward window for post-negation cues.
        if self._check_forward(text, match_end):
            return True

        return False

    def _check_lookback(
        self,
        text: str,
        match_start: int,
        negation_phrases: list[str],
    ) -> bool:
        """Check the lookback window for negation cues, respecting scope terminators."""
        lookback_start = max(0, match_start - _LOOKBACK_WINDOW)
        lookback_text = text[lookback_start:match_start]

        # Find the rightmost negation cue in the lookback window.
        best_cue_end = -1
        for phrase in negation_phrases:
            try:
                pattern = re.compile(phrase, re.IGNORECASE)
            except re.error:
                continue
            # Check lookback window only
            for m in pattern.finditer(lookback_text):
                if m.end() > best_cue_end:
                    best_cue_end = m.end()
            # Also check if the negation phrase spans into the match (e.g., "-SI"
            # where the phrase is `(?:\-|negative) (?:for )?SI` matching across
            # the lookback/match boundary).
            extended = text[lookback_start : match_start + 20]
            for m in pattern.finditer(extended):
                # The cue must start before the match
                if m.start() + lookback_start < match_start and m.end() > best_cue_end:
                    best_cue_end = m.end()

        if best_cue_end < 0:
            return False

        # Check for scope terminators between the cue and the match.
        between = lookback_text[best_cue_end:]
        return not self._has_scope_terminator(between)

    def _check_forward(self, text: str, match_end: int) -> bool:
        """Check the forward window for post-negation cues."""
        forward_end = min(len(text), match_end + _FORWARD_WINDOW)
        forward_text = text[match_end:forward_end]

        for post_neg in _POST_NEGATION:
            if post_neg.search(forward_text):
                return True
        return False

    @staticmethod
    def _has_scope_terminator(text: str) -> bool:
        """Check if text contains a scope terminator (punctuation or keyword)."""
        for char in text:
            if char in _SCOPE_TERMINATORS:
                return True

        # Check for word-level scope terminators.
        words = text.lower().split()
        for word in words:
            # Strip punctuation for word matching.
            clean = word.strip(".,;:!?")
            if clean in _SCOPE_TERMINATOR_WORDS:
                return True

        return False
