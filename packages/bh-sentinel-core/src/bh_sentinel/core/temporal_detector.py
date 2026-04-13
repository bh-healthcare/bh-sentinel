"""Temporal context detection (past vs. present tense, historical references)."""

from __future__ import annotations

import re

# Past markers -- indicate historical context.
_PAST_MARKERS = [
    re.compile(r"\bhistory of\b", re.IGNORECASE),
    re.compile(r"\bhx of\b", re.IGNORECASE),
    re.compile(r"\bpmh\b", re.IGNORECASE),
    re.compile(r"\bprior\b", re.IGNORECASE),
    re.compile(r"\bprevious\b", re.IGNORECASE),
    re.compile(r"\bpast\b", re.IGNORECASE),
    re.compile(r"\bformer\b", re.IGNORECASE),
    re.compile(r"\bremote\b", re.IGNORECASE),
    re.compile(r"\byears? ago\b", re.IGNORECASE),
    re.compile(r"\bmonths? ago\b", re.IGNORECASE),
    re.compile(r"\bin \d{4}\b", re.IGNORECASE),
    re.compile(r"\bas a (?:child|teenager|adolescent|kid|youth)\b", re.IGNORECASE),
    re.compile(r"\bused to\b", re.IGNORECASE),
    re.compile(r"\bhad been\b", re.IGNORECASE),
    re.compile(r"\bresolved\b", re.IGNORECASE),
    re.compile(r"\bno longer\b", re.IGNORECASE),
]

# Present markers -- indicate current or recent context. Present wins when both found.
_PRESENT_MARKERS = [
    re.compile(r"\bcurrently\b", re.IGNORECASE),
    re.compile(r"\bnow\b", re.IGNORECASE),
    re.compile(r"\btoday\b", re.IGNORECASE),
    re.compile(r"\bthis week\b", re.IGNORECASE),
    re.compile(r"\blast night\b", re.IGNORECASE),
    re.compile(r"\brecently\b", re.IGNORECASE),
    re.compile(r"\bright now\b", re.IGNORECASE),
    re.compile(r"\bactively\b", re.IGNORECASE),
    re.compile(r"\bpresently\b", re.IGNORECASE),
    re.compile(r"\bongoing\b", re.IGNORECASE),
    re.compile(r"\bagain\b", re.IGNORECASE),
    re.compile(r"\bstarted\b", re.IGNORECASE),
    re.compile(r"\bresumed\b", re.IGNORECASE),
    re.compile(r"\breturned\b", re.IGNORECASE),
    re.compile(r"\bcame back\b", re.IGNORECASE),
    re.compile(r"\breports?\b", re.IGNORECASE),
    re.compile(r"\bendorses?\b", re.IGNORECASE),
]

# Window sizes for temporal marker search.
_LOOKBACK_WINDOW = 80
_FORWARD_WINDOW = 40


class TemporalDetector:
    """Detects temporal qualifiers in clinical text (e.g., 'used to cut',
    'history of attempt') to distinguish current risk from historical context.
    """

    def classify(self, text: str, match_start: int, match_end: int) -> str:
        """Classify the temporal context of a match as 'past' or 'present'.

        Resolution rule: if present marker found -> 'present' (always);
        if only past markers -> 'past'; if neither -> 'present' (safe default).
        """
        if not text:
            return "present"

        window_start = max(0, match_start - _LOOKBACK_WINDOW)
        window_end = min(len(text), match_end + _FORWARD_WINDOW)
        window = text[window_start:window_end]

        has_present = any(m.search(window) for m in _PRESENT_MARKERS)
        has_past = any(m.search(window) for m in _PAST_MARKERS)

        if has_present:
            return "present"
        if has_past:
            return "past"
        return "present"
