"""Text normalization, sentence splitting, and character offset tracking."""

from __future__ import annotations

import re

from bh_sentinel.core._types import PreprocessedText, SentenceBoundary

# Clinical abbreviations that should NOT trigger sentence splits.
# Lowercase for case-insensitive comparison.
_ABBREVIATIONS = frozenset(
    {
        "dr",
        "mr",
        "mrs",
        "ms",
        "pt",
        "pts",
        "hx",
        "dx",
        "tx",
        "rx",
        "sx",
        "fx",
        "cx",
        "pmh",
        "vs",
        "no",
        "approx",
        "dept",
        "est",
        "e.g",
        "i.e",
        "etc",
        "vol",
        "avg",
        "min",
        "max",
        "inc",
        "ltd",
    }
)

# Regex for splitting on sentence-ending punctuation followed by whitespace.
# Uses a lookahead to preserve the punctuation in the preceding sentence.
_SENT_END = re.compile(r"([.!?])\s+")

# Regex for semicolons as separators.
_SEMICOLON = re.compile(r";\s*")


class TextPreprocessor:
    """Splits input text into sentences, normalizes whitespace, and tracks
    character offsets so downstream layers can map flags back to source spans.
    """

    def process(self, text: str) -> PreprocessedText:
        if not text or not text.strip():
            return PreprocessedText(original=text, sentences=())

        sentences: list[SentenceBoundary] = []
        index = 0

        # Phase 1: Split on newlines
        for line_match in re.finditer(r"[^\n]+", text):
            line = line_match.group()
            line_start = line_match.start()

            if not line.strip():
                continue

            # Phase 2: Split on semicolons within line
            semi_segments = self._split_semicolons(line, line_start)

            # Phase 3: Split on sentence-ending punctuation within each segment
            for seg_text, seg_start in semi_segments:
                for sent_text, sent_start in self._split_sentences(seg_text, seg_start):
                    stripped = sent_text.strip()
                    if not stripped:
                        continue

                    # Find the offset of the stripped text within the original
                    lstrip_offset = len(sent_text) - len(sent_text.lstrip())
                    abs_start = sent_start + lstrip_offset
                    abs_end = abs_start + len(stripped)

                    sentences.append(
                        SentenceBoundary(
                            text=stripped,
                            index=index,
                            char_start=abs_start,
                            char_end=abs_end,
                        )
                    )
                    index += 1

        return PreprocessedText(original=text, sentences=tuple(sentences))

    def _split_semicolons(self, text: str, offset: int) -> list[tuple[str, int]]:
        """Split text on semicolons, returning (segment, abs_offset) pairs."""
        parts: list[tuple[str, int]] = []
        last_end = 0
        for m in _SEMICOLON.finditer(text):
            parts.append((text[last_end : m.start()], offset + last_end))
            last_end = m.end()
        parts.append((text[last_end:], offset + last_end))
        return parts

    def _split_sentences(self, text: str, offset: int) -> list[tuple[str, int]]:
        """Split text on sentence-ending punctuation, respecting abbreviations."""
        parts: list[tuple[str, int]] = []
        last_end = 0

        for m in _SENT_END.finditer(text):
            punct_pos = m.start()
            # Check if this period is an abbreviation
            if m.group(1) == ".":
                word_before = self._word_before_period(text, punct_pos)
                if word_before and word_before.lower() in _ABBREVIATIONS:
                    continue

            # Include the punctuation in the sentence
            sent = text[last_end : punct_pos + 1]
            parts.append((sent, offset + last_end))
            last_end = m.end()

        # Remainder
        if last_end < len(text):
            parts.append((text[last_end:], offset + last_end))

        return parts

    @staticmethod
    def _word_before_period(text: str, period_pos: int) -> str | None:
        """Extract the word immediately before a period."""
        i = period_pos - 1
        while i >= 0 and text[i].isalpha():
            i -= 1
        word = text[i + 1 : period_pos]
        # Handle abbreviations with dots like "e.g" or "i.e"
        if i >= 1 and text[i] == "." and text[i - 1].isalpha():
            word = text[i - 1 : period_pos]
        return word if word else None
