"""Layer 1: Compiled regex pattern matching with negation and temporal awareness."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from bh_sentinel.core._types import PatternMatchCandidate, PreprocessedText
from bh_sentinel.core.negation_detector import NegationDetector
from bh_sentinel.core.taxonomy import FlagTaxonomy
from bh_sentinel.core.temporal_detector import TemporalDetector


@dataclass(slots=True)
class _CompiledFlag:
    """Internal: a flag with pre-compiled regex patterns."""

    flag_id: str
    domain: str
    name: str
    default_severity: str
    confidence: float
    patterns: list[re.Pattern[str]]
    negation_phrases: list[str]


class PatternMatcher:
    """Deterministic regex/keyword detection engine.

    Loads patterns from YAML configuration, compiles them into regex objects,
    and matches against preprocessed text with negation and temporal context.
    """

    def __init__(
        self,
        patterns_path: Path,
        taxonomy: FlagTaxonomy,
        negation_detector: NegationDetector,
        temporal_detector: TemporalDetector,
    ) -> None:
        self._taxonomy = taxonomy
        self._negation = negation_detector
        self._temporal = temporal_detector
        self._compiled: list[_CompiledFlag] = []
        self._total_patterns = 0

        with open(patterns_path) as f:
            data = yaml.safe_load(f)

        for domain_id, flags in data.items():
            if domain_id.startswith("_"):
                continue

            if not isinstance(flags, dict):
                continue

            for flag_id, flag_data in flags.items():
                if not isinstance(flag_data, dict):
                    continue

                tax_flag = taxonomy.get_flag(flag_id)
                if tax_flag is None:
                    continue

                compiled_patterns: list[re.Pattern[str]] = []

                # Compile main patterns
                for p in flag_data.get("patterns", []):
                    try:
                        compiled_patterns.append(re.compile(p, re.IGNORECASE))
                    except re.error:
                        continue

                # Compile clinical shorthand patterns
                for p in flag_data.get("clinical_shorthand", []):
                    try:
                        compiled_patterns.append(re.compile(p, re.IGNORECASE))
                    except re.error:
                        continue

                self._total_patterns += len(compiled_patterns)

                self._compiled.append(
                    _CompiledFlag(
                        flag_id=flag_id,
                        domain=domain_id,
                        name=tax_flag["name"],
                        default_severity=tax_flag["default_severity"],
                        confidence=flag_data.get("confidence", 0.85),
                        patterns=compiled_patterns,
                        negation_phrases=flag_data.get("negation_phrases", []),
                    )
                )

    @property
    def pattern_count(self) -> int:
        return self._total_patterns

    def covered_flag_ids(self) -> list[str]:
        return [cf.flag_id for cf in self._compiled if cf.patterns]

    def match(self, preprocessed: PreprocessedText) -> list[PatternMatchCandidate]:
        """Match all patterns against preprocessed text.

        Returns ALL candidates including negated ones (needed for signal-density
        rules COMP-003/004). Negated candidates have `negated=True`.
        """
        all_candidates: list[PatternMatchCandidate] = []
        original = preprocessed.original

        for sent in preprocessed.sentences:
            sent_candidates: dict[str, PatternMatchCandidate] = {}

            for cf in self._compiled:
                for pattern in cf.patterns:
                    for m in pattern.finditer(sent.text):
                        # Map to absolute offsets in original document
                        abs_start = sent.char_start + m.start()
                        abs_end = sent.char_start + m.end()

                        # Check negation
                        negated = self._negation.is_negated(
                            original, abs_start, abs_end, cf.negation_phrases
                        )

                        # Check temporal context
                        temporal = self._temporal.classify(original, abs_start, abs_end)

                        candidate = PatternMatchCandidate(
                            flag_id=cf.flag_id,
                            domain=cf.domain,
                            name=cf.name,
                            default_severity=cf.default_severity,
                            confidence=cf.confidence,
                            sentence_index=sent.index,
                            char_start=abs_start,
                            char_end=abs_end,
                            pattern_text=pattern.pattern,
                            basis_description=(
                                f"Pattern match detected language consistent with "
                                f"{cf.name.lower()}."
                            ),
                            matched_context_hint=cf.name.lower(),
                            negated=negated,
                            temporal_context=temporal,
                        )

                        # Within-sentence deduplication: per (flag_id, sentence_index),
                        # keep longest match, prefer non-negated.
                        key = f"{cf.flag_id}:{sent.index}"
                        if key in sent_candidates:
                            existing = sent_candidates[key]
                            match_len = abs_end - abs_start
                            existing_len = existing.char_end - existing.char_start
                            # Prefer non-negated, then longest
                            if (not negated and existing.negated) or (
                                negated == existing.negated and match_len > existing_len
                            ):
                                sent_candidates[key] = candidate
                        else:
                            sent_candidates[key] = candidate

            all_candidates.extend(sent_candidates.values())

        return all_candidates
