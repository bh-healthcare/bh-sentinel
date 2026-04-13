"""Pipeline orchestrator for multi-layer analysis."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

from pydantic import ValidationError

from bh_sentinel.core._config import (
    default_emotion_lexicon_path,
    default_flag_taxonomy_path,
    default_patterns_path,
    default_rules_path,
)
from bh_sentinel.core._types import EmotionScores, PatternMatchCandidate
from bh_sentinel.core.emotion_lexicon import EmotionLexicon
from bh_sentinel.core.models.flags import Domain, LayerStatus, Severity
from bh_sentinel.core.models.request import AnalysisConfig, AnalysisRequest
from bh_sentinel.core.models.response import (
    AnalysisResponse,
    AnalysisSummary,
    EmotionResult,
    ErrorCode,
    ErrorResponse,
    PipelineStatus,
)
from bh_sentinel.core.negation_detector import NegationDetector
from bh_sentinel.core.pattern_matcher import PatternMatcher
from bh_sentinel.core.preprocessor import TextPreprocessor
from bh_sentinel.core.rules_engine import RulesEngine
from bh_sentinel.core.taxonomy import FlagTaxonomy
from bh_sentinel.core.temporal_detector import TemporalDetector


class Pipeline:
    """Orchestrates parallel execution of pattern matching, transformer
    classification, and emotion lexicon layers, then feeds results through
    the rules engine for final flag determination.
    """

    def __init__(
        self,
        enable_patterns: bool = True,
        enable_transformer: bool = False,
        enable_emotion_lexicon: bool = True,
        taxonomy_path: Path | None = None,
        patterns_path: Path | None = None,
        rules_path: Path | None = None,
        lexicon_path: Path | None = None,
    ) -> None:
        self._enable_patterns = enable_patterns
        self._enable_transformer = enable_transformer
        self._enable_emotion_lexicon = enable_emotion_lexicon

        tax_path = taxonomy_path or default_flag_taxonomy_path()
        self._taxonomy = FlagTaxonomy(tax_path)
        self._preprocessor = TextPreprocessor()
        self._negation = NegationDetector()
        self._temporal = TemporalDetector()

        if enable_patterns:
            pat_path = patterns_path or default_patterns_path()
            self._pattern_matcher = PatternMatcher(
                pat_path, self._taxonomy, self._negation, self._temporal
            )
        else:
            self._pattern_matcher = None

        if enable_emotion_lexicon:
            lex_path = lexicon_path or default_emotion_lexicon_path()
            try:
                self._emotion_lexicon = EmotionLexicon(lex_path)
            except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
                self._emotion_lexicon = None
        else:
            self._emotion_lexicon = None

        rul_path = rules_path or default_rules_path()
        self._rules_engine = RulesEngine(rul_path, self._taxonomy)

    async def analyze(
        self,
        text: str,
        config: AnalysisConfig | None = None,
    ) -> AnalysisResponse | ErrorResponse:
        """Run the full analysis pipeline."""
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        # Validate input.
        try:
            request = AnalysisRequest(text=text)
            validated_text = request.text
        except ValidationError as e:
            return self._make_error(request_id, text, e)

        if config is None:
            config = AnalysisConfig()

        # Preprocess.
        preprocessed = self._preprocessor.process(validated_text)

        # Layer statuses.
        ps = PipelineStatus()

        # Run L1 and L3 in parallel.
        l1_candidates: list[PatternMatchCandidate] = []
        emotion_scores = EmotionScores()

        async def run_l1():
            nonlocal l1_candidates
            if self._pattern_matcher:
                l1_candidates = self._pattern_matcher.match(preprocessed)

        async def run_l3():
            nonlocal emotion_scores
            if self._emotion_lexicon:
                emotion_scores = self._emotion_lexicon.score(validated_text)

        await asyncio.gather(run_l1(), run_l3())

        if self._pattern_matcher:
            ps.layer_1_pattern = LayerStatus.COMPLETED
        else:
            ps.layer_1_pattern = LayerStatus.SKIPPED

        ps.layer_2_transformer = LayerStatus.SKIPPED

        if self._emotion_lexicon:
            ps.layer_3_emotion_lexicon = LayerStatus.COMPLETED
        elif not self._enable_emotion_lexicon:
            ps.layer_3_emotion_lexicon = LayerStatus.SKIPPED
        else:
            ps.layer_3_emotion_lexicon = LayerStatus.FAILED

        # L4: Rules engine.
        rules_result = self._rules_engine.evaluate(l1_candidates, emotion_scores)
        ps.layer_4_rules = LayerStatus.COMPLETED

        # Separate protective factors.
        risk_flags = [f for f in rules_result.flags if f.domain != Domain.PROTECTIVE_FACTORS]
        protective = [f for f in rules_result.flags if f.domain == Domain.PROTECTIVE_FACTORS]

        # Apply domain filter.
        if config.domains:
            domain_set = set(config.domains)
            risk_flags = [f for f in risk_flags if f.domain in domain_set]

        # Apply min_severity filter.
        # Severity rank: CRITICAL=4, HIGH=3, MEDIUM=2, LOW=1, POSITIVE=0
        sev_rank = {
            Severity.POSITIVE: 0,
            Severity.LOW: 1,
            Severity.MEDIUM: 2,
            Severity.HIGH: 3,
            Severity.CRITICAL: 4,
        }
        min_rank = sev_rank[config.min_severity]
        risk_flags = [f for f in risk_flags if sev_rank[f.severity] >= min_rank]

        # Build summary.
        max_sev = Severity.LOW
        if risk_flags:
            max_sev = max(risk_flags, key=lambda f: sev_rank[f.severity]).severity
        elif protective:
            max_sev = Severity.POSITIVE

        domains_flagged = sorted(set(f.domain for f in risk_flags))

        recommended_action = None
        if rules_result.recommended_actions:
            recommended_action = "; ".join(rules_result.recommended_actions)

        summary = AnalysisSummary(
            max_severity=max_sev,
            total_flags=len(risk_flags),
            domains_flagged=domains_flagged,
            requires_immediate_review=rules_result.requires_immediate_review,
            recommended_action=recommended_action,
        )

        # Build emotion result.
        emotions = None
        if config.include_emotions:
            emotions = EmotionResult(
                primary=emotion_scores.primary,
                secondary=emotion_scores.secondary,
                category_scores=emotion_scores.scores,
                comprehend_available=False,
            )

        elapsed = (time.perf_counter() - start) * 1000

        return AnalysisResponse(
            request_id=request_id,
            processing_time_ms=elapsed,
            taxonomy_version=self._taxonomy.version,
            flags=risk_flags,
            emotions=emotions,
            protective_factors=protective if config.include_protective else [],
            summary=summary,
            pipeline_status=ps,
        )

    def analyze_sync(
        self,
        text: str,
        config: AnalysisConfig | None = None,
    ) -> AnalysisResponse | ErrorResponse:
        """Synchronous wrapper for analyze(). Cannot be called from async context."""
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "analyze_sync() cannot be called from within a running event loop. "
                "Use 'await pipeline.analyze()' instead."
            )
        except RuntimeError as e:
            if "no current event loop" in str(e) or "no running event loop" in str(e):
                return asyncio.run(self.analyze(text, config))
            raise

    @staticmethod
    def _make_error(request_id: str, text: str, error: ValidationError) -> ErrorResponse:
        """Build a PHI-safe error response from a validation error."""
        err_str = str(error)
        if "at least" in err_str or "too_short" in err_str:
            if not text or not text.strip():
                return ErrorResponse(
                    request_id=request_id,
                    error_code=ErrorCode.VALIDATION_TEXT_EMPTY,
                    message=f"Text is empty or whitespace-only. Request ID: {request_id}",
                    http_status=400,
                )
            return ErrorResponse(
                request_id=request_id,
                error_code=ErrorCode.VALIDATION_TEXT_TOO_SHORT,
                message=f"Text is too short. Request ID: {request_id}",
                http_status=400,
            )
        if "at most" in err_str or "too_long" in err_str:
            return ErrorResponse(
                request_id=request_id,
                error_code=ErrorCode.VALIDATION_TEXT_TOO_LONG,
                message=f"Text exceeds maximum length. Request ID: {request_id}",
                http_status=400,
            )
        return ErrorResponse(
            request_id=request_id,
            error_code=ErrorCode.VALIDATION_TEXT_EMPTY,
            message=f"Invalid input. Request ID: {request_id}",
            http_status=400,
        )
