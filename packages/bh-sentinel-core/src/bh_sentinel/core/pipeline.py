"""Pipeline orchestrator for multi-layer analysis."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from bh_sentinel.core._config import (
    default_emotion_lexicon_path,
    default_flag_taxonomy_path,
    default_patterns_path,
    default_rules_path,
)
from bh_sentinel.core._types import EmotionScores, PatternMatchCandidate
from bh_sentinel.core.emotion_lexicon import EmotionLexicon
from bh_sentinel.core.models.flags import (
    DetectionLayer,
    Domain,
    LayerStatus,
    Severity,
)
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


def _load_ml() -> dict[str, Any]:
    """Lazy-load the bh-sentinel-ml package.

    Raises ImportError with a clear install hint if the ml package is
    not installed. Returns a dict with the classes/functions Pipeline
    needs -- kept as a dict so tests can patch a single entry point.

    This indirection is the seam that keeps bh-sentinel-core zero-dep
    on onnxruntime, tokenizers, and huggingface_hub. The import only
    happens when enable_transformer=True.
    """
    try:
        from bh_sentinel.ml._config import load_hypotheses, load_ml_config
        from bh_sentinel.ml.calibration import FixedDiscount, TemperatureScaling
        from bh_sentinel.ml.merge import merge_candidates
        from bh_sentinel.ml.model_cache import resolve_model_path
        from bh_sentinel.ml.transformer import TransformerClassifier
        from bh_sentinel.ml.zero_shot import ZeroShotClassifier
    except ImportError as exc:
        raise ImportError(
            "bh-sentinel-ml is not installed. Install it with "
            "`pip install bh-sentinel-ml` to enable the Layer 2 "
            "transformer classifier."
        ) from exc

    return {
        "ZeroShotClassifier": ZeroShotClassifier,
        "TransformerClassifier": TransformerClassifier,
        "merge_candidates": merge_candidates,
        "resolve_model_path": resolve_model_path,
        "load_ml_config": load_ml_config,
        "load_hypotheses": load_hypotheses,
        "FixedDiscount": FixedDiscount,
        "TemperatureScaling": TemperatureScaling,
    }


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
        transformer_model_path: Path | None = None,
        transformer_auto_download: bool = True,
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

        # Layer 2 wiring. On any failure between _load_ml (other than
        # ImportError, which is a hard fail for misconfigured installs)
        # we record the error and mark L2 FAILED at analyze() time.
        self._zero_shot: Any | None = None
        self._merge_candidates_fn: Any | None = None
        self._l2_init_error: Exception | None = None

        if enable_transformer:
            self._init_l2(
                model_path=transformer_model_path,
                auto_download=transformer_auto_download,
            )

    def _init_l2(
        self,
        *,
        model_path: Path | None,
        auto_download: bool,
    ) -> None:
        """Load the ml package and construct the Layer 2 classifier.

        ImportError (ml package missing) is a hard failure -- it means
        the deployment is misconfigured and there is no point starting.
        Any other failure (model cache miss, SHA mismatch, ONNX load
        error, tokenizer load error) is caught here so the pipeline
        still serves L1+L3+L4 with L2 marked FAILED per analyze().
        """
        try:
            ml = _load_ml()
        except ImportError:
            raise
        except Exception as e:
            # A stubbed _load_ml in tests can raise non-ImportError to
            # simulate an incomplete install. Treat like init failure.
            self._l2_init_error = e
            return

        # Tests may stub _load_ml to return a tuple of
        # (ZeroShotClassifier, TransformerClassifier, _) -- in that
        # case the ZeroShotClassifier is already a pre-built stub, and
        # we skip the real transformer construction.
        if isinstance(ml, tuple):
            ZeroShotCls, _TransformerCls, *_rest = ml
            try:
                from bh_sentinel.ml.merge import merge_candidates as merge_fn
            except ImportError:
                merge_fn = None
            self._merge_candidates_fn = merge_fn
            try:
                self._zero_shot = ZeroShotCls()
            except Exception as e:
                self._l2_init_error = e
                self._zero_shot = None
            return

        # Production path: dict-shaped ml bundle.
        self._merge_candidates_fn = ml["merge_candidates"]
        try:
            ml_cfg = ml["load_ml_config"]()
            transformer = self._build_transformer(
                ml,
                ml_cfg=ml_cfg,
                model_path=model_path,
                auto_download=auto_download,
            )
            hypotheses = ml["load_hypotheses"]()
            calibrator = ml["FixedDiscount"](factor=ml_cfg.calibration.get("discount", 0.85))
            self._zero_shot = ml["ZeroShotClassifier"](
                transformer=transformer,
                hypotheses=hypotheses,
                taxonomy=self._taxonomy,
                calibrator=calibrator,
                temporal=self._temporal,
                min_emit_confidence=ml_cfg.min_emit_confidence,
            )
        except Exception as e:
            self._l2_init_error = e
            self._zero_shot = None

    @staticmethod
    def _build_transformer(
        ml: dict[str, Any],
        *,
        ml_cfg: Any,
        model_path: Path | None,
        auto_download: bool,
    ) -> Any:
        """Resolve model path and construct the TransformerClassifier."""
        resolved = ml["resolve_model_path"](
            model_path=model_path,
            auto_download=auto_download,
            onnx_filename=ml_cfg.onnx_filename,
            model_repo=ml_cfg.model_repo,
            model_revision=ml_cfg.model_revision,
        )
        onnx_path = resolved / ml_cfg.onnx_filename
        tokenizer_path = resolved / "tokenizer.json"
        return ml["TransformerClassifier"](
            model_path=onnx_path,
            tokenizer_path=tokenizer_path,
            expected_sha256=ml_cfg.model_sha256,
            max_length=ml_cfg.max_sentence_length,
            max_batch_size=ml_cfg.max_batch_size,
        )

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

        preprocessed = self._preprocessor.process(validated_text)

        ps = PipelineStatus()

        # Run L1, L2, L3 in parallel.
        l1_candidates: list[PatternMatchCandidate] = []
        l2_candidates: list[PatternMatchCandidate] = []
        emotion_scores = EmotionScores()
        l2_failed = False

        async def run_l1():
            nonlocal l1_candidates
            if self._pattern_matcher:
                l1_candidates = self._pattern_matcher.match(preprocessed)

        async def run_l2():
            nonlocal l2_candidates, l2_failed
            if not self._enable_transformer:
                return
            if self._zero_shot is None or self._l2_init_error is not None:
                l2_failed = True
                return
            try:
                l2_candidates = self._zero_shot.classify(preprocessed)
            except Exception:
                l2_failed = True

        async def run_l3():
            nonlocal emotion_scores
            if self._emotion_lexicon:
                emotion_scores = self._emotion_lexicon.score(validated_text)

        await asyncio.gather(run_l1(), run_l2(), run_l3())

        if self._pattern_matcher:
            ps.layer_1_pattern = LayerStatus.COMPLETED
        else:
            ps.layer_1_pattern = LayerStatus.SKIPPED

        if not self._enable_transformer:
            ps.layer_2_transformer = LayerStatus.SKIPPED
        elif l2_failed:
            ps.layer_2_transformer = LayerStatus.FAILED
        else:
            ps.layer_2_transformer = LayerStatus.COMPLETED

        if self._emotion_lexicon:
            ps.layer_3_emotion_lexicon = LayerStatus.COMPLETED
        elif not self._enable_emotion_lexicon:
            ps.layer_3_emotion_lexicon = LayerStatus.SKIPPED
        else:
            ps.layer_3_emotion_lexicon = LayerStatus.FAILED

        # Merge L1 + L2 candidates (architecture 4.7). Collapses
        # duplicate flag_ids with max-confidence + corroboration metadata.
        merged_candidates, corroboration = self._merge_layers(l1_candidates, l2_candidates)

        # Layer 4: rules engine. Merged candidates drive severity;
        # raw l2_candidates are available for signal-density counting.
        rules_result = self._rules_engine.evaluate(
            merged_candidates,
            emotion_scores,
            l2_candidates=l2_candidates,
        )
        ps.layer_4_rules = LayerStatus.COMPLETED

        # Hydrate corroborating_layers onto each Flag.
        for flag in rules_result.flags:
            cor = corroboration.get(flag.flag_id)
            if cor:
                flag.corroborating_layers = [DetectionLayer(name) for name in cor]

        risk_flags = [f for f in rules_result.flags if f.domain != Domain.PROTECTIVE_FACTORS]
        protective = [f for f in rules_result.flags if f.domain == Domain.PROTECTIVE_FACTORS]

        if config.domains:
            domain_set = set(config.domains)
            risk_flags = [f for f in risk_flags if f.domain in domain_set]

        sev_rank = {
            Severity.POSITIVE: 0,
            Severity.LOW: 1,
            Severity.MEDIUM: 2,
            Severity.HIGH: 3,
            Severity.CRITICAL: 4,
        }
        min_rank = sev_rank[config.min_severity]
        risk_flags = [f for f in risk_flags if sev_rank[f.severity] >= min_rank]

        max_sev = Severity.LOW
        if risk_flags:
            max_sev = max(risk_flags, key=lambda f: sev_rank[f.severity]).severity
        elif protective:
            max_sev = Severity.POSITIVE

        domains_flagged = sorted({f.domain for f in risk_flags})

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

    def _merge_layers(
        self,
        l1_candidates: list[PatternMatchCandidate],
        l2_candidates: list[PatternMatchCandidate],
    ) -> tuple[list[PatternMatchCandidate], dict[str, list[str]]]:
        """Merge L1+L2 candidates with corroboration tracking.

        When L2 is not enabled or the ml package's merge function isn't
        loaded (pre-init-error, or test stub that didn't supply one),
        fall back to L1-only with an empty corroboration map -- this
        preserves v0.1 behavior byte-for-byte for the zero-dep path.
        """
        if not self._enable_transformer or self._merge_candidates_fn is None:
            return l1_candidates, {}
        if not l2_candidates:
            return l1_candidates, {}
        result = self._merge_candidates_fn(l1_candidates, l2_candidates)
        return result.candidates, result.corroborating_layers

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
