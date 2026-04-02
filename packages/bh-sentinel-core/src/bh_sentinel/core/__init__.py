"""bh-sentinel-core: Pattern-based clinical safety signal detection."""

__version__ = "0.1.0"

from bh_sentinel.core.emotion_lexicon import EmotionLexicon
from bh_sentinel.core.models import (
    AnalysisConfig,
    AnalysisContext,
    AnalysisRequest,
    AnalysisResponse,
    AnalysisSummary,
    DetectionLayer,
    Domain,
    EmotionResult,
    EvidenceSpan,
    Flag,
    LayerStatus,
    PipelineStatus,
    Severity,
)
from bh_sentinel.core.negation_detector import NegationDetector
from bh_sentinel.core.pattern_matcher import PatternMatcher
from bh_sentinel.core.pipeline import Pipeline
from bh_sentinel.core.preprocessor import TextPreprocessor
from bh_sentinel.core.rules_engine import RulesEngine
from bh_sentinel.core.taxonomy import FlagTaxonomy
from bh_sentinel.core.temporal_detector import TemporalDetector

__all__ = [
    "__version__",
    # Models
    "AnalysisConfig",
    "AnalysisContext",
    "AnalysisRequest",
    "AnalysisResponse",
    "AnalysisSummary",
    "DetectionLayer",
    "Domain",
    "EmotionResult",
    "EvidenceSpan",
    "Flag",
    "LayerStatus",
    "PipelineStatus",
    "Severity",
    # Core classes
    "EmotionLexicon",
    "FlagTaxonomy",
    "NegationDetector",
    "PatternMatcher",
    "Pipeline",
    "RulesEngine",
    "TextPreprocessor",
    "TemporalDetector",
]
