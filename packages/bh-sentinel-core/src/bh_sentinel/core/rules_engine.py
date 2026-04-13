"""Layer 4: Business logic rules for severity escalation and flag resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bh_sentinel.core._types import EmotionScores, PatternMatchCandidate
from bh_sentinel.core.models.flags import (
    DetectionLayer,
    Domain,
    EvidenceSpan,
    Flag,
    Severity,
)
from bh_sentinel.core.taxonomy import FlagTaxonomy

_SEVERITY_ORDER = ["POSITIVE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]


@dataclass(slots=True)
class RulesResult:
    """Result of rules engine evaluation."""

    flags: list[Flag] = field(default_factory=list)
    requires_immediate_review: bool = False
    recommended_actions: list[str] = field(default_factory=list)


class RulesEngine:
    """Combines signals from pattern matching, transformer, and emotion lexicon
    into final flag determinations with severity escalation and de-escalation.
    """

    def __init__(self, rules_path: Path, taxonomy: FlagTaxonomy) -> None:
        with open(rules_path) as f:
            data = json.load(f)

        self._taxonomy = taxonomy
        self._escalation_rules: list[dict[str, Any]] = data.get("escalation_rules", [])
        self._de_escalation_rules: list[dict[str, Any]] = data.get("de_escalation_rules", [])
        self._compound_rules: list[dict[str, Any]] = data.get("compound_rules", [])
        self._action_rules: list[dict[str, Any]] = data.get("action_rules", [])

    def evaluate(
        self,
        candidates: list[PatternMatchCandidate],
        emotion_scores: EmotionScores,
        l2_candidates: list[PatternMatchCandidate] | None = None,
    ) -> RulesResult:
        """Evaluate rules against candidates and emotion scores.

        Order: escalation -> de-escalation -> compound -> action.
        """
        if l2_candidates is None:
            l2_candidates = []

        # Filter out negated candidates for the working set.
        active = [c for c in candidates if not c.negated]
        # Keep all (including negated) for signal-density counting.
        all_candidates = list(candidates)

        if not active:
            return RulesResult()

        # Build working severity map (can be mutated by rules).
        severity_map: dict[str, str] = {}
        for c in active:
            severity_map[c.flag_id] = c.default_severity

        # Build context for condition evaluation.
        ctx = _EvalContext(
            active=active,
            all_candidates=all_candidates,
            l2_candidates=l2_candidates,
            severity_map=severity_map,
            emotion_scores=emotion_scores,
        )

        result = RulesResult()

        # 1. Escalation rules
        for rule in self._escalation_rules:
            if self._eval_condition(rule["condition"], ctx):
                self._apply_action(rule["action"], ctx, result)

        # 2. De-escalation rules
        for rule in self._de_escalation_rules:
            self._apply_deescalation(rule, ctx)

        # 3. Compound rules
        for rule in self._compound_rules:
            if self._eval_condition(rule["condition"], ctx):
                self._apply_action(rule["action"], ctx, result)

        # 4. Action rules
        for rule in self._action_rules:
            if self._eval_condition(rule["condition"], ctx):
                self._apply_action(rule["action"], ctx, result)

        # Hydrate candidates into Flag models.
        result.flags = self._hydrate_flags(active, ctx.severity_map)

        return result

    def _eval_condition(self, condition: dict[str, Any], ctx: _EvalContext) -> bool:
        """Recursively evaluate a condition dict."""
        if "all_of" in condition:
            return all(self._eval_condition(c, ctx) for c in condition["all_of"])

        if "any_of" in condition:
            return any(self._eval_condition(c, ctx) for c in condition["any_of"])

        if "flag_present" in condition:
            flag_id = condition["flag_present"]
            min_conf = condition.get("min_confidence", 0.0)
            return any(c.flag_id == flag_id and c.confidence >= min_conf for c in ctx.active)

        if "any_flag_present" in condition:
            flag_ids = set(condition["any_flag_present"])
            min_conf = condition.get("min_confidence", 0.0)
            return any(c.flag_id in flag_ids and c.confidence >= min_conf for c in ctx.active)

        if "domain_present" in condition:
            domain = condition["domain_present"]
            return any(c.domain == domain for c in ctx.active)

        if "domain_severity" in condition:
            spec = condition["domain_severity"]
            domain = spec["domain"]
            min_sev = spec["min_severity"]
            min_idx = _SEVERITY_ORDER.index(min_sev)
            for c in ctx.active:
                if c.domain == domain:
                    current_sev = ctx.severity_map.get(c.flag_id, c.default_severity)
                    if _SEVERITY_ORDER.index(current_sev) >= min_idx:
                        return True
            return False

        if "temporal_context" in condition:
            target = condition["temporal_context"]
            return any(c.temporal_context == target for c in ctx.active)

        if "emotion_above" in condition:
            spec = condition["emotion_above"]
            cat = spec["category"]
            threshold = spec["threshold"]
            return ctx.emotion_scores.scores.get(cat, 0.0) > threshold

        if "domain_flag_count" in condition:
            spec = condition["domain_flag_count"]
            domain = spec["domain"]
            min_count = spec["min_count"]
            include_l2 = spec.get("include_l2_candidates", False)
            count = sum(1 for c in ctx.active if c.domain == domain)
            if include_l2:
                count += sum(1 for c in ctx.l2_candidates if c.domain == domain)
            return count >= min_count

        return False

    def _apply_action(self, action: dict[str, Any], ctx: _EvalContext, result: RulesResult) -> None:
        """Apply a rule action (escalation, flags, recommended actions)."""
        new_sev = action.get("new_severity")

        if "escalate_flag" in action and new_sev:
            fid = action["escalate_flag"]
            if fid in ctx.severity_map:
                ctx.severity_map[fid] = new_sev

        if "escalate_flags" in action and new_sev:
            for fid in action["escalate_flags"]:
                if fid in ctx.severity_map:
                    ctx.severity_map[fid] = new_sev

        if action.get("set_requires_immediate_review"):
            result.requires_immediate_review = True

        if "recommended_action" in action:
            ra = action["recommended_action"]
            if ra not in result.recommended_actions:
                result.recommended_actions.append(ra)

    def _apply_deescalation(self, rule: dict[str, Any], ctx: _EvalContext) -> None:
        """Apply de-escalation to candidates matching the condition."""
        condition = rule["condition"]
        action = rule["action"]
        reduce_by = action.get("reduce_severity_by", 0)

        if "temporal_context" in condition:
            target_temporal = condition["temporal_context"]
            for c in ctx.active:
                if c.temporal_context == target_temporal:
                    current = ctx.severity_map.get(c.flag_id, c.default_severity)
                    new = self._reduce_severity(current, reduce_by)
                    ctx.severity_map[c.flag_id] = new

    @staticmethod
    def _reduce_severity(severity: str, levels: int) -> str:
        """Reduce severity by N levels. Floor is LOW (never crosses into POSITIVE)."""
        idx = _SEVERITY_ORDER.index(severity)
        if severity == "POSITIVE":
            return "POSITIVE"
        # Floor at LOW (index 1) -- de-escalation never crosses into POSITIVE
        new_idx = max(1, idx - levels)
        return _SEVERITY_ORDER[new_idx]

    def _hydrate_flags(
        self,
        active: list[PatternMatchCandidate],
        severity_map: dict[str, str],
    ) -> list[Flag]:
        """Convert PatternMatchCandidates to Flag Pydantic models.

        Per flag_id, select the highest-confidence candidate.
        """
        best: dict[str, PatternMatchCandidate] = {}
        for c in active:
            if c.flag_id not in best or c.confidence > best[c.flag_id].confidence:
                best[c.flag_id] = c

        flags: list[Flag] = []
        for c in best.values():
            final_severity = severity_map.get(c.flag_id, c.default_severity)
            flags.append(
                Flag(
                    flag_id=c.flag_id,
                    domain=Domain(c.domain),
                    name=c.name,
                    severity=Severity(final_severity),
                    confidence=c.confidence,
                    detection_layer=DetectionLayer.PATTERN_MATCH,
                    matched_context_hint=c.matched_context_hint,
                    basis_description=c.basis_description,
                    evidence_span=EvidenceSpan(
                        sentence_index=c.sentence_index,
                        char_start=c.char_start,
                        char_end=c.char_end,
                    ),
                    temporal_context=c.temporal_context,
                )
            )

        return flags


@dataclass(slots=True)
class _EvalContext:
    """Internal context passed to condition evaluators."""

    active: list[PatternMatchCandidate]
    all_candidates: list[PatternMatchCandidate]
    l2_candidates: list[PatternMatchCandidate]
    severity_map: dict[str, str]
    emotion_scores: EmotionScores
