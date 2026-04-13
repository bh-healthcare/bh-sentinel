"""Tests for RulesEngine -- exhaustive rule-by-rule and condition-type coverage."""

from __future__ import annotations

from bh_sentinel.core._types import EmotionScores, PatternMatchCandidate


def make_candidate(**overrides) -> PatternMatchCandidate:
    """Build a PatternMatchCandidate with sensible defaults."""
    defaults = dict(
        flag_id="SH-001",
        domain="self_harm",
        name="Passive death wish",
        default_severity="HIGH",
        confidence=0.92,
        sentence_index=0,
        char_start=0,
        char_end=10,
        pattern_text="test",
        basis_description="test",
        matched_context_hint="test",
    )
    defaults.update(overrides)
    return PatternMatchCandidate(**defaults)


def empty_emotions() -> EmotionScores:
    """EmotionScores with all categories at 0.0."""
    return EmotionScores(
        scores={
            "hopelessness": 0.0,
            "agitation": 0.0,
            "anxiety": 0.0,
            "anger": 0.0,
            "sadness": 0.0,
            "guilt": 0.0,
            "shame": 0.0,
            "mania": 0.0,
            "dissociation": 0.0,
            "positive_valence": 0.0,
            "negative_valence": 0.0,
        }
    )


# ---------------------------------------------------------------------------
# Escalation rules (ESC-001 through ESC-010)
# ---------------------------------------------------------------------------
class TestEscalationRules:
    def test_esc_001_passive_si_plus_hopelessness(self, rules_engine):
        candidates = [
            make_candidate(flag_id="SH-001", confidence=0.92),
            make_candidate(
                flag_id="CD-001",
                domain="clinical_deterioration",
                default_severity="HIGH",
                confidence=0.85,
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        sh001 = [f for f in result.flags if f.flag_id == "SH-001"]
        assert len(sh001) == 1
        assert sh001[0].severity == "CRITICAL"
        assert any("C-SSRS" in a for a in result.recommended_actions)

    def test_esc_002_nssi_plus_substance(self, rules_engine):
        candidates = [
            make_candidate(flag_id="SH-007", default_severity="HIGH", confidence=0.91),
            make_candidate(
                flag_id="SU-001",
                domain="substance_use",
                default_severity="MEDIUM",
                confidence=0.85,
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        sh007 = [f for f in result.flags if f.flag_id == "SH-007"]
        assert len(sh007) == 1
        assert sh007[0].severity == "CRITICAL"

    def test_esc_003_active_si_plus_auditory_hallucinations(self, rules_engine):
        candidates = [
            make_candidate(flag_id="SH-002", default_severity="CRITICAL", confidence=0.95),
            make_candidate(
                flag_id="CD-005a",
                domain="clinical_deterioration",
                default_severity="HIGH",
                confidence=0.85,
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        assert result.requires_immediate_review is True
        cd005a = [f for f in result.flags if f.flag_id == "CD-005a"]
        assert len(cd005a) == 1
        assert cd005a[0].severity == "CRITICAL"

    def test_esc_004_hi_plus_substance(self, rules_engine):
        candidates = [
            make_candidate(
                flag_id="HO-001",
                domain="harm_to_others",
                default_severity="CRITICAL",
                confidence=0.9,
            ),
            make_candidate(
                flag_id="SU-001",
                domain="substance_use",
                default_severity="MEDIUM",
                confidence=0.85,
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        ho001 = [f for f in result.flags if f.flag_id == "HO-001"]
        assert len(ho001) == 1
        assert ho001[0].severity == "CRITICAL"
        assert result.requires_immediate_review is True

    def test_esc_005_mania_plus_risky_substance(self, rules_engine):
        candidates = [
            make_candidate(
                flag_id="CD-008",
                domain="clinical_deterioration",
                default_severity="HIGH",
                confidence=0.85,
            ),
            make_candidate(
                flag_id="SU-005",
                domain="substance_use",
                default_severity="HIGH",
                confidence=0.85,
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        cd008 = [f for f in result.flags if f.flag_id == "CD-008"]
        assert len(cd008) == 1
        assert cd008[0].severity == "CRITICAL"
        assert result.requires_immediate_review is True

    def test_esc_006_prior_attempt_plus_current_si(self, rules_engine):
        candidates = [
            make_candidate(flag_id="SH-008", default_severity="HIGH", confidence=0.9),
            make_candidate(flag_id="SH-001", confidence=0.92),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        sh001 = [f for f in result.flags if f.flag_id == "SH-001"]
        assert len(sh001) == 1
        assert sh001[0].severity == "CRITICAL"

    def test_esc_007_hopelessness_plus_isolation(self, rules_engine):
        candidates = [
            make_candidate(
                flag_id="CD-001",
                domain="clinical_deterioration",
                default_severity="HIGH",
                confidence=0.85,
            ),
            make_candidate(
                flag_id="CD-002",
                domain="clinical_deterioration",
                default_severity="MEDIUM",
                confidence=0.85,
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        cd001 = [f for f in result.flags if f.flag_id == "CD-001"]
        cd002 = [f for f in result.flags if f.flag_id == "CD-002"]
        assert cd001[0].severity == "CRITICAL"
        assert cd002[0].severity == "CRITICAL"
        assert result.requires_immediate_review is True

    def test_esc_008_active_si_plus_destabilizer(self, rules_engine):
        """ESC-008 uses any_of nested inside all_of -- recursive evaluation."""
        candidates = [
            make_candidate(flag_id="SH-002", default_severity="CRITICAL", confidence=0.95),
            make_candidate(
                flag_id="CD-002",
                domain="clinical_deterioration",
                default_severity="MEDIUM",
                confidence=0.85,
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        assert result.requires_immediate_review is True
        sh002 = [f for f in result.flags if f.flag_id == "SH-002"]
        assert sh002[0].severity == "CRITICAL"

    def test_esc_008_with_substance_destabilizer(self, rules_engine):
        """ESC-008 also fires when the destabilizer is substance_use domain."""
        candidates = [
            make_candidate(flag_id="SH-003", default_severity="CRITICAL", confidence=0.96),
            make_candidate(
                flag_id="SU-001",
                domain="substance_use",
                default_severity="MEDIUM",
                confidence=0.85,
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        assert result.requires_immediate_review is True

    def test_esc_009_active_si_plus_agitation(self, rules_engine):
        candidates = [
            make_candidate(flag_id="SH-002", default_severity="CRITICAL", confidence=0.95),
        ]
        emotions = EmotionScores(scores={**empty_emotions().scores, "agitation": 0.5})
        result = rules_engine.evaluate(candidates, emotions)
        assert result.requires_immediate_review is True

    def test_esc_010_hi_plus_anger(self, rules_engine):
        candidates = [
            make_candidate(
                flag_id="HO-001",
                domain="harm_to_others",
                default_severity="CRITICAL",
                confidence=0.9,
            ),
        ]
        emotions = EmotionScores(scores={**empty_emotions().scores, "anger": 0.5})
        result = rules_engine.evaluate(candidates, emotions)
        assert result.requires_immediate_review is True


# ---------------------------------------------------------------------------
# De-escalation rules
# ---------------------------------------------------------------------------
class TestDeEscalation:
    def test_de_001_historical_reduces_severity(self, rules_engine):
        candidates = [
            make_candidate(
                flag_id="SH-008",
                default_severity="HIGH",
                confidence=0.9,
                temporal_context="past",
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        sh008 = [f for f in result.flags if f.flag_id == "SH-008"]
        assert len(sh008) == 1
        assert sh008[0].severity == "MEDIUM"

    def test_de_001_critical_reduces_to_high(self, rules_engine):
        candidates = [
            make_candidate(
                flag_id="SH-002",
                default_severity="CRITICAL",
                confidence=0.95,
                temporal_context="past",
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        sh002 = [f for f in result.flags if f.flag_id == "SH-002"]
        assert sh002[0].severity == "HIGH"

    def test_de_001_low_stays_low(self, rules_engine):
        candidates = [
            make_candidate(
                flag_id="SH-001",
                default_severity="LOW",
                confidence=0.92,
                temporal_context="past",
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        sh001 = [f for f in result.flags if f.flag_id == "SH-001"]
        assert sh001[0].severity == "LOW"

    def test_de_001_positive_stays_positive(self, rules_engine):
        candidates = [
            make_candidate(
                flag_id="PF-001",
                domain="protective_factors",
                default_severity="POSITIVE",
                confidence=0.85,
                temporal_context="past",
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        pf001 = [f for f in result.flags if f.flag_id == "PF-001"]
        assert pf001[0].severity == "POSITIVE"


# ---------------------------------------------------------------------------
# Compound rules (COMP-001 through COMP-008)
# ---------------------------------------------------------------------------
class TestCompoundRules:
    def test_comp_001_substance_plus_self_harm(self, rules_engine):
        candidates = [
            make_candidate(flag_id="SH-002", default_severity="CRITICAL", confidence=0.95),
            make_candidate(
                flag_id="SU-001",
                domain="substance_use",
                default_severity="MEDIUM",
                confidence=0.85,
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        assert result.requires_immediate_review is True

    def test_comp_002_anxiety_plus_psychosis(self, rules_engine):
        candidates = [
            make_candidate(
                flag_id="CD-007",
                domain="clinical_deterioration",
                default_severity="MEDIUM",
                confidence=0.85,
            ),
            make_candidate(
                flag_id="CD-005a",
                domain="clinical_deterioration",
                default_severity="HIGH",
                confidence=0.85,
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        assert result.requires_immediate_review is True
        cd005a = [f for f in result.flags if f.flag_id == "CD-005a"]
        assert cd005a[0].severity == "CRITICAL"

    def test_comp_003_signal_density_self_harm(self, rules_engine):
        """domain_flag_count >= 3 in self_harm triggers immediate review."""
        candidates = [
            make_candidate(flag_id="SH-001", confidence=0.92),
            make_candidate(flag_id="SH-002", default_severity="CRITICAL", confidence=0.95),
            make_candidate(flag_id="SH-007", default_severity="HIGH", confidence=0.91),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        assert result.requires_immediate_review is True

    def test_comp_004_signal_density_harm_to_others(self, rules_engine):
        """domain_flag_count >= 3 in harm_to_others triggers immediate review."""
        candidates = [
            make_candidate(
                flag_id="HO-001",
                domain="harm_to_others",
                default_severity="CRITICAL",
                confidence=0.9,
            ),
            make_candidate(
                flag_id="HO-002",
                domain="harm_to_others",
                default_severity="CRITICAL",
                confidence=0.9,
            ),
            make_candidate(
                flag_id="HO-004",
                domain="harm_to_others",
                default_severity="HIGH",
                confidence=0.85,
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        assert result.requires_immediate_review is True

    def test_comp_005_med_misuse_plus_self_harm(self, rules_engine):
        candidates = [
            make_candidate(
                flag_id="MED-003",
                domain="medication",
                default_severity="HIGH",
                confidence=0.85,
            ),
            make_candidate(flag_id="SH-001", confidence=0.92),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        assert result.requires_immediate_review is True

    def test_comp_006_hi_plus_paranoia(self, rules_engine):
        candidates = [
            make_candidate(
                flag_id="HO-001",
                domain="harm_to_others",
                default_severity="CRITICAL",
                confidence=0.9,
            ),
            make_candidate(
                flag_id="CD-005c",
                domain="clinical_deterioration",
                default_severity="HIGH",
                confidence=0.85,
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        assert result.requires_immediate_review is True

    def test_comp_007_nssi_plus_passive_si(self, rules_engine):
        candidates = [
            make_candidate(flag_id="SH-007", default_severity="HIGH", confidence=0.91),
            make_candidate(flag_id="SH-001", confidence=0.92),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        assert result.requires_immediate_review is True

    def test_comp_008_multi_domain_destabilization(self, rules_engine):
        """COMP-008: any_of with nested all_of -- deepest nesting in rules.json."""
        candidates = [
            make_candidate(flag_id="SH-001", confidence=0.92),
            make_candidate(
                flag_id="SU-001",
                domain="substance_use",
                default_severity="MEDIUM",
                confidence=0.85,
            ),
            make_candidate(
                flag_id="CD-001",
                domain="clinical_deterioration",
                default_severity="HIGH",
                confidence=0.85,
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        assert result.requires_immediate_review is True
        assert any("Multi-domain" in a for a in result.recommended_actions)


# ---------------------------------------------------------------------------
# Action rules
# ---------------------------------------------------------------------------
class TestActionRules:
    def test_act_001_critical_self_harm_prompts_cssrs(self, rules_engine):
        """ACT-001 uses domain_severity which checks post-escalation severity."""
        candidates = [
            make_candidate(flag_id="SH-001", confidence=0.92),
            make_candidate(
                flag_id="CD-001",
                domain="clinical_deterioration",
                default_severity="HIGH",
                confidence=0.85,
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        # ESC-001 escalates SH-001 to CRITICAL, then ACT-001 fires
        assert any("C-SSRS" in a for a in result.recommended_actions)

    def test_act_001_does_not_fire_below_critical(self, rules_engine):
        """ACT-001 should not fire if no self_harm flag reaches CRITICAL."""
        candidates = [
            make_candidate(flag_id="SH-001", confidence=0.92),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        cssrs_actions = [a for a in result.recommended_actions if "C-SSRS" in a]
        assert len(cssrs_actions) == 0


# ---------------------------------------------------------------------------
# Condition evaluator tests (dedicated per condition type)
# ---------------------------------------------------------------------------
class TestConditionEvaluators:
    def test_condition_flag_present(self, rules_engine):
        candidates = [make_candidate(flag_id="SH-001", confidence=0.92)]
        result = rules_engine.evaluate(candidates, empty_emotions())
        assert len(result.flags) == 1

    def test_condition_flag_present_min_confidence(self, rules_engine):
        """ESC-001 requires min_confidence 0.7 -- 0.5 should not trigger."""
        candidates = [
            make_candidate(flag_id="SH-001", confidence=0.5),
            make_candidate(
                flag_id="CD-001",
                domain="clinical_deterioration",
                default_severity="HIGH",
                confidence=0.5,
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        sh001 = [f for f in result.flags if f.flag_id == "SH-001"]
        assert sh001[0].severity == "HIGH"  # Not escalated

    def test_condition_any_flag_present(self, rules_engine):
        """ESC-003 uses any_flag_present for SH-002..005."""
        candidates = [
            make_candidate(flag_id="SH-004", default_severity="CRITICAL", confidence=0.96),
            make_candidate(
                flag_id="CD-005a",
                domain="clinical_deterioration",
                default_severity="HIGH",
                confidence=0.85,
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        assert result.requires_immediate_review is True

    def test_condition_domain_present(self, rules_engine):
        """ESC-004 uses domain_present for harm_to_others and substance_use."""
        candidates = [
            make_candidate(
                flag_id="HO-004",
                domain="harm_to_others",
                default_severity="HIGH",
                confidence=0.85,
            ),
            make_candidate(
                flag_id="SU-003",
                domain="substance_use",
                default_severity="HIGH",
                confidence=0.85,
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        assert result.requires_immediate_review is True

    def test_condition_domain_severity_post_escalation(self, rules_engine):
        """ACT-001 checks domain_severity after escalation has run."""
        # SH-001 starts HIGH, ESC-001 escalates to CRITICAL, then ACT-001 sees CRITICAL
        candidates = [
            make_candidate(flag_id="SH-001", confidence=0.92),
            make_candidate(
                flag_id="CD-001",
                domain="clinical_deterioration",
                default_severity="HIGH",
                confidence=0.85,
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        assert any("C-SSRS" in a for a in result.recommended_actions)

    def test_condition_temporal_context(self, rules_engine):
        """DE-001 checks temporal_context == 'past'."""
        candidates = [
            make_candidate(
                flag_id="SH-008",
                default_severity="HIGH",
                confidence=0.9,
                temporal_context="past",
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        sh008 = [f for f in result.flags if f.flag_id == "SH-008"]
        assert sh008[0].severity == "MEDIUM"

    def test_condition_emotion_above(self, rules_engine):
        """ESC-009 checks agitation > 0.3."""
        candidates = [
            make_candidate(flag_id="SH-002", default_severity="CRITICAL", confidence=0.95),
        ]
        # Below threshold
        low_agit = EmotionScores(scores={**empty_emotions().scores, "agitation": 0.2})
        result_low = rules_engine.evaluate(candidates, low_agit)
        # ESC-009 should NOT fire (agitation 0.2 <= 0.3)
        # But COMP-001 etc. might. Check that ESC-009 specifically didn't add its action.
        esc009_actions = [a for a in result_low.recommended_actions if "agitation" in a.lower()]
        assert len(esc009_actions) == 0

        # Above threshold
        high_agit = EmotionScores(scores={**empty_emotions().scores, "agitation": 0.5})
        result_high = rules_engine.evaluate(candidates, high_agit)
        assert result_high.requires_immediate_review is True

    def test_condition_domain_flag_count(self, rules_engine):
        """COMP-003 checks domain_flag_count >= 3 in self_harm."""
        # Only 2 flags -- should NOT trigger
        two = [
            make_candidate(flag_id="SH-001", confidence=0.92),
            make_candidate(flag_id="SH-002", default_severity="CRITICAL", confidence=0.95),
        ]
        result_two = rules_engine.evaluate(two, empty_emotions())
        multi_sh_actions = [a for a in result_two.recommended_actions if "Multiple self-harm" in a]
        assert len(multi_sh_actions) == 0

    def test_condition_all_of(self, rules_engine):
        """ESC-001 uses all_of: both SH-001 AND CD-001 must be present."""
        # Only SH-001 -- should not escalate
        only_sh = [make_candidate(flag_id="SH-001", confidence=0.92)]
        result = rules_engine.evaluate(only_sh, empty_emotions())
        sh001 = [f for f in result.flags if f.flag_id == "SH-001"]
        assert sh001[0].severity == "HIGH"  # Not escalated

    def test_condition_any_of(self, rules_engine):
        """ESC-008 inner any_of: any destabilizer suffices."""
        # MED-003 is one of the destabilizers in ESC-008's any_of
        candidates = [
            make_candidate(flag_id="SH-002", default_severity="CRITICAL", confidence=0.95),
            make_candidate(
                flag_id="MED-003",
                domain="medication",
                default_severity="HIGH",
                confidence=0.85,
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        assert result.requires_immediate_review is True

    def test_condition_nested_any_of_in_all_of(self, rules_engine):
        """ESC-008 pattern: all_of[any_flag_present, any_of[...]]."""
        # SH-005 + SU-004 should trigger ESC-008
        candidates = [
            make_candidate(flag_id="SH-005", default_severity="CRITICAL", confidence=0.97),
            make_candidate(
                flag_id="SU-004",
                domain="substance_use",
                default_severity="HIGH",
                confidence=0.85,
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        assert result.requires_immediate_review is True


# ---------------------------------------------------------------------------
# Rule ordering
# ---------------------------------------------------------------------------
class TestRuleOrdering:
    def test_escalation_before_deescalation(self, rules_engine):
        """Escalation raises severity, then de-escalation lowers historical flags."""
        candidates = [
            make_candidate(
                flag_id="SH-001",
                confidence=0.92,
                temporal_context="past",
            ),
            make_candidate(
                flag_id="CD-001",
                domain="clinical_deterioration",
                default_severity="HIGH",
                confidence=0.85,
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        sh001 = [f for f in result.flags if f.flag_id == "SH-001"]
        assert len(sh001) == 1
        # Escalated CRITICAL by ESC-001, then de-escalated to HIGH by DE-001
        assert sh001[0].severity == "HIGH"

    def test_compound_runs_after_escalation(self, rules_engine):
        """Compound rules see post-escalation state."""
        # ESC-002 escalates SH-007 to CRITICAL, then COMP-001 sees substance+self_harm
        candidates = [
            make_candidate(flag_id="SH-007", default_severity="HIGH", confidence=0.91),
            make_candidate(
                flag_id="SU-001",
                domain="substance_use",
                default_severity="MEDIUM",
                confidence=0.85,
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        assert result.requires_immediate_review is True

    def test_action_runs_after_compound(self, rules_engine):
        """Action rules see final state after escalation+compound."""
        # ESC-001 escalates SH-001 to CRITICAL -> ACT-001 fires
        candidates = [
            make_candidate(flag_id="SH-001", confidence=0.92),
            make_candidate(
                flag_id="CD-001",
                domain="clinical_deterioration",
                default_severity="HIGH",
                confidence=0.85,
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        assert any("C-SSRS" in a for a in result.recommended_actions)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_no_flags_no_rules_fire(self, rules_engine):
        result = rules_engine.evaluate([], empty_emotions())
        assert len(result.flags) == 0
        assert result.requires_immediate_review is False
        assert len(result.recommended_actions) == 0

    def test_negated_candidates_filtered(self, rules_engine):
        candidates = [
            make_candidate(flag_id="SH-002", default_severity="CRITICAL", negated=True),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        assert len(result.flags) == 0

    def test_confidence_below_threshold_skips_rule(self, rules_engine):
        candidates = [
            make_candidate(flag_id="SH-001", confidence=0.5),
            make_candidate(
                flag_id="CD-001",
                domain="clinical_deterioration",
                default_severity="HIGH",
                confidence=0.5,
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        sh001 = [f for f in result.flags if f.flag_id == "SH-001"]
        assert len(sh001) == 1
        assert sh001[0].severity == "HIGH"

    def test_recommended_actions_accumulate(self, rules_engine):
        """Multiple rules firing should accumulate recommended_actions."""
        candidates = [
            make_candidate(flag_id="SH-001", confidence=0.92),
            make_candidate(
                flag_id="CD-001",
                domain="clinical_deterioration",
                default_severity="HIGH",
                confidence=0.85,
            ),
        ]
        result = rules_engine.evaluate(candidates, empty_emotions())
        # ESC-001 adds one action, ACT-001 adds another
        assert len(result.recommended_actions) >= 2

    def test_l2_candidates_none_treated_as_empty_list(self, rules_engine):
        candidates = [make_candidate(flag_id="SH-001", confidence=0.92)]
        # Passing l2_candidates=None should not raise
        result = rules_engine.evaluate(candidates, empty_emotions(), l2_candidates=None)
        assert len(result.flags) == 1
