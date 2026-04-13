"""Real-world text validation suite.

Tests bh-sentinel against public domain literature, realistic clinical
vignettes, and true-negative everyday text to validate detection quality
on language that wasn't used to author the patterns.

These are not pass/fail assertions — they document what the pipeline
detects and verify the output is structurally sound and clinically
plausible. Failures here indicate detection quality issues, not bugs.
"""

from __future__ import annotations

import pytest

from bh_sentinel.core.models.response import AnalysisResponse
from bh_sentinel.core.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Public domain literary excerpts with mental health themes
# ---------------------------------------------------------------------------
# Sources are out of copyright (pre-1928 US publication)

WOOLF_MRS_DALLOWAY_1925 = (
    # Septimus Warren Smith's internal monologue — PTSD, suicidal ideation
    "He would kill himself rather than let them get at him. "
    "He did not want to die. Life was good. The sun hot. "
    "Only human beings — what did they want? He could not feel anything. "
    "Every power poured its treasures on his head, and his hand lay there "
    "on the back of the sofa, as he had seen his hand lie when he was bathing, "
    "floating, on the top of the waves, while far away on shore he heard dogs "
    "barking and barking far away. Fear no more, says the heart in the body; "
    "fear no more. He did not want to die. He was not afraid."
)

GILMAN_YELLOW_WALLPAPER_1892 = (
    # Charlotte Perkins Gilman — clinical deterioration, dissociation
    "I don't sleep much at night, for it is so interesting to watch "
    "developments; but I sleep a good deal in the daytime. In the daytime "
    "it is tiresome and perplexing. I cry at nothing, and cry most of the time. "
    "It is getting to be a great effort for me to think straight. "
    "I am getting angry enough to do something desperate. "
    "I sometimes fancy that in my condition if I had less opposition and "
    "more society and stimulus — but John says the very worst thing I can "
    "do is to think about my condition."
)

TOLSTOY_ANNA_KARENINA_1878 = (
    # Levin's existential crisis — passive ideation, hopelessness
    "Without knowing what I am and why I am here, life is impossible. "
    "And that I cannot know, so I cannot live. In infinite time, in infinite "
    "matter, in infinite space, is formed a bubble organism, and that bubble "
    "lasts a while and bursts, and that bubble is me. I was keeping myself "
    "from suicide with a rope, so as not to hang myself. And I could not "
    "stop thinking about it."
)

DOSTOEVSKY_CRIME_PUNISHMENT_1866 = (
    # Raskolnikov's deterioration — isolation, agitation, paranoia
    "He had become so completely absorbed in himself, and isolated from "
    "his fellows that he dreaded meeting, not only his landlady, but anyone "
    "at all. He was crushed by poverty, but the anxieties of his position "
    "had of late ceased to weigh upon him. He had given up attending to "
    "matters of practical importance. He was not afraid of any landlady. "
    "But to be stopped on the stairs, to be forced to listen to her idle, "
    "persistent chatter, all that whining about what he owed, threats, "
    "complaints, and to have to think up excuses, to prevaricate — no, "
    "rather he would creep down the stairs like a cat and slip out unseen."
)

# ---------------------------------------------------------------------------
# Realistic synthetic clinical vignettes
# (Not from real patients — authored to test detection patterns)
# ---------------------------------------------------------------------------

VIGNETTE_CRISIS_INTAKE = (
    "32 yo female presents to PHP intake. Reports she has not been sleeping "
    "for three days and stopped taking her Seroquel two weeks ago because she "
    "couldn't afford the copay. Endorses active suicidal ideation with a vague "
    "plan but denies intent to act. History of one prior attempt by overdose in "
    "2021. Reports drinking heavily since losing her job last month, approximately "
    "a pint of vodka daily. Hearing voices intermittently that tell her she is "
    "worthless. No identified social supports. Lives alone with her two cats."
)

VIGNETTE_ROUTINE_SESSION = (
    "45 yo male seen for individual therapy. Mood is euthymic today. Reports "
    "sleeping seven hours per night. Taking medications as prescribed and finds "
    "the current regimen helpful. Attended all three group sessions this week and "
    "reports finding the DBT skills group particularly useful. His wife has been "
    "very supportive. He used breathing exercises when he felt anxious yesterday "
    "and it helped. Looking forward to his son's baseball game this weekend. "
    "Denies any suicidal or homicidal ideation. No psychotic symptoms."
)

VIGNETTE_MIXED_PRESENTATION = (
    "28 yo nonbinary patient seen for crisis follow-up. They report the suicidal "
    "thoughts have decreased since starting Lexapro but still feel hopeless about "
    "their situation at home. They haven't left the house in five days. Their "
    "partner has been checking in daily which they find helpful. They expressed "
    "commitment to continuing therapy. Reports using the crisis line twice this "
    "week when feeling overwhelmed. Denies any plan or intent."
)

VIGNETTE_JOURNAL_ENTRY = (
    "I can't do this anymore. Everything feels pointless and I'm so tired of "
    "pretending I'm okay. I stopped taking my meds because what's the point. "
    "Mom keeps calling but I don't pick up. Haven't eaten in two days. "
    "The only thing keeping me going is my dog. I have therapy tomorrow and "
    "I'm going to be honest about how bad things have gotten."
)

# ---------------------------------------------------------------------------
# True negatives — everyday text that should produce zero flags
# ---------------------------------------------------------------------------

TRUE_NEGATIVE_WEATHER = (
    "The forecast calls for sunny skies this weekend with temperatures reaching "
    "the mid-seventies. Perfect weather for the farmers market on Saturday. "
    "Don't forget your sunscreen and a reusable bag for produce."
)

TRUE_NEGATIVE_RECIPE = (
    "Preheat oven to 375 degrees. Mix flour, sugar, and butter until crumbly. "
    "Press into a 9-inch pan and bake for 12 minutes. Let cool completely "
    "before adding the cream cheese filling. Refrigerate for at least 2 hours."
)

TRUE_NEGATIVE_SPORTS = (
    "The team rallied in the fourth quarter with a stunning 14-point comeback. "
    "The quarterback threw for 340 yards and three touchdowns. The defense "
    "held strong in the final minutes, sealing the victory 31-28."
)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------
class TestRealWorldValidation:
    """Run real-world text through the pipeline and validate output quality."""

    @pytest.fixture(scope="class")
    def pipeline(self) -> Pipeline:
        return Pipeline()

    # -- Public domain literature --

    def test_woolf_mrs_dalloway(self, pipeline):
        """Septimus Warren Smith — contains SI language in Victorian literary style.

        Layer 1 (pattern matching) does not detect this because the language
        uses conditional mood and indirect construction ("He would kill himself
        rather than let them get at him") rather than clinical disclosure patterns.
        This is the expected boundary of deterministic pattern matching —
        Layer 2 (transformer) is designed to catch these semantic signals.
        """
        result = pipeline.analyze_sync(WOOLF_MRS_DALLOWAY_1925)
        assert isinstance(result, AnalysisResponse)
        _print_result("Woolf - Mrs Dalloway (Septimus)", result)

    def test_gilman_yellow_wallpaper(self, pipeline):
        """Charlotte Perkins Gilman — describes clinical deterioration through
        metaphor and period-specific language. Pattern-based detection has
        limited coverage here; confirms the value of the transformer layer.
        """
        result = pipeline.analyze_sync(GILMAN_YELLOW_WALLPAPER_1892)
        assert isinstance(result, AnalysisResponse)
        _print_result("Gilman - The Yellow Wallpaper", result)

    def test_tolstoy_anna_karenina(self, pipeline):
        """Levin's existential crisis — explicit suicide references but in
        philosophical/literary construction. Documents Layer 1 boundary:
        'keeping myself from suicide with a rope' is not a clinical disclosure
        pattern. Layer 2 semantic understanding would catch this.
        """
        result = pipeline.analyze_sync(TOLSTOY_ANNA_KARENINA_1878)
        assert isinstance(result, AnalysisResponse)
        _print_result("Tolstoy - Anna Karenina (Levin)", result)

    def test_dostoevsky_crime_punishment(self, pipeline):
        """Raskolnikov — isolation, possible clinical deterioration."""
        result = pipeline.analyze_sync(DOSTOEVSKY_CRIME_PUNISHMENT_1866)
        assert isinstance(result, AnalysisResponse)
        # This is subtle — isolation and avoidance, may or may not trigger
        _print_result("Dostoevsky - Crime and Punishment", result)

    # -- Clinical vignettes --

    def test_crisis_intake(self, pipeline):
        """High-acuity intake — multiple domains, immediate review expected."""
        result = pipeline.analyze_sync(VIGNETTE_CRISIS_INTAKE)
        assert isinstance(result, AnalysisResponse)
        flag_ids = {f.flag_id for f in result.flags}
        domains = {f.domain for f in result.flags}
        assert "SH-002" in flag_ids or "SH-005" in flag_ids, "Expected SI detection"
        assert result.summary.requires_immediate_review is True, (
            "Crisis intake should trigger immediate review"
        )
        assert len(domains) >= 2, f"Expected multi-domain flags, got {domains}"
        _print_result("Clinical Vignette - Crisis Intake", result)

    def test_routine_session(self, pipeline):
        """Routine stable session — protective factors, minimal risk flags."""
        result = pipeline.analyze_sync(VIGNETTE_ROUTINE_SESSION)
        assert isinstance(result, AnalysisResponse)
        # Should have mostly protective factors, minimal risk
        assert result.summary.requires_immediate_review is False, (
            "Routine session should not trigger immediate review"
        )
        _print_result("Clinical Vignette - Routine Session", result)

    def test_mixed_presentation(self, pipeline):
        """Nonbinary patient with mixed risk and protective — nuanced detection."""
        result = pipeline.analyze_sync(VIGNETTE_MIXED_PRESENTATION)
        assert isinstance(result, AnalysisResponse)
        # Should detect hopelessness + isolation but also protective factors
        assert len(result.flags) > 0 or len(result.protective_factors) > 0, (
            "Expected at least some detection in mixed presentation"
        )
        _print_result("Clinical Vignette - Mixed Presentation", result)

    def test_journal_entry(self, pipeline):
        """First-person journal — passive ideation, med non-adherence, isolation."""
        result = pipeline.analyze_sync(VIGNETTE_JOURNAL_ENTRY)
        assert isinstance(result, AnalysisResponse)
        flag_ids = {f.flag_id for f in result.flags}
        assert len(result.flags) >= 2, f"Expected multiple flags in journal entry, got {flag_ids}"
        _print_result("Clinical Vignette - Journal Entry", result)

    # -- True negatives --

    def test_weather_report(self, pipeline):
        """Weather text should produce zero flags."""
        result = pipeline.analyze_sync(TRUE_NEGATIVE_WEATHER)
        assert isinstance(result, AnalysisResponse)
        assert len(result.flags) == 0, (
            f"Weather report should have no flags, got {[f.flag_id for f in result.flags]}"
        )

    def test_recipe(self, pipeline):
        """Recipe text should produce zero flags."""
        result = pipeline.analyze_sync(TRUE_NEGATIVE_RECIPE)
        assert isinstance(result, AnalysisResponse)
        assert len(result.flags) == 0, (
            f"Recipe should have no flags, got {[f.flag_id for f in result.flags]}"
        )

    def test_sports_recap(self, pipeline):
        """Sports text should produce zero flags."""
        result = pipeline.analyze_sync(TRUE_NEGATIVE_SPORTS)
        assert isinstance(result, AnalysisResponse)
        assert len(result.flags) == 0, (
            f"Sports recap should have no flags, got {[f.flag_id for f in result.flags]}"
        )


def _print_result(label: str, result: AnalysisResponse) -> None:
    """Print detection results for documentation."""
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    if result.flags:
        print(f"  Risk flags ({len(result.flags)}):")
        for f in result.flags:
            print(
                f"    [{f.severity}] {f.flag_id}: {f.name} "
                f"(conf={f.confidence}, temporal={f.temporal_context})"
            )
    else:
        print("  No risk flags detected.")
    if result.protective_factors:
        print(f"  Protective factors ({len(result.protective_factors)}):")
        for f in result.protective_factors:
            print(f"    [POSITIVE] {f.flag_id}: {f.name} (conf={f.confidence})")
    print(
        f"  Summary: max_severity={result.summary.max_severity}, "
        f"immediate_review={result.summary.requires_immediate_review}"
    )
    if result.summary.recommended_action:
        print(f"  Recommended: {result.summary.recommended_action}")
    if result.emotions and result.emotions.primary:
        top = sorted(
            result.emotions.category_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        top_str = ", ".join(f"{k}={v:.2f}" for k, v in top if v > 0)
        if top_str:
            print(f"  Top emotions: {top_str}")
    print()
