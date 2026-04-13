"""PHI safety tests -- verify no raw text leaks into response fields."""

from __future__ import annotations

from bh_sentinel.core.models.response import AnalysisResponse, ErrorResponse
from bh_sentinel.core.pipeline import Pipeline


def pipeline() -> Pipeline:
    return Pipeline()


class TestPHISafety:
    def test_no_raw_text_in_flags(self):
        text = "Patient reports suicidal ideation for the past two days."
        result = pipeline().analyze_sync(text)
        if isinstance(result, AnalysisResponse):
            for flag in result.flags:
                # matched_context_hint should be a category name, not verbatim text
                assert len(flag.matched_context_hint) < 100
                # The hint should NOT be the full input text
                assert flag.matched_context_hint != text

    def test_error_messages_are_static(self):
        result = pipeline().analyze_sync("ab")  # too short
        assert isinstance(result, ErrorResponse)
        assert "123-45-6789" not in result.message
        assert "SSN" not in result.message

    def test_basis_description_is_generic(self):
        text = "Patient reports suicidal ideation."
        result = pipeline().analyze_sync(text)
        if isinstance(result, AnalysisResponse):
            for flag in result.flags:
                assert "Pattern match detected" in flag.basis_description
