"""Tests for Pydantic model validation logic."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bh_sentinel.core import (
    AnalysisRequest,
    DetectionLayer,
    Domain,
    EvidenceSpan,
    Flag,
    Severity,
)


class TestAnalysisRequestValidation:
    def test_request_rejects_empty_text(self):
        with pytest.raises(ValidationError):
            AnalysisRequest(text="   ")

    def test_request_rejects_too_short(self):
        with pytest.raises(ValidationError):
            AnalysisRequest(text="ab")

    def test_request_rejects_too_long(self):
        with pytest.raises(ValidationError):
            AnalysisRequest(text="a" * 50_001)

    def test_request_strips_null_bytes(self):
        req = AnalysisRequest(text="hello\x00world test")
        assert "\x00" not in req.text

    def test_request_nfc_normalizes(self):
        import unicodedata

        composed = unicodedata.normalize("NFC", "\u00e9")
        decomposed = unicodedata.normalize("NFD", "\u00e9")
        req = AnalysisRequest(text=f"caf{decomposed} test")
        assert composed in req.text

    def test_request_strips_whitespace(self):
        req = AnalysisRequest(text="  hello world  ")
        assert req.text == "hello world"


class TestFlagModel:
    def _make_flag(self, **overrides):
        defaults = dict(
            flag_id="SH-001",
            domain=Domain.SELF_HARM,
            name="test",
            severity=Severity.HIGH,
            confidence=0.9,
            detection_layer=DetectionLayer.PATTERN_MATCH,
            matched_context_hint="test",
            basis_description="test",
            evidence_span=EvidenceSpan(sentence_index=0, char_start=0, char_end=5),
        )
        defaults.update(overrides)
        return Flag(**defaults)

    def test_flag_confidence_accepts_bounds(self):
        assert self._make_flag(confidence=0.0).confidence == 0.0
        assert self._make_flag(confidence=1.0).confidence == 1.0

    def test_flag_confidence_rejects_negative(self):
        with pytest.raises(ValidationError):
            self._make_flag(confidence=-0.1)

    def test_flag_confidence_rejects_over_one(self):
        with pytest.raises(ValidationError):
            self._make_flag(confidence=1.1)

    def test_flag_corroborating_layers_defaults_empty(self):
        assert self._make_flag().corroborating_layers == []

    def test_flag_temporal_context_defaults_present(self):
        assert self._make_flag().temporal_context == "present"


class TestEvidenceSpanValidation:
    def test_evidence_span_accepts_zero_indices(self):
        es = EvidenceSpan(sentence_index=0, char_start=0, char_end=0)
        assert es.sentence_index == 0

    def test_evidence_span_rejects_negative_char_start(self):
        with pytest.raises(ValidationError):
            EvidenceSpan(sentence_index=0, char_start=-1, char_end=5)

    def test_evidence_span_rejects_negative_char_end(self):
        with pytest.raises(ValidationError):
            EvidenceSpan(sentence_index=0, char_start=0, char_end=-1)

    def test_evidence_span_rejects_negative_sentence_index(self):
        with pytest.raises(ValidationError):
            EvidenceSpan(sentence_index=-1, char_start=0, char_end=5)
