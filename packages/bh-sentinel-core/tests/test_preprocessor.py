"""Tests for TextPreprocessor -- written before implementation (TDD)."""

from __future__ import annotations

from bh_sentinel.core.preprocessor import TextPreprocessor


def pp() -> TextPreprocessor:
    return TextPreprocessor()


class TestSentenceSplitting:
    def test_single_sentence(self):
        result = pp().process("Patient reports feeling fine.")
        assert len(result.sentences) == 1
        assert result.sentences[0].text == "Patient reports feeling fine."

    def test_two_sentences(self):
        result = pp().process("Denies SI. Reports hopelessness.")
        assert len(result.sentences) == 2

    def test_multiple_with_different_punctuation(self):
        result = pp().process("Is he safe? Yes! He said so.")
        assert len(result.sentences) == 3

    def test_abbreviation_dr_not_split(self):
        result = pp().process("Dr. Smith reports the patient is stable.")
        assert len(result.sentences) == 1

    def test_abbreviation_pt_not_split(self):
        result = pp().process("Pt. reports SI for two days.")
        assert len(result.sentences) == 1

    def test_abbreviation_hx_not_split(self):
        result = pp().process("Hx. of depression noted.")
        assert len(result.sentences) == 1

    def test_newline_splits(self):
        result = pp().process("First sentence\nSecond sentence")
        assert len(result.sentences) == 2

    def test_empty_lines_skipped(self):
        result = pp().process("First sentence\n\n\nSecond sentence")
        assert len(result.sentences) == 2

    def test_no_terminal_punctuation(self):
        result = pp().process("Patient reports SI")
        assert len(result.sentences) == 1
        assert result.sentences[0].text == "Patient reports SI"

    def test_semicolon_splits(self):
        result = pp().process("Denies SI; reports hopelessness")
        assert len(result.sentences) == 2


class TestOffsetTracking:
    def test_offsets_map_back_to_original(self):
        text = "First sentence. Second sentence."
        result = pp().process(text)
        for sent in result.sentences:
            assert text[sent.char_start : sent.char_end] == sent.text

    def test_offsets_with_leading_whitespace(self):
        text = "  Hello world. Goodbye."
        result = pp().process(text)
        for sent in result.sentences:
            assert text[sent.char_start : sent.char_end] == sent.text

    def test_offsets_with_newlines(self):
        text = "First line\nSecond line"
        result = pp().process(text)
        for sent in result.sentences:
            assert text[sent.char_start : sent.char_end] == sent.text

    def test_sentence_indices_sequential(self):
        result = pp().process("One. Two. Three.")
        for i, sent in enumerate(result.sentences):
            assert sent.index == i


class TestClinicalText:
    def test_clinical_shorthand_plus_si(self):
        result = pp().process("+SI noted during intake.")
        assert len(result.sentences) == 1

    def test_clinical_shorthand_minus_hi(self):
        result = pp().process("-HI per patient report.")
        assert len(result.sentences) == 1

    def test_semicolons_as_separators(self):
        result = pp().process("Reports SI; denies HI; no plan")
        assert len(result.sentences) == 3


class TestEdgeCases:
    def test_empty_string_returns_empty(self):
        result = pp().process("")
        assert len(result.sentences) == 0

    def test_whitespace_only_returns_empty(self):
        result = pp().process("   \n\t  ")
        assert len(result.sentences) == 0

    def test_unicode_offsets(self):
        text = "Caf\u00e9 visit. Normal follow-up."
        result = pp().process(text)
        for sent in result.sentences:
            assert text[sent.char_start : sent.char_end] == sent.text
