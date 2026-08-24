"""Unit tests for ingestion.normalize.

These use string literals rather than PDFs on purpose: normalize is a pure
function, so its behaviour can be pinned exactly, including the cases this
corpus does not happen to contain today.

Visible Arabic is written literally so the cases stay readable. Anything
invisible -- bidi controls, zero-width characters, tatweel -- is written as a
``\\uXXXX`` escape: a test asserting that invisible characters get removed is
not reviewable if those characters are sitting invisibly in the test file.
"""

from __future__ import annotations

import unicodedata

import pytest

from ingestion.normalize import (
    ARABIC_DIGIT_BOUNDARY,
    BIDI_CONTROLS,
    TAIL_FRAGMENT,
    TATWEEL,
    ZERO_WIDTH,
    census,
    collapse_whitespace,
    explain,
    is_clean,
    normalize,
)

# "مصرف" (bank) as pdftotext emits it -- contextual presentation forms ...
MASRAF_FORMS = "ﻣﺼﺮﻑ"
# ... and as anyone would type it.
MASRAF_BASE = "مصرف"

RLE = "\u202b"  # RIGHT-TO-LEFT EMBEDDING
LRE = "\u202a"  # LEFT-TO-RIGHT EMBEDDING
PDF_ = "\u202c"  # POP DIRECTIONAL FORMATTING


class TestPresentationForms:
    """The bug this module exists for."""

    def test_presentation_forms_become_base_letters(self):
        # The premise, in one line: same word on screen, different codepoints.
        assert MASRAF_FORMS != MASRAF_BASE
        assert normalize(MASRAF_FORMS) == MASRAF_BASE

    @pytest.mark.parametrize(
        "block", [(0xFB50, 0xFE00), (0xFE70, 0xFF00)], ids=["forms-a", "forms-b"]
    )
    def test_no_resolvable_presentation_form_survives(self, block):
        # Sweep the whole block rather than the handful this corpus contains.
        text = "".join(
            chr(cp) for cp in range(*block) if unicodedata.name(chr(cp), "")
        )
        assert census(normalize(text)).presentation_forms == 0

    def test_lam_alef_ligature_splits_into_two_letters(self):
        assert normalize("ﻻ") == "لا"
        assert normalize("ﻼ") == "لا"

    def test_allah_ligature_expands(self):
        assert normalize("ﷲ") == "الله"

    def test_multi_word_ligature_expands_and_whitespace_is_tidied_after(self):
        # U+FDFA decomposes into a phrase *containing spaces*, which is why
        # whitespace collapsing has to run after NFKC rather than before.
        out = normalize("ﷺ")
        assert " " in out
        assert "  " not in out
        assert out == out.strip()


class TestInvisibles:
    """What NFKC leaves behind."""

    def test_normalize_removes_bidi_controls(self):
        assert normalize(RLE + MASRAF_FORMS + LRE + PDF_) == MASRAF_BASE

    def test_every_invisible_character_is_stripped(self):
        # The whole set, not just the three this corpus uses -- other
        # typesetters emit others.
        for char in sorted(BIDI_CONTROLS | ZERO_WIDTH):
            assert normalize(f"a{char}b") == "ab", f"U+{ord(char):04X}"

    def test_tatweel_is_stripped(self):
        assert normalize("م" + TATWEEL + "ص") == "مص"

    def test_tatweel_produced_by_nfkc_is_also_stripped(self):
        # U+FE71 decomposes to TATWEEL + FATHATAN. Stripping tatweel before
        # NFKC would miss this one, so ordering inside normalize matters.
        assert unicodedata.normalize("NFKC", "ﹱ") == TATWEEL + "ً"
        assert normalize("ﹱ") == "ً"


class TestSymbolsAreContentNotCorruption:
    """Presentation-block codepoints that NFKC cannot fold.

    Around forty codepoints in these blocks have no compatibility decomposition:
    Quranic annotation symbols, ornate parentheses, and honorific ligatures.
    They survive NFKC because there is no base form to fold them into -- they
    are content. Only the tail fragment is genuinely meaningless.
    """

    def test_tail_fragment_is_stripped(self):
        assert unicodedata.decomposition(TAIL_FRAGMENT) == ""
        assert normalize(f"a{TAIL_FRAGMENT}b") == "ab"

    # Written as codepoints, not literals. Typing these by eye is how you end up
    # asserting against U+FDFC RIAL SIGN, which does decompose, and quietly
    # testing the opposite of what you meant.
    def test_symbols_are_preserved(self):
        for codepoint in (0xFDFD, 0xFD3E, 0xFBB2):  # bismillah, ornate paren, dot
            symbol = chr(codepoint)
            assert unicodedata.decomposition(symbol) == "", "premise: nothing to fold"
            assert symbol in normalize(f"x {symbol} y")

    def test_symbols_do_not_count_as_corruption(self):
        c = census(chr(0xFDFD))  # ARABIC LIGATURE BISMILLAH AR-RAHMAN AR-RAHEEM
        assert c.arabic_symbols == 1
        assert c.presentation_forms == 0
        assert c.is_clean

    def test_a_resolvable_form_still_counts_as_corruption(self):
        c = census("ﻣ")
        assert c.presentation_b == 1
        assert c.arabic_symbols == 0
        assert not c.is_clean


class TestWhitespace:
    def test_layout_column_padding_collapses(self):
        assert normalize("Company        SABIC") == "Company SABIC"

    def test_paragraph_breaks_survive(self):
        assert normalize("one\n\ntwo") == "one\n\ntwo"

    def test_long_blank_runs_collapse_to_one_paragraph_break(self):
        assert normalize("one\n\n\n\n\ntwo") == "one\n\ntwo"

    def test_single_newlines_survive(self):
        assert normalize("one\ntwo") == "one\ntwo"

    def test_form_feed_becomes_a_paragraph_break(self):
        assert normalize("page one\fpage two") == "page one\n\npage two"

    def test_trailing_and_leading_whitespace_removed(self):
        assert normalize("  \n  hello  \n  ") == "hello"


class TestArabicDigitSeams:
    """pdftotext drops the space where Arabic meets a numeral."""

    def test_a_numeral_before_an_arabic_word_is_separated(self):
        assert normalize("435مليون") == "435 مليون"

    def test_an_arabic_word_before_a_numeral_is_separated(self):
        assert normalize("الربع1.2") == "الربع 1.2"

    def test_a_decimal_point_is_not_split(self):
        assert normalize("1.75دولار") == "1.75 دولار"

    def test_both_seams_in_one_run(self):
        assert normalize("ارتفاع17بالمئة") == "ارتفاع 17 بالمئة"

    def test_a_real_extracted_line(self):
        assert normalize("حياد: التوصية2222 أرامكو") == "حياد: التوصية 2222 أرامكو"

    def test_an_existing_space_is_not_doubled(self):
        assert normalize("435 مليون") == "435 مليون"

    def test_it_only_inserts_so_the_original_is_recoverable(self):
        # The whole justification for doing this and not repairing punctuation:
        # deleting the inserted space gets the source text back.
        source = "بلغت 34.33مليار ريال بتراجع 7بالمئة"
        assert normalize(source).replace(" ", "") == source.replace(" ", "")

    def test_it_runs_after_the_bidi_controls_are_stripped(self):
        # The control sits exactly on the seam. Strip it too late and the digit
        # and the letter are never adjacent for the pattern to see.
        assert normalize(f"435{PDF_}مليون") == "435 مليون"

    def test_latin_words_and_numerals_are_left_alone(self):
        assert normalize("SAR4.07 billion in Q2 2025") == "SAR4.07 billion in Q2 2025"

    def test_an_isin_is_not_broken_up(self):
        assert normalize("ISIN SA0007879121") == "ISIN SA0007879121"

    def test_arabic_indic_digits_are_digits_not_letters(self):
        # Both runs live inside the 0600-06FF block. Treating them as letters
        # would push a space into the middle of a number.
        for digits in ("٢٠٢٥", "۲۰۲۵"):
            assert normalize(f"{digits}5") == f"{digits}5"

    def test_a_diacritic_beside_a_numeral_is_not_a_word_boundary(self):
        assert normalize("ً5") == "ً5"

    def test_the_seam_pattern_finds_nothing_after_normalising(self):
        text = normalize("بلغت 34.33مليار ريال بتراجع 7بالمئة سنويا")
        assert ARABIC_DIGIT_BOUNDARY.search(text) is None


class TestDocumentedNonBehaviour:
    """Decisions recorded in the module docstring, pinned so they cannot drift."""

    def test_lossy_folding_is_not_done(self):
        # Alef variants, alef maksura, teh marbuta and diacritics all survive.
        # Each would merge two distinct spellings; bge-m3 handles them unfolded.
        for char in ("أ", "إ", "آ", "ى", "ة", "مَ"):
            assert normalize(char) == char

    def test_english_text_is_untouched(self):
        text = "SABIC reported SAR 4.07 billion of impairments in Q2 2025."
        assert normalize(text) == text


class TestProperties:
    def test_idempotent(self):
        dirty = f"{RLE}{MASRAF_FORMS}{LRE} 2010 {PDF_}\n\n\nﻻ  x\f"
        once = normalize(dirty)
        assert normalize(once) == once

    def test_output_is_always_clean(self):
        dirty = RLE + MASRAF_FORMS + TATWEEL + "\u200b" + PDF_
        assert not is_clean(dirty)
        assert is_clean(normalize(dirty))

    def test_empty_and_whitespace_only_input(self):
        assert normalize("") == ""
        assert normalize("   \n\n\f  ") == ""


class TestCensus:
    def test_counts_the_raw_forms(self):
        c = census(RLE + MASRAF_FORMS + PDF_)
        assert c.presentation_b == 4
        assert c.bidi_controls == 2
        assert c.arabic_base == 0
        assert c.total == 6
        assert not c.is_clean

    def test_counts_the_normalised_form(self):
        c = census(MASRAF_BASE)
        assert c.arabic_base == 4
        assert c.presentation_forms == 0
        assert c.is_clean

    def test_tatweel_is_not_counted_as_arabic_base(self):
        # Tatweel sits inside 0600-06FF, so it must be classified before the
        # base-letter range or the census would call dirty text clean.
        c = census(TATWEEL)
        assert c.decorative == 1
        assert c.arabic_base == 0
        assert not c.is_clean


class TestExplain:
    """The per-character audit trail the --inspect CLI prints."""

    def test_it_labels_every_source_character(self):
        text = RLE + MASRAF_FORMS + PDF_
        changes = explain(text)
        assert len(changes) == len(text)
        assert changes[0].reason == "removed: bidi control"
        assert changes[1].result == "م"

    def test_concatenating_results_matches_normalize(self):
        # The per-character view the CLI shows must not be a fiction: for this
        # corpus, per-character folding agrees with whole-string NFKC.
        text = RLE + MASRAF_FORMS + LRE + "ﻻ 2010" + PDF_
        rebuilt = "".join(c.result for c in explain(text))
        assert collapse_whitespace(rebuilt) == normalize(text)
