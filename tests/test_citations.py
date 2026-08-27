"""Tests for tools.citations.

Nothing here calls a model or builds an index -- the checker reads a string and
a chunk count.

The answers below are shapes Gemini returns for this corpus: the citation before
the full stop and after it, grouped as "[1, 2]", grouped with an Arabic comma,
and refusals that cite nothing.
"""

from __future__ import annotations

from schemas import DocumentChunk, RAGSearchResult
from tools import citations as c


def result_with(n: int) -> RAGSearchResult:
    """A search result carrying ``n`` extracts. Only the count is ever read."""
    return RAGSearchResult(
        chunks=[
            DocumentChunk(text=f"extract {i}", source=f"doc_{i}", score=0.5)
            for i in range(1, n + 1)
        ]
    )


class TestFindingTheCitations:
    def test_a_plain_reference(self):
        assert c.markers("Margins are under pressure [1].") == [1]

    def test_several_in_one_sentence(self):
        assert c.markers("Fee income offset it [1][3].") == [1, 3]

    def test_a_grouped_reference_is_two_references(self):
        # The prompt asks for "[1] or [2]"; the model groups them anyway.
        assert c.markers("Both notes agree [1, 2].") == [1, 2]

    def test_an_arabic_comma_groups_them_too(self):
        assert c.markers("تشير التقارير إلى ذلك [1، 3].") == [1, 3]

    def test_spacing_inside_the_brackets_is_tolerated(self):
        assert c.markers("As noted [ 2 ].") == [2]

    def test_order_is_the_order_written(self):
        assert c.markers("First [3]. Then [1]. Again [3].") == [3, 1, 3]

    def test_brackets_that_are_not_citations_are_ignored(self):
        assert c.markers("See [see note] and [Table A].") == []

    def test_an_answer_with_no_citations_finds_none(self):
        assert c.markers("We hold no research covering this.") == []

    def test_empty_input_is_not_an_error(self):
        assert c.markers("") == []
        assert c.markers(None) == []


class TestACitationThatPointsAtNothing:
    """The failure this module exists for."""

    def test_a_number_past_the_last_extract_is_out_of_range(self):
        report = c.check_citations("Capex steps up sharply [7].", result_with(5))
        assert report.out_of_range == (7,)
        assert not report.ok

    def test_the_last_extract_is_in_range(self):
        # Off-by-one guard: [5] of 5 is fine, and must not be flagged.
        report = c.check_citations("Capex steps up sharply [5].", result_with(5))
        assert report.out_of_range == ()
        assert report.ok

    def test_zero_is_out_of_range(self):
        # Extracts are numbered from 1. [0] is not a real reference.
        report = c.check_citations("As shown [0].", result_with(5))
        assert report.out_of_range == (0,)

    def test_citing_anything_when_nothing_was_retrieved_fails(self):
        # The refusal path. Retrieval found nothing, so any citation is invented.
        report = c.check_citations("SABIC looks well positioned [1].", result_with(0))
        assert report.out_of_range == (1,)
        assert not report.ok

    def test_every_bad_number_is_reported_not_just_the_first(self):
        report = c.check_citations("One [7]. Two [9].", result_with(5))
        assert report.out_of_range == (7, 9)

    def test_a_good_answer_passes(self):
        answer = "Margins fell [1]. Fee income offset it [2]."
        assert c.check_citations(answer, result_with(5)).ok


class TestSentencesWithoutACitation:
    """Counted, never failed -- a refusal cites nothing and is still correct."""

    def test_an_uncited_sentence_is_counted(self):
        report = c.check_citations("Margins fell [1]. SABIC is a good buy.",
                                   result_with(5))
        assert report.uncited == ("SABIC is a good buy.",)

    def test_but_it_is_still_ok(self):
        report = c.check_citations("Margins fell [1]. SABIC is a good buy.",
                                   result_with(5))
        assert report.ok

    def test_a_refusal_cites_nothing_and_is_still_ok(self):
        report = c.check_citations(
            "Our research does not cover Tesla.", result_with(0))
        assert report.uncited
        assert report.ok

    def test_a_citation_after_the_full_stop_still_counts(self):
        # "... pressure. [1]" -- the fragment belongs to the sentence before it.
        report = c.check_citations("Margins are under pressure. [1]", result_with(5))
        assert report.uncited == ()
        assert report.sentences == 1

    def test_bullets_are_one_sentence_each(self):
        answer = "- Margins fell [1]\n- Capex rose [2]\n- Rating unchanged"
        report = c.check_citations(answer, result_with(5))
        assert report.sentences == 3
        assert report.uncited == ("- Rating unchanged",)

    def test_an_arabic_question_mark_ends_a_sentence(self):
        report = c.check_citations("ما المخاطر؟ ترتفع التكاليف [1].", result_with(5))
        assert report.sentences == 2

    def test_punctuation_only_fragments_are_not_sentences(self):
        report = c.check_citations("Margins fell [1].\n\n---\n\n2026.", result_with(5))
        assert report.sentences == 1
        assert report.uncited == ()

    def test_an_empty_answer_has_no_sentences(self):
        report = c.check_citations("", result_with(5))
        assert report.sentences == 0
        assert report.ok


class TestWhatItReportsBack:
    def test_it_records_how_many_extracts_there_were(self):
        assert c.check_citations("Margins fell [1].", result_with(5)).extracts == 5

    def test_cited_is_deduplicated_in_first_appearance_order(self):
        report = c.check_citations("A [3]. B [1]. C [3].", result_with(5))
        assert report.cited == (3, 1)

    def test_extracts_the_answer_never_used_are_named(self):
        report = c.check_citations("A [2].", result_with(4))
        assert report.unused == (1, 3, 4)

    def test_unused_extracts_are_not_a_failure(self):
        # Five are retrieved because five is a useful shortlist, not because the
        # answer has to mention all of them.
        assert c.check_citations("A [2].", result_with(4)).ok

    def test_a_plain_dict_works_as_the_result(self):
        # So an eval can build one without importing the schema.
        report = c.check_citations("A [1].", {"chunks": [1, 2, 3]})
        assert report.extracts == 3
        assert report.ok

    def test_a_missing_result_counts_as_no_extracts(self):
        assert c.check_citations("A [1].", None).out_of_range == (1,)

    def test_the_summary_names_the_bad_numbers(self):
        summary = c.check_citations("A [7].", result_with(5)).summary()
        assert "[7]" in summary
        assert "out of range" in summary

    def test_the_summary_of_a_clean_answer_says_what_was_cited(self):
        summary = c.check_citations("A [1]. B [2].", result_with(5)).summary()
        assert "2/5" in summary
        assert "out of range" not in summary
