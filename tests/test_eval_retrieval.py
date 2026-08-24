"""Tests for ingestion.eval_retrieval.

Deliberately small. The retrieval this module measures is already tested
elsewhere; what is worth testing here is the arithmetic, because a scoring bug
does not announce itself -- it just reports a number that is wrong in a
believable direction. Running the eval for real is the CLI's job.
"""

from __future__ import annotations

import pytest

from ingestion import eval_retrieval as ev


def question(*expect: str, language: str = "en") -> ev.Question:
    return ev.Question(question="q", expect=expect, language=language, why="")


def result(*ranked: str, expect: str = "target", language: str = "en") -> ev.Result:
    return ev.Result(question=question(expect, language=language), ranked=ranked)


class TestRank:
    def test_rank_is_one_based_not_zero_based(self):
        # An off-by-one here would silently inflate every recall@1.
        assert result("target", "other").rank == 1
        assert result("other", "target").rank == 2

    def test_an_absent_document_has_no_rank(self):
        assert result("other").rank is None

    def test_either_acceptable_document_counts(self):
        # Some questions genuinely have two sources; see data/eval/retrieval.csv.
        both = ev.Result(question=question("a", "b"), ranked=("b", "c"))
        assert both.rank == 1


class TestRecall:
    def test_a_hit_below_the_cutoff_does_not_count(self):
        assert result("x", "y", "target").hit_at(2) is False

    def test_a_hit_at_the_cutoff_counts(self):
        assert result("x", "y", "target").hit_at(3) is True

    def test_recall_is_the_fraction_of_questions_hit(self):
        results = [result("target"), result("other")]
        assert ev.recall_at(results, 5) == 0.5

    def test_an_empty_run_scores_zero_rather_than_dividing_by_zero(self):
        assert ev.recall_at([], 5) == 0.0


class TestMrr:
    def test_first_place_scores_one(self):
        assert ev.mrr([result("target")]) == 1.0

    def test_second_place_scores_a_half(self):
        assert ev.mrr([result("other", "target")]) == 0.5

    def test_a_miss_contributes_nothing(self):
        assert ev.mrr([result("target"), result("nope")]) == 0.5


class TestTheQuestionSet:
    def test_it_loads(self):
        assert len(ev.load_questions()) > 10

    def test_every_expected_document_exists(self):
        # A typo in a doc_id would show up as a permanent, unexplained miss.
        from ingestion.catalog import load_catalog

        known = {entry.doc_id for entry in load_catalog().documents}
        unknown = {
            doc_id
            for q in ev.load_questions()
            for doc_id in q.expect
            if doc_id not in known
        }
        assert unknown == set()

    def test_both_languages_are_covered(self):
        languages = {q.language for q in ev.load_questions()}
        assert {"ar", "en"} <= languages

    def test_a_missing_file_says_so(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ev.load_questions(tmp_path / "nothing.csv")


class TestReport:
    def test_it_breaks_down_by_language(self):
        # The column the whole file exists for.
        text = ev.report([result("target", language="ar"), result("x", language="en")])
        assert "ar" in text and "en" in text

    def test_it_names_what_was_missed(self):
        text = ev.report([result("wrong-doc", expect="wanted-doc")])
        assert "wanted-doc" in text and "wrong-doc" in text

    def test_a_clean_run_lists_no_misses(self):
        assert "Missed" not in ev.report([result("target")])
