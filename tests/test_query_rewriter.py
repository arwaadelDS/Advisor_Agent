"""Unit tests for tools/query_rewriter.py.

Mocks _grounding()/_raw_grounding() and _rewriter_llm() rather than the DB/
model beneath them: the rewriter's contract is "grounding text + question ->
verified RewrittenQuery", and that contract shouldn't need a real DB
connection or a real API call to test.
"""

import pytest

from schemas import RewrittenQuery
from tools import query_rewriter as qr


class FakeLLM:
    """Stands in for get_llm().with_structured_output(RewrittenQuery).

    Only .invoke is exercised here -- query_rewriter calls _rewriter_llm()
    once per rewrite_query() call and never touches with_structured_output
    itself (that's built once, inside _rewriter_llm, not per-call), so the
    fake only needs to match the Retrying interface's invoke() surface.
    """

    def __init__(self, response):
        self._response = response

    def invoke(self, messages):
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


FAKE_RAW_GROUNDING = {
    "instruments": [
        ("2010", "SABIC", "سابك"),
        ("2222", "Saudi Aramco", "أرامكو السعودية"),
    ],
    "clients": [
        ("C001", "Faisal Al-Otaibi"),
        ("C002", "Noura Al-Sudairi"),
    ],
    "risk_profiles": ["Conservative", "Balanced", "Aggressive"],
    "aum_tiers": ["HNW", "Ultra-HNW"],
    "sectors": ["Petrochemicals", "Energy", "Banking"],
    "asset_classes": ["Equity"],
}


@pytest.fixture(autouse=True)
def clear_caches():
    """_raw_grounding and _rewriter_llm are lru_cache'd; each test needs a
    clean slate, or one test's monkeypatch leaks into the next via the cache."""
    qr._raw_grounding.cache_clear()
    qr._rewriter_llm.cache_clear()
    yield
    qr._raw_grounding.cache_clear()
    qr._rewriter_llm.cache_clear()


def _patch_llm(monkeypatch, response):
    monkeypatch.setattr(qr, "_rewriter_llm", lambda: FakeLLM(response))


def _patch_raw_grounding(monkeypatch, raw=None):
    monkeypatch.setattr(qr, "_raw_grounding", lambda: raw or FAKE_RAW_GROUNDING)


class TestRewriteQuery:
    def test_empty_question_raises(self, monkeypatch):
        _patch_raw_grounding(monkeypatch)
        with pytest.raises(qr.QueryRewriterError, match="cannot be empty"):
            qr.rewrite_query("   ")

    def test_successful_rewrite_sets_original_deterministically(self, monkeypatch):
        _patch_raw_grounding(monkeypatch)
        fake_response = RewrittenQuery(
            original="whatever the model echoed",  # should be overwritten
            rewritten="What is the market value of ticker 2010 for this client?",
            corrections=["سابك -> 2010"],
        )
        _patch_llm(monkeypatch, fake_response)

        result = qr.rewrite_query("ما هي قيمة سابك لهذا العميل؟")

        assert result.original == "ما هي قيمة سابك لهذا العميل؟"
        assert result.rewritten == "What is the market value of ticker 2010 for this client?"
        # 2010 is a real ticker in the fixture -- passes verification unchanged.
        assert result.corrections == ["سابك -> 2010"]
        assert result.ambiguous == []
        assert result.needs_clarification is False

    def test_ambiguous_term_is_carried_through(self, monkeypatch):
        _patch_raw_grounding(monkeypatch)
        fake_response = RewrittenQuery(
            original="",
            rewritten="What is the value of شركة غير معروفة for this client?",
            ambiguous=["شركة غير معروفة"],
            needs_clarification=True,
        )
        _patch_llm(monkeypatch, fake_response)

        result = qr.rewrite_query("ما هي قيمة شركة غير معروفة لهذا العميل؟")

        assert result.ambiguous == ["شركة غير معروفة"]
        assert result.needs_clarification is True

    def test_non_rewritten_query_return_raises(self, monkeypatch):
        _patch_raw_grounding(monkeypatch)
        _patch_llm(monkeypatch, "just a plain string, not a RewrittenQuery")

        with pytest.raises(qr.QueryRewriterError, match="did not return a RewrittenQuery"):
            qr.rewrite_query("some question")

    def test_llm_exception_wrapped_as_rewriter_error(self, monkeypatch):
        _patch_raw_grounding(monkeypatch)
        _patch_llm(monkeypatch, RuntimeError("API rate limited"))

        with pytest.raises(qr.QueryRewriterError, match="rewrite failed"):
            qr.rewrite_query("some question")

    def test_grounding_failure_raises_before_llm_call(self, monkeypatch):
        def broken_raw_grounding():
            raise qr.QueryRewriterError(
                "could not load grounding values from the DB:\nboom"
            )
        monkeypatch.setattr(qr, "_raw_grounding", broken_raw_grounding)

        with pytest.raises(qr.QueryRewriterError, match="could not load grounding values"):
            qr.rewrite_query("some question")


class TestVerifyCorrections:
    """The deterministic floor under the model's own matching -- see the
    _verify_corrections docstring for why this exists at all."""

    def test_exact_match_passes_through_unchanged(self, monkeypatch):
        _patch_raw_grounding(monkeypatch)
        fake_response = RewrittenQuery(
            original="", rewritten="...", corrections=["أرامكو -> Saudi Aramco"],
        )
        _patch_llm(monkeypatch, fake_response)

        result = qr.rewrite_query("سؤال عن أرامكو")

        assert result.corrections == ["أرامكو -> Saudi Aramco"]
        assert result.needs_clarification is False

    def test_near_miss_is_fuzzy_corrected(self, monkeypatch):
        """'SABIK' isn't a real value, but it's close enough to 'SABIC'
        (the fixture's real ticker-adjacent value) to be corrected rather
        than rejected."""
        _patch_raw_grounding(monkeypatch)
        fake_response = RewrittenQuery(
            original="", rewritten="...", corrections=["سابك -> SABIK"],
        )
        _patch_llm(monkeypatch, fake_response)

        result = qr.rewrite_query("سؤال عن سابك")

        assert result.corrections == ["سابك -> SABIC"]
        assert result.ambiguous == []
        assert result.needs_clarification is False

    def test_no_confident_match_is_demoted_to_ambiguous(self, monkeypatch):
        """Nothing in the fixture is close to this -- must be rejected, not
        silently passed through as if it were a verified match."""
        _patch_raw_grounding(monkeypatch)
        fake_response = RewrittenQuery(
            original="", rewritten="...", corrections=["شركة وهمية -> Fictional Corp"],
        )
        _patch_llm(monkeypatch, fake_response)

        result = qr.rewrite_query("سؤال عن شركة وهمية")

        assert result.corrections == []
        assert result.ambiguous == ["شركة وهمية -> Fictional Corp"]
        assert result.needs_clarification is True

    def test_malformed_correction_string_kept_as_is(self, monkeypatch):
        """A correction with no '->' can't be checked -- shouldn't crash the
        verification pass, and shouldn't be silently dropped either."""
        _patch_raw_grounding(monkeypatch)
        fake_response = RewrittenQuery(
            original="", rewritten="...", corrections=["not the expected shape"],
        )
        _patch_llm(monkeypatch, fake_response)

        result = qr.rewrite_query("سؤال")

        assert result.corrections == ["not the expected shape"]
        assert result.needs_clarification is False

    def test_pre_existing_ambiguous_entries_are_preserved(self, monkeypatch):
        """Verification appends to ambiguous, it must not clobber whatever
        the model already flagged on its own."""
        _patch_raw_grounding(monkeypatch)
        fake_response = RewrittenQuery(
            original="",
            rewritten="...",
            corrections=["شركة وهمية -> Fictional Corp"],
            ambiguous=["some other unresolved term"],
            needs_clarification=True,
        )
        _patch_llm(monkeypatch, fake_response)

        result = qr.rewrite_query("سؤال")

        assert "some other unresolved term" in result.ambiguous
        assert "شركة وهمية -> Fictional Corp" in result.ambiguous
        assert len(result.ambiguous) == 2

    def test_known_values_pools_across_all_columns(self, monkeypatch):
        """A client name, not just an instrument, should verify cleanly --
        _known_values must not be scoped to instruments only."""
        _patch_raw_grounding(monkeypatch)
        fake_response = RewrittenQuery(
            original="", rewritten="...", corrections=["فيصل العتيبي -> C001"],
        )
        _patch_llm(monkeypatch, fake_response)

        result = qr.rewrite_query("سؤال عن فيصل العتيبي")

        assert result.corrections == ["فيصل العتيبي -> C001"]
        assert result.needs_clarification is False


class TestRawGrounding:
    def test_db_failure_wrapped_as_rewriter_error(self, monkeypatch):
        """_raw_grounding itself, not mocked this time -- a broken DB
        connection should surface as QueryRewriterError, not a raw
        SQLAlchemy/sqlite3 exception leaking out of tools/query_rewriter.py."""
        class BrokenEngine:
            def connect(self):
                raise RuntimeError("no such table: instruments")

        monkeypatch.setattr(qr, "agent_engine", BrokenEngine())
        with pytest.raises(qr.QueryRewriterError, match="could not load grounding values"):
            qr._raw_grounding()

class TestClientSelectedContext:
    def test_deictic_client_reference_not_flagged_ambiguous_when_selected(self, monkeypatch):
        """Regression for the eval failure: 'this client' was being treated
        as an unmatched name instead of a context-resolved reference."""
        _patch_raw_grounding(monkeypatch)
        fake_response = RewrittenQuery(
            original="", rewritten="this client's holdings",
            corrections=[], ambiguous=[], needs_clarification=False,
        )
        _patch_llm(monkeypatch, fake_response)

        result = qr.rewrite_query("this client's holdings", client_selected=True)

        assert result.needs_clarification is False
        assert result.ambiguous == []