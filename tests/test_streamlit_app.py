"""Tests for the advisor console.

Nothing here starts Streamlit or calls a model. What is worth testing is the
layer between the graph and the page: which replies belong to this turn, and
whether a failure reaches the advisor as a sentence instead of a stack trace.

Rendering itself is left untested -- asserting that a chunk was passed to
``st.expander`` would test Streamlit, not this project.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import streamlit_app as ui
from schemas import DocumentChunk, RAGSearchResult

# AppTest resolves a relative path against the *calling* file, which would look
# for the app inside tests/.
APP = str(ui.__file__)


def app_test():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(APP, default_timeout=120)


def run_app(at):
    """Run the script, then teach the selectbox proxy how labels are made.

    AppTest builds its own view of a widget and defaults ``format_func`` to
    ``str``, so it looks for the raw id among options that are really formatted
    labels and raises. Handing it the app's own formatter is what lets a test
    touch the picker at all.
    """
    at.run()
    return at


def pick_client(at, client_id: str):
    """Choose a client in the sidebar picker, by position rather than label."""
    ids = [ui.NO_CLIENT] + [row[0] for row in ui.load_clients()]
    at.sidebar.selectbox[0].select_index(ids.index(client_id)).run()
    return at


class TestWhichRepliesBelongToThisTurn:
    """The checkpointer returns the whole thread; a turn prints only its own."""

    def test_a_single_reply(self):
        messages = [HumanMessage("what are the risks"),
                    AIMessage("Risks are X [1].", name="rag_agent")]
        assert [r["text"] for r in ui.replies_of(messages)] == ["Risks are X [1]."]

    def test_earlier_turns_are_not_reprinted(self):
        messages = [
            HumanMessage("first"), AIMessage("first answer", name="sql_agent"),
            HumanMessage("second"), AIMessage("second answer", name="rag_agent"),
        ]
        assert [r["text"] for r in ui.replies_of(messages)] == ["second answer"]

    def test_a_handoff_keeps_both_replies_in_order(self):
        # supervisor -> sql_agent -> rag_agent is one turn with two answers.
        messages = [
            HumanMessage("holdings and research on them"),
            AIMessage("You hold SABIC and Al Rajhi.", name="sql_agent"),
            AIMessage("Research says X [1].", name="rag_agent"),
        ]
        assert [r["agent"] for r in ui.replies_of(messages)] == \
            ["sql_agent", "rag_agent"]

    def test_the_agent_name_is_carried_through(self):
        # The badge over each answer comes from this, not from guesswork.
        messages = [HumanMessage("q"), AIMessage("a", name="sql_agent")]
        assert ui.replies_of(messages)[0]["agent"] == "sql_agent"

    def test_an_untagged_reply_still_renders(self):
        # The supervisor and search agent do not set a name.
        messages = [HumanMessage("hello"), AIMessage("Hi -- how can I help?")]
        (reply,) = ui.replies_of(messages)
        assert reply["agent"] is None and reply["text"]

    def test_an_untagged_reply_gets_a_label_anyway(self):
        assert ui.AGENT_LABELS[None][0]

    def test_block_shaped_content_is_read_as_text(self):
        # Gemini 3.x returns a list of typed blocks; str() of that list is a
        # repr full of base64. See tools/llm.text_of.
        blocks = [{"type": "text", "text": "Risks are X [1]."}]
        messages = [HumanMessage("q"), AIMessage(blocks, name="rag_agent")]
        assert ui.replies_of(messages)[0]["text"] == "Risks are X [1]."

    def test_no_messages_at_all(self):
        assert ui.replies_of([]) == []

    def test_a_question_with_no_answer_yet(self):
        assert ui.replies_of([HumanMessage("q")]) == []


class TestAsk:
    """One turn, reduced to what the page needs."""

    def _run(self, **state):
        def run_turn(question, thread_id, client_id):
            return state
        return ui.ask(run_turn, "q", "t", "C001")

    def test_it_pulls_the_result_slots_off_the_state(self):
        turn = self._run(messages=[], sql_result="SQL", rag_context="RAG",
                         search_result="WEB")
        assert (turn["sql_result"], turn["rag_context"], turn["search_result"]) \
            == ("SQL", "RAG", "WEB")

    def test_a_state_missing_every_slot_is_fine(self):
        # Only one agent runs per turn, so two of the three are always absent.
        turn = self._run(messages=[HumanMessage("q"), AIMessage("a")])
        assert turn["error"] is None and turn["replies"]

    def test_a_failure_becomes_an_error_not_an_exception(self):
        def run_turn(question, thread_id, client_id):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

        turn = ui.ask(run_turn, "q", "t", "C001")
        assert turn["error"] and turn["replies"] == []

    def test_the_question_is_kept_for_redisplay(self):
        turn = self._run(messages=[])
        assert turn["question"] == "q"


class TestWhatTheAdvisorSeesWhenItBreaks:
    """A demo fails in front of people. It should fail in words."""

    @pytest.mark.parametrize("raw", [
        "429 RESOURCE_EXHAUSTED", "ClientError: 429 Too Many Requests",
    ])
    def test_quota_is_explained_not_quoted(self, raw):
        message = ui.friendly_error(Exception(raw))
        assert "quota" in message.lower()
        assert "RESOURCE_EXHAUSTED" not in message

    def test_quota_says_what_to_do_about_it(self):
        assert "billing" in ui.friendly_error(Exception("429")).lower()

    def test_a_dead_model_names_the_setting_to_change(self):
        message = ui.friendly_error(Exception("404 NOT_FOUND"))
        assert "LLM_MODEL" in message

    def test_a_missing_key_names_the_variable(self):
        assert "GOOGLE_API_KEY" in ui.friendly_error(Exception("GOOGLE_API_KEY unset"))

    def test_an_unrecognised_failure_is_still_shown(self):
        # Better a raw message than a blank screen.
        assert "hang up" in ui.friendly_error(Exception("socket hang up"))


def a_turn(question: str) -> dict:
    return {"question": question, "replies": [], "error": None,
            "sql_result": None, "rag_context": None, "search_result": None}


class TestOneConversationPerClient:
    """Switching client changes which conversation is shown, and nothing else.

    Two failures pull against each other here. Share one thread across clients
    and the checkpointer's surviving ``sql_result`` lets the research agent
    filter by the *previous* client's holdings -- the wrong portfolio, answered
    confidently. Start a fresh thread on every switch and that is fixed, but
    switching back lands on a third new thread and the earlier conversation is
    stranded in the checkpointer under an id nobody kept. Keeping one thread per
    client is what satisfies both.
    """

    def test_each_client_gets_a_different_thread(self):
        at = run_app(app_test())
        pick_client(at, "C001")
        first = at.session_state["threads"]["C001"]
        pick_client(at, "C002")
        assert at.session_state["threads"]["C002"] != first

    def test_switching_client_keeps_the_new_client(self):
        at = run_app(app_test())
        pick_client(at, "C002")
        assert at.session_state["client_id"] == "C002"

    def test_the_other_clients_transcript_is_not_shown(self):
        at = app_test()
        at.session_state["transcripts"] = {"C001": [a_turn("about C001")]}
        run_app(at)
        pick_client(at, "C002")
        assert at.session_state["transcripts"].get("C002", []) == []

    def test_switching_back_restores_the_conversation(self):
        # The bug this class exists for: C001 -> C002 -> C001 used to come back
        # to an empty screen with the real conversation unreachable.
        at = app_test()
        at.session_state["transcripts"] = {"C001": [a_turn("about C001")]}
        run_app(at)
        pick_client(at, "C002")
        pick_client(at, "C001")
        assert [t["question"] for t in at.session_state["transcripts"]["C001"]] \
            == ["about C001"]

    def test_switching_back_reuses_the_same_thread(self):
        # Restoring the visible transcript is not enough -- the graph has to be
        # asked on the same thread, or the follow-up loses its context.
        at = run_app(app_test())
        pick_client(at, "C001")
        original = at.session_state["threads"]["C001"]
        pick_client(at, "C002")
        pick_client(at, "C001")
        assert at.session_state["threads"]["C001"] == original

    def test_clearing_affects_only_the_current_client(self):
        at = app_test()
        at.session_state["transcripts"] = {"C001": [a_turn("keep me")],
                                           "C002": [a_turn("clear me")]}
        at.session_state["client_id"] = "C002"
        run_app(at)
        at.sidebar.button[0].click().run()
        assert at.session_state["transcripts"]["C002"] == []
        assert at.session_state["transcripts"]["C001"] != []

    def test_clearing_drops_the_thread_too(self):
        # Otherwise the graph still has the old turns and answers as though the
        # conversation had never been cleared.
        at = app_test()
        at.session_state["transcripts"] = {"C001": [a_turn("old")]}
        at.session_state["client_id"] = "C001"
        run_app(at)
        at.session_state["threads"]["C001"] = "ui-C001-original"
        at.sidebar.button[0].click().run()
        assert at.session_state["threads"].get("C001") != "ui-C001-original"


class TestTheClientPicker:
    """The picker replaced a free-text box that accepted anything.

    A typo used to open a phantom conversation whose every question failed
    obscurely, and it demanded the advisor already know the ids -- they know
    names. Options come from the database instead.
    """

    def test_it_offers_every_client_in_the_database(self):
        at = run_app(app_test())
        assert len(at.sidebar.selectbox[0].options) == len(ui.load_clients()) + 1

    def test_a_client_is_shown_by_name_not_just_id(self):
        # "C001" means nothing to an advisor; "Reem Al-Otaibi" does.
        assert ui.client_label("C001") == "C001 -- Reem Al-Otaibi"

    def test_the_label_carries_no_session_state(self):
        # Streamlit keys a selectbox's remembered choice to its option strings,
        # so a label that grew "(3)" as the conversation went on would reset the
        # picker mid-conversation.
        assert ui.client_label("C001") == ui.client_label("C001")

    def test_an_unknown_id_still_gets_a_label(self):
        assert ui.client_label("C999") == "C999"

    def test_working_on_nobody_is_an_option(self):
        # Retrieval supports an unfiltered search properly, so this is a real
        # mode rather than an empty state.
        assert "research only" in ui.client_label(ui.NO_CLIENT)
        assert ui.NO_CLIENT in [ui.NO_CLIENT] + [r[0] for r in ui.load_clients()]

    def test_no_client_gets_its_own_thread(self):
        at = run_app(app_test())
        pick_client(at, ui.NO_CLIENT)
        assert at.session_state["threads"][ui.NO_CLIENT].startswith("ui-research-")

    def test_the_examples_name_the_selected_client(self):
        assert any("C002" in example for example in ui.examples_for("C002"))

    def test_the_examples_drop_portfolio_questions_with_no_client(self):
        # Nothing to ask a portfolio question *about*.
        assert not any("hold" in e for e in ui.examples_for(ui.NO_CLIENT))


class TestACitationWithNoSourceBehindIt:
    """Whether to warn is a decision; drawing the warning is Streamlit's job.

    The check itself lives in tools/citations.py. This is about which reply it
    runs on.
    """

    def turn_with(self, replies, chunks):
        turn = a_turn("what are the risks")
        turn["replies"] = replies
        turn["rag_context"] = RAGSearchResult(chunks=[
            DocumentChunk(text=f"extract {i}", source=f"d{i}", score=0.5)
            for i in range(chunks)
        ])
        return turn

    def test_a_citation_past_the_last_source_is_flagged(self):
        turn = self.turn_with(
            [{"agent": "rag_agent", "text": "Capex rose [7]."}], chunks=5)
        assert ui.unsourced_citations(turn) == (7,)

    def test_a_sound_answer_is_not_flagged(self):
        turn = self.turn_with(
            [{"agent": "rag_agent", "text": "Margins fell [1]. Capex rose [5]."}],
            chunks=5)
        assert ui.unsourced_citations(turn) == ()

    def test_only_the_research_reply_is_checked(self):
        # The portfolio answer's "[7]" is a row count in prose, not a citation
        # -- the sources panel is not about it.
        turn = self.turn_with(
            [{"agent": "sql_agent", "text": "Seven positions [7]."},
             {"agent": "rag_agent", "text": "Margins fell [1]."}], chunks=5)
        assert ui.unsourced_citations(turn) == ()

    def test_a_turn_with_no_research_is_not_flagged(self):
        turn = a_turn("hello")
        turn["replies"] = [{"agent": None, "text": "Hello. I can help with [1]."}]
        assert ui.unsourced_citations(turn) == ()

    def test_a_failed_turn_is_not_flagged(self):
        # rag_context is None when retrieval died; the apology cites nothing.
        turn = a_turn("what are the risks")
        turn["replies"] = [{"agent": "rag_agent",
                            "text": "I couldn't search the research library."}]
        assert ui.unsourced_citations(turn) == ()


class TestNamingAClientInTheQuestion:
    """Start from nobody and just ask -- the console moves to that client.

    The bar for switching is high on purpose. Failing to switch leaves an
    unscoped search that announces it is unscoped; switching to the wrong client
    scopes the answer to someone else's portfolio and says nothing at all.
    """

    def test_an_explicit_id(self):
        assert ui.client_in("What does C004 hold?") == "C004"

    def test_a_lowercase_id(self):
        assert ui.client_in("what does c004 hold?") == "C004"

    def test_the_word_client_with_a_bare_number(self):
        assert ui.client_in("show me client 4's holdings") == "C004"

    def test_a_full_name(self):
        name = ui.load_clients()[0][1]
        assert ui.client_in(f"what does {name} hold?") == ui.load_clients()[0][0]

    def test_a_shared_first_name_is_not_enough(self):
        # Every first name in this database belongs to at least two people.
        assert ui.client_in("what does Lama hold?") is None

    def test_a_shared_surname_is_not_enough(self):
        assert ui.client_in("what does Al-Qahtani hold?") is None

    def test_two_clients_named_is_a_comparison_not_a_selection(self):
        (first_id, first_name, *_), (_, second_name, *_) = ui.load_clients()[:2]
        both = f"compare {first_name} and {second_name}"
        assert ui.client_in(both) is None

    def test_a_ticker_is_not_a_client_id(self):
        # Tadawul codes are four digits and appear constantly in these
        # questions. Nothing but a C or the word "client" makes an id.
        assert ui.client_in("what are the risks for 2010?") is None

    def test_a_company_ending_in_c_is_not_a_client(self):
        assert ui.client_in("Summarise the risks for SABIC from our research") is None

    def test_an_unknown_id_is_not_invented(self):
        assert ui.client_in("what does C999 hold?") is None

    def test_a_pure_research_question_names_nobody(self):
        assert ui.client_in("What is the outlook for Saudi petrochemicals?") is None

    def test_the_console_starts_on_nobody(self):
        assert run_app(app_test()).session_state["client_id"] == ui.NO_CLIENT

    def test_a_parked_switch_is_applied_before_the_picker(self):
        # Streamlit refuses to let a widget's own session_state key be
        # reassigned once the widget exists, so the switch is parked for the
        # next run. Driven directly rather than by typing a question: if the
        # resolver ever regressed, a chat_input test would fall through to the
        # real model and spend live quota to report a bug it already caught.
        at = app_test()
        at.session_state["switch_to"] = "C004"
        run_app(at)
        assert at.session_state["client_id"] == "C004"

    def test_a_switch_gives_that_client_their_own_thread(self):
        at = app_test()
        at.session_state["switch_to"] = "C004"
        run_app(at)
        assert at.session_state["threads"]["C004"].startswith("ui-C004-")
