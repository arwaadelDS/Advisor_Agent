"""Advisor console -- a browser front end for the multi-agent graph.

    uv run streamlit run streamlit_app.py

One text box, four specialists behind it. The supervisor routes; whichever agent
answers writes back into ``AdvisorState``, and this renders whatever it left
there.

Showing the working is the point. An advisor is being asked to relay these
answers to a client, so the interface puts the evidence next to the prose: the
holdings table beside the portfolio answer, the page-level citation beside every
research claim, the generated SQL one click away. A chat bubble on its own is
not something a compliance function can sign off.

Each turn is stored whole rather than as text, so the transcript can be
re-rendered without asking the graph again -- and so a failed turn still shows
the research it managed to retrieve before the model call died.
"""

from __future__ import annotations

import re

import streamlit as st

st.set_page_config(page_title="SNB Capital Advisor Console",
                   page_icon="\N{BANK}", layout="wide")

# Names the graph writes onto AIMessage.name, mapped to something an advisor
# would recognise. The supervisor and search agent do not tag their replies, so
# an untagged message falls back to the last entry here.
AGENT_LABELS = {
    "sql_agent": ("Portfolio", "blue"),
    "rag_agent": ("Research", "violet"),
    "search_agent": ("Market", "orange"),
    None: ("Assistant", "gray"),
}

# Selected when the advisor is not working on anyone in particular. Retrieval
# supports this properly -- no holdings filter, and the answer says so -- so it
# is a real mode rather than an empty state.
NO_CLIENT = ""


def examples_for(client_id: str) -> list[str]:
    """Starter questions, worded for whoever is selected."""
    if not client_id:
        return [
            "Summarise the risks for SABIC from our research",
            "ما هي مخاطر سابك؟",
            "What is the outlook for Saudi petrochemicals?",
        ]
    return [
        f"What does {client_id} hold?",
        "Summarise the risks for SABIC from our research",
        "ما هي مخاطر سابك؟",
        f"Show me {client_id}'s holdings and any research risk analysis on them",
    ]


def client_label(client_id: str) -> str:
    """One line in the picker.

    Deliberately a pure function of the id and the client table: no question
    counts, no session state. Streamlit keys a selectbox's remembered choice to
    its option strings, so a label that grew "(3)" as the conversation went on
    would reset the picker mid-conversation. Counts live in the captions below
    it instead.
    """
    if not client_id:
        return "No client -- research only"
    for other, name, *_ in load_clients():
        if other == client_id:
            return f"{client_id} -- {name}"
    return client_id


@st.cache_data(show_spinner=False)
def load_clients() -> list[tuple[str, str, str, str]]:
    """Every client in the database, for the picker.

    Returns ``[]`` rather than raising if the database is missing or unreadable,
    so the console still opens on a machine that has the research index but no
    portfolio data -- the sidebar falls back to a plain text box.
    """
    try:
        from sqlalchemy import text

        from tools.db import agent_engine

        with agent_engine.connect() as connection:
            rows = connection.execute(text(
                "SELECT client_id, name, risk_profile, aum_tier "
                "FROM clients ORDER BY client_id"
            )).fetchall()
        return [tuple(row) for row in rows]
    except Exception:  # noqa: BLE001 - a missing database must not stop the page
        return []


# "C001", "c 001", "client 007". Anchored on the C or the word "client" so a
# bare number is never taken for a client id -- Tadawul tickers are four digits
# and appear in these questions constantly.
_CLIENT_ID = re.compile(
    r"\bc\s*-?\s*(\d{1,4})\b|\bclient\s+(?:no\.?\s*)?(\d{1,4})\b",
    re.IGNORECASE,
)


def client_in(question: str) -> str | None:
    """The client a question names, if it names exactly one. Otherwise ``None``.

    Lets an advisor start from nobody in particular and just ask "what does
    C004 hold" -- the console moves to that client's conversation rather than
    making them find the picker first.

    Silence is the answer whenever there is any doubt, because the cost is
    asymmetric: failing to switch leaves an unscoped search that says it is
    unscoped, while switching to the wrong client scopes the answer to someone
    else's portfolio and says nothing. Every first name and surname in this
    database is shared by at least two people, so only a full name or an
    explicit id is ever conclusive.
    """
    clients = load_clients()
    if not clients:
        return None
    known = {client_id for client_id, *_ in clients}

    for match in _CLIENT_ID.finditer(question):
        digits = match.group(1) or match.group(2)
        candidate = f"C{int(digits):03d}"
        if candidate in known:
            return candidate

    lowered = question.lower()
    full = {cid for cid, name, *_ in clients if name.lower() in lowered}
    if full:
        # Two full names in one question is a comparison, not a selection.
        return next(iter(full)) if len(full) == 1 else None

    words = set(re.findall(r"[\w'-]+", lowered))
    partial = {cid for cid, name, *_ in clients
               if words & {part.lower() for part in name.split()}}
    return next(iter(partial)) if len(partial) == 1 else None


@st.cache_resource(show_spinner="Loading agents and the embedding model...")
def load_graph():
    """Import the graph once per process.

    Importing pulls in sentence-transformers and the local embedder, which takes
    a while. Streamlit re-runs this script top to bottom on every interaction,
    so without caching every keystroke would pay that cost. The cached value
    also keeps the graph's in-process checkpointer alive between runs, which is
    what makes ``thread_id`` mean anything.
    """
    from graph.workflow import run_turn

    return run_turn


def friendly_error(exc: Exception) -> str:
    """Turn an exception into something worth showing an advisor.

    The quota case gets its own sentence because it is the one that will
    actually happen during a demo, and "RESOURCE_EXHAUSTED" tells the person
    watching nothing about what to do next.
    """
    text = str(exc)
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        return ("The daily quota for this API key is used up (the free tier "
                "allows 20 requests a day, and one question can cost several). "
                "Try again tomorrow, or switch to a key with billing enabled.")
    if "PERMISSION_DENIED" in text or "403" in text:
        return "This API key's project has been denied access to the model."
    if "NOT_FOUND" in text or "404" in text:
        return ("The configured model is not available to this API key. "
                "`gemini-2.5-flash` is closed to keys created recently -- "
                "set `LLM_MODEL` to a 3.x model in `.env`.")
    if "GOOGLE_API_KEY" in text:
        return "No API key is configured. Set `GOOGLE_API_KEY` in `.env`."
    return f"Something went wrong: {text}"


def render_citations(rag_context) -> None:
    """The research behind a research answer, numbered to match the prose.

    The numbering has to line up with the `[1]`/`[2]` markers the model wrote,
    which means rendering ``chunks`` in order and never filtering it -- the
    model was shown all of them.
    """
    if not rag_context or not rag_context.chunks:
        return

    st.caption(f"Sources ({len(rag_context.chunks)})")
    for number, chunk in enumerate(rag_context.chunks, start=1):
        label = chunk.citation or f"{chunk.title or chunk.doc_id}, page {chunk.pages}"
        with st.expander(f"[{number}] {label}"):
            if chunk.language == "ar":
                # Right-to-left, or the Arabic renders as visual nonsense.
                st.markdown(
                    f"<div dir='rtl' style='text-align:right'>{chunk.text}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(chunk.text)
            st.caption(
                f"{chunk.doc_id} · section: {chunk.section or '--'} · "
                f"score: {chunk.score} · {chunk.language}"
            )


def render_holdings(sql_result) -> None:
    """The rows the portfolio answer was composed from, and the SQL that got them."""
    if not sql_result:
        return

    if sql_result.error:
        st.warning(f"The query did not complete: {sql_result.error}")

    if sql_result.holdings:
        st.caption(f"Holdings ({sql_result.row_count})")
        st.dataframe(
            [
                {
                    "Symbol": h.symbol,
                    "Name": h.name_en,
                    "Sector": h.sector,
                    "Quantity": h.quantity,
                    "Market value": h.market_value,
                }
                for h in sql_result.holdings
            ],
            hide_index=True,
            width="stretch",
        )

    if sql_result.query_used:
        # Shown because an advisor who cannot see the query has to take the
        # number on faith, and because it is the fastest way to spot a wrong
        # answer that reads plausibly.
        with st.expander("Generated SQL"):
            st.code(sql_result.query_used, language="sql")


def render_turn(turn: dict) -> None:
    """One question and everything the graph produced answering it."""
    with st.chat_message("user"):
        st.markdown(turn["question"])

    if turn["error"]:
        with st.chat_message("assistant"):
            st.error(turn["error"])
        return

    for reply in turn["replies"]:
        label, colour = AGENT_LABELS.get(reply["agent"], AGENT_LABELS[None])
        with st.chat_message("assistant"):
            st.badge(label, color=colour)
            st.markdown(reply["text"])

    rag_context = turn["rag_context"]
    sql_result = turn["sql_result"]

    if rag_context and rag_context.note:
        # The note says *why* retrieval returned what it did -- an unscoped
        # search, or a client whose holdings nothing covers. Suppressing it
        # would leave sector research looking like advice about this client.
        st.info(rag_context.note, icon="\N{INFORMATION SOURCE}")

    if sql_result or rag_context or turn["search_result"]:
        with st.container(border=True):
            render_holdings(sql_result)
            render_citations(rag_context)
            if turn["search_result"]:
                with st.expander("Raw web search findings"):
                    st.markdown(turn["search_result"])


def replies_of(messages: list) -> list[dict]:
    """The agent replies belonging to the newest question.

    The checkpointer hands back the whole thread, but a turn should only print
    what it just produced. Everything after the last human message is exactly
    that -- one reply usually, two when the portfolio agent hands off to
    research. Read off the transcript rather than tracked in a counter, because
    a turn that raised part-way through leaves any counter wrong for good.
    """
    last_question = -1
    for index, message in enumerate(messages):
        if getattr(message, "type", None) == "human":
            last_question = index

    from tools.llm import text_of

    return [
        {"agent": getattr(message, "name", None), "text": text_of(message)}
        for message in messages[last_question + 1:]
        if getattr(message, "type", None) == "ai"
    ]


def ask(run_turn, question: str, thread_id: str, client_id: str) -> dict:
    """Run one turn and reduce the resulting state to what the page renders."""
    turn: dict = {
        "question": question, "replies": [], "error": None,
        "sql_result": None, "rag_context": None, "search_result": None,
    }
    try:
        state = run_turn(question, thread_id=thread_id, client_id=client_id)
    except Exception as exc:  # noqa: BLE001 - the page must survive any failure
        turn["error"] = friendly_error(exc)
        return turn

    turn["replies"] = replies_of(state.get("messages") or [])
    turn["sql_result"] = state.get("sql_result")
    turn["rag_context"] = state.get("rag_context")
    turn["search_result"] = state.get("search_result")
    return turn


def thread_for(client_id: str) -> str:
    """This client's conversation id, created the first time they are asked about.

    One thread per client, kept side by side, rather than one thread that
    follows whoever is currently selected. Two reasons, and they pull in
    opposite directions until you separate the threads:

    Sharing a thread across clients is unsafe -- the checkpointer keeps
    ``sql_result`` alive between turns, so the research agent would go on
    filtering by the previous client's holdings while the sidebar showed the new
    one. Making every switch start a *fresh* thread fixes that but throws the
    old conversation away: the checkpointer still holds it, under an id nobody
    kept. Remembering the id per client is what satisfies both.
    """
    from uuid import uuid4

    threads = st.session_state.threads
    if client_id not in threads:
        threads[client_id] = f"ui-{client_id or 'research'}-{uuid4().hex[:6]}"
    return threads[client_id]


def turns_for(client_id: str) -> list:
    """The rendered transcript for one client."""
    return st.session_state.transcripts.setdefault(client_id, [])


def clear_conversation() -> None:
    """Forget the current client's conversation. Other clients keep theirs.

    Dropping the thread id is what makes this a real reset: the next question
    gets a new one, so the checkpointer's copy of the old turns is not reachable
    either.
    """
    client_id = st.session_state.client_id
    st.session_state.threads.pop(client_id, None)
    st.session_state.transcripts[client_id] = []


def main() -> None:
    # Applied before the picker is built: Streamlit refuses to let a widget's
    # own session_state key be reassigned once the widget exists, so a question
    # that named a client parks the switch here for the following run.
    if "switch_to" in st.session_state:
        st.session_state.client_id = st.session_state.pop("switch_to")

    st.session_state.setdefault("client_id", NO_CLIENT)
    st.session_state.setdefault("threads", {})
    st.session_state.setdefault("transcripts", {})

    clients = load_clients()
    names = {client_id: name for client_id, name, _, _ in clients}

    with st.sidebar:
        st.subheader("Client")

        if clients:
            options = [NO_CLIENT] + [client_id for client_id, *_ in clients]
            if st.session_state.client_id not in options:
                st.session_state.client_id = NO_CLIENT
            st.selectbox("Working on", options, format_func=client_label,
                         key="client_id", label_visibility="collapsed")
        else:
            # No database. Keep the console usable for research questions.
            st.text_input("Client ID", key="client_id",
                          help="No client database found, so this is not checked.")

        client_id = st.session_state.client_id
        turns = turns_for(client_id)
        thread_id = thread_for(client_id)

        if not client_id:
            st.caption("Research is searched in full, with no holdings filter. "
                       "Name a client in your question and the console will "
                       "switch to them.")

        first_name = names.get(client_id, "").split(" ")[0]
        st.button(f"Start over with {first_name}" if first_name else "Start over",
                  on_click=clear_conversation, width="stretch", disabled=not turns)
        st.caption(
            "Each client keeps a separate conversation, and earlier questions "
            "stay in context -- so you can follow up without repeating "
            "yourself. Switching away and back brings it back."
        )

        others = sorted(
            other for other, kept in st.session_state.transcripts.items()
            if kept and other != client_id
        )
        if others:
            st.caption("Also in progress: "
                       + ", ".join(f"{client_label(o)} ({len(st.session_state.transcripts[o])})"
                                 for o in others))

        st.divider()
        st.caption(
            "**Portfolio** answers come from the client database, and show the "
            "rows and the query behind them. **Research** answers come only "
            "from SNB Capital's own published research, each claim carrying the "
            "page it came from. **Market** answers come from a live web search "
            "-- outside sources, not internal research."
        )

    st.title("SNB Capital Advisor Console")
    st.caption("Portfolios, internal research, and live market context"
               + (f" · working on **{client_label(client_id)}**" if client_id else ""))

    for turn in turns:
        render_turn(turn)

    examples = examples_for(client_id)
    if not turns:
        st.markdown("**Try one of these:**")
        for column, example in zip(st.columns(len(examples)), examples):
            if column.button(example, width="stretch"):
                st.session_state.pending = example
                st.rerun()

    question = st.chat_input(
        f"Ask about {names.get(client_id) or client_id}, our research, or the market"
        if client_id else "Ask about a client by name or id, our research, or the market"
    )
    if not question:
        question = st.session_state.pop("pending", None)
    if not question:
        return

    mentioned = client_in(question)
    if mentioned and mentioned != client_id:
        # The question named someone else. Move to their conversation and ask
        # it there, so the answer is scoped to the client it is about and joins
        # the right transcript. Both are parked for the next run because the
        # picker has already been built with the old value.
        st.session_state.switch_to = mentioned
        st.session_state.pending = question
        st.rerun()

    run_turn = load_graph()
    with st.spinner("Routing and answering..."):
        # The graph's own word for "nobody in particular" -- it goes into the
        # supervisor's context line, so an empty string would read as a bug.
        turn = ask(run_turn, question, thread_id, client_id or "unspecified")
    turns.append(turn)
    st.rerun()


if __name__ == "__main__":
    main()
