# Advisor Agent

A multi-agent assistant for SNB Capital relationship advisors. One question box;
a supervisor decides whether the answer comes from the client's portfolio, from
internal research, or from live market data, and the console shows the evidence
next to the prose.

The corpus and the client data are synthetic. The documents in `data/documents`
are written to look like Saudi equity research and are marked
`is_synthetic=true` in `data/manifest.csv`; the clients, holdings and
instruments in `data/mock` are generated. Nothing here is real client data or a
real published note.

## What runs where

```
                         supervisor
                             |
        +--------------------+--------------------+
        |                    |                    |
    sql_agent            rag_agent           search_agent
   client data         internal research     live market news
        |                    |
        +-------> rag_agent (same turn)
```

The supervisor picks one worker per turn. The one exception is `sql_agent`,
which can hand off to `rag_agent` within the same turn when the question asks
for research on the holdings it just retrieved — so a turn can be
`supervisor -> sql_agent -> rag_agent`. Routing lives in `graph/workflow.py`.

| Agent | Answers from | Backed by |
| --- | --- | --- |
| `sql_agent` | the SQLite portfolio DB | `tools/query_rewriter.py`, `tools/sql_tools.py` |
| `rag_agent` | the PDF research corpus | `tools/rag_tools.py`, `ingestion/vector_store.py` |
| `search_agent` | web search | `tools/web_search_tools.py` |

Two things are deliberate rather than incidental:

- **Retrieval always runs before the research agent answers.** It is not a tool
  the model may skip. An unsourced sentence about a real security is
  indistinguishable downstream from a sourced one.
- **Holdings come from state, not from the model.** A paraphrase of the
  question cannot widen the document filter to names the client does not hold.

Every research answer is checked against the extracts it was shown
(`tools/citations.py`): a `[7]` after five extracts is flagged in the UI. Whether
extract `[2]` actually supports the sentence citing it is not checked — that
needs a second model.

## Setup

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and a `pdftotext`
binary from **poppler** (see the warning below).

```bash
uv sync
cp .env.example .env        # then fill in GOOGLE_API_KEY
```

Embeddings run locally through sentence-transformers, so ingestion and
retrieval need no API key at all and the corpus never leaves the machine. A key
is only needed for the answering step:

| Variable | Needed for |
| --- | --- |
| `GOOGLE_API_KEY` | every LLM call — supervisor, SQL, research |
| `SEARCH_API_KEY` | the search agent only (OpenRouter) |

Then build the two data stores. Both are derived from files in the repo, and
neither is committed:

```bash
uv run python ingestion/seed_mock_db.py        # CSVs -> data/mock/advisor_mock.db
uv run python -m ingestion.doctor              # check the toolchain first
uv run python -m ingestion.build_vector_index  # PDFs  -> data/index
```

Run `doctor` before the index build, especially on a fresh clone. It answers the
two questions that decide whether the index would be quietly wrong: whether this
machine extracts the Arabic, and whether every document still joins to an
instrument.

### The pdftotext trap on Windows

Git for Windows ships an Xpdf `pdftotext` that is usually first on `PATH` and
defaults to Latin-1. It drops every Arabic codepoint, exits 0, and prints
nothing to stderr — so half the corpus indexes as whitespace with no error
anywhere. Poppler's build is the one that works. `ingestion.doctor` reports
which one it found; if it picks the wrong one, set `POPPLER_PATH` in `.env` to
the poppler executable's absolute path.

## Running it

```bash
uv run streamlit run streamlit_app.py     # the console, on :8501
uv run python -m graph.workflow           # the same graph, in the terminal
```

Or with Docker, which installs poppler for you:

```bash
docker compose up --build
```

The compose file mounts `data/index` and `data/mock` from the host, so build
them once and the container reuses them.

## Working on it

```bash
uv run pytest -m "not integration"        # the default suite; no API calls
uv run pytest -m integration              # hits the real DB and LLM
```

Integration tests spend API quota. `-m "not integration"` is not the default in
`pyproject.toml` yet, so a bare `pytest` will call the model.

Useful entry points while developing:

```bash
# Ask the corpus a question the way the agent does
uv run python -m tools.rag_tools "ما هي مخاطر سابك؟" --ticker 2010
uv run python -m tools.rag_tools "What are SABIC's risks?" --context

# What is actually in the index right now
uv run python -m ingestion.build_vector_index --status

# Score retrieval against the labelled question set in data/eval/retrieval.csv
uv run python -m ingestion.eval_retrieval
uv run python -m ingestion.eval_retrieval --rerank    # with the cross-encoder

# Run the SQL question bank end to end and write a CSV report
uv run python ingestion/eval_sql_agent.py
```

The reranker is off by default (`RERANKER_MODEL=` empty). It is a separate model
download and costs a few seconds a question on CPU, so it has to earn its place
against the eval rather than be assumed to help.

## Layout

```
agents/       one module per worker, plus the supervisor
graph/        the LangGraph state and the compiled workflow
tools/        what the agents call: retrieval, SQL, citations, the LLM wrapper
ingestion/    the document pipeline, the DB seeder, and the evals
data/         PDFs, the manifest, the mock CSVs, the eval question sets
tests/        pytest suite
streamlit_app.py
```

`config.py` reads `.env` into a `Settings` object; everything else imports
`settings` from there rather than reading the environment directly.

