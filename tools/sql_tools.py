"""Generation, validation, execution, and repair for the SQL agent.

One generator handles both a single client's data and cross-client
aggregates -- there is no separate aggregate pipeline. Which of
SQLQueryResult.holdings / .rows gets filled depends only on whether the
executed query's columns match ClientHolding's shape; see _shape_rows.

client_profile is never part of the generated SQL. It's a fixed, tiny lookup
(SELECT * FROM clients WHERE client_id = ...) run deterministically whenever
client_id is known, so the LLM's only job is ever "answer the question," never
"also remember to fetch the client's tier." Keeping it out of generation also
keeps the holdings/rows shape check unambiguous -- a generated query never
has profile columns mixed into it to confuse that check.

Validation is a hard gate, not a suggestion: nothing this module executes
skips validate_sql first. See tools/query_rewriter.py's docstring for why
that separation matters -- same principle, one step earlier in the pipeline.

client_id being present in session no longer forces every query to filter on
it -- an aggregate question ("how many clients...", "total across all
clients") asked while a client happens to be selected must still be able to
span all clients. What IS still enforced when no client is selected: an
unscoped query returning raw, individually-identifiable client rows
(name/client_id, not truly aggregated -- see _is_aggregated and
_touches_client_identity) is rejected outright. This closes two real gaps
found via manual testing: GROUP BY name/client_id still discloses one row per
client despite the SUM() wrapper, and SELECT * is represented by sqlglot as
exp.Star rather than exp.Column, so a naive column-name scan misses it
entirely.

Statement-type and multi-statement checks are done via sqlglot's parsed AST
(exp.Select / parse() returning >1 statement) rather than string matching on
keywords or semicolons -- a regex on raw text is fragile in ways an AST check
is not (comments, string-literal semicolons, alternate casing/whitespace all
survive a naive text scan unchanged but not a parse).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import sqlglot
from sqlglot import exp
from sqlalchemy import text

from tools.db import agent_engine
from tools.llm import get_llm
from schemas import ClientHolding, ClientProfile, GeneratedQuery, PipelineResult, SQLQueryResult

DIALECT = "sqlite"
MAX_ROWS = 200
MAX_REPAIR_ATTEMPTS = 2  # attempts after the first try, not counting it


class SQLToolError(RuntimeError):
    """Raised when the pipeline cannot run at all -- no model, no DB connection."""


ALLOWED_COLUMNS: dict[str, set[str]] = {
    "clients": {"client_id", "name", "risk_profile", "aum_tier"},
    "instruments": {"ticker", "isin", "name_en", "name_ar", "sector", "asset_class", "shariah_flag"},
    "holdings": {"client_id", "ticker", "quantity", "market_value"},
}
ALLOWED_TABLES = set(ALLOWED_COLUMNS)

HOLDING_COLUMNS = {"symbol", "name_en", "name_ar", "sector", "asset_class", "quantity", "market_value"}

CLIENT_FILTER_PATTERN = __import__("re").compile(r":client_id\b")



SCHEMA_DESCRIPTION = """\
Tables and columns (SQLite):

clients(client_id, name, risk_profile, aum_tier)
instruments(ticker, isin, name_en, name_ar, sector, asset_class, shariah_flag)
holdings(client_id, ticker, quantity, market_value)

holdings.ticker joins to instruments.ticker. holdings.client_id joins to
clients.client_id. shariah_flag is always empty -- do not filter on it.
Both holdings and instruments have a "ticker" column -- always qualify it
(holdings.ticker or instruments.ticker) in any query that joins them.
"""

SYSTEM_PROMPT = """\
You write SQLite SELECT queries against a fixed 3-table schema. You never
generate anything except SELECT statements.

{schema}

Rules:
- Only SELECT. Never INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/TRUNCATE, and
  never more than one statement.
- Only the tables and columns listed above. Do not invent columns.
- Always include a LIMIT.
- {client_rule}
- If a question about one client's positions is being answered, alias the
  result columns exactly to: ticker AS symbol, name_en, name_ar, sector,
  asset_class, quantity, market_value -- in that column order. This applies
  even if the question only asks about one specific instrument or a subset
  of positions (e.g. "their Almarai position") -- still return the full row
  shape for the matching holding(s), never just ticker/quantity alone. This
  lets the caller recognise a holdings-shaped result without guessing.
- If the question is an aggregate or spans multiple clients (averages,
  counts, group-bys, "which clients..."), the result columns can be whatever
  the question needs -- there is no fixed shape to match for those. If no
  client is selected and the question would otherwise return individual
  client records (names, client_ids) across all clients with no filter, you
  must aggregate in a way that does not still expose one row per client --
  grouping by name or client_id does not count as aggregation for this
  purpose, since it still discloses each client's exact figure.
- If the question is ambiguous or unanswerable from this schema, set
  needs_clarification=true and ask a specific question instead of guessing.
  - If the question asks for something beyond this schema (e.g. research,
  opinions, news, risk commentary) in addition to data this schema can
  answer, do NOT set needs_clarification because of that -- generate SQL
  for the data-retrievable part only. A separate step handles the
  non-schema part; your job is only the data half.
"""

CLIENT_SCOPED_RULE = (
    "A client (client_id) is selected in this session, but that does not mean "
    "every question is about that one client. Read the question itself: "
    "if it refers to 'this client', 'the client', 'their', or similar -- filter "
    "on holdings.client_id = :client_id (a bind parameter, never a literal ID). "
    "If the question is about multiple/all clients, an aggregate, or a count "
    "across clients ('how many clients...', 'total across all clients', "
    "'average client...') -- do NOT filter by client_id, even though one is "
    "selected. The session's selected client is available context, not a "
    "forced scope."
)
UNSCOPED_RULE = (
    "No client is selected in this session. Do not filter by any client_id "
    "-- this question spans clients or the whole table."
)

USER_PROMPT = "Question:\n{question}"


@lru_cache(maxsize=1)
def _generator_llm():
    return get_llm().with_structured_output(GeneratedQuery)


def generate_query(
    question: str,
    client_id: Optional[str] = None,
    prior_error: Optional[str] = None,
) -> GeneratedQuery:
    """Generate one candidate SQL query for the question.

    client_id being set means a client is selected in the session, not that
    the query must filter on it -- see CLIENT_SCOPED_RULE. The model decides
    scope from the question's own wording; the validator no longer enforces
    either direction here, only that an unresolved client_id isn't referenced
    when none is selected, and that an unscoped query never returns raw
    individual client records ungrouped. See validate_sql.

    ``prior_error`` is set on repair attempts -- see run_query_pipeline --
    and is appended to the user turn so the model sees exactly what went
    wrong last time, not just the original question again.
    """
    question = question.strip()
    if not question:
        raise SQLToolError("a question to generate SQL for cannot be empty")

    client_rule = CLIENT_SCOPED_RULE if client_id else UNSCOPED_RULE
    system = SYSTEM_PROMPT.format(schema=SCHEMA_DESCRIPTION, client_rule=client_rule)
    user = USER_PROMPT.format(question=question)
    if prior_error:
        user += f"\n\nYour previous attempt failed:\n{prior_error}\nFix it."

    try:
        result = _generator_llm().invoke([("system", system), ("user", user)])
    except Exception as exc:
        raise SQLToolError(f"SQL generation failed:\n{exc}") from exc

    if not isinstance(result, GeneratedQuery):
        raise SQLToolError(f"model did not return a GeneratedQuery: {result!r}")
    return result



def _find_ambiguous_columns(parsed, tables: set[str]) -> list[str]:
    """Unqualified column references that exist in more than one joined
    table -- SQLite accepts these syntactically and only fails at execution,
    costing a repair attempt for something checkable statically for free.
    """
    if len(tables) < 2:
        return []
    ambiguous = []
    for column in parsed.find_all(exp.Column):
        if column.table:
            continue
        owners = [t for t in tables if column.name.lower() in ALLOWED_COLUMNS.get(t, set())]
        if len(owners) > 1:
            ambiguous.append(column.name)
    return ambiguous


def _is_aggregated(parsed) -> bool:
    """True if the query aggregates *without* grouping by individual client
    identity. GROUP BY name/client_id still produces one row per client --
    that's raw per-client disclosure with a SUM() attached, not aggregation
    that actually hides who's who. A plain aggregate with no GROUP BY
    (COUNT(*), a single SUM over the whole table) collapses to one summary
    row and is safe regardless.
    """
    if parsed.find(exp.AggFunc) is None:
        return False
    group = parsed.args.get("group")
    if group is not None:
        group_columns = {c.name.lower() for c in group.find_all(exp.Column)}
        if group_columns & {"name", "client_id"}:
            return False
    return True


def _touches_client_identity(parsed, tables: set[str]) -> bool:
    """True if the query selects columns that identify individual clients
    (name, client_id) rather than only aggregate/derived values.

    SELECT * is checked explicitly: sqlglot represents a wildcard as
    exp.Star, not as exp.Column, so a plain find_all(exp.Column) walk is
    silently blind to "SELECT * FROM clients" -- the single most direct way
    to ask for everything.
    """
    if "clients" not in tables:
        return False
    if parsed.find(exp.Star) is not None:
        return True
    for column in parsed.find_all(exp.Column):
        if column.name.lower() in {"name", "client_id"}:
            return True
    return False


def validate_sql(sql: str, client_id: Optional[str]) -> list[str]:
    """Policy + syntactic checks. An empty list means the SQL may run.

    client_id present no longer requires a client_id filter -- a client can
    be selected in session while the question itself is cross-client (an
    aggregate, a count across clients). Scope is a generation-time decision
    made from the question's wording (see CLIENT_SCOPED_RULE), not something
    this validator can safely force. Two things ARE still enforced regardless
    of whether a client is selected: a query must not reference :client_id at
    all when no client is in context (that can only be a hallucinated or
    leftover filter), and a query returning raw, individually-identifiable
    client rows must be genuinely aggregated -- see _touches_client_identity /
    _is_aggregated. The second check is keyed on whether the query itself
    filters by client (``has_client_filter``), not on whether a client
    happens to be selected in session: CLIENT_SCOPED_RULE explicitly allows a
    cross-client aggregate to be generated even with a client selected, so
    gating this guard on ``client_id`` alone left that legitimate branch
    completely unprotected -- any unscoped, non-aggregated, client-identifying
    query got a free pass the instant any client had ever been selected in
    the thread. See tests/test_sql_tools.py for the injection case this closes.

    A non-SELECT statement does not short-circuit the rest of the checks --
    a DELETE against a non-allowlisted table should still report both
    problems, not just the first one found.
    """
    errors: list[str] = []
    stripped = sql.strip().rstrip(";")

    try:
        statements = sqlglot.parse(stripped, read=DIALECT)
    except Exception as exc:
        errors.append(f"SQL parse error: {exc}")
        return errors

    non_empty_statements = [s for s in statements if s is not None]
    if len(non_empty_statements) == 0:
        errors.append("SQL statement is empty.")
        return errors
    if len(non_empty_statements) > 1:
        errors.append("Multi-statement SQL is not allowed.")
        return errors

    parsed = non_empty_statements[0]

    if not isinstance(parsed, exp.Select):
        errors.append(
            f"Only SELECT statements are allowed (got {type(parsed).__name__})."
        )
       

    has_client_filter = bool(CLIENT_FILTER_PATTERN.search(sql))
    if not client_id and has_client_filter:
        errors.append("No client is in context; the query must not reference :client_id.")

    tables = {t.name.lower() for t in parsed.find_all(exp.Table)}
    unknown_tables = tables - ALLOWED_TABLES
    if unknown_tables:
        errors.append(f"References non-allowlisted tables: {sorted(unknown_tables)}")

    allowed_used_tables = tables & ALLOWED_TABLES

    ambiguous_cols = _find_ambiguous_columns(parsed, allowed_used_tables)
    if ambiguous_cols:
        errors.append(f"Ambiguous unqualified columns (qualify with table name): {sorted(set(ambiguous_cols))}")

    if not has_client_filter and _touches_client_identity(parsed, allowed_used_tables) and not _is_aggregated(parsed):
        errors.append(
            "A query returning individual client records must be aggregated "
            "in a way that does not still disclose one row per client "
            "(grouping by name/client_id does not count) -- raw or per-client "
            "rows across clients are not allowed without a client filter, "
            "regardless of whether a client happens to be selected in session."
        )

    for column in parsed.find_all(exp.Column):
        table_name = column.table.lower() if column.table else None
        if table_name and table_name in ALLOWED_COLUMNS:
            if column.name.lower() not in ALLOWED_COLUMNS[table_name]:
                errors.append(f"Unknown column {table_name}.{column.name}")

    if isinstance(parsed, exp.Select) and parsed.args.get("limit") is None:
        errors.append("A LIMIT is required.")

    return errors

def fetch_client_profile(client_id: str) -> Optional[ClientProfile]:
    """The deterministic client-facts lookup. Never touches generated SQL."""
    try:
        with agent_engine.connect() as conn:
            row = conn.execute(
                text("SELECT client_id, name, risk_profile, aum_tier FROM clients WHERE client_id = :client_id"),
                {"client_id": client_id},
            ).fetchone()
    except Exception as exc:
        raise SQLToolError(f"client profile lookup failed:\n{exc}") from exc

    if row is None:
        return None
    return ClientProfile(client_id=row[0], name=row[1], risk_profile=row[2], aum_tier=row[3])


def _shape_rows(columns: list[str], rows: list[tuple]) -> tuple[list[ClientHolding], list[dict]]:
    """Split a raw result into (holdings, rows) by column shape.

    Exact match against HOLDING_COLUMNS, in any order, is what makes this a
    holdings result -- the generation prompt asks for that exact alias set
    when the question is client-scoped, so this is checking that the model
    followed the convention, not guessing at intent.
    """
    if set(columns) == HOLDING_COLUMNS:
        holdings = [ClientHolding(**dict(zip(columns, row))) for row in rows]
        return holdings, []
    return [], [dict(zip(columns, row)) for row in rows]


def execute_sql(sql: str, client_id: Optional[str]) -> SQLQueryResult:
    """Run already-validated SQL. Never call this on unvalidated SQL."""
    params = {"client_id": client_id} if client_id else {}
    try:
        with agent_engine.connect() as conn:
            result = conn.execute(text(sql), params)
            rows = result.fetchmany(MAX_ROWS)
            columns = list(result.keys())
    except Exception as exc:
        return SQLQueryResult(client_id=client_id, error=str(exc))

    holdings, generic_rows = _shape_rows(columns, rows)
    return SQLQueryResult(
        client_id=client_id,
        holdings=holdings,
        rows=generic_rows,
        row_count=len(rows),
        query_used=sql,
    )



def run_query_pipeline(question: str, client_id: Optional[str] = None) -> PipelineResult:
    """Generate -> validate -> execute, retrying on either failure.

    A validation failure and an execution failure are both fed back to
    generate_query as prior_error and count against the same repair budget --
    the model doesn't need to know which kind of failure it was to fix it,
    just what the error said.
    """
    attempts = 0
    generated = generate_query(question, client_id)

    while True:
        if generated.needs_clarification:
            return PipelineResult(
                question=question,
                generated=generated,
                repair_attempts=attempts,
                needs_clarification=True,
                clarification_question=generated.clarification_question,
            )

        errors = validate_sql(generated.sql, client_id)
        if errors:
            validation_error = "; ".join(errors)
            if attempts >= MAX_REPAIR_ATTEMPTS:
                return PipelineResult(
                    question=question,
                    generated=generated,
                    validation_error=validation_error,
                    repair_attempts=attempts,
                )
            attempts += 1
            generated = generate_query(question, client_id, prior_error=validation_error)
            continue

        result = execute_sql(generated.sql, client_id)
        if result.error and attempts < MAX_REPAIR_ATTEMPTS:
            attempts += 1
            generated = generate_query(question, client_id, prior_error=result.error)
            continue

        if client_id and not result.error and CLIENT_FILTER_PATTERN.search(generated.sql):
            profile = fetch_client_profile(client_id)
            result = result.model_copy(update={"client_profile": profile})

        return PipelineResult(question=question, generated=generated, result=result, repair_attempts=attempts)


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline from the command line.

        uv run python -m tools.sql_tools "average market value by risk profile"
        uv run python -m tools.sql_tools "this client's holdings" --client C001
    """
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="tools.sql_tools")
    parser.add_argument("question")
    parser.add_argument("--client", default=None, dest="client_id")
    args = parser.parse_args(argv)

    outcome = run_query_pipeline(args.question, args.client_id)
    print(f"SQL: {outcome.generated.sql}")
    print(f"repair_attempts: {outcome.repair_attempts}")
    if outcome.needs_clarification:
        print(f"needs_clarification: {outcome.clarification_question}")
        return 0
    if outcome.validation_error:
        print(f"validation_error: {outcome.validation_error}")
        return 1
    if outcome.result and outcome.result.error:
        print(f"execution error: {outcome.result.error}")
        return 1
    if outcome.result:
        if outcome.result.client_profile:
            print(f"profile: {outcome.result.client_profile}")
        for h in outcome.result.holdings:
            print(f"  {h.symbol}  {h.name_en}  qty={h.quantity}  value={h.market_value}")
        for r in outcome.result.rows:
            print(f"  {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())