"""This module turns a (already rewritten) natural-language question into a
validated, executed SQL query against the read-only advisor mock DB, and
returns it in the fixed SQLQueryResult shape the RAG branch expects
(see schemas.py). This pipeline currently answers exactly one question
family: "show holdings for client X" — every generated query is a
5-column projection (symbol, name_en, quantity, market_value, sector)
joined across clients/holdings/instruments and filtered to one client_id.
Broader aggregate questions ("total portfolio value", "compare sectors")
are out of scope for this result shape and are not handled here.

Pipeline, mirroring query_rewriter.py's step boundaries:

1. Schema grounding: a cached, DB-introspected snapshot of tables/columns
   is injected into the generation prompt.

2. Structured-output generation: the LLM returns {sql, confidence,
   needs_clarification, clarification_question} via a pydantic schema,
   always targeting the fixed output projection. Ambiguous questions
   ("which client?") get flagged rather than guessed at — same principle
   as query_rewriter.py's ambiguous/needs_clarification. When flagged,
   the returned SQL is NEVER validated or executed, even though the
   model still returns a best-guess query — the model's own uncertainty
   is enough reason to not touch the DB at all.

3. Validation: sqlglot AST-level checks (single SELECT, only known
   tables) replace keyword matching. client_id is pulled structurally out
   of the WHERE clause instead of string-matched from the question text.

4. Execution: read-only engine, wall-clock timeout, row cap.

5. Row-shape validation: each row is mapped into ClientHolding. A missing
   expected column is NOT a crash — it's fed back into the repair loop as
   an actionable error, since that's exactly the KeyError class of bug
   the eval caught previously.

6. Repair loop: bounded attempts, stops early on a repeated identical
   error.

Functions here only detect/report/execute — except for generation-level
ambiguity (needs_clarification), which is a hard stop: the returned SQL is
never validated or executed, even though the model still returns a
best-guess query. The model's own uncertainty is reason enough not to
touch the database at all.
"""

import concurrent.futures
from functools import lru_cache

import sqlglot
from sqlglot import exp
from sqlalchemy import text

from tools.llm import get_llm
from tools.db import agent_engine
from schemas import ClientHolding, SQLQueryResult, GeneratedQuery, PipelineResult


ALLOWED_TABLES = {"clients", "instruments", "holdings"}
DIALECT = "sqlite"
DEFAULT_MAX_ROWS = 200
DEFAULT_TIMEOUT_SECONDS = 5
MAX_REPAIR_ATTEMPTS = 2

REQUIRED_COLUMNS = {
    "symbol": "symbol",
    "name_en": "name_en",
    "quantity": "quantity",
    "market_value": "market_value",
    "sector": "sector",
}


@lru_cache(maxsize=1)
def _schema_snapshot() -> str:
    lines = []
    with agent_engine.connect() as conn:
        for table in sorted(ALLOWED_TABLES):
            cols = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
            col_desc = ", ".join(f"{c[1]} {c[2]}" for c in cols)
            lines.append(f"{table}({col_desc})")
    lines.append("")
    lines.append("Foreign keys: holdings.client_id -> clients.client_id, "
                  "holdings.ticker -> instruments.ticker")
    return "\n".join(lines)



GENERATE_SYSTEM_PROMPT = """You translate a natural-language question about
a single client's portfolio holdings into a SQLite SELECT query.

Schema:
{schema}

The query MUST always:
- SELECT exactly these 5 columns, aliased exactly like this:
  holdings.ticker AS symbol, instruments.name_en AS name_en,
  holdings.quantity AS quantity, holdings.market_value AS market_value,
  instruments.sector AS sector
- JOIN holdings to clients (on client_id) and to instruments (on ticker)
- WHERE clients.client_id = '<the client id>'

Rules:
- If the question doesn't clearly identify one client (no client_id,
  or a name that could match more than one client), set
  needs_clarification=true and explain what's ambiguous in
  clarification_question. Still return your best-guess sql.
- Set confidence to how sure you are this is the client the user meant.
- Do not include a LIMIT unless the question asks for a specific count;
  one will be added automatically.
"""

REPAIR_SUFFIX = """

The previous attempt failed:
Previous SQL: {prev_sql}
Error: {error}

Fix the query so it executes successfully and still follows the required
column/alias/join shape above.
"""


def generate_query(question: str, prior_error: tuple[str, str] | None = None) -> GeneratedQuery:
    schema = _schema_snapshot()
    system_prompt = GENERATE_SYSTEM_PROMPT.format(schema=schema)
    if prior_error:
        prev_sql, error = prior_error
        system_prompt += REPAIR_SUFFIX.format(prev_sql=prev_sql, error=error)

    llm = get_llm().with_structured_output(GeneratedQuery)
    return llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ])



def validate_sql(sql: str, max_rows: int = DEFAULT_MAX_ROWS) -> tuple[str | None, str | None]:
    """Returns (safe_sql, error). Structural checks only:
    - exactly one statement (blocks stacked-query injection)
    - statement is a SELECT (DROP/DELETE/UPDATE/INSERT/PRAGMA/ATTACH all
      parse to a non-Select node type regardless of surrounding text, so
      no raw keyword-substring pre-filter is needed — that approach
      false-positives on legitimate literals like WHERE name = 'DROP...')
    - every referenced table is in ALLOWED_TABLES
    - LIMIT is injected if missing, capped down if set too high
    """
    try:
        statements = sqlglot.parse(sql, dialect=DIALECT)
    except Exception as e:
        return None, f"SQL failed to parse: {e}"

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        return None, f"expected exactly one statement, found {len(statements)}"

    stmt = statements[0]
    if not isinstance(stmt, exp.Select):
        return None, f"only SELECT statements are allowed, got {type(stmt).__name__}"

    tables = {t.name for t in stmt.find_all(exp.Table)}
    disallowed = tables - ALLOWED_TABLES
    if disallowed:
        return None, f"query references disallowed table(s): {', '.join(sorted(disallowed))}"

    existing_limit = stmt.args.get("limit")
    if existing_limit is None:
        stmt.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
    else:
        try:
            requested = int(existing_limit.expression.this)
            if requested > max_rows:
                stmt.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
        except (AttributeError, ValueError):
            stmt.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))

    return stmt.sql(dialect=DIALECT), None


def _extract_client_id(sql: str) -> str | None:
    """Pulls the client_id filter value out of the WHERE clause via the
    AST rather than string-matching the question — structural, not
    textual, consistent with the rest of this module's validation."""
    try:
        stmt = sqlglot.parse_one(sql, dialect=DIALECT)
    except Exception:
        return None
    for eq in stmt.find_all(exp.EQ):
        left = eq.this
        right = eq.expression
        if isinstance(left, exp.Column) and left.name.lower() == "client_id" \
                and isinstance(right, exp.Literal):
            return str(right.this)
    return None


def _execute(sql: str, timeout_seconds: int) -> tuple[list[dict] | None, str | None]:
    def _run():
        with agent_engine.connect() as conn:
            cursor_result = conn.execute(text(sql))
            columns = list(cursor_result.keys())
            return [dict(zip(columns, row)) for row in cursor_result.fetchall()]

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run)
        try:
            return future.result(timeout=timeout_seconds), None
        except concurrent.futures.TimeoutError:
            return None, f"query exceeded {timeout_seconds}s timeout"
        except Exception as e:
            return None, str(e)



def _map_rows(rows: list[dict]) -> tuple[list[ClientHolding] | None, str | None]:
    """Maps raw result rows into ClientHolding. A missing expected column
    is reported as an error string (repair-loop fodder), never raised as
    a bare KeyError."""
    if not rows:
        return [], None

    missing = set(REQUIRED_COLUMNS) - set(rows[0].keys())
    if missing:
        return None, (f"query result is missing required column(s): "
                       f"{', '.join(sorted(missing))} — got columns: "
                       f"{', '.join(rows[0].keys())}")

    holdings = []
    for row in rows:
        try:
            holdings.append(ClientHolding(
                symbol=row["symbol"],
                name_en=row["name_en"],
                quantity=row["quantity"],
                market_value=row["market_value"],
                sector=row["sector"],
            ))
        except Exception as e:
            return None, f"row failed to map to ClientHolding: {e}"
    return holdings, None


def run_query_pipeline(question: str, max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
                        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
                        max_rows: int = DEFAULT_MAX_ROWS) -> PipelineResult:
    generated = generate_query(question)
    if generated.needs_clarification:
        return PipelineResult(
            question=question, generated=generated,
            needs_clarification=True,
            clarification_question=generated.clarification_question,
        )

    last_error: str | None = None
    attempts = 0

    while True:
        safe_sql, validation_error = validate_sql(generated.sql, max_rows=max_rows)

        if validation_error:
            if attempts >= max_repair_attempts:
                result = SQLQueryResult(
                    client_id=_extract_client_id(generated.sql) or "",
                    query_used=generated.sql,
                    error=validation_error,
                )
                return PipelineResult(
                    question=question, generated=generated, result=result,
                    validation_error=validation_error, repair_attempts=attempts,
                    needs_clarification=generated.needs_clarification,
                    clarification_question=generated.clarification_question,
                )
            attempts += 1
            generated = generate_query(question, prior_error=(generated.sql, validation_error))
            continue

        rows, exec_error = _execute(safe_sql, timeout_seconds)
        step_error = exec_error

        holdings = None
        if step_error is None:
            holdings, map_error = _map_rows(rows)
            step_error = map_error

        if step_error is None:
            client_id = _extract_client_id(safe_sql) or ""
            result = SQLQueryResult(
                client_id=client_id,
                holdings=holdings,
                row_count=len(holdings),
                query_used=safe_sql,
                error=None,
            )
            return PipelineResult(
                question=question, generated=generated, result=result,
                repair_attempts=attempts,
                needs_clarification=generated.needs_clarification,
                clarification_question=generated.clarification_question,
            )

        if attempts >= max_repair_attempts or step_error == last_error:
            result = SQLQueryResult(
                client_id=_extract_client_id(safe_sql) or "",
                query_used=safe_sql,
                error=step_error,
            )
            return PipelineResult(
                question=question, generated=generated, result=result,
                repair_attempts=attempts,
                needs_clarification=generated.needs_clarification,
                clarification_question=generated.clarification_question,
            )

        last_error = step_error
        attempts += 1
        generated = generate_query(question, prior_error=(safe_sql, step_error))