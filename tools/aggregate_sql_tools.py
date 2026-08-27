"""Cross-client / aggregate portfolio questions -- "total value across all
clients", "how many clients are Aggressive risk", "which sector has the
most holdings". Deliberately a SEPARATE module from tools/sql_tools.py,
not a modification of it: the existing single-client holdings pipeline
(generate_query / GeneratedQuery / SQLQueryResult / ClientHolding) is
what the RAG branch's contract depends on, so it stays untouched. This
file only ADDS a parallel capability; nothing here changes behavior for
a question that doesn't match is_aggregate_question.

Reused as-is from tools/sql_tools.py, unmodified:
- _schema_snapshot()  -- schema grounding is identical for both pipelines
- validate_sql()      -- never hardcoded the single-client 5-column
                          projection; it only checks statement count,
                          SELECT-only, allowed tables, LIMIT enforcement
                          -- all equally correct for an aggregate query
- _execute()           -- plain SQL execution, no shape assumptions

NOT reused: generate_query (wrong prompt/shape), _map_rows (assumes
ClientHolding's fixed columns), run_query_pipeline (wired to
SQLQueryResult specifically). Aggregate rows have no fixed schema across
questions, so there's nothing to map to here -- raw dict rows are the
whole point.
"""

import re

from tools.llm import get_llm
from tools.sql_tools import (
    _schema_snapshot,
    validate_sql,
    _execute,
    DEFAULT_MAX_ROWS,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_REPAIR_ATTEMPTS,
)
from schemas import AggregateQuery, AggregateQueryResult


_AGGREGATE_INTENT_PATTERN = re.compile(
    r"\b(total|average|overall|how many|count of|across all|all clients|"
    r"most|least|highest|lowest|which sector|by sector|by risk|breakdown)\b",
    re.IGNORECASE,
)


def is_aggregate_question(question: str) -> bool:
    """Cheap keyword heuristic, same lightweight-regex approach as
    agents/sql_agent.py's _wants_research_followup rather than an extra
    LLM call. The real distinction isn't single-client vs multi-client --
    it's whether the answer is one computed number/breakdown (SUM, COUNT,
    GROUP BY) or a per-holding row list. "C009's total market value" is
    correctly an aggregate question even though it's about one client:
    the aggregate LLM can write WHERE client_id='C009' itself from the
    question text, same as it can omit that filter for a cross-client
    question -- this classifier only decides which OUTPUT SHAPE is
    needed, not which clients are involved.
    """
    return bool(_AGGREGATE_INTENT_PATTERN.search(question))

AGGREGATE_SYSTEM_PROMPT = """You translate a natural-language question about
aggregate or cross-client portfolio statistics into a SQLite SELECT query.

Schema:
{schema}

This is for questions whose answer is a computed number or breakdown --
totals, counts, averages, grouped by sector/risk_profile/aum_tier -- NOT
for a single client's list of individual holdings.

Rules:
- Only SELECT statements. Only the tables listed above.
- Use GROUP BY / COUNT / SUM / AVG / MIN / MAX as needed. Give result
  columns clear, descriptive aliases (e.g. AS total_value, AS client_count).
- If the question mentions a specific client, filter to that client -- an
  aggregate query can still be scoped to one client (e.g. "C009's total
  market value" is SUM(market_value) WHERE client_id = 'C009').
- If the question is ambiguous (e.g. "biggest" without a specified
  metric), set needs_clarification=true, explain what's ambiguous in
  clarification_question, and still return your best-guess sql.
- Set confidence to how sure you are this answers exactly what was asked.
- Do not include a LIMIT unless the question asks for a specific count;
  one will be added automatically.
"""

REPAIR_SUFFIX = """

The previous attempt failed:
Previous SQL: {prev_sql}
Error: {error}

Fix the query so it executes successfully.
"""


def generate_aggregate_query(question: str,
                              prior_error: tuple[str, str] | None = None) -> AggregateQuery:
    schema = _schema_snapshot()
    system_prompt = AGGREGATE_SYSTEM_PROMPT.format(schema=schema)
    if prior_error:
        prev_sql, error = prior_error
        system_prompt += REPAIR_SUFFIX.format(prev_sql=prev_sql, error=error)

    llm = get_llm().with_structured_output(AggregateQuery)
    return llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ])


def run_aggregate_query_pipeline(question: str,
                                  max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
                                  timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
                                  max_rows: int = DEFAULT_MAX_ROWS) -> AggregateQueryResult:
    generated = generate_aggregate_query(question)

    # Same fix as sql_tools.run_query_pipeline: the model itself flagging
    # this question as unresolved (e.g. "biggest" with no metric
    # specified) must stop everything here, before validate_sql/_execute
    # ever run. Without this, a best-guess aggregate query still executes
    # cleanly against the real DB and returns real rows attached to a
    # question the model never actually resolved -- exactly the bug this
    # mirrors from the single-client pipeline.
    if generated.needs_clarification:
        return AggregateQueryResult(
            needs_clarification=True,
            clarification_question=generated.clarification_question,
        )

    last_error: str | None = None
    attempts = 0

    while True:
        safe_sql, validation_error = validate_sql(generated.sql, max_rows=max_rows)

        if validation_error:
            if attempts >= max_repair_attempts:
                return AggregateQueryResult(
                    query_used=generated.sql, error=validation_error,
                    needs_clarification=generated.needs_clarification,
                    clarification_question=generated.clarification_question,
                )
            attempts += 1
            generated = generate_aggregate_query(question, prior_error=(generated.sql, validation_error))
            continue

        rows, exec_error = _execute(safe_sql, timeout_seconds)

        if exec_error is None:
            return AggregateQueryResult(
                rows=rows, row_count=len(rows), query_used=safe_sql, error=None,
                needs_clarification=generated.needs_clarification,
                clarification_question=generated.clarification_question,
            )

        if attempts >= max_repair_attempts or exec_error == last_error:
            return AggregateQueryResult(
                query_used=safe_sql, error=exec_error,
                needs_clarification=generated.needs_clarification,
                clarification_question=generated.clarification_question,
            )

        last_error = exec_error
        attempts += 1
        generated = generate_aggregate_query(question, prior_error=(safe_sql, exec_error))