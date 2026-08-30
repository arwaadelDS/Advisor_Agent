"""Translates an advisor's question into terms that match the DB, not the
other way around. Sits between the advisor's raw question and SQL generation --
see agents/sql_agent.py.

The DB stores client names and enum values (risk_profile, aum_tier) as Latin
strings; ``instruments`` is the only table with a genuine Arabic column
(``name_ar``). An Arabic question routinely needs to reference all of these,
so this cannot be a lookup against one table -- it has to be a translation
grounded in whatever values actually exist across the schema, generated fresh
rather than hand-mapped. A hardcoded Arabic-phrase dictionary only covers the
phrasing someone thought to add; grounding on live distinct values means a new
sector or instrument needs no rewriter change to be handled correctly.

Client-name resolution is the one case that stays best-effort: instrument
names and enum values are closed sets pulled straight from their columns, but
a client's name has no Arabic-script counterpart in the DB at all -- only a
Latin transliteration -- so matching "فيصل العتيبي" to "Faisal Al-Otaibi" is
open-ended in a way the other two are not. In practice this rarely matters:
``client_id`` normally comes from session state (see graph/state.py), not
parsed out of the question, so this path only fires when a question names a
client explicitly.
"""

from __future__ import annotations

import difflib
from functools import lru_cache
from typing import Optional

from sqlalchemy import text

from tools.db import agent_engine
from tools.llm import get_llm
from schemas import RewrittenQuery


class QueryRewriterError(RuntimeError):
    """Raised when the rewriter cannot run at all -- no grounding data, no model."""


# How close a model-resolved value has to be to a real DB value to be accepted
# as a near-miss correction rather than rejected outright. 0.8 catches typos
# and minor casing/spacing drift ("SABIK" -> "SABIC") without accepting a
# different instrument entirely.
FUZZY_CUTOFF = 0.8


@lru_cache(maxsize=1)
def _raw_grounding() -> dict[str, list]:
    """One DB round trip, shared by the prompt text and the verification set.

    Cached once per process -- same reasoning as rag_tools.catalog(): static
    mock CSVs, re-querying per question is pure waste. Call
    ``_raw_grounding.cache_clear()`` if the CSVs change without a restart.
    """
    try:
        with agent_engine.connect() as conn:
            return {
                "instruments": conn.execute(text(
                    "SELECT ticker, name_en, name_ar FROM instruments"
                )).fetchall(),
                "clients": conn.execute(text(
                    "SELECT client_id, name FROM clients"
                )).fetchall(),
                "risk_profiles": [r[0] for r in conn.execute(text(
                    "SELECT DISTINCT risk_profile FROM clients"
                )).fetchall()],
                "aum_tiers": [r[0] for r in conn.execute(text(
                    "SELECT DISTINCT aum_tier FROM clients"
                )).fetchall()],
                "sectors": [r[0] for r in conn.execute(text(
                    "SELECT DISTINCT sector FROM instruments"
                )).fetchall()],
                "asset_classes": [r[0] for r in conn.execute(text(
                    "SELECT DISTINCT asset_class FROM instruments"
                )).fetchall()],
            }
    except Exception as exc:
        raise QueryRewriterError(f"could not load grounding values from the DB:\n{exc}") from exc


def _grounding() -> str:
    """The raw grounding, formatted for the prompt."""
    raw = _raw_grounding()
    lines = ["Instruments (ticker | English name | Arabic name):"]
    lines += [f"- {t} | {en} | {ar}" for t, en, ar in raw["instruments"]]
    lines.append("")
    lines.append("Clients (client_id | name, Latin transliteration):")
    lines += [f"- {cid} | {name}" for cid, name in raw["clients"]]
    lines.append("")
    lines.append("risk_profile values: " + ", ".join(sorted(raw["risk_profiles"])))
    lines.append("aum_tier values: " + ", ".join(sorted(raw["aum_tiers"])))
    lines.append("sector values: " + ", ".join(sorted(raw["sectors"])))
    lines.append("asset_class values: " + ", ".join(sorted(raw["asset_classes"])))
    return "\n".join(lines)


def _known_values() -> set[str]:
    """Every real DB value a rewritten reference could legitimately resolve to.

    Pooled into one flat set rather than checked column-by-column: the
    verification step only needs to know "is this a real value anywhere
    relevant", not which column it came from -- the SQL generator figures out
    the column from context, same as it already does for anything else in the
    rewritten question.
    """
    raw = _raw_grounding()
    values: set[str] = set()
    for ticker, name_en, name_ar in raw["instruments"]:
        values.update({ticker, name_en, name_ar})
    for client_id, name in raw["clients"]:
        values.update({client_id, name})
    values.update(raw["risk_profiles"])
    values.update(raw["aum_tiers"])
    values.update(raw["sectors"])
    values.update(raw["asset_classes"])
    return values


def _try_resolve(term: str, known: set[str]) -> Optional[str]:
    """Attempt to resolve one term against real DB values: exact match,
    then a close-typo fuzzy match, then unique-substring containment
    (a short name like "Noura" against a full "Noura Al-Zahrani").

    Order matters: fuzzy runs before substring so a genuine near-miss typo
    ("SABIK") is corrected to the right value ("SABIC") rather than being
    mistaken for a substring of something else. Substring only fires when
    exactly one known value contains the term, so an ambiguous partial
    reference (matching two different clients, say) is never silently
    guessed -- it stays unresolved instead.
    """
    if term in known:
        return term
    close = difflib.get_close_matches(term, known, n=1, cutoff=FUZZY_CUTOFF)
    if close:
        return close[0]
    substring_hits = [
        v for v in known
        if term.lower() in v.lower() and term.lower() != v.lower()
    ]
    if len(substring_hits) == 1:
        return substring_hits[0]
    return None


def _verify_corrections(result: RewrittenQuery) -> RewrittenQuery:
    """Check every correction the model made against real DB values, and
    attempt to rescue anything the model itself flagged as ambiguous.

    The model is asked to only resolve references it can match exactly, but
    nothing enforces that on its own -- a confident near-miss ("SABIK" instead
    of "SABIC") would otherwise reach SQL generation looking exactly like a
    verified match, and fail later at execution instead of here where the
    failure is cheap and specific. This is the deterministic floor under the
    model's own judgment, not a replacement for it.

    Pre-existing ``ambiguous`` entries get the same rescue attempt as failed
    corrections: a short reference like "Noura" often never becomes a
    ``corrections`` entry at all -- the model, told not to guess, puts it
    straight into ``ambiguous`` -- so resolving it has to happen here too,
    not only on the corrections side.
    """
    known = _known_values()
    kept_corrections: list[str] = []
    new_ambiguous: list[str] = []
    needs_clarification = result.needs_clarification

    for correction in result.corrections:
        if "->" not in correction:
            kept_corrections.append(correction)
            continue
        original, resolved = (part.strip() for part in correction.split("->", 1))

        if original.lower() == resolved.lower():
            continue  # no-op "correction" -- not a real substitution

        resolved_final = _try_resolve(resolved, known)
        if resolved_final:
            kept_corrections.append(f"{original} -> {resolved_final}")
        else:
            new_ambiguous.append(correction)
            needs_clarification = True

    for entry in result.ambiguous:
        term = entry.split("->", 1)[0].strip() if "->" in entry else entry
        resolved_final = _try_resolve(term, known)
        if resolved_final:
            kept_corrections.append(f"{term} -> {resolved_final}")
        else:
            new_ambiguous.append(entry)
            needs_clarification = True

    return result.model_copy(update={
        "corrections": kept_corrections,
        "ambiguous": new_ambiguous,
        "needs_clarification": needs_clarification,
    })


SYSTEM_PROMPT = """\
You rewrite an advisor's question so every reference in it matches a value
that actually exists in the database below -- exact spelling, exact casing.

You are not answering the question. You are translating references in it:
- A company mentioned in Arabic or English -> its ticker.
- A risk tolerance, AUM tier, sector, or asset class *value* mentioned in
  Arabic or English, or informally -> the exact column value listed below
  (e.g. "aggressive risk" -> "Aggressive").
- A client named in the question -> the client_id, only if you are confident
  which client is meant.
- If every entity in the question (client, company, value) matches correctly
  but the question asks for information this schema doesn't have at all
  (e.g. a home address, a phone number, anything not in the tables listed),
  do not flag the correctly-matched entity as ambiguous. Leave the question
  as rewritten and let it fail downstream where "no such column" belongs,
  rather than reporting a real match as unmatched.
  

Database values:
{grounding}

Rules:
- Only rewrite references that map to something in the list above. Do not
  invent tickers, IDs, or values that are not listed.
- If a reference in the question does not clearly match anything above, leave
  it in the rewritten question as-is and list it in `ambiguous`. Do not guess.
- `corrections` is ONLY for data-value substitutions you can verify against
  the list above -- a ticker, a client_id, or an exact enum value (like
  "Conservative" or "HNW"). Each entry's resolved side must be copied
  verbatim from the list above, nothing else.
- Do NOT put column-name or field-name translations in `corrections` (e.g.
  translating "risk level"/"مستوى المخاطرة" to the concept of risk_profile is
  just phrasing the rewritten question in English/normal terms -- it is not a
  value substitution and must not appear in `corrections`).
- When substituting a ticker into the rewritten question text itself, phrase
  it as "ticker 2010" or "the instrument with ticker 2010" so a reader with
  no access to this rewrite step cannot mistake it for a year -- but the
  `corrections` entry for that same substitution must still show the bare
  ticker on the resolved side, e.g. "سابك -> 2010", not the full phrase.
- Set `needs_clarification=true` only if an unresolved reference is essential
  to answering the question (e.g. an unmatched company or client). A vague but
  answerable question does not need clarification.
- The rewritten question should read naturally, in the same language as the
  original, with only the resolved terms substituted in.
  - Set `wants_research` to true if the question asks for anything beyond raw
  data about the holdings themselves -- research, risk assessment, opinion,
  sentiment, "should [client] be worried", "what does research say" -- in
  addition to or instead of just retrieving the positions. A question that
  only asks what a client holds, without asking for any judgment or research
  on those positions, leaves this false.
"""

USER_PROMPT = "Advisor's question:\n{question}"


@lru_cache(maxsize=1)
def _rewriter_llm():
    return get_llm().with_structured_output(RewrittenQuery)


def rewrite_query(question: str, client_selected: bool = False) -> RewrittenQuery:
    """Ground an advisor's question against the DB's actual values.

    client_selected tells the model a client is already chosen in this
    session, so a generic reference to "this client"/"the client" (in any
    language) is not something to resolve by name -- it's resolved by
    session context downstream, in tools/sql_tools.py's CLIENT_SCOPED_RULE.
    Without this flag the rewriter tries to match "this client" against real
    client names, fails, and wrongly demands clarification for a perfectly
    answerable question.
    """
    question = question.strip()
    if not question:
        raise QueryRewriterError("a question to rewrite cannot be empty")

    grounding = _grounding()
    client_note = (
        "\nA client is already selected in this session. Do not flag generic "
        "references to that client ('this client', 'the client', 'my client', "
        "'their', or equivalents in any language) as ambiguous -- those are "
        "resolved by context, not by name. Only flag an unresolved reference "
        "if the question names a *specific* client or company you cannot "
        "match in the list above.\n"
        if client_selected else ""
    )
    messages = [
        ("system", SYSTEM_PROMPT.format(grounding=grounding) + client_note),
        ("user", USER_PROMPT.format(question=question)),
    ]

    try:
        result = _rewriter_llm().invoke(messages)
    except Exception as exc:
        raise QueryRewriterError(f"rewrite failed:\n{exc}") from exc

    if not isinstance(result, RewrittenQuery):
        raise QueryRewriterError(f"model did not return a RewrittenQuery: {result!r}")

    result = result.model_copy(update={"original": question})
    return _verify_corrections(result)


def main(argv: list[str] | None = None) -> int:
    """Rewrite a question from the command line.

        uv run python -m tools.query_rewriter "ما هي مقتنيات العميل في قطاع البنوك؟"
        uv run python -m tools.query_rewriter "this client's holdings" --client-selected
    """
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="tools.query_rewriter",
        description="Ground an advisor's question against actual DB values.",
    )
    parser.add_argument("question")
    parser.add_argument("--client-selected", action="store_true",
                        help="simulate a client already being selected in session")
    args = parser.parse_args(argv)

    try:
        result = rewrite_query(args.question, client_selected=args.client_selected)
    except QueryRewriterError as exc:
        print("FAILED")
        print(str(exc))
        return 1

    print(f"Original:  {result.original}")
    print(f"Rewritten: {result.rewritten}")
    if result.corrections:
        print("Corrections:")
        for c in result.corrections:
            print(f"  - {c}")
    if result.ambiguous:
        print("Ambiguous:")
        for a in result.ambiguous:
            print(f"  - {a}")
    print(f"needs_clarification: {result.needs_clarification}")
    print(f"wants_research: {result.wants_research}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())