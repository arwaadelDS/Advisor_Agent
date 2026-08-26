"""This module handles the preprocessing and normalization of natural language
queries submitted by users before they are translated into SQL statements.
1- llm rewrite: correct spelling mistakes, grammar, phrasing in the query

2- fuzzy matching: pull vocab from db and do token-level matching to correct
values that may have typos with the actual db values format. Covers both
single-token categorical fields and multi-word phrase/entity fields.

3- ambiguity detection: when a fuzzy match ties between two or more real
values, do NOT guess — surface it for clarification instead. A confident
wrong guess against client/portfolio data is worse than asking the user.
"""

from pydantic import BaseModel
from sqlalchemy import text
from rapidfuzz import process, fuzz
from tools.llm import get_llm, text_of
from tools.db import agent_engine
from schemas import RewrittenQuery



REWRITE_SYSTEM_PROMPT = """You correct spelling, grammar, and phrasing in a user's
question about client portfolio data. Do NOT change the meaning, add filters,
or guess at entity values (client IDs, names, sectors) — just fix language.
Return ONLY the corrected question text, nothing else."""


def llm_rewrite(question: str) -> str:
    llm = get_llm()
    resp = llm.invoke([
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ])
    # Gemini can return empty content for prompts it flags as adversarial/
    # unsafe (e.g. SQL-injection-looking text) rather than raising an
    # exception. An empty rewritten question would otherwise propagate
    # downstream into generate_query's LLM call and crash the whole
    # pipeline with "contents are required" -- fall back to the original
    # question text instead of ever returning "".
    content = text_of(resp)
    return content if content else question


CATEGORICAL_FIELDS = {
    "client_id": ("clients", "client_id"),
    "risk_profile": ("clients", "risk_profile"),
    "aum_tier": ("clients", "aum_tier"),
    "sector": ("instruments", "sector"),
    "ticker": ("instruments", "ticker"),
}

ENTITY_FIELDS = {
    "client_name": ("clients", "name"),
    "instrument_name_en": ("instruments", "name_en"),
    "instrument_name_ar": ("instruments", "name_ar"),
}


def _load_distinct_values(table: str, column: str) -> list[str]:
    with agent_engine.connect() as conn:
        rows = conn.execute(text(f"SELECT DISTINCT {column} FROM {table}")).fetchall()
    return [r[0] for r in rows if r[0]]


def _load_known_values() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    categorical = {
        field: _load_distinct_values(table, col)
        for field, (table, col) in CATEGORICAL_FIELDS.items()
    }
    entities = {
        field: _load_distinct_values(table, col)
        for field, (table, col) in ENTITY_FIELDS.items()
    }
    return categorical, entities


def _tied_top_matches(candidate: str, values: list[str], scorer, processor=None) -> tuple[set[str], float]:
    """Every real value that ties for the top score against `candidate`,
    and that score. len(tied) > 1 means multiple real values are equally
    close — a genuine ambiguity, not just 'the closest one' being slightly
    better. Shared by both correction functions below so production
    behavior and anything testing it can never drift apart."""
    results = process.extract(candidate, values, scorer=scorer, processor=processor, limit=None)
    if not results:
        return set(), 0
    top_score = results[0][1]
    tied = {r[0] for r in results if r[1] == top_score}
    return tied, top_score


def fuzzy_correct_categorical(question: str, categorical: dict[str, list[str]],
                               threshold: int = 85) -> tuple[str, list[str], list[str]]:
    """Token-level correction for single-word bounded-vocabulary fields.
    Scoring is case-insensitive (processor=str.lower) so pure-case typos like
    'c007' vs 'C007' are found; the correction decision is case-sensitive
    (best_match != stripped) so a pure-case fix still counts as a real
    change. Competition is across ALL categorical values (not just one
    field) since that's what a real query token could plausibly refer to.
    When a candidate ties between two or more real values, it is left
    uncorrected and reported in `ambiguous` instead of guessed."""
    corrections = []
    ambiguous = []
    tokens = question.split()
    corrected_tokens = []

    all_values = [v for values in categorical.values() for v in values]
    value_to_field = {v: field for field, values in categorical.items() for v in values}

    for token in tokens:
        stripped = token.strip(".,?!")
        tied, score = _tied_top_matches(stripped, all_values, fuzz.ratio, str.lower)

        if not tied or score < threshold:
            corrected_tokens.append(token)
            continue

        if len(tied) == 1:
            best_match = next(iter(tied))
            if best_match != stripped:
                field = value_to_field[best_match]
                corrections.append(f"'{stripped}' -> '{best_match}' ({field}, {score:.1f}%)")
                corrected_tokens.append(token.replace(stripped, best_match))
            else:
                corrected_tokens.append(token)
        else:
            candidates = sorted(tied)
            ambiguous.append(
                f"'{stripped}' is ambiguous ({score:.1f}%) between: {', '.join(candidates)} — not auto-corrected"
            )
            corrected_tokens.append(token)

    return " ".join(corrected_tokens), corrections, ambiguous


def fuzzy_correct_entities(question: str, entities: dict[str, list[str]],
                            threshold: int = 85) -> tuple[str, list[str], list[str]]:
    """Whole-phrase / sliding-window matching for multi-word proper nouns.
    Matches are flagged as hints, never auto-substituted into the question —
    silently rewriting free text around a proper noun risks mangling the
    sentence or guessing wrong on an ambiguous partial name. When the best
    score found for a field ties between two or more real entities (a
    shared family name, a common word like 'Bank', a near-identical
    suffix), all tied candidates are reported in `ambiguous` instead of
    picking one and presenting it as a confident single match."""
    corrections = []
    ambiguous = []
    tokens = [t.strip(".,?!") for t in question.split()]
    n = len(tokens)

    for field, values in entities.items():
        max_words = max(len(v.split()) for v in values)
        best_score_per_value: dict[str, float] = {}

        for window_size in range(1, min(max_words, n) + 1):
            for i in range(n - window_size + 1):
                candidate = " ".join(tokens[i:i + window_size])
                if len(candidate) < 3:
                    continue
                for match, score, _ in process.extract(
                    candidate, values, scorer=fuzz.partial_ratio, limit=None
                ):
                    if score > best_score_per_value.get(match, 0):
                        best_score_per_value[match] = score

        if not best_score_per_value:
            continue

        top_score = max(best_score_per_value.values())
        if top_score < threshold:
            continue

        tied = sorted(v for v, s in best_score_per_value.items() if s == top_score)

        if len(tied) == 1:
            match = tied[0]
            if match.lower() not in question.lower():
                corrections.append(
                    f"possible entity match: '{match}' ({field}, {top_score:.1f}%) — not auto-replaced"
                )
        else:
            ambiguous.append(
                f"possible entity match is ambiguous ({field}, {top_score:.1f}%) between: "
                f"{', '.join(tied)} — not auto-replaced"
            )

    return question, corrections, ambiguous


def rewrite_question(question: str) -> RewrittenQuery:
    llm_fixed = llm_rewrite(question)

    categorical, entities = _load_known_values()

    step1, cat_corrections, cat_ambiguous = fuzzy_correct_categorical(llm_fixed, categorical)
    step2, entity_notes, entity_ambiguous = fuzzy_correct_entities(step1, entities)

    all_corrections = cat_corrections + entity_notes
    if llm_fixed != question:
        all_corrections.insert(0, f"language: '{question}' -> '{llm_fixed}'")

    all_ambiguous = cat_ambiguous + entity_ambiguous

    return RewrittenQuery(
        original=question,
        rewritten=step2,
        corrections=all_corrections,
        ambiguous=all_ambiguous,
        needs_clarification=bool(all_ambiguous),
    )