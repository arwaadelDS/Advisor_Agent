"""
Test suite for tools/query_rewriter.py

Three layers:

1. REGRESSION — hand-picked known cases against your real seeded DB. Anchors:
   if one of these breaks, something in the core logic broke.

2. GENERIC / DB-DRIVEN — derived programmatically from whatever is currently
   in the DB, split by qr._tied_top_matches (the SAME tie-detection the
   production code uses — not a re-implementation in the test) into:
     - unambiguous cases: must recover / flag the original value
     - ambiguous cases: must NOT guess a single winner; must be reported
       via the `ambiguous` list instead

3. Diagnostics with no real invariant to assert (ticker digit-mutation risk,
   threshold boundary probe) stay print-only.

Requires a live, seeded DB reachable via tools.db.agent_engine.
"""

import pytest
from rapidfuzz import fuzz
from tools import query_rewriter as qr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_llm(monkeypatch):
    class FakeResponse:
        def __init__(self, content):
            self.content = content

    class FakeLLM:
        def __init__(self):
            self.last_messages = None
            self.next_response = None

        def invoke(self, messages):
            self.last_messages = messages
            return FakeResponse(self.next_response)

    fake = FakeLLM()
    monkeypatch.setattr(qr, "get_llm", lambda: fake)
    return fake


CATEGORICAL, ENTITIES = qr._load_known_values()
ALL_CATEGORICAL_VALUES = [v for values in CATEGORICAL.values() for v in values]


def _drop_middle_char(value: str) -> str:
    """Produce a plausible 1-edit typo by removing the middle character."""
    mid = len(value) // 2
    return value[:mid] + value[mid + 1:]


# ---------------------------------------------------------------------------
# REGRESSION — existing known-case tests, updated for the 3-tuple return
# ---------------------------------------------------------------------------

def test_load_known_values_from_real_db():
    categorical, entities = qr._load_known_values()

    print("\n--- categorical values loaded ---")
    for field, values in categorical.items():
        print(f"{field}: {values}")

    print("\n--- entity values loaded ---")
    for field, values in entities.items():
        print(f"{field}: {values}")

    assert "C001" in categorical["client_id"]
    assert len(categorical["client_id"]) == 18
    assert "Energy" in categorical["sector"]
    assert "Banking" in categorical["sector"]
    assert set(categorical["risk_profile"]) == {"Conservative", "Balanced", "Aggressive"}
    assert set(categorical["aum_tier"]) == {"HNW", "Ultra-HNW"}
    assert "Saudi Aramco" in entities["instrument_name_en"]


def test_fixes_client_id_casing_real_data():
    result, corrections, ambiguous = qr.fuzzy_correct_categorical("show holdings for c007", CATEGORICAL)
    print(f"\nresult: {result}")
    print(f"corrections: {corrections}")
    print(f"ambiguous: {ambiguous}")

    assert "C007" in result
    assert len(corrections) == 1
    assert ambiguous == []


def test_fixes_real_sector_typo():
    result, corrections, ambiguous = qr.fuzzy_correct_categorical(
        "compare Energey vs Bnaking sectors", CATEGORICAL
    )
    print(f"\nresult: {result}")
    print(f"corrections: {corrections}")
    print(f"ambiguous: {ambiguous}")

    assert "Energy" in result
    assert ambiguous == []


def test_leaves_correct_value_unchanged_real_data():
    result, corrections, ambiguous = qr.fuzzy_correct_categorical("show Energy holdings", CATEGORICAL)
    assert result == "show Energy holdings"
    assert corrections == []
    assert ambiguous == []


def test_no_false_positive_on_plain_words():
    result, corrections, ambiguous = qr.fuzzy_correct_categorical("show all holdings please", CATEGORICAL)
    print(f"\nresult: {result}")
    print(f"corrections: {corrections}")
    assert corrections == []
    assert ambiguous == []


def test_entity_match_flagged_real_data():
    result, corrections, ambiguous = qr.fuzzy_correct_entities("show holdings in Aramco", ENTITIES)
    print(f"\ncorrections: {corrections}")
    print(f"ambiguous: {ambiguous}")
    assert result == "show holdings in Aramco"
    assert len(corrections) == 1
    assert "Saudi Aramco" in corrections[0]
    assert ambiguous == []


def test_llm_rewrite_sends_question_and_returns_content(fake_llm):
    fake_llm.next_response = "show holdings for client C007"
    result = qr.llm_rewrite("shwo hodings for cleint c007")

    assert result == "show holdings for client C007"
    assert fake_llm.last_messages[1]["content"] == "shwo hodings for cleint c007"


def test_rewrite_question_full_pipeline_real_db(fake_llm):
    fake_llm.next_response = "show holdings for client c007 in Energey sector"

    result = qr.rewrite_question("shwo hodings for cleint c007 in Energey sector")

    print(f"\noriginal:  {result.original}")
    print(f"rewritten: {result.rewritten}")
    print(f"corrections: {result.corrections}")
    print(f"ambiguous: {result.ambiguous}")
    print(f"needs_clarification: {result.needs_clarification}")

    assert "C007" in result.rewritten
    assert "Energy" in result.rewritten
    assert any("language:" in c for c in result.corrections)
    assert result.needs_clarification is False


def test_rewrite_question_flags_ambiguous_input_for_clarification(fake_llm):
    """An input built to collide (a client-id fragment shared by two real
    IDs) should come back needing clarification, not a confident guess."""
    fake_llm.next_response = "show holdings for client C01"

    result = qr.rewrite_question("show holdings for client C01")

    print(f"\nrewritten: {result.rewritten}")
    print(f"ambiguous: {result.ambiguous}")
    print(f"needs_clarification: {result.needs_clarification}")

    assert result.needs_clarification is True
    assert result.ambiguous != []


# ---------------------------------------------------------------------------
# GENERIC — derived from the live DB, split by real (not approximated)
# ambiguity using the production tie-detection function.
# ---------------------------------------------------------------------------

CATEGORICAL_TYPO_CASES = []
CATEGORICAL_TYPO_AMBIGUOUS = []

for field, values in CATEGORICAL.items():
    if field == "ticker":
        continue
    for value in values:
        if len(value) < 4:
            continue
        typo = _drop_middle_char(value)
        tied, score = qr._tied_top_matches(typo, ALL_CATEGORICAL_VALUES, fuzz.ratio, str.lower)
        if tied == {value}:
            CATEGORICAL_TYPO_CASES.append((field, value, typo))
        elif len(tied) > 1:
            CATEGORICAL_TYPO_AMBIGUOUS.append((field, value, typo, sorted(tied)))
        # tied == {some other value}: a clean miss unrelated to ambiguity;
        # not expected given how the typo was generated, so not bucketed.


@pytest.mark.parametrize("field,original,typo", CATEGORICAL_TYPO_CASES)
def test_categorical_typo_recovery_generic(field, original, typo):
    """For every real categorical value whose one-character-dropped typo is
    an empirically unambiguous top match, that typo should fuzzy-correct
    back to the original."""
    question = f"show records where {field} is {typo}"
    result, corrections, ambiguous = qr.fuzzy_correct_categorical(question, CATEGORICAL)
    assert original in result, f"{typo!r} (typo of {original!r}) did not correct back; got {corrections}"
    assert ambiguous == []


@pytest.mark.parametrize("field,original,typo,tied_with", CATEGORICAL_TYPO_AMBIGUOUS)
def test_categorical_ambiguous_typo_not_guessed(field, original, typo, tied_with):
    """When a typo ties in score between two or more real values, the
    function must NOT silently guess one — the token stays uncorrected and
    the ambiguity is surfaced instead."""
    question = f"show records where {field} is {typo}"
    result, corrections, ambiguous = qr.fuzzy_correct_categorical(question, CATEGORICAL)
    print(f"\nfield={field} original={original!r} typo={typo!r} tied_with={tied_with} "
          f"-> result={result!r} corrections={corrections} ambiguous={ambiguous}")

    assert typo in result, f"expected {typo!r} left uncorrected, got {result!r}"
    assert any(typo in a for a in ambiguous), f"expected an ambiguity note for {typo!r}, got {ambiguous}"
    for candidate in tied_with:
        assert any(candidate in a for a in ambiguous), f"expected {candidate!r} listed in {ambiguous}"


@pytest.mark.parametrize("ticker", CATEGORICAL.get("ticker", []))
def test_ticker_exact_match_not_altered(ticker):
    """Exact real tickers should never get 'corrected' to a different real
    ticker — this would silently change which instrument a query targets."""
    question = f"show holdings for ticker {ticker}"
    result, corrections, ambiguous = qr.fuzzy_correct_categorical(question, CATEGORICAL)
    assert ticker in result
    assert corrections == [], f"unexpected correction fired on an exact ticker: {corrections}"


def test_ticker_single_digit_typo_diagnostic():
    """Diagnostic, not pass/fail: prints whether a ticker one digit off from
    a real value gets confidently 'corrected' to a *different* real ticker
    (a unique-but-wrong top match, which ambiguity detection alone can't
    catch since there's no tie). Read the printed output — if this fires,
    ticker should probably be dropped from fuzzy correction entirely."""
    tickers = [t for t in CATEGORICAL.get("ticker", []) if t.isdigit() and len(t) >= 2]
    for real in tickers:
        last_digit = int(real[-1])
        mutated = real[:-1] + str((last_digit + 1) % 10)
        question = f"show holdings for ticker {mutated}"
        result, corrections, ambiguous = qr.fuzzy_correct_categorical(question, CATEGORICAL)
        print(f"\nreal={real}  mutated={mutated}  result={result!r}  corrections={corrections}  ambiguous={ambiguous}")


def test_threshold_boundary_probe_generic():
    """Diagnostic, not pass/fail: for each categorical value, prints the
    score at 1 and 2 dropped characters so you can judge whether
    threshold=85 is actually a sane cutoff."""
    for field, values in CATEGORICAL.items():
        if field == "ticker":
            continue
        for value in values:
            if len(value) < 6:
                continue
            variants = {
                "-1 char": _drop_middle_char(value),
                "-2 char": _drop_middle_char(_drop_middle_char(value)),
            }
            for label, variant in variants.items():
                question = f"show {field} {variant}"
                result, corrections, ambiguous = qr.fuzzy_correct_categorical(question, CATEGORICAL)
                print(f"\n{field}={value!r} {label}={variant!r} -> result={result!r} "
                      f"corrections={corrections} ambiguous={ambiguous}")


ENTITY_PARTIAL_CASES = []
ENTITY_PARTIAL_AMBIGUOUS = []

for field, values in ENTITIES.items():
    multi_word = [v for v in values if len(v.split()) > 1]
    for v in multi_word:
        last = v.split()[-1]
        if len(last) < 4:
            continue
        tied, score = qr._tied_top_matches(last, values, fuzz.partial_ratio)
        if tied == {v}:
            ENTITY_PARTIAL_CASES.append((field, v, last))
        elif len(tied) > 1:
            ENTITY_PARTIAL_AMBIGUOUS.append((field, v, last, sorted(tied)))


@pytest.mark.parametrize("field,full_value,partial", ENTITY_PARTIAL_CASES)
def test_entity_partial_name_flagged_generic(field, full_value, partial):
    """For every real multi-word entity whose last word is an empirically
    unambiguous top match within its field, asking about just that last
    word should flag the full value as a possible match."""
    question = f"show holdings related to {partial}"
    result, corrections, ambiguous = qr.fuzzy_correct_entities(question, ENTITIES)
    found = any(full_value.lower() in c.lower() for c in corrections)
    assert found, f"expected a flag for {full_value!r} from partial {partial!r}, got {corrections}"
    assert ambiguous == []


@pytest.mark.parametrize("field,full_value,partial,shared_by", ENTITY_PARTIAL_AMBIGUOUS)
def test_entity_ambiguous_partial_flags_all_candidates(field, full_value, partial, shared_by):
    """When a fragment ties in score between multiple real entities, the
    function must flag the ambiguity — listing every tied candidate —
    instead of silently returning one as if it were confident."""
    question = f"show holdings related to {partial}"
    result, corrections, ambiguous = qr.fuzzy_correct_entities(question, ENTITIES)
    print(f"\nfield={field} full_value={full_value!r} partial={partial!r} shared_by={shared_by} "
          f"-> corrections={corrections} ambiguous={ambiguous}")

    assert corrections == [], f"expected no confident single correction, got {corrections}"
    assert ambiguous != [], f"expected an ambiguity note for {partial!r}"
    note = ambiguous[0]
    for candidate in shared_by:
        assert candidate in note, f"expected {candidate!r} listed among tied candidates in {note!r}"


@pytest.mark.parametrize("field,fragment", [
    (field, value[:2])
    for field, values in CATEGORICAL.items()
    for value in values
    if len(value) >= 6
])
def test_categorical_short_fragment_diagnostic(field, fragment):
    """Diagnostic, not pass/fail: a short, ambiguous fragment of a real
    value being the closest match among a small vocabulary isn't the same
    as it being confidently a typo of that value."""
    question = f"show records where {field} is {fragment}"
    result, corrections, ambiguous = qr.fuzzy_correct_categorical(question, CATEGORICAL)
    print(f"\nfield={field} fragment={fragment!r} -> result={result!r} "
          f"corrections={corrections} ambiguous={ambiguous}")


@pytest.mark.parametrize("clean_question", [
    "show all holdings",
    "what is the total portfolio value",
    "list clients by tier",
    "compare holdings across sectors",
    "how many instruments are in the database",
    "show the largest holding by value",
    "what is the average market value per client",
    "list all instruments",
])
def test_no_spurious_corrections_on_clean_questions(clean_question):
    """False-positive control: plain questions with no typos and no entity
    mentions should trigger zero corrections and zero ambiguity notes."""
    cat_result, cat_corrections, cat_ambiguous = qr.fuzzy_correct_categorical(clean_question, CATEGORICAL)
    _, entity_corrections, entity_ambiguous = qr.fuzzy_correct_entities(clean_question, ENTITIES)
    print(f"\ninput: {clean_question}")
    print(f"categorical corrections: {cat_corrections}  ambiguous: {cat_ambiguous}")
    print(f"entity corrections: {entity_corrections}  ambiguous: {entity_ambiguous}")
    assert cat_corrections == [], f"false positive categorical correction(s): {cat_corrections}"
    assert cat_ambiguous == [], f"false positive categorical ambiguity flag(s): {cat_ambiguous}"
    assert entity_corrections == [], f"false positive entity flag(s): {entity_corrections}"
    assert entity_ambiguous == [], f"false positive entity ambiguity flag(s): {entity_ambiguous}"