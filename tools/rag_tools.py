"""Retrieval as the agent uses it: a client's holdings, a question, cited chunks.

``vector_store`` knows how to search the index; this decides what the agent is
allowed to ask for.

Only ``.symbol`` is read off a holding, so the SQL side can keep reshaping
``ClientHolding`` without breaking retrieval. Bare tickers and dict rows work
too.

"No client", "found nothing" and "cover nothing" stay three separate answers --
the store's ``doc_ids=None`` vs ``[]`` distinction, carried up to where it can
be said out loud in ``RAGSearchResult.note``. Data conditions return; a missing or stale
index raises, because a deployment fault dressed up as "no results" is how an
agent ends up telling an advisor there is no research on SABIC.

``PER_DOCUMENT`` caps how much one chatty note can crowd out the others, then
relaxes to fill remaining slots so a single-holding client is not starved.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ingestion.catalog import Catalog, CatalogError, load_catalog
from ingestion.rerank import RerankError, maybe_rerank
from ingestion.vector_store import Hit, VectorStoreError, search
from schemas import DocumentChunk, RAGSearchResult

DEFAULT_K = 5

# At most this many chunks from any one document, while other documents are
# still competing for slots. See the module docstring.
PER_DOCUMENT = 2

# How many candidates to pull before capping. Enough that the cap has something
# to choose from -- and, when a reranker is configured, enough for it to promote
# something the dense search ranked low; small enough that it stays one query.
OVERFETCH = 4


class RagToolError(RuntimeError):
    """Raised when retrieval cannot run at all -- no index, or a stale one.

    Distinct from an empty result, which is a legitimate answer. See the module
    docstring.
    """


@lru_cache(maxsize=1)
def catalog() -> Catalog:
    """The document/instrument join, loaded once per process.

    It is a few kilobytes and every call needs it, so re-reading two CSVs per
    question would be pure waste. ``lru_cache`` does not cache exceptions, so a
    catalogue fixed on disk is picked up by the next call.
    """
    try:
        return load_catalog()
    except CatalogError as exc:
        raise RagToolError(
            f"the document catalogue will not load, so nothing can be "
            f"retrieved:\n{exc}"
        ) from exc


def tickers_of(holdings: Iterable[Any]) -> list[str]:
    """Reduce whatever the SQL side hands over to a list of Tadawul tickers.

    Accepts ``ClientHolding`` objects, plain dict rows, and bare ticker strings.
    Order is preserved and duplicates dropped, so a portfolio holding the same
    name in two accounts does not weight the filter twice.
    """
    seen: dict[str, None] = {}
    for holding in holdings:
        if isinstance(holding, str):
            ticker = holding
        elif isinstance(holding, Mapping):
            ticker = holding.get("symbol") or holding.get("ticker") or ""
        else:
            ticker = getattr(holding, "symbol", "") or getattr(holding, "ticker", "")
        ticker = str(ticker).strip()
        if ticker:
            seen.setdefault(ticker, None)
    return list(seen)


def _diversify(hits: Sequence[Hit], k: int, per_document: int) -> list[Hit]:
    """Take ``k`` hits, spreading them across documents where possible.

    Two passes: the first honours the per-document cap, the second fills any
    remaining slots from what the cap held back. Both walk ``hits`` in score
    order, so the result is still ranked -- the cap changes which chunks are
    present, never the order they appear in.

    Ordering is by ``rank_score``, which is the cross-encoder's score once a
    reranker has run and the cosine score otherwise.
    """
    if per_document <= 0:
        return list(hits[:k])

    kept: list[Hit] = []
    held_back: list[Hit] = []
    per_doc: dict[str, int] = {}
    for hit in hits:
        if per_doc.get(hit.doc_id, 0) < per_document:
            per_doc[hit.doc_id] = per_doc.get(hit.doc_id, 0) + 1
            kept.append(hit)
        else:
            held_back.append(hit)

    kept = kept[:k]
    if len(kept) < k:
        kept.extend(held_back[: k - len(kept)])
    # The top-up appends lower-scoring hits, so re-sort rather than assume.
    return sorted(kept, key=lambda hit: hit.rank_score, reverse=True)


def to_chunk(hit: Hit) -> DocumentChunk:
    """A store ``Hit`` as the typed chunk the rest of the app passes around."""
    isins = str(hit.metadata.get("isins", ""))
    return DocumentChunk(
        text=hit.text,
        source=hit.doc_id,
        # The score that decided the ordering, so the column stays monotonic
        # whether or not a reranker ran.
        score=round(hit.rank_score, 4),
        chunk_id=hit.chunk_id,
        doc_id=hit.doc_id,
        title=hit.title,
        section=hit.section,
        pages=str(hit.metadata.get("pages", "")),
        published_at=str(hit.metadata.get("published_at", "")),
        language=hit.language,
        # Stored comma-joined because Chroma metadata must be scalar.
        isins=[isin for isin in isins.split(",") if isin],
        citation=hit.citation,
    )


def coverage(tickers: Sequence[str]) -> tuple[list[str], list[str], list[str]]:
    """Split tickers into (covered, known-but-uncovered, unknown).

    The three cases mean different things and only the first can be searched.
    "Known but uncovered" is normal -- we do not publish on everything. "Unknown"
    means the SQL side returned a ticker that is not in instruments.csv, which is
    a data fault worth saying out loud rather than silently dropping.
    """
    cat = catalog()
    covered, uncovered, unknown = [], [], []
    for ticker in tickers:
        if cat.instrument_by_ticker(ticker) is None:
            unknown.append(ticker)
        elif cat.doc_ids_for_tickers([ticker]):
            covered.append(ticker)
        else:
            uncovered.append(ticker)
    return covered, uncovered, unknown


def _note(
    covered: list[str],
    uncovered: list[str],
    unknown: list[str],
    found: int,
    scoped: bool = True,
) -> str:
    """The sentence the answering model reads to know what it may claim.

    ``scoped`` is whether a client's holdings restricted the search at all.
    Without it ``covered`` is empty in two unrelated situations -- a client we
    cover no research for, and no client in context -- and the second would be
    reported as the first. The prompt then turns a corpus-wide search that found
    good research into a refusal.

    The total-non-coverage and partial-non-coverage sentences are kept lexically
    distinct on purpose (neither opens with the same phrase) -- a shared prefix
    like "No research covers..." is exactly the kind of surface pattern a model
    following "if the Note says no research covers the holdings, say that" will
    match on without checking whether chunks are actually present above it.
    """
    parts: list[str] = []
    if not scoped:
        if not found:
            parts.append("Nothing in the index answers this question.")
    elif not covered:
        parts.append(
            "No research in the index covers any of this client's holdings, so "
            "there is nothing to cite. Say so rather than answering from "
            "general knowledge."
        )
    elif not found:
        parts.append(
            f"Research covering {', '.join(covered)} was searched but nothing in "
            "it answers this question."
        )
    if uncovered:
        parts.append(
            f"Research is available for {', '.join(covered) or 'other holdings'}. "
            f"Separately, no research is indexed for: {', '.join(uncovered)}."
        )
    if unknown:
        parts.append(
            f"Not in the instrument list, so not searched: {', '.join(unknown)}."
        )
    return " ".join(parts)


def search_research(
    question: str,
    holdings: Iterable[Any] | None = None,
    k: int = DEFAULT_K,
    per_document: int = PER_DOCUMENT,
    path: Path | None = None,
) -> RAGSearchResult:
    """Search the research covering a client's holdings.

    ``holdings=None`` searches the whole corpus -- the right behaviour for a
    general market question with no client in context. An empty ``holdings``
    searches nothing, because a client with no positions has no research to be
    answered from. Those two must not collapse; see the module docstring.

    ``path`` defaults to the configured index and exists so tests can point at
    a temporary one, as everywhere else in ingestion.
    """
    question = question.strip()
    if not question:
        raise RagToolError("a retrieval question cannot be empty")

    if holdings is None:
        doc_ids = None
        covered, uncovered, unknown = [], [], []
    else:
        tickers = tickers_of(holdings)
        if not tickers:
            return RAGSearchResult(
                chunks=[],
                query=question,
                note="No holdings were supplied, so no research was searched.",
            )
        covered, uncovered, unknown = coverage(tickers)
        doc_ids = catalog().doc_ids_for_tickers(covered)

    try:
        hits = search(question, doc_ids, k=max(k * OVERFETCH, k), path=path)
        # Reordering only. The per-document cap below still decides what
        # survives; the reranker decides what it is choosing between.
        hits = maybe_rerank(question, hits)
    except (VectorStoreError, RerankError) as exc:
        raise RagToolError(str(exc)) from exc

    chosen = _diversify(hits, k, per_document)
    return RAGSearchResult(
        chunks=[to_chunk(hit) for hit in chosen],
        query=question,
        searched_tickers=covered,
        uncovered_tickers=uncovered + unknown,
        note=_note(covered, uncovered, unknown, len(chosen), holdings is not None),
    )


def format_context(result: RAGSearchResult) -> str:
    """The retrieved chunks as the block of text a model is given.

    Numbered so the answer can refer to ``[2]``, and each block leads with its
    citation so the source travels with the text instead of being reattached
    afterwards from memory -- which is where invented page numbers come from.
    """
    if not result.chunks:
        return result.note or "No research was retrieved."

    # The heading is the citation only. The chunk body already opens with its
    # own title and section -- the prefix that makes it self-identifying to the
    # embedder -- so repeating them here would say the same thing three times.
    blocks = [
        f"[{number}] {chunk.citation}\n{chunk.text}"
        for number, chunk in enumerate(result.chunks, start=1)
    ]

    context = "\n\n".join(blocks)
    if result.note:
        context += f"\n\nNote: {result.note}"
    return context


def main(argv: list[str] | None = None) -> int:
    """Ask the corpus a question from the command line.

        uv run python -m tools.rag_tools "ما هي مخاطر سابك؟" --ticker 2010
    """
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="tools.rag_tools",
        description="Search the research corpus the way the agent does.",
    )
    parser.add_argument("question")
    parser.add_argument(
        "--ticker", action="append", default=[], metavar="CODE",
        help="restrict to research covering this holding; repeatable. "
             "Omit to search everything.",
    )
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    parser.add_argument("--context", action="store_true",
                        help="print the formatted model context instead of a table")
    args = parser.parse_args(argv)

    try:
        result = search_research(args.question, args.ticker or None, k=args.k)
    except RagToolError as exc:
        print("FAILED")
        print(str(exc))
        return 1

    if args.context:
        print(format_context(result))
        return 0

    for chunk in result.chunks:
        print(f"{chunk.score:6.3f}  {chunk.language}  {chunk.citation}")
        print(f"        {chunk.section or '(no heading)'}")
    if result.note:
        print(f"\n{result.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
