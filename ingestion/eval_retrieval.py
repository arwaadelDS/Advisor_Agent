"""Measure whether retrieval finds the right document::

    uv run python -m ingestion.eval_retrieval

Everything upstream is tested for correctness. None of it says the search is any
good -- a pipeline can be right in every component and still rank the wrong note
first.

The question set is data/eval/retrieval.csv so it can grow without touching
code. A row may name two acceptable documents where the corpus genuinely has
two.

The headline is recall@5, but the split worth watching is by question language:
if Arabic drops well below English the fault is upstream, in extraction,
normalisation or the embedder, and no prompt work will fix it.

``--rerank`` scores the same questions through the cross-encoder, so the two
runs are directly comparable and reranking has to earn its place rather than be
assumed to help.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from ingestion.rerank import CANDIDATES, DEFAULT_MODEL, RerankError, rerank
from ingestion.vector_store import VectorStoreError, search

QUESTIONS_PATH = Path("data/eval/retrieval.csv")

# Reported at 1, 3 and 5. One is what a "just show me the answer" UI shows;
# five is what the agent actually puts in front of the model.
CUTOFFS = (1, 3, 5)


@dataclass(frozen=True)
class Question:
    question: str
    expect: tuple[str, ...]
    language: str
    why: str


@dataclass(frozen=True)
class Result:
    question: Question
    ranked: tuple[str, ...]  # doc_ids in rank order, deduplicated

    @property
    def rank(self) -> int | None:
        """1-based rank of the first acceptable document, or None if unfound."""
        for position, doc_id in enumerate(self.ranked, start=1):
            if doc_id in self.question.expect:
                return position
        return None

    def hit_at(self, k: int) -> bool:
        return self.rank is not None and self.rank <= k


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    """Read the labelled set. ``expect`` is ``;``-separated."""
    if not path.is_file():
        raise FileNotFoundError(f"no evaluation set at {path}")
    # utf-8-sig for the same reason as the catalogue: Excel writes a BOM.
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        Question(
            question=row["question"].strip(),
            expect=tuple(p.strip() for p in row["expect"].split(";") if p.strip()),
            language=row["language"].strip(),
            why=row.get("why", "").strip(),
        )
        for row in rows
        if row.get("question", "").strip()
    ]


def evaluate(
    questions: list[Question],
    k: int = 5,
    path: Path | None = None,
    reranker: str | None = None,
    candidates: int = CANDIDATES,
) -> list[Result]:
    """Run every question unfiltered and record where the right document ranked.

    Deliberately unfiltered: the holdings filter can only ever help, so
    measuring with it on would flatter the ranking. This is the honest number.

    ``reranker`` names a cross-encoder to re-score the shortlist with. It is a
    model id rather than a flag so the report can say which model produced the
    number. ``candidates`` is how deep a shortlist it sees, which is the knob
    that trades its cost against what it is able to rescue.
    """
    results = []
    for question in questions:
        # A reranker can only promote what it is shown, so the shortlist has to
        # be wider than k whenever one is running.
        depth = max(candidates, k) if reranker else k
        hits = search(question.question, None, k=depth, path=path)
        if reranker:
            hits = rerank(question.question, hits, top_n=k, name=reranker)
        ranked: list[str] = []
        for hit in hits:  # several chunks can share a document; rank documents
            if hit.doc_id not in ranked:
                ranked.append(hit.doc_id)
        results.append(Result(question=question, ranked=tuple(ranked)))
    return results


def recall_at(results: list[Result], k: int) -> float:
    if not results:
        return 0.0
    return sum(r.hit_at(k) for r in results) / len(results)


def mrr(results: list[Result]) -> float:
    """Mean reciprocal rank -- rewards being first, not merely present."""
    if not results:
        return 0.0
    return sum(1.0 / r.rank if r.rank else 0.0 for r in results) / len(results)


def _line(label: str, results: list[Result]) -> str:
    scores = "  ".join(f"@{k} {recall_at(results, k):5.0%}" for k in CUTOFFS)
    return f"  {label:10} n={len(results):<3} {scores}   MRR {mrr(results):.2f}"


def report(results: list[Result], note: str = "") -> str:
    lines = [f"Retrieval quality{note}", _line("overall", results)]

    languages = sorted({r.question.language for r in results})
    if len(languages) > 1:
        for language in languages:
            lines.append(_line(language, [r for r in results if r.question.language == language]))

    missed = [r for r in results if not r.hit_at(max(CUTOFFS))]
    if missed:
        lines.append("")
        lines.append(f"Missed at {max(CUTOFFS)} ({len(missed)}):")
        for result in missed:
            lines.append(f"  {result.question.question}")
            lines.append(f"    wanted {'/'.join(result.question.expect)}")
            lines.append(f"    got    {', '.join(result.ranked[:3]) or '(nothing)'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="ingestion.eval_retrieval",
        description="Score retrieval against the labelled question set.",
    )
    parser.add_argument("-k", type=int, default=max(CUTOFFS))
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument(
        "--min-recall", type=float, default=0.0, metavar="F",
        help="exit non-zero if recall@%d falls below this (e.g. 0.9)" % max(CUTOFFS),
    )
    parser.add_argument(
        "--rerank", nargs="?", const=DEFAULT_MODEL, default=None, metavar="MODEL",
        help=f"re-score the shortlist with a cross-encoder (default {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--candidates", type=int, default=CANDIDATES, metavar="N",
        help="how many hits the reranker sees; its cost scales with this",
    )
    args = parser.parse_args(argv)

    try:
        questions = load_questions(args.questions)
        results = evaluate(
            questions, k=args.k, reranker=args.rerank, candidates=args.candidates
        )
    except (FileNotFoundError, VectorStoreError, RerankError) as exc:
        print("FAILED")
        print(str(exc))
        return 1

    note = f"  (reranked with {args.rerank}, top {args.candidates})" if args.rerank else ""
    print(report(results, note))
    achieved = recall_at(results, max(CUTOFFS))
    if achieved < args.min_recall:
        print(f"\nBelow the {args.min_recall:.0%} floor.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
