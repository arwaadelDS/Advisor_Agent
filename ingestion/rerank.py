"""Re-score the shortlist with a cross-encoder before the model sees it.

Dense search compares a question vector with a chunk vector, each built without
having seen the other. A cross-encoder reads the pair together and scores it
directly: far more accurate, and far too slow to run over a whole corpus. So it
runs over the shortlist the dense search has already narrowed things down to.

The shortlist has to be wider than the answer. Reranking five chunks can only
reorder five; the point is to pull up something dense retrieval put tenth.
``rag_tools`` already over-fetches to feed its per-document cap, so the
candidates are there to be re-scored.

Off unless ``settings.reranker_model`` is set. The model is a separate download
and every other stage has to keep working on a machine that has never pulled it.

Scores land in ``Hit.rerank_score`` and the cosine score stays where it was, so
a chunk that ranked badly can still be traced to the stage that misranked it.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import Sequence

from config import settings
from ingestion.vector_store import Hit

# Multilingual, and the same family as the bge-m3 embedder. An English-only
# cross-encoder is worse than none here: it would reorder the Arabic half of
# the corpus by nothing at all, confidently.
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

# How many candidates a reranker sees. Measured on data/eval/retrieval.csv:
# ten and twenty score identically (recall@5 100%, MRR 0.97) and five does not
# (96%, 0.96) -- the one document worth rescuing sits between sixth and tenth in
# the dense ranking. Cost is linear in this, so twenty is paid for nothing.
CANDIDATES = 10


class RerankError(RuntimeError):
    """Raised when a reranker was asked for and could not be loaded.

    Distinct from reranking being switched off, which is not an error.
    """


def model_name() -> str:
    """The configured cross-encoder, or ``""`` when reranking is off."""
    return settings.reranker_model.strip()


def is_enabled() -> bool:
    return bool(model_name())


@lru_cache(maxsize=2)
def _cross_encoder(name: str):
    """The cross-encoder, loaded once per process.

    Imported lazily for the same reason the embedder is: chunking, extraction
    and the catalogue must not pay for torch.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover - dependency is declared
        # Carry the original message: this fires for a shadowed module or a
        # broken install just as readily as a missing one, and "run uv sync"
        # sends you the wrong way for both.
        raise RerankError(f"could not import CrossEncoder: {exc}") from exc
    try:
        return CrossEncoder(name)
    except Exception as exc:
        raise RerankError(
            f"could not load the reranker {name!r}: {exc}. It is a separate "
            "download; unset reranker_model to search without it."
        ) from exc


def rerank(
    query: str,
    hits: Sequence[Hit],
    top_n: int | None = None,
    name: str | None = None,
) -> list[Hit]:
    """Re-score ``hits`` against ``query``, best first.

    The returned score is whatever the cross-encoder reports -- bge rerankers
    squash it to [0, 1], others do not. It is an ordering, not a quantity, and
    is only comparable inside one result set.
    """
    hits = list(hits)
    if not hits:
        return []

    encoder = _cross_encoder(name or model_name() or DEFAULT_MODEL)
    scores = encoder.predict(
        [(query, hit.text) for hit in hits], show_progress_bar=False
    )

    scored = [
        replace(hit, rerank_score=float(score)) for hit, score in zip(hits, scores)
    ]
    scored.sort(key=lambda hit: hit.rank_score, reverse=True)
    return scored[:top_n] if top_n else scored


def maybe_rerank(query: str, hits: Sequence[Hit]) -> list[Hit]:
    """``rerank`` when one is configured, otherwise ``hits`` untouched.

    The call sites should not each have to remember that reranking is optional.
    """
    if not is_enabled():
        return list(hits)
    # Deeper than CANDIDATES the cross-encoder costs more without changing the
    # answer, and it is the expensive stage by roughly thirty-five to one.
    return rerank(query, list(hits)[:CANDIDATES])
