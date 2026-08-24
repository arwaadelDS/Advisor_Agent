"""Embed the chunked corpus and serve ISIN-filtered similarity search.

This is where the document side becomes queryable. Build it with::

    uv run python -m ingestion.build_vector_index

Embeddings are computed here and passed in, never left to Chroma: its default
embedding function is ``all-MiniLM-L6-v2``, which is English-only, and it would
embed half this corpus with a model that has never seen Arabic without erroring.
For the same reason the collection records which model built it and refuses a
query embedded by another one -- a mismatch returns a confident ranking of
unrelated chunks, not an exception.

Filtering is on ``doc_id`` rather than ISIN. Documents and instruments are
many-to-many and Chroma metadata must be scalar, so ``Catalog`` resolves the
join in memory first and the store filter is a single ``$in``.

Query and passage prefixes are a property of the configured model (the e5 family
needs them, bge-m3 does not); getting that wrong costs recall silently.

The index is rebuilt rather than updated -- it takes seconds, and an incremental
path would need to detect which documents changed, which is a source of stale
chunks no test would catch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from config import settings
from ingestion.catalog import Catalog, load_catalog
from ingestion.chunk import Chunk, chunk_corpus
from ingestion.extract import Document, extract_corpus

COLLECTION = "research"

# Cosine, to match the normalised vectors written below. Chroma's default is
# squared L2, which for un-normalised vectors ranks by magnitude as much as by
# direction -- a longer chunk would score differently for being longer.
SPACE = "cosine"

# Chroma writes in batches; 163 chunks fits in one, but the corpus is meant to
# grow and the API caps a single add.
BATCH = 256

# Prefixes the configured embedder expects, if any. Keyed by a substring of the
# model id so a family is matched rather than one pinned revision.
_PREFIXES: dict[str, tuple[str, str]] = {
    "e5": ("query: ", "passage: "),
    "gte": ("", ""),
    "bge-m3": ("", ""),
}


class VectorStoreError(RuntimeError):
    """Raised when the index is missing, stale, or built by another model."""


def prefixes(model_name: str) -> tuple[str, str]:
    """``(query_prefix, passage_prefix)`` for a model id."""
    for key, pair in _PREFIXES.items():
        if key in model_name.lower():
            return pair
    return "", ""


@lru_cache(maxsize=2)
def _model(name: str):
    """The sentence-transformers model, loaded once per process.

    Imported lazily: chunking and extraction must not pay for torch, and
    ``ingestion.doctor`` should still run on a machine where the model has never
    been downloaded.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise VectorStoreError(
            "sentence-transformers is not installed. Run `uv sync`."
        ) from exc
    return SentenceTransformer(name)


def embed(
    texts: list[str], model_name: str | None = None, *, as_query: bool = False
) -> list[list[float]]:
    """Embed passages, or a query, with the configured model.

    Vectors are normalised so that a cosine distance is a plain dot product and
    chunk length cannot influence the ranking.
    """
    model_name = model_name or settings.embedding_model
    query_prefix, passage_prefix = prefixes(model_name)
    prefix = query_prefix if as_query else passage_prefix
    vectors = _model(model_name).encode(
        [prefix + text for text in texts],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in vectors]


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk, with enough metadata to cite it."""

    chunk_id: str
    doc_id: str
    text: str
    score: float  # cosine similarity in [-1, 1]; higher is closer
    metadata: dict[str, Any]

    # Set only once a cross-encoder has re-scored this hit. Kept beside the
    # cosine score rather than replacing it, so a chunk that ranked badly can
    # be traced to the stage that misranked it. See ingestion/rerank.py.
    rerank_score: float | None = None


    @property
    def rank_score(self) -> float:
        """What to order this hit by -- the cross-encoder's score if it ran."""
        return self.score if self.rerank_score is None else self.rerank_score

    @property
    def title(self) -> str:
        return str(self.metadata.get("title", ""))

    @property
    def section(self) -> str:
        return str(self.metadata.get("section", ""))

    @property
    def language(self) -> str:
        return str(self.metadata.get("language", ""))

    @property
    def citation(self) -> str:
        """What the agent should show the advisor under an answer."""
        pages = self.metadata.get("pages", "")
        published = self.metadata.get("published_at", "")
        where = f", page {pages}" if pages else ""
        return f"{self.title}{where} ({published})"


def _client(path: Path):
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    # Chroma reports usage to a third party by default. This index is built
    # from client research, so telemetry is off.
    return chromadb.PersistentClient(
        path=str(path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def build_index(
    chunks: list[Chunk],
    documents: list[Document],
    path: Path | None = None,
    model_name: str | None = None,
) -> int:
    """(Re)build the index from scratch. Returns the number of chunks written.

    Rebuilds rather than updates. At this size it takes seconds, and an
    incremental path would need to detect which documents changed -- a source of
    stale chunks that no test would catch, in exchange for saving no time worth
    having.
    """
    path = Path(settings.vector_store_path) if path is None else path
    model_name = model_name or settings.embedding_model
    if not chunks:
        raise VectorStoreError("refusing to build an empty index")

    by_id = {document.doc_id: document for document in documents}
    missing = {chunk.doc_id for chunk in chunks} - by_id.keys()
    if missing:
        raise VectorStoreError(
            f"chunks reference documents that were not passed in: {sorted(missing)}"
        )

    path.mkdir(parents=True, exist_ok=True)
    client = _client(path)
    if COLLECTION in {c.name for c in client.list_collections()}:
        client.delete_collection(COLLECTION)
    collection = client.create_collection(
        name=COLLECTION,
        metadata={
            "hnsw:space": SPACE,
            "embedding_model": model_name,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )

    for start in range(0, len(chunks), BATCH):
        batch = chunks[start : start + BATCH]
        collection.add(
            ids=[chunk.chunk_id for chunk in batch],
            documents=[chunk.text for chunk in batch],
            embeddings=embed([chunk.text for chunk in batch], model_name),
            metadatas=[chunk.metadata(by_id[chunk.doc_id]) for chunk in batch],
        )
    return len(chunks)


def open_collection(path: Path | None = None, model_name: str | None = None):
    """Open the built index, refusing a model mismatch."""
    path = Path(settings.vector_store_path) if path is None else path
    model_name = model_name or settings.embedding_model
    if not path.is_dir():
        raise VectorStoreError(
            f"no index at {path}. Build it with:\n"
            "  uv run python -m ingestion.build_vector_index"
        )
    client = _client(path)
    try:
        collection = client.get_collection(COLLECTION)
    except Exception as exc:
        raise VectorStoreError(
            f"no '{COLLECTION}' collection in {path}. Rebuild the index with:\n"
            "  uv run python -m ingestion.build_vector_index"
        ) from exc

    built_with = (collection.metadata or {}).get("embedding_model")
    if built_with != model_name:
        # Silent-failure guard: a query embedded by a different model still
        # returns a confident ranking, of the wrong chunks.
        raise VectorStoreError(
            f"the index was built with {built_with!r} but EMBEDDING_MODEL is "
            f"{model_name!r}. Vectors from two models are not comparable -- "
            "queries would return confident nonsense. Rebuild the index."
        )
    return collection


def search(
    query: str,
    doc_ids: Iterable[str] | None = None,
    k: int = 5,
    path: Path | None = None,
    model_name: str | None = None,
) -> list[Hit]:
    """Nearest chunks to ``query``, restricted to ``doc_ids`` when given.

    ``doc_ids=None`` searches everything; an *empty* sequence returns nothing,
    which is the honest answer when a client's holdings resolve to no covered
    document. Those two cases must not collapse into each other -- an unfiltered
    search for a client who holds nothing we cover would answer from another
    client's research.
    """
    if doc_ids is not None:
        doc_ids = list(doc_ids)
        if not doc_ids:
            return []

    collection = open_collection(path, model_name)
    where = {"doc_id": {"$in": doc_ids}} if doc_ids is not None else None
    result = collection.query(
        query_embeddings=embed([query], model_name, as_query=True),
        n_results=min(k, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for chunk_id, text, metadata, distance in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        hits.append(
            Hit(
                chunk_id=chunk_id,
                doc_id=str(metadata.get("doc_id", "")),
                text=text,
                # Chroma returns cosine *distance*; the agent reasons about
                # closeness, so it is flipped once, here.
                score=1.0 - float(distance),
                metadata=dict(metadata),
            )
        )
    return hits


def search_for_tickers(
    query: str,
    tickers: Iterable[str],
    k: int = 5,
    catalog: Catalog | None = None,
    path: Path | None = None,
) -> list[Hit]:
    """Search the research covering a client's holdings.

    The entry point the agent side wants: hand it the tickers the SQL agent
    returned and it resolves them to documents before searching.
    """
    catalog = catalog or load_catalog()
    return search(query, catalog.doc_ids_for_tickers(tickers), k=k, path=path)


def index_status(path: Path | None = None) -> dict[str, Any]:
    """What is in the index right now, for doctor and the CLI."""
    path = Path(settings.vector_store_path) if path is None else path
    if not path.is_dir():
        return {"built": False, "path": str(path)}
    try:
        collection = _client(path).get_collection(COLLECTION)
    except Exception:
        return {"built": False, "path": str(path)}
    metadata = collection.metadata or {}
    return {
        "built": True,
        "path": str(path),
        "chunks": collection.count(),
        "embedding_model": metadata.get("embedding_model"),
        "built_at": metadata.get("built_at"),
        "matches_config": metadata.get("embedding_model") == settings.embedding_model,
    }


def build(path: Path | None = None) -> int:
    """Extract, chunk and index the whole corpus. Returns the chunk count."""
    documents = extract_corpus(load_catalog())
    chunks = chunk_corpus(documents)
    return build_index(chunks, documents, path)


def main(argv: list[str] | None = None) -> int:
    """``uv run python -m ingestion.vector_store`` -- query the built index."""
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="ingestion.vector_store",
        description="Query the research index, optionally as a client's holdings.",
    )
    parser.add_argument("query", nargs="?", help="what to search for")
    parser.add_argument("--ticker", action="append", default=[],
                        help="restrict to research covering this holding (repeatable)")
    parser.add_argument("-k", type=int, default=5, help="how many chunks to return")
    parser.add_argument("--full", action="store_true", help="print whole chunks")
    args = parser.parse_args(argv)

    status = index_status()
    if not status["built"]:
        print(f"No index at {status['path']}.")
        print("Build it with: uv run python -m ingestion.build_vector_index")
        return 1
    print(
        f"{status['chunks']} chunks, built {status['built_at']} "
        f"with {status['embedding_model']}"
    )
    if not args.query:
        return 0

    catalog = load_catalog()
    if args.ticker:
        doc_ids = catalog.doc_ids_for_tickers(args.ticker)
        held = ", ".join(
            catalog.instrument_by_ticker(t).label
            if catalog.instrument_by_ticker(t) else f"{t} (unknown)"
            for t in args.ticker
        )
        print(f"holdings: {held}")
        print(f"filter:   {len(doc_ids)} of {len(catalog.documents)} documents")
    else:
        doc_ids = None

    hits = search(args.query, doc_ids, k=args.k)
    print(f"\nquery:    {args.query!r}\n")
    if not hits:
        print("  (nothing matched)")
        return 0
    for rank, hit in enumerate(hits, start=1):
        print(f"{rank}. {hit.score:.3f}  [{hit.language}]  {hit.chunk_id}")
        print(f"     {hit.citation}  -- {hit.section}")
        body = hit.text if args.full else hit.text.replace("\n", " ")[:150] + "..."
        print(f"     {body}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
