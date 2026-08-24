"""Build the research index from the PDFs on disk.

    uv run python -m ingestion.build_vector_index

Runs the whole document pipeline in one pass -- catalogue, extraction,
normalisation, chunking, embedding -- and writes a Chroma index to
``settings.vector_store_path``. Safe to re-run: it rebuilds from scratch.

The index is gitignored on purpose. It is derived from data/documents and the
manifest, so it is reproducible by running this, and a committed index would go
stale silently the moment a document or the chunker changed.

Everything it does is in ingestion/vector_store.py; this is the entry point.
"""

from __future__ import annotations

import sys
import time

from config import settings
from ingestion.catalog import CatalogError, load_catalog
from ingestion.chunk import chunk_corpus
from ingestion.extract import ExtractionError, extract_corpus
from ingestion.pdf_backend import PdfBackendError
from ingestion.vector_store import VectorStoreError, build_index, index_status


def main(argv: list[str] | None = None) -> int:
    import argparse

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="ingestion.build_vector_index",
        description="Extract, chunk and embed the corpus into a Chroma index.",
    )
    parser.add_argument("--status", action="store_true",
                        help="report what is in the index and exit")
    args = parser.parse_args(argv)

    if args.status:
        status = index_status()
        if not status["built"]:
            print(f"No index at {status['path']}.")
            return 1
        for key, value in status.items():
            print(f"  {key:16} {value}")
        return 0 if status["matches_config"] else 1

    started = time.monotonic()
    try:
        catalog = load_catalog()
        print(f"catalogue  {len(catalog.documents)} documents, "
              f"{len(catalog.instruments)} instruments")

        documents = extract_corpus(catalog)
        pages = sum(len(d.pages) for d in documents)
        print(f"extracted  {len(documents)} documents, {pages} pages")

        chunks = chunk_corpus(documents)
        tokens = sum(c.token_count for c in chunks)
        print(f"chunked    {len(chunks)} chunks, {tokens:,} tokens")

        print(f"embedding  {settings.embedding_model} "
              "(first run downloads the model)")
        written = build_index(chunks, documents)
    except (CatalogError, ExtractionError, PdfBackendError, VectorStoreError) as exc:
        print("\nFAILED")
        print(str(exc))
        return 1

    elapsed = time.monotonic() - started
    print(f"indexed    {written} chunks -> {settings.vector_store_path} "
          f"in {elapsed:.0f}s")
    print("\nTry it:")
    print('  uv run python -m ingestion.vector_store "ما هي المخاطر" --ticker 2010')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
