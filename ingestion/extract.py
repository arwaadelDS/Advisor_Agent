"""Turn the PDF corpus into ``Document`` records: clean text plus its metadata.

    pdf_backend.extract_raw -> normalize -> catalog metadata -> Document

Splits on the form feed *before* normalising, since ``normalize`` turns form
feeds into paragraph breaks and the page boundary is unrecoverable afterwards.
Pages are worth keeping: "SABIC note, page 2" is a citation a human can check.

A page's ``language`` means "more Arabic than Latin", nothing stronger -- no
page here is monolingual, and the Arabic share of an "Arabic" page runs
0.59-0.78 against 0.04-0.28 for an English one. Chunking re-classifies with
``classify`` at chunk granularity, or the English disclaimer on page 2 gets
labelled Arabic.

Boilerplate is kept deliberately: extracted text has to correspond to the PDF,
so dropping it is chunking's job. ``repeated_paragraphs`` finds it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ingestion import normalize as N
from ingestion import pdf_backend
from ingestion.catalog import Catalog, ManifestEntry, load_catalog

# Below this, a PDF is almost certainly a scan with no text layer, or an
# extraction that failed without saying so. The shortest real document here is
# around 2,400 characters, so this triggers only on genuine breakage.
MIN_DOCUMENT_CHARS = 400

ARABIC = "ar"
ENGLISH = "en"
UNDETERMINED = "und"


class ExtractionError(RuntimeError):
    """Raised when a document cannot be turned into text we would index."""


def classify(text: str) -> str:
    """``"ar"``, ``"en"`` or ``"und"`` for a page, paragraph or chunk.

    Dominant script by character count, with no threshold to tune: Arabic
    letters against Latin letters, digits and punctuation ignored because they
    are shared. Text with neither is ``"und"`` -- a table of figures, or a page
    that is only a page number.

    Unnormalised Arabic counts too. Callers should normalise first, but a
    caller that forgot should be told the text is Arabic rather than told it is
    undetermined, which would hide the mistake instead of surfacing it.
    """
    counts = N.census(text)
    arabic = counts.arabic_base + counts.arabic_symbols + counts.presentation_forms
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if arabic == latin == 0:
        return UNDETERMINED
    return ARABIC if arabic > latin else ENGLISH


def split_pages(raw: str) -> list[str]:
    """Raw pdftotext output -> one string per page.

    pdftotext writes a form feed *after* every page including the last, so a
    naive split leaves a trailing empty string. Dropping only trailing empties
    keeps a genuinely blank interior page in place, so page numbers still line
    up with the PDF.
    """
    pages = raw.split("\f")
    while pages and not pages[-1].strip():
        pages.pop()
    return pages


@dataclass(frozen=True)
class Page:
    """One page of normalised text."""

    number: int  # 1-based, matching the PDF viewer
    text: str
    language: str

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass(frozen=True)
class Document:
    """A research document with its text read and its instruments resolved."""

    entry: ManifestEntry
    isins: tuple[str, ...]
    pages: tuple[Page, ...]

    @property
    def doc_id(self) -> str:
        return self.entry.doc_id

    @property
    def title(self) -> str:
        return self.entry.title

    @property
    def published_at(self) -> date:
        return self.entry.published_at

    @property
    def text(self) -> str:
        """The whole document, pages joined by a paragraph break."""
        return "\n\n".join(page.text for page in self.pages if not page.is_empty)

    @property
    def detected_languages(self) -> tuple[str, ...]:
        """Languages actually found in the text, in page order, deduplicated."""
        seen: list[str] = []
        for page in self.pages:
            if page.language != UNDETERMINED and page.language not in seen:
                seen.append(page.language)
        return tuple(seen)

    @property
    def language_mismatch(self) -> str | None:
        """Where the manifest and the extracted text disagree, if they do.

        A cross-check, not a formality. The manifest is hand-written, and if it
        claims a document is bilingual while extraction found only English, then
        either the manifest is wrong or a page failed to extract. Both are
        conditions where retrieval quietly answers Arabic questions from
        English text.
        """
        claimed = set(self.entry.languages)
        found = set(self.detected_languages)
        if claimed == found:
            return None
        missing = ", ".join(sorted(claimed - found))
        extra = ", ".join(sorted(found - claimed))
        parts = []
        if missing:
            parts.append(f"manifest claims {missing} but the text has none")
        if extra:
            parts.append(f"the text contains {extra}, unlisted in the manifest")
        return "; ".join(parts)

    def metadata(self) -> dict[str, str | int | bool]:
        """Flat, scalar-only metadata for the vector store.

        Chroma metadata values must be scalars, so the ISIN list is joined for
        display. That is safe here because retrieval does *not* filter on this
        field: ``Catalog.doc_ids_for_isins`` resolves holdings to ``doc_id``
        values in memory first, and the store filter is a single ``$in`` over
        ``doc_id``. Keeping the ISINs anyway means a retrieved chunk can be
        cited without a second lookup.
        """
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "publisher": self.entry.publisher,
            "published_at": self.published_at.isoformat(),
            "doc_type": self.entry.doc_type,
            "classification": self.entry.classification,
            "tickers": ",".join(self.entry.tickers),
            "isins": ",".join(self.isins),
            "is_synthetic": self.entry.is_synthetic,
            "page_count": len(self.pages),
        }


def extract_document(
    entry: ManifestEntry,
    catalog: Catalog,
    backend: pdf_backend.Backend | None = None,
) -> Document:
    """Read one document and enforce the post-conditions ingestion depends on.

    Raises rather than returning a degraded record. Every failure mode here --
    a missing file, an empty text layer, surviving presentation forms -- is one
    that produces an index that looks fine and answers badly.
    """
    path = entry.path(catalog.documents_root)
    if not path.is_file():
        raise ExtractionError(f"{entry.doc_id}: {path} is missing")

    raw = pdf_backend.extract_raw(path, backend)
    pages = []
    for number, page in enumerate(split_pages(raw), start=1):
        text = N.normalize(page)
        pages.append(Page(number=number, text=text, language=classify(text)))

    document = Document(
        entry=entry, isins=catalog.isins_for(entry), pages=tuple(pages)
    )
    text = document.text

    if len(text) < MIN_DOCUMENT_CHARS:
        raise ExtractionError(
            f"{entry.doc_id}: only {len(text)} characters of text "
            f"(expected at least {MIN_DOCUMENT_CHARS}).\n"
            "The PDF probably has no text layer, or the extraction failed silently."
        )

    # The Phase 1 invariant, enforced where documents are actually built rather
    # than only in the inspection CLI. If this ever fires, something upstream
    # changed and the Arabic is about to be embedded as presentation forms.
    counts = N.census(text)
    if not counts.is_clean:
        raise ExtractionError(
            f"{entry.doc_id}: normalisation left {counts.presentation_forms} "
            f"presentation form(s), {counts.bidi_controls} bidi control(s) and "
            f"{counts.zero_width} zero-width character(s) in the text.\n"
            "Indexing this would corrupt Arabic retrieval silently -- see "
            "ingestion/normalize.py."
        )

    if not document.isins:
        raise ExtractionError(
            f"{entry.doc_id}: resolves to no ISIN, so no client holding can "
            "retrieve it. Check its instruments column in data/manifest.csv."
        )

    return document


def extract_corpus(catalog: Catalog | None = None) -> list[Document]:
    """Every document in the manifest, in publication order (oldest first).

    Verifies the pdftotext backend once before reading anything: it is cheaper
    to fail on the probe than to extract sixteen documents with a backend that
    drops Arabic.
    """
    catalog = catalog or load_catalog()
    backend = pdf_backend.get_backend()
    entries = sorted(catalog.documents, key=lambda e: e.published_at)
    return [extract_document(entry, catalog, backend) for entry in entries]


def repeated_paragraphs(
    documents: list[Document], min_documents: int = 3, min_chars: int = 40
) -> list[tuple[int, str]]:
    """Paragraphs that recur across documents, most widespread first.

    Boilerplate: disclaimers, rating-scale definitions, publisher footers. It is
    left in the extracted text on purpose, but chunking should drop it -- a
    paragraph appearing in sixteen documents carries no information that could
    distinguish one from another, and indexing it produces sixteen chunks that
    all match a query equally well and crowd out the real answer.
    """
    seen: dict[str, set[str]] = {}
    for document in documents:
        for page in document.pages:
            for paragraph in page.text.split("\n\n"):
                paragraph = paragraph.strip()
                if len(paragraph) >= min_chars:
                    seen.setdefault(paragraph, set()).add(document.doc_id)
    hits = [
        (len(doc_ids), paragraph)
        for paragraph, doc_ids in seen.items()
        if len(doc_ids) >= min_documents
    ]
    hits.sort(key=lambda item: (-item[0], item[1]))
    return hits


def main(argv: list[str] | None = None) -> int:
    """``uv run python -m ingestion.extract`` -- extract everything and report."""
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="ingestion.extract",
        description="Extract the corpus to normalised text and check it.",
    )
    parser.add_argument("--doc", default=None,
                        help="print one document's text, by doc_id")
    parser.add_argument("--boilerplate", action="store_true",
                        help="list paragraphs repeated across documents")
    args = parser.parse_args(argv)

    catalog = load_catalog()
    documents = extract_corpus(catalog)

    if args.doc:
        matches = [d for d in documents if d.doc_id == args.doc]
        if not matches:
            raise SystemExit(f"no document with doc_id {args.doc!r}")
        (document,) = matches
        for key, value in document.metadata().items():
            print(f"  {key:16} {value}")
        for page in document.pages:
            print(f"\n{'-' * 70}\npage {page.number}  [{page.language}]\n{'-' * 70}")
            print(page.text)
        return 0

    if args.boilerplate:
        hits = repeated_paragraphs(documents)
        print(f"{len(hits)} paragraph(s) repeated across 3+ documents\n")
        for count, paragraph in hits:
            print(f"  in {count:>2} documents  [{classify(paragraph)}]  "
                  f"{len(paragraph):>4} chars")
            print(f"    {paragraph[:120]}...")
        return 0

    print(f"{'document':40} {'pages':>5} {'chars':>7} {'languages':>12}  ISINs")
    for document in documents:
        pages = "/".join(page.language for page in document.pages)
        print(
            f"{document.doc_id:40} {len(document.pages):>5} {len(document.text):>7} "
            f"{pages:>12}  {len(document.isins)}"
        )

    mismatches = [(d.doc_id, d.language_mismatch) for d in documents
                  if d.language_mismatch]
    print()
    for doc_id, problem in mismatches:
        print(f"  warning: {doc_id}: {problem}")

    total = sum(len(d.text) for d in documents)
    print(
        f"\n{len(documents)} documents, {total:,} characters, "
        f"{sum(len(d.pages) for d in documents)} pages"
    )
    print(f"{len(repeated_paragraphs(documents))} repeated paragraph(s) "
          "(see --boilerplate)")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
