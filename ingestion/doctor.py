"""Pre-flight check for the document ingestion toolchain.

Run this before building an index, and when a teammate first clones the repo::

    uv run python -m ingestion.doctor

It answers the two questions that decide whether an index built here would be
quietly wrong:

* Will this machine extract the Arabic in data/documents, or silently drop it?
  See ingestion/pdf_backend.py for why that is a real risk rather than a
  theoretical one.
* Does every document still join to an instrument? A document whose ticker is
  not in instruments.csv is never retrieved and never complained about --
  see ingestion/catalog.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

from config import settings
from ingestion import pdf_backend
from ingestion.catalog import CatalogError, load_catalog
from ingestion.pdf_backend import PdfBackendError

DOCS_DIR = Path("data/documents")
MANIFEST = Path("data/manifest.csv")
INSTRUMENTS = Path("data/mock/instruments.csv")


def _out(line: str = "") -> None:
    print(line)


def report_backends() -> pdf_backend.Backend:
    found = pdf_backend.discover()
    chosen = pdf_backend.resolve()

    _out("pdftotext builds found")
    if not found:
        _out("  (none)")
    for backend in found:
        mark = "->" if backend.exe == chosen.exe else "  "
        _out(f"  {mark} {backend.flavour:8} {backend.version:10} {backend.exe}")

    _out()
    if settings.poppler_path:
        _out(f"Using (pinned via poppler_path): {chosen}")
    else:
        _out(f"Using (auto-discovered): {chosen}")

    # The situation this whole module exists for: the right build is installed
    # but a different one shadows it on PATH.
    also_xpdf = [b for b in found if b.exe != chosen.exe and b.flavour == "xpdf"]
    if also_xpdf and chosen.flavour == "poppler" and not settings.poppler_path:
        _out()
        _out("  Note: an Xpdf build is also installed. It is selected against")
        _out("  here, but anything calling pdftotext straight off PATH may still")
        _out("  reach it. Pin poppler_path in .env to remove the ambiguity.")

    if chosen.flavour == "xpdf":
        _out()
        _out("  Warning: this is an Xpdf build. It only keeps Arabic because we")
        _out("  pass -enc UTF-8 explicitly. Prefer poppler where available.")

    return chosen


def report_extraction(backend: pdf_backend.Backend) -> None:
    result = pdf_backend.check(backend)
    pct = 100.0 * result.arabic_chars / max(result.total_chars, 1)
    _out()
    _out(f"Arabic extraction check on {result.probe.name}")
    _out(f"  {result.total_chars} characters extracted")
    _out(f"  {result.arabic_chars} of them Arabic ({pct:.0f}%)")
    _out("  OK -- the backend decodes Arabic rather than dropping it.")


def report_corpus() -> int:
    """Report the corpus and the document/instrument join. Returns an exit code."""
    _out()
    _out("Corpus")
    pdfs = sorted(DOCS_DIR.glob("*.pdf")) if DOCS_DIR.is_dir() else []
    _out(f"  {len(pdfs)} PDFs in {DOCS_DIR}")
    for path, label in ((MANIFEST, "manifest"), (INSTRUMENTS, "instruments")):
        state = "present" if path.is_file() else "MISSING"
        _out(f"  {label}: {state} ({path})")

    try:
        catalog = load_catalog(strict=False)
    except CatalogError as exc:
        _out()
        _out("FAILED")
        _out(str(exc))
        return 1

    report = catalog.validate()
    _out(
        f"  {len(catalog.documents)} manifest rows joined to "
        f"{len(catalog.instruments)} instruments"
    )
    coverage = [
        len(catalog.documents_for_instrument(inst.isin)) for inst in catalog.instruments
    ]
    if coverage:
        # How much corpus survives the ISIN filter for a single holding. If the
        # floor is 0, some client's question has nothing to retrieve from.
        _out(
            f"  {min(coverage)}-{max(coverage)} documents per instrument "
            f"({sum(coverage)} links in total)"
        )

    for warning in report.warnings:
        _out(f"  warning: {warning}")
    for error in report.errors:
        _out(f"  ERROR:   {error}")
    return 0 if report.ok else 1


def main() -> int:
    # These docs are Arabic; a cp1252 console would raise on the first print.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        backend = report_backends()
        report_extraction(backend)
    except PdfBackendError as exc:
        _out()
        _out("FAILED")
        _out(str(exc))
        return 1

    code = report_corpus()
    _out()
    _out("Toolchain and catalogue OK." if code == 0
         else "Toolchain OK, but the catalogue has errors -- see above.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
