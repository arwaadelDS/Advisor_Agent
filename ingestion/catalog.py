"""Load the document manifest and the instrument list, and join them.

Why this module exists
----------------------
The agent's retrieval step is not "search the documents". It is "search the
documents *that belong to this client's holdings*". The SQL side returns
positions identified by Tadawul ticker; the research corpus is identified by
filename. Neither knows about the other. This module is the join between them::

    holding (ticker 2010) -> instruments.csv -> ISIN SA0007879121
                                             -> manifest.csv -> two documents

Two things make that join worth its own module rather than a dict comprehension
somewhere in the retriever.

**ISINs cannot be read out of the documents.** The four sector notes mention no
ISIN at all, and the company notes print them inside bidi-wrapped Arabic
paragraphs. Scraping them from text would silently under-link exactly the
documents that cover several companies at once. The manifest is the authority.

**A broken link is invisible at runtime.** If a manifest row names a ticker that
instruments.csv does not list, that document is simply never retrieved. No
error, no empty result -- the advisor gets an answer built from the *other*
documents and no indication that one was skipped. So the links are validated up
front, loudly, instead of being discovered in a demo.

Ownership
---------
``data/manifest.csv`` is ours. ``data/mock/instruments.csv`` belongs to the SQL
side and will be replaced, so this module validates its shape on every load and
names the offending column when it does not match, rather than raising a
``KeyError`` three call frames deeper.

Both files are read as UTF-8 with a tolerated BOM: they contain Arabic names,
they get opened in Excel, and Python on Windows would otherwise decode them with
the ANSI code page and mangle every Arabic column without erroring.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

MANIFEST_PATH = Path("data/manifest.csv")
INSTRUMENTS_PATH = Path("data/mock/instruments.csv")
DOCUMENTS_ROOT = Path("data/documents")

MANIFEST_COLUMNS = frozenset(
    {
        "filename",
        "title",
        "publisher",
        "published_at",
        "language",
        "doc_type",
        "classification",
        "instruments",
        "is_synthetic",
    }
)
INSTRUMENT_COLUMNS = frozenset(
    {"isin", "ticker", "name_en", "name_ar", "sector", "asset_class", "shariah_flag"}
)

# Two letters of country code, nine alphanumerics, one check digit.
_ISIN_SHAPE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")

_LIST_SEPARATOR = ";"
_TRUTHY = frozenset({"true", "yes", "1", "y"})


class CatalogError(RuntimeError):
    """Raised when the catalogue cannot be loaded, or is loaded strict and invalid."""


def _split(value: str) -> tuple[str, ...]:
    """``"1120;1180"`` -> ``("1120", "1180")``, tolerating stray whitespace."""
    return tuple(part.strip() for part in value.split(_LIST_SEPARATOR) if part.strip())


def isin_check_digit(isin: str) -> str:
    """The digit an ISIN *should* end with, per ISO 6166.

    Letters expand to their two-digit ordinal (A=10 ... Z=35), then a Luhn sum
    over the resulting digit string. Catches single-character typos and adjacent
    transpositions -- the two ways an ISIN gets copied wrong by hand.
    """
    digits = "".join(str(ord(c) - 55) if c.isalpha() else c for c in isin[:-1])
    total = 0
    double = True
    for char in reversed(digits):
        value = int(char)
        if double:
            value *= 2
            if value > 9:
                value -= 9
        total += value
        double = not double
    return str((10 - total % 10) % 10)


def isin_is_valid(isin: str) -> bool:
    return bool(_ISIN_SHAPE.match(isin)) and isin[-1] == isin_check_digit(isin)


@dataclass(frozen=True)
class Instrument:
    """One tradeable security, as the SQL side and the advisor both see it."""

    isin: str
    ticker: str
    name_en: str
    name_ar: str
    sector: str
    asset_class: str
    shariah_flag: str | None  # unpopulated in the draft file; None, not ""

    @property
    def label(self) -> str:
        """How this instrument should read in a citation or a log line."""
        return f"{self.name_en} ({self.ticker})"


@dataclass(frozen=True)
class ManifestEntry:
    """One research document, before its text has been read."""

    filename: str
    title: str
    publisher: str
    published_at: date
    languages: tuple[str, ...]
    doc_type: str
    classification: str
    tickers: tuple[str, ...]
    is_synthetic: bool

    @property
    def doc_id(self) -> str:
        """Stable identifier used everywhere downstream: the filename stem.

        Human-readable on purpose. It ends up in chunk metadata, in retrieval
        filters and in citations, and a reviewer should be able to tell which
        document a chunk came from without a lookup.
        """
        return Path(self.filename).stem

    def path(self, root: Path = DOCUMENTS_ROOT) -> Path:
        return root / self.filename


@dataclass(frozen=True)
class CatalogReport:
    """What validation found. Errors block ingestion; warnings do not."""

    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        if self.ok and not self.warnings:
            return "catalogue consistent"
        parts = []
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        return ", ".join(parts)


def _read_rows(path: Path, required: frozenset[str], what: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise CatalogError(f"{what} not found at {path} (run from the repo root).")

    # utf-8-sig, not utf-8: Excel writes a BOM, and a BOM on the header line
    # turns "isin" into "﻿isin" and every lookup into a KeyError.
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = required - columns
        if missing:
            raise CatalogError(
                f"{what} at {path} is missing column(s): {', '.join(sorted(missing))}.\n"
                f"Found: {', '.join(sorted(columns)) or '(no header row)'}"
            )
        rows = [
            {key: (value or "").strip() for key, value in row.items() if key is not None}
            for row in reader
        ]

    if not rows:
        raise CatalogError(f"{what} at {path} has a header but no rows.")
    return rows


def load_instruments(path: Path = INSTRUMENTS_PATH) -> tuple[Instrument, ...]:
    rows = _read_rows(path, INSTRUMENT_COLUMNS, "instruments file")
    return tuple(
        Instrument(
            isin=row["isin"].upper(),
            # Deliberately a string. Tadawul codes are identifiers, not numbers:
            # int() would drop a leading zero and make the join miss silently.
            ticker=row["ticker"],
            name_en=row["name_en"],
            name_ar=row["name_ar"],
            sector=row["sector"],
            asset_class=row["asset_class"],
            shariah_flag=row["shariah_flag"] or None,
        )
        for row in rows
    )


def load_manifest(path: Path = MANIFEST_PATH) -> tuple[ManifestEntry, ...]:
    rows = _read_rows(path, MANIFEST_COLUMNS, "manifest")
    entries = []
    for line_no, row in enumerate(rows, start=2):  # +1 header, +1 to 1-base
        try:
            published_at = date.fromisoformat(row["published_at"])
        except ValueError as exc:
            raise CatalogError(
                f"{path} line {line_no}: published_at {row['published_at']!r} "
                "is not an ISO date (YYYY-MM-DD)."
            ) from exc
        entries.append(
            ManifestEntry(
                filename=row["filename"],
                title=row["title"],
                publisher=row["publisher"],
                published_at=published_at,
                languages=_split(row["language"]),
                doc_type=row["doc_type"],
                classification=row["classification"],
                tickers=_split(row["instruments"]),
                is_synthetic=row["is_synthetic"].lower() in _TRUTHY,
            )
        )
    return tuple(entries)


class Catalog:
    """The joined view of documents and instruments.

    Built once at ingestion time and kept in memory -- it is a few kilobytes.
    Retrieval uses it to turn a client's ISINs into the set of ``doc_id`` values
    worth searching, which is what keeps the vector-store filter to a single
    ``$in`` over a scalar field instead of a multi-valued metadata match.
    """

    def __init__(
        self,
        documents: tuple[ManifestEntry, ...],
        instruments: tuple[Instrument, ...],
        documents_root: Path = DOCUMENTS_ROOT,
    ) -> None:
        self.documents = documents
        self.instruments = instruments
        self.documents_root = documents_root

        # Last row wins on a duplicate key; validate() reports the duplicate
        # separately rather than letting the index quietly disagree with the file.
        self._by_isin = {inst.isin: inst for inst in instruments}
        self._by_ticker = {inst.ticker: inst for inst in instruments}
        self._by_doc_id = {entry.doc_id: entry for entry in documents}

    # -- lookups ---------------------------------------------------------

    def instrument(self, isin: str) -> Instrument | None:
        return self._by_isin.get(isin.upper())

    def instrument_by_ticker(self, ticker: str) -> Instrument | None:
        return self._by_ticker.get(str(ticker).strip())

    def entry(self, doc_id: str) -> ManifestEntry | None:
        return self._by_doc_id.get(doc_id)

    def isins_for(self, entry: ManifestEntry) -> tuple[str, ...]:
        """The ISINs a document covers, resolved through the ticker join.

        Tickers with no matching instrument are dropped here and reported by
        ``validate`` -- this returns what is genuinely reachable, not what the
        manifest hoped for.
        """
        resolved = (self._by_ticker.get(ticker) for ticker in entry.tickers)
        return tuple(inst.isin for inst in resolved if inst is not None)

    def doc_ids_for_isins(self, isins: list[str] | tuple[str, ...]) -> list[str]:
        """Documents covering any of these ISINs, newest first.

        This is the retrieval pre-filter. Newest first because two documents
        covering the same company are usually a company note and an older sector
        note, and ties should fall to the more recent view.
        """
        wanted = {isin.upper() for isin in isins}
        matched = [
            entry
            for entry in self.documents
            if wanted.intersection(self.isins_for(entry))
        ]
        matched.sort(key=lambda e: e.published_at, reverse=True)
        return [entry.doc_id for entry in matched]

    def doc_ids_for_tickers(self, tickers: list[str] | tuple[str, ...]) -> list[str]:
        """Same, keyed by ticker -- the form holdings actually arrive in."""
        isins = [
            inst.isin
            for inst in (self.instrument_by_ticker(t) for t in tickers)
            if inst is not None
        ]
        return self.doc_ids_for_isins(isins)

    def documents_for_instrument(self, isin: str) -> list[ManifestEntry]:
        return [
            entry for entry in self.documents if isin.upper() in self.isins_for(entry)
        ]

    # -- validation ------------------------------------------------------

    def validate(self) -> CatalogReport:
        """Check every link that ingestion and retrieval depend on.

        Errors are conditions under which a document would be silently
        unreachable or unreadable. Warnings are conditions a human should look
        at but that do not corrupt the index.
        """
        errors: list[str] = []
        warnings: list[str] = []

        errors.extend(self._duplicate_errors())
        errors.extend(self._link_errors())
        errors.extend(self._file_errors())
        warnings.extend(self._coverage_warnings())
        warnings.extend(self._isin_warnings())

        return CatalogReport(tuple(errors), tuple(warnings))

    def _duplicate_errors(self) -> list[str]:
        errors = []
        for label, values in (
            ("ISIN", [i.isin for i in self.instruments]),
            ("ticker", [i.ticker for i in self.instruments]),
            ("doc_id", [d.doc_id for d in self.documents]),
        ):
            seen: set[str] = set()
            for value in values:
                if value in seen:
                    errors.append(f"duplicate {label} {value!r}")
                seen.add(value)
        return errors

    def _link_errors(self) -> list[str]:
        errors = []
        for entry in self.documents:
            if not entry.tickers:
                errors.append(
                    f"{entry.doc_id}: no instruments listed, so no client holding "
                    "will ever retrieve it"
                )
            for ticker in entry.tickers:
                if ticker not in self._by_ticker:
                    errors.append(
                        f"{entry.doc_id}: ticker {ticker!r} is not in the instruments "
                        "file, so that link resolves to no ISIN"
                    )
        return errors

    def _file_errors(self) -> list[str]:
        errors = []
        listed = {entry.filename for entry in self.documents}
        for entry in self.documents:
            if not entry.path(self.documents_root).is_file():
                errors.append(f"{entry.doc_id}: {entry.filename} is missing from disk")
        if self.documents_root.is_dir():
            for pdf in sorted(self.documents_root.glob("*.pdf")):
                if pdf.name not in listed:
                    errors.append(
                        f"{pdf.name} is on disk but not in the manifest, so it would "
                        "never be indexed"
                    )
        return errors

    def _coverage_warnings(self) -> list[str]:
        covered = {isin for entry in self.documents for isin in self.isins_for(entry)}
        return [
            f"{inst.label}: no research document covers this instrument"
            for inst in self.instruments
            if inst.isin not in covered
        ]

    def _isin_warnings(self) -> list[str]:
        # A warning rather than an error: these ISINs are real today, but the
        # instruments file is the SQL side's to replace and may arrive with
        # invented ones. Worth flagging, not worth blocking the demo.
        warnings = []
        for inst in self.instruments:
            if not _ISIN_SHAPE.match(inst.isin):
                warnings.append(
                    f"{inst.label}: ISIN {inst.isin!r} is not 12 characters of "
                    "country code + 9 alphanumerics + check digit"
                )
            elif inst.isin[-1] != isin_check_digit(inst.isin):
                warnings.append(
                    f"{inst.label}: ISIN {inst.isin} fails its check digit "
                    f"(expected ...{isin_check_digit(inst.isin)}) -- likely a typo"
                )
        return warnings


def load_catalog(
    manifest_path: Path = MANIFEST_PATH,
    instruments_path: Path = INSTRUMENTS_PATH,
    documents_root: Path = DOCUMENTS_ROOT,
    *,
    strict: bool = True,
) -> Catalog:
    """Load and join both files.

    ``strict`` (the default) refuses to hand back a catalogue with broken links.
    Ingestion should keep it: an index built from a catalogue with a dangling
    ticker is missing a document and looks exactly like one that is not. Pass
    ``strict=False`` to inspect a broken catalogue rather than only hear about it.
    """
    catalog = Catalog(
        documents=load_manifest(manifest_path),
        instruments=load_instruments(instruments_path),
        documents_root=documents_root,
    )
    if strict:
        report = catalog.validate()
        if not report.ok:
            listed = "\n  - ".join(report.errors)
            raise CatalogError(
                f"catalogue has {len(report.errors)} broken link(s):\n  - {listed}\n"
                "Fix data/manifest.csv or data/mock/instruments.csv, or load with "
                "strict=False to inspect."
            )
    return catalog


def main(argv: list[str] | None = None) -> int:
    """``uv run python -m ingestion.catalog`` -- print the join and check it."""
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="ingestion.catalog",
        description="Show how documents and instruments join, and validate the links.",
    )
    parser.add_argument("--isin", action="append", default=[],
                        help="show the documents an ISIN resolves to (repeatable)")
    parser.add_argument("--ticker", action="append", default=[],
                        help="same, by Tadawul ticker (repeatable)")
    args = parser.parse_args(argv)

    catalog = load_catalog(strict=False)
    report = catalog.validate()

    if args.isin or args.ticker:
        doc_ids = catalog.doc_ids_for_isins(args.isin) if args.isin else []
        if args.ticker:
            doc_ids += [
                d for d in catalog.doc_ids_for_tickers(args.ticker) if d not in doc_ids
            ]
        print(f"Documents for {', '.join(args.isin + args.ticker)}:")
        for doc_id in doc_ids:
            entry = catalog.entry(doc_id)
            assert entry is not None
            print(f"  {entry.published_at}  {entry.doc_type:12}  {doc_id}")
            print(f"    {entry.title}")
        if not doc_ids:
            print("  (none)")
        return 0

    print(
        f"{len(catalog.documents)} documents, {len(catalog.instruments)} instruments\n"
    )
    print(f"{'instrument':46} {'ticker':>7}  documents")
    for inst in catalog.instruments:
        entries = catalog.documents_for_instrument(inst.isin)
        names = ", ".join(e.doc_id for e in entries) or "(none)"
        print(f"{inst.label[:46]:46} {inst.ticker:>7}  {len(entries)}  {names}")

    print()
    print(f"{'document':40} {'type':12} {'lang':6} ISINs")
    for entry in sorted(catalog.documents, key=lambda e: e.published_at):
        isins = ",".join(catalog.isins_for(entry)) or "(unresolved)"
        print(
            f"{entry.doc_id:40} {entry.doc_type:12} "
            f"{'/'.join(entry.languages):6} {isins}"
        )

    print()
    for warning in report.warnings:
        print(f"  warning: {warning}")
    for error in report.errors:
        print(f"  ERROR:   {error}")
    print(f"\n{report.summary()}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
