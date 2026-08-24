"""Show what normalisation does to a PDF, character by character.

    uv run python -m ingestion.inspect_pdf                      # the Arabic probe
    uv run python -m ingestion.inspect_pdf data/documents/2026-08-06_sabic_2010.pdf
    uv run python -m ingestion.inspect_pdf <pdf> --line 12 --chars 20
    uv run python -m ingestion.inspect_pdf --corpus             # census every document

Prints codepoints rather than text: the corruption this guards against is
invisible by construction, since raw and normalised Arabic render identically
in a terminal and only the codepoints differ.

Separate from normalize.py so that module stays pure string functions with no
I/O, testable against literals rather than PDFs.
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

from ingestion import normalize as N
from ingestion import pdf_backend

_CENSUS_ROWS = (
    ("Arabic base (0600-06FF)", "arabic_base"),
    ("Presentation Forms-A", "presentation_a"),
    ("Presentation Forms-B", "presentation_b"),
    ("Arabic symbols (kept)", "arabic_symbols"),
    ("bidi controls", "bidi_controls"),
    ("decorative (tatweel/tail)", "decorative"),
    ("zero-width", "zero_width"),
    ("other non-ASCII", "other_non_ascii"),
)


def _name(ch: str) -> str:
    return unicodedata.name(ch, "<unnamed>")


def _describe(text: str) -> str:
    """Codepoints of a short string, for the right-hand side of a mapping."""
    if not text:
        return "(removed)"
    return " ".join(f"U+{ord(c):04X}" for c in text)


def print_census(raw: str, normalised: str) -> None:
    before, after = N.census(raw), N.census(normalised)
    print("Character census")
    print(f"  {'class':28} {'raw':>8} {'normalised':>12}")
    for label, field in _CENSUS_ROWS:
        b, a = getattr(before, field), getattr(after, field)
        flag = "  <-- fixed" if b and not a else ""
        print(f"  {label:28} {b:>8} {a:>12}{flag}")
    print(f"  {'total characters':28} {before.total:>8} {after.total:>12}")
    print()
    print(f"  raw safe to embed?         {'yes' if before.is_clean else 'NO'}")
    print(f"  normalised safe to embed?  {'yes' if after.is_clean else 'NO'}")


def pick_line(raw: str, index: int | None) -> tuple[int, str]:
    """The line to show: the requested one, else the first containing Arabic."""
    lines = raw.split("\n")
    if index is not None:
        if not 0 <= index < len(lines):
            raise SystemExit(f"--line {index} out of range (0..{len(lines) - 1})")
        return index, lines[index]
    for i, line in enumerate(lines):
        c = N.census(line)
        if c.arabic_base or c.presentation_forms:
            return i, line
    return 0, lines[0]


def print_sample(line_no: int, line: str, limit: int) -> None:
    normalised = N.normalize(line)
    print(f"Sample: line {line_no}")
    print(f"  raw         {line.strip()[:100]}")
    print(f"  normalised  {normalised[:100]}")
    print()
    print(f"  per character (first {limit} non-space):")
    print(f"    {'source':42} {'result':14} what happened")

    shown = 0
    for change in N.explain(line):
        if change.source.isspace():
            continue
        source = f"U+{ord(change.source):04X}  {_name(change.source)}"
        print(f"    {source:42} {_describe(change.result):14} {change.reason}")
        shown += 1
        if shown >= limit:
            break


def print_corpus(docs: Path) -> int:
    """Census every document, and fail if any would reach the index dirty."""
    print(f"{'document':46} {'pres':>6} {'bidi':>6} {'arabic':>7}  clean")
    dirty = 0
    for pdf in sorted(docs.glob("*.pdf")):
        raw = pdf_backend.extract_raw(pdf)
        before, after = N.census(raw), N.census(N.normalize(raw))
        ok = after.is_clean
        dirty += not ok
        print(
            f"{pdf.name:46} {before.presentation_forms:>6} {before.bidi_controls:>6} "
            f"{after.arabic_base:>7}  {'yes' if ok else 'NO'}"
        )
    print()
    if dirty:
        print(f"{dirty} document(s) still contain characters that would corrupt embeddings.")
        return 1
    print("All documents normalise clean.")
    return 0


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="ingestion.inspect_pdf",
        description="Show what Arabic normalisation does to an extracted PDF.",
    )
    parser.add_argument(
        "pdf", nargs="?", default=str(pdf_backend.PROBE_PDF),
        help="PDF to inspect (default: the Arabic-only probe document)",
    )
    parser.add_argument("--line", type=int, default=None,
                        help="line index to sample (default: first line with Arabic)")
    parser.add_argument("--chars", type=int, default=12,
                        help="how many characters to map (default: 12)")
    parser.add_argument("--full", action="store_true",
                        help="print the whole normalised document")
    parser.add_argument("--corpus", action="store_true",
                        help="census every document in data/documents instead")
    args = parser.parse_args(argv)

    if args.corpus:
        return print_corpus(Path("data/documents"))

    pdf = Path(args.pdf)
    backend = pdf_backend.resolve()
    raw = pdf_backend.extract_raw(pdf, backend)
    normalised = N.normalize(raw)

    print(f"Document   {pdf.name}")
    print(f"Backend    {backend.flavour} {backend.version}")
    print()
    print_census(raw, normalised)
    print()
    line_no, line = pick_line(raw, args.line)
    print_sample(line_no, line, args.chars)

    if args.full:
        print()
        print("Normalised text")
        print("-" * 70)
        print(normalised)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
