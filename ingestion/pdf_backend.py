"""Locate and validate the ``pdftotext`` executable used to extract PDF text.

This module exists because two different ``pdftotext`` builds are commonly on
PATH on Windows, and one of them destroys the Arabic half of this corpus without
reporting an error:

* **Xpdf 4.00**, bundled with Git for Windows and usually *first* on PATH,
  defaults to Latin-1 output. Every codepoint outside Latin-1 is dropped, so an
  Arabic research note extracts as a few Latin fragments and a lot of
  whitespace. Exit code 0, nothing on stderr.
* **Poppler** (24.04.0 here, bundled with MiKTeX) defaults to UTF-8 and keeps
  the Arabic.

Passing ``-enc UTF-8`` makes both correct, and ``extract_raw`` always does. But
an index built by a backend that silently returned no Arabic looks exactly like
a working one until an advisor asks an Arabic question in the demo, so
``check()`` runs a real extraction against a known Arabic document and raises
rather than letting that reach the vector store.

The text returned here is deliberately still *raw*: Arabic arrives as Unicode
presentation forms (U+FE70-U+FEFF) with bidi control characters interleaved.
Normalising that belongs to ingestion/normalize.py, not here.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from config import settings

# Arabic as it arrives from a PDF: base letters plus the two presentation-form
# blocks. We count both, because at this stage either is a sign the backend
# decoded the font correctly -- converting one to the other is normalize.py's
# job downstream.
_ARABIC_RANGES = (
    (0x0600, 0x06FF),  # Arabic
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)

# Extra places to look when PATH resolves to the wrong build or to nothing.
_EXTRA_DIRS = (
    Path(os.environ.get("LOCALAPPDATA", ".")) / "Programs/MiKTeX/miktex/bin/x64",
    Path("C:/Program Files/poppler/bin"),
    Path("C:/Program Files/poppler/Library/bin"),
    Path("C:/poppler/bin"),
    Path(os.environ.get("USERPROFILE", ".")) / "anaconda3/Library/bin",
)

# An Arabic-only note: if a backend can read this, it can read the whole corpus.
PROBE_PDF = Path("data/documents/2026-08-17_sector-petrochemicals.pdf")

# The probe carries ~1900 Arabic characters. A backend that drops the script
# returns single digits, so anything in the hundreds is unambiguously working.
_MIN_PROBE_ARABIC = 500


class PdfBackendError(RuntimeError):
    """Raised when no usable pdftotext can be found, or the one found is broken."""


@dataclass(frozen=True)
class Backend:
    exe: Path
    flavour: str  # "poppler" | "xpdf" | "unknown"
    version: str

    def __str__(self) -> str:
        return f"{self.flavour} {self.version} ({self.exe})"


def count_arabic(text: str) -> int:
    """Number of Arabic characters in ``text``, in any form."""
    return sum(
        1
        for ch in text
        if any(lo <= ord(ch) <= hi for lo, hi in _ARABIC_RANGES)
    )


def _identify(exe: Path) -> Backend | None:
    """Run ``pdftotext -v`` and work out which build this is."""
    try:
        proc = subprocess.run([str(exe), "-v"], capture_output=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None

    # -v writes to stderr on poppler and to stdout on some Xpdf builds.
    banner = (proc.stdout + proc.stderr).decode("utf-8", errors="replace")

    version = "unknown"
    for line in banner.splitlines():
        if "version" in line.lower():
            version = line.strip().split()[-1]
            break

    lowered = banner.lower()
    if "poppler" in lowered:
        flavour = "poppler"
    elif "glyph & cog" in lowered or "xpdf" in lowered:
        flavour = "xpdf"
    else:
        flavour = "unknown"

    return Backend(exe=exe.resolve(), flavour=flavour, version=version)


def discover() -> list[Backend]:
    """Every pdftotext we can find, best first.

    Poppler sorts ahead of Xpdf: both are correct once ``-enc UTF-8`` is passed,
    but poppler is correct by default and is what the team installs.
    """
    candidates: list[Path] = []
    on_path = shutil.which("pdftotext")
    if on_path:
        candidates.append(Path(on_path))
    for directory in _EXTRA_DIRS:
        candidates.append(directory / "pdftotext.exe")
        candidates.append(directory / "pdftotext")

    seen: set[Path] = set()
    found: list[Backend] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        backend = _identify(resolved)
        if backend is not None:
            found.append(backend)

    rank = {"poppler": 0, "unknown": 1, "xpdf": 2}
    found.sort(key=lambda b: rank.get(b.flavour, 3))
    return found


def resolve(preferred: str | None = None) -> Backend:
    """Pick the pdftotext to use.

    ``preferred`` (or ``settings.poppler_path``) wins if set, so a machine with
    an unusual layout can pin the binary in .env rather than reordering PATH.
    """
    explicit = preferred if preferred is not None else settings.poppler_path
    if explicit:
        exe = Path(explicit)
        if exe.is_dir():
            exe = exe / "pdftotext.exe"
        if not exe.is_file():
            raise PdfBackendError(
                f"poppler_path points at {exe}, which is not a file. Set it to a "
                "pdftotext executable, or leave it empty to auto-discover."
            )
        backend = _identify(exe)
        if backend is None:
            raise PdfBackendError(f"{exe} exists but would not run.")
        return backend

    found = discover()
    if not found:
        raise PdfBackendError(
            "No pdftotext found on PATH or in the usual install locations.\n"
            "Install poppler, or set poppler_path in .env to the executable."
        )
    return found[0]


def extract_raw(
    pdf: Path | str,
    backend: Backend | None = None,
    *,
    layout: bool = True,
    first_page: int | None = None,
    last_page: int | None = None,
) -> str:
    """Extract text from ``pdf`` as UTF-8, without normalising it.

    ``-enc UTF-8`` is not optional: without it an Xpdf backend returns the
    document with every Arabic character removed.
    """
    backend = backend or get_backend()
    pdf = Path(pdf)
    if not pdf.is_file():
        raise FileNotFoundError(pdf)

    cmd = [str(backend.exe), "-enc", "UTF-8"]
    if layout:
        cmd.append("-layout")
    if first_page is not None:
        cmd += ["-f", str(first_page)]
    if last_page is not None:
        cmd += ["-l", str(last_page)]
    cmd += [str(pdf), "-"]

    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise PdfBackendError(f"pdftotext failed on {pdf.name}: {stderr}")

    return proc.stdout.decode("utf-8", errors="replace").lstrip("\ufeff")


@dataclass(frozen=True)
class CheckResult:
    backend: Backend
    probe: Path
    arabic_chars: int
    total_chars: int


def check(backend: Backend | None = None, probe: Path | str = PROBE_PDF) -> CheckResult:
    """Prove the backend can actually read Arabic, or raise.

    Called once at ingestion startup. The cost is one subprocess; what it buys is
    that a corpus can never be indexed with its Arabic silently missing.
    """
    backend = backend or resolve()
    probe = Path(probe)
    if not probe.is_file():
        raise PdfBackendError(
            f"Probe document {probe} is missing, so the backend cannot be verified."
        )

    text = extract_raw(probe, backend)
    arabic = count_arabic(text)

    if arabic < _MIN_PROBE_ARABIC:
        raise PdfBackendError(
            f"{backend} extracted only {arabic} Arabic characters from "
            f"{probe.name} (expected at least {_MIN_PROBE_ARABIC}).\n"
            "This backend is dropping Arabic instead of decoding it -- an index "
            "built with it would answer Arabic questions from English text only.\n"
            "Install poppler and set poppler_path in .env to its pdftotext."
        )

    return CheckResult(
        backend=backend,
        probe=probe,
        arabic_chars=arabic,
        total_chars=len(text),
    )


@lru_cache(maxsize=1)
def get_backend() -> Backend:
    """The verified backend for this process. Checked once, then cached."""
    return check().backend
