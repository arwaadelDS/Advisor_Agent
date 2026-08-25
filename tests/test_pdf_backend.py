"""Tests for ingestion.pdf_backend.

These are integration tests: they run the real pdftotext against the real
corpus, because the failure they guard against is a property of the installed
binary, not of our code. A mocked pdftotext would pass on a machine where
ingestion is silently dropping every Arabic character.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestion import pdf_backend
from ingestion.normalize import census, normalize
from ingestion.pdf_backend import PdfBackendError

DOCS = Path("data/documents")
ARABIC_DOC = DOCS / "2026-08-17_sector-petrochemicals.pdf"
BILINGUAL_DOC = DOCS / "2026-08-06_sabic_2010.pdf"
ENGLISH_DOC = DOCS / "2026-08-18_sector-banking.pdf"

pytestmark = pytest.mark.skipif(
    not DOCS.is_dir(), reason="corpus not present"
)


@pytest.fixture(scope="module")
def backend():
    return pdf_backend.resolve()


class TestDiscovery:
    def test_finds_at_least_one_backend(self):
        assert pdf_backend.discover()

    def test_prefers_poppler_over_xpdf(self):
        found = pdf_backend.discover()
        flavours = [b.flavour for b in found]
        if "poppler" in flavours and "xpdf" in flavours:
            assert found[0].flavour == "poppler"


class TestExplicitPath:
    def test_nonexistent_path_raises_a_useful_error(self):
        with pytest.raises(PdfBackendError, match="not a file"):
            pdf_backend.resolve("C:/definitely/not/here/pdftotext.exe")

    def test_a_directory_resolves_to_the_executable_inside(self, backend):
        resolved = pdf_backend.resolve(str(backend.exe.parent))
        assert resolved.exe == backend.exe


class TestExtraction:
    def test_extracts_arabic_from_the_arabic_document(self, backend):
        text = pdf_backend.extract_raw(ARABIC_DOC, backend)
        assert pdf_backend.count_arabic(text) > 1000

    def test_extracts_both_scripts_from_the_bilingual_document(self, backend):
        text = pdf_backend.extract_raw(BILINGUAL_DOC, backend)
        assert pdf_backend.count_arabic(text) > 500
        assert "SABIC" in text

    def test_missing_file_raises(self, backend):
        with pytest.raises(FileNotFoundError):
            pdf_backend.extract_raw(DOCS / "nope.pdf", backend)


class TestArabicGuard:
    """The check that stops a silently-Arabic-free corpus reaching the index."""

    def test_passes_on_the_arabic_probe(self, backend):
        result = pdf_backend.check(backend)
        assert result.arabic_chars > 1000

    def test_fails_when_the_extracted_text_has_no_arabic(self, backend):
        # An English-only document stands in for a backend that dropped the
        # script: from check()'s point of view the two are indistinguishable,
        # which is exactly the condition it must refuse.
        with pytest.raises(PdfBackendError, match="dropping Arabic"):
            pdf_backend.check(backend, probe=ENGLISH_DOC)

    def test_fails_when_the_probe_is_missing(self, backend):
        with pytest.raises(PdfBackendError, match="missing"):
            pdf_backend.check(backend, probe=DOCS / "nope.pdf")


class TestCorpusNormalisesClean:
    """End to end: every document must be embeddable after normalisation."""

    def test_every_document_normalises_clean(self, backend):
        for pdf in sorted(DOCS.glob("*.pdf")):
            result = census(normalize(pdf_backend.extract_raw(pdf, backend)))
            assert result.is_clean, f"{pdf.name} would corrupt embeddings"

    def test_arabic_documents_actually_contain_arabic_afterwards(self, backend):
        # Guards against a normaliser that "cleans" text by deleting it.
        text = normalize(pdf_backend.extract_raw(ARABIC_DOC, backend))
        assert census(text).arabic_base > 1000
