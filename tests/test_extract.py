"""Tests for ingestion.extract.

``classify`` and ``split_pages`` are pure and tested against literals. The rest
runs against the real corpus: extraction is a composition of a subprocess, a
normaliser and two CSVs, and the thing worth asserting is that the composition
holds on the actual documents, not that a mock returns what the mock was told to.

The corpus tests share one extraction pass -- sixteen subprocess calls is the
expensive part, and none of these tests mutate anything.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from ingestion.catalog import load_catalog
from ingestion.extract import (
    ARABIC,
    ENGLISH,
    MIN_DOCUMENT_CHARS,
    UNDETERMINED,
    Document,
    ExtractionError,
    classify,
    extract_corpus,
    extract_document,
    repeated_paragraphs,
    split_pages,
)
from ingestion.normalize import census, normalize


class TestClassify:
    def test_arabic_text(self):
        assert classify("مصرف الراجحي حقق أرباحا قياسية") == ARABIC

    def test_english_text(self):
        assert classify("Al Rajhi Bank reported record profit") == ENGLISH

    def test_dominant_script_wins_in_mixed_text(self):
        # Every "Arabic" page in this corpus carries an English disclaimer and
        # every ticker in Latin digits, so purity is never the test.
        assert classify("مصرف الراجحي مصرف الراجحي (ticker 1120)") == ARABIC

    def test_digits_and_punctuation_do_not_decide(self):
        assert classify("2026 -- 4.07 (2010) 1,120") == UNDETERMINED

    def test_presentation_forms_still_count_as_arabic(self):
        # Should never happen post-normalisation, but classify must not report
        # unnormalised Arabic as undetermined and hide the fact.
        assert classify("ﻣﺼﺮﻑ") == ARABIC


class TestSplitPages:
    def test_pages_are_separated_on_the_form_feed(self):
        assert split_pages("one\ftwo") == ["one", "two"]

    def test_the_trailing_form_feed_does_not_create_a_page(self):
        # pdftotext emits a form feed after every page, including the last.
        assert split_pages("one\ftwo\f") == ["one", "two"]

    def test_a_blank_interior_page_is_kept_so_numbering_stays_right(self):
        assert split_pages("one\f   \fthree\f") == ["one", "   ", "three"]

    def test_splitting_happens_before_normalising(self):
        # normalize turns form feeds into paragraph breaks, so a caller that
        # normalised first could not recover the page boundary at all.
        assert "\f" not in normalize("one\ftwo")
        assert len(split_pages("one\ftwo")) == 2


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


@pytest.fixture(scope="module")
def documents(catalog):
    return extract_corpus(catalog)


class TestTheCorpusExtracts:
    def test_every_manifest_document_is_extracted(self, catalog, documents):
        assert len(documents) == len(catalog.documents)

    def test_documents_come_back_in_publication_order(self, documents):
        dates = [d.published_at for d in documents]
        assert dates == sorted(dates)

    def test_every_document_has_substantial_text(self, documents):
        for document in documents:
            assert len(document.text) > MIN_DOCUMENT_CHARS, document.doc_id

    def test_no_document_reaches_the_index_dirty(self, documents):
        # The Phase 1 invariant, asserted on real extracted text rather than on
        # literals: nothing survives that would corrupt an Arabic embedding.
        for document in documents:
            assert census(document.text).is_clean, document.doc_id

    def test_every_document_resolves_to_an_isin(self, documents):
        for document in documents:
            assert document.isins, document.doc_id


class TestLanguageDetection:
    def test_nothing_contradicts_the_manifest(self, documents):
        # If this fires, either the manifest is wrong or a page stopped
        # extracting -- both mean Arabic questions get answered from English.
        mismatched = {d.doc_id: d.language_mismatch for d in documents
                      if d.language_mismatch}
        assert mismatched == {}

    def test_bilingual_notes_have_an_english_page_then_an_arabic_one(self, documents):
        bilingual = [d for d in documents if set(d.entry.languages) == {"ar", "en"}]
        assert len(bilingual) == 12
        for document in bilingual:
            assert [p.language for p in document.pages] == [ENGLISH, ARABIC], (
                document.doc_id
            )

    def test_the_arabic_only_notes_have_no_english_page(self, documents):
        arabic_only = [d for d in documents if d.entry.languages == ("ar",)]
        assert len(arabic_only) == 2
        for document in arabic_only:
            assert all(p.language == ARABIC for p in document.pages), document.doc_id

    def test_the_probe_document_is_arabic(self, documents):
        # pdf_backend uses this one to prove a backend can read Arabic at all.
        (probe,) = [d for d in documents if "petrochemicals" in d.doc_id]
        assert probe.detected_languages == (ARABIC,)
        assert census(probe.text).arabic_base > 1000

    def test_an_arabic_page_still_contains_latin(self, documents):
        # Documents the design note: page language is dominant, not pure, so
        # chunking has to re-classify rather than inherit the page label.
        (sabic,) = [d for d in documents if d.doc_id.endswith("sabic_2010")]
        arabic_page = [p for p in sabic.pages if p.language == ARABIC][0]
        assert any(ch.isascii() and ch.isalpha() for ch in arabic_page.text)


class TestMetadata:
    def test_pages_are_numbered_from_one(self, documents):
        for document in documents:
            assert [p.number for p in document.pages] == list(
                range(1, len(document.pages) + 1)
            ), document.doc_id
            assert document.metadata()["page_count"] == len(document.pages)

    def test_values_are_all_scalars(self, documents):
        # Chroma rejects lists, and a list here would only be discovered at
        # index-build time.
        for document in documents:
            for key, value in document.metadata().items():
                assert isinstance(value, (str, int, float, bool)), key

    def test_it_carries_the_identifiers_a_citation_needs(self, documents):
        metadata = documents[0].metadata()
        assert metadata["doc_id"] and metadata["title"]
        assert metadata["published_at"] == documents[0].published_at.isoformat()

    def test_isins_are_joined_for_display_not_for_filtering(self, documents):
        (sector,) = [d for d in documents if "sector-banking" in d.doc_id]
        assert metadata_isins(sector) == list(sector.isins)
        assert len(sector.isins) == 3


def metadata_isins(document: Document) -> list[str]:
    return str(document.metadata()["isins"]).split(",")


class TestGuards:
    def test_a_missing_pdf_raises(self, catalog):
        broken = replace(catalog.documents[0], filename="does-not-exist.pdf")
        with pytest.raises(ExtractionError, match="missing"):
            extract_document(broken, catalog)

    def test_a_document_with_no_instruments_raises(self, catalog):
        # Extraction would succeed; the document would just never be retrieved.
        orphan = replace(catalog.documents[0], tickers=())
        with pytest.raises(ExtractionError, match="no ISIN"):
            extract_document(orphan, catalog)


class TestBoilerplate:
    def test_the_disclaimer_is_found_in_both_languages(self, documents):
        hits = repeated_paragraphs(documents)
        languages = {classify(paragraph) for _, paragraph in hits}
        assert languages == {ARABIC, ENGLISH}

    def test_boilerplate_is_left_in_the_extracted_text(self, documents):
        # Removing it is chunking's decision: extracted text has to correspond
        # to what is actually in the PDF.
        _, paragraph = repeated_paragraphs(documents)[0]
        assert sum(paragraph in d.text for d in documents) == 14

    def test_document_specific_prose_is_not_reported(self, documents):
        # Only the two disclaimers repeat. If real analysis started showing up
        # here, the synthetic corpus would be too templated to evaluate against.
        hits = {paragraph for _, paragraph in repeated_paragraphs(documents)}
        assert len(hits) == 2
        (sabic,) = [d for d in documents if d.doc_id.endswith("sabic_2010")]
        body = [p for p in sabic.text.split("\n\n") if len(p) >= 40]
        assert [p for p in body if p not in hits], "every paragraph is boilerplate"
