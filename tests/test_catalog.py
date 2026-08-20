"""Tests for ingestion.catalog.

Two layers. Most tests build small CSVs in a tmp directory, because the failure
modes worth pinning -- a dangling ticker, a duplicate ISIN, a file listed but
absent -- are exactly the ones the real, healthy corpus does not contain. A
final class then asserts the real files are healthy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.catalog import (
    Catalog,
    CatalogError,
    isin_check_digit,
    isin_is_valid,
    load_catalog,
    load_instruments,
    load_manifest,
)

INSTRUMENT_HEADER = "isin,ticker,name_en,name_ar,sector,asset_class,shariah_flag"
MANIFEST_HEADER = (
    "filename,title,publisher,published_at,language,doc_type,classification,"
    "instruments,is_synthetic"
)

SABIC = "SA0007879121,2010,SABIC,سابك,Petrochemicals,equity,"
KAYAN = "SA000A0MQCJ2,2350,Saudi Kayan,كيان السعودية,Petrochemicals,equity,"

NOTE = (
    "note.pdf,SABIC Q2,Meridian Gulf Research,2026-08-06,ar;en,company_note,"
    "internal,2010,true"
)
SECTOR = (
    "sector.pdf,Petrochemicals Review,Meridian Gulf Research,2026-08-17,ar,"
    "sector_note,internal,2010;2350,true"
)


@pytest.fixture
def build(tmp_path: Path):
    """Write a catalogue into tmp_path and load it.

    ``pdfs`` names the files that exist on disk; by default the ones the
    manifest lists, so tests opt in to a mismatch rather than tripping over one.
    """

    def _build(
        instruments: list[str] = None,
        documents: list[str] = None,
        pdfs: list[str] | None = None,
        *,
        strict: bool = False,
        encoding: str = "utf-8",
    ) -> Catalog:
        instruments = [SABIC, KAYAN] if instruments is None else instruments
        documents = [NOTE, SECTOR] if documents is None else documents

        instruments_path = tmp_path / "instruments.csv"
        manifest_path = tmp_path / "manifest.csv"
        instruments_path.write_text(
            "\n".join([INSTRUMENT_HEADER, *instruments]) + "\n", encoding=encoding
        )
        manifest_path.write_text(
            "\n".join([MANIFEST_HEADER, *documents]) + "\n", encoding=encoding
        )

        root = tmp_path / "documents"
        root.mkdir(exist_ok=True)
        if pdfs is None:
            pdfs = [row.split(",")[0] for row in documents]
        for name in pdfs:
            (root / name).write_bytes(b"%PDF-1.4")

        return load_catalog(manifest_path, instruments_path, root, strict=strict)

    return _build


class TestLoading:
    def test_missing_file_names_the_path(self, tmp_path):
        with pytest.raises(CatalogError, match="nope.csv"):
            load_instruments(tmp_path / "nope.csv")

    def test_missing_column_is_named(self, tmp_path):
        path = tmp_path / "instruments.csv"
        path.write_text("isin,ticker,name_en\nSA0007879121,2010,SABIC\n", encoding="utf-8")
        with pytest.raises(CatalogError, match="name_ar"):
            load_instruments(path)

    def test_header_with_no_rows_is_an_error(self, tmp_path):
        path = tmp_path / "instruments.csv"
        path.write_text(INSTRUMENT_HEADER + "\n", encoding="utf-8")
        with pytest.raises(CatalogError, match="no rows"):
            load_instruments(path)

    def test_excel_byte_order_mark_does_not_break_the_header(self, build):
        # Excel writes a BOM on save. Without utf-8-sig the first column would
        # be named "﻿isin" and every lookup would KeyError.
        catalog = build(encoding="utf-8-sig")
        assert catalog.instrument("SA0007879121") is not None

    def test_arabic_names_survive_the_read(self, build):
        instrument = build().instrument("SA0007879121")
        assert instrument.name_ar == "سابك"

    def test_ticker_stays_a_string(self, build):
        # int(ticker) would turn a leading-zero code into a different one and
        # make the join miss without erroring.
        catalog = build(instruments=[SABIC.replace(",2010,", ",0210,")])
        assert catalog.instruments[0].ticker == "0210"

    def test_blank_shariah_flag_becomes_none(self, build):
        assert build().instrument("SA0007879121").shariah_flag is None

    def test_isin_lookup_is_case_insensitive(self, build):
        assert build().instrument("sa0007879121") is not None

    def test_bad_date_names_the_line(self, tmp_path):
        path = tmp_path / "manifest.csv"
        path.write_text(
            "\n".join([MANIFEST_HEADER, NOTE.replace("2026-08-06", "06/08/2026")]) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(CatalogError, match="line 2"):
            load_manifest(path)

    def test_semicolon_lists_are_split(self, build):
        sector = build().entry("sector")
        assert sector.tickers == ("2010", "2350")
        assert sector.languages == ("ar",)

    def test_doc_id_is_the_filename_stem(self, build):
        entry = build().entry("note")
        assert entry.filename == "note.pdf"
        assert entry.doc_id == "note"


class TestJoin:
    def test_tickers_resolve_to_isins(self, build):
        catalog = build()
        assert catalog.isins_for(catalog.entry("sector")) == (
            "SA0007879121",
            "SA000A0MQCJ2",
        )

    def test_a_holding_finds_both_its_documents(self, build):
        assert set(build().doc_ids_for_isins(["SA0007879121"])) == {"note", "sector"}

    def test_documents_come_back_newest_first(self, build):
        # A company note and a sector note both cover SABIC; the more recent
        # view should rank first when a tie has to be broken.
        assert build().doc_ids_for_isins(["SA0007879121"]) == ["sector", "note"]

    def test_holdings_can_be_passed_as_tickers(self, build):
        assert build().doc_ids_for_tickers(["2350"]) == ["sector"]

    def test_an_unheld_instrument_matches_nothing(self, build):
        assert build().doc_ids_for_isins(["SA0007879113"]) == []

    def test_no_duplicates_when_two_holdings_share_a_document(self, build):
        doc_ids = build().doc_ids_for_tickers(["2010", "2350"])
        assert doc_ids == ["sector", "note"]
        assert len(doc_ids) == len(set(doc_ids))

    def test_unknown_tickers_are_ignored_rather_than_raising(self, build):
        assert build().doc_ids_for_tickers(["9999"]) == []

    def test_unresolvable_tickers_are_dropped_from_the_isin_list(self, build):
        # The link is broken and validate() reports it; isins_for returns what
        # is genuinely reachable, not what the manifest hoped for.
        catalog = build(instruments=[SABIC])
        assert catalog.isins_for(catalog.entry("sector")) == ("SA0007879121",)


class TestValidation:
    def test_a_healthy_catalogue_is_clean(self, build):
        report = build().validate()
        assert report.ok and not report.warnings

    def test_dangling_ticker_is_an_error(self, build):
        report = build(instruments=[SABIC]).validate()
        assert any("2350" in e for e in report.errors)

    def test_document_with_no_instruments_is_an_error(self, build):
        orphan = NOTE.replace(",internal,2010,true", ",internal,,true")
        report = build(documents=[orphan]).validate()
        assert any("no instruments listed" in e for e in report.errors)

    def test_missing_pdf_is_an_error(self, build):
        report = build(pdfs=["note.pdf"]).validate()
        assert any("sector.pdf is missing from disk" in e for e in report.errors)

    def test_pdf_not_in_the_manifest_is_an_error(self, build):
        # It would simply never be indexed, and nothing would say so.
        report = build(pdfs=["note.pdf", "sector.pdf", "stray.pdf"]).validate()
        assert any("stray.pdf" in e and "never be indexed" in e for e in report.errors)

    @pytest.mark.parametrize(
        "field,row",
        [("ISIN", SABIC), ("ticker", SABIC.replace("SA0007879121", "SA0007879048"))],
    )
    def test_duplicate_instrument_keys_are_errors(self, build, field, row):
        report = build(instruments=[SABIC, row, KAYAN]).validate()
        assert any(f"duplicate {field}" in e for e in report.errors)

    def test_duplicate_doc_id_is_an_error(self, build):
        report = build(documents=[NOTE, NOTE]).validate()
        assert any("duplicate doc_id" in e for e in report.errors)

    def test_instrument_with_no_coverage_is_a_warning_not_an_error(self, build):
        # Legitimate: a client can hold something no analyst has written about.
        report = build(documents=[NOTE]).validate()
        assert report.ok
        assert any("no research document covers" in w for w in report.warnings)

    def test_bad_check_digit_is_a_warning_not_an_error(self, build):
        # The instruments file belongs to the SQL side and may arrive with
        # invented ISINs, so a failed check digit must not block the demo.
        typo = SABIC.replace("SA0007879121", "SA0007879122")
        report = build(instruments=[typo, KAYAN]).validate()
        assert report.ok
        assert any("check digit" in w for w in report.warnings)

    def test_strict_load_refuses_a_broken_catalogue(self, build):
        with pytest.raises(CatalogError, match="broken link"):
            build(instruments=[SABIC], strict=True)

    def test_non_strict_load_returns_it_for_inspection(self, build):
        assert not build(instruments=[SABIC], strict=False).validate().ok


class TestIsinCheckDigit:
    @pytest.mark.parametrize(
        "isin",
        ["SA0007879121", "SA14TG012N13", "SA000A0MQCJ2", "SA1510P1UMH1"],
    )
    def test_real_saudi_isins_pass(self, isin):
        assert isin_is_valid(isin)

    def test_a_single_wrong_digit_is_caught(self):
        assert not isin_is_valid("SA0007879122")

    def test_a_transposition_is_caught(self):
        assert not isin_is_valid("SA0007879211")

    def test_wrong_shape_is_rejected_before_the_check_digit(self):
        assert not isin_is_valid("2010")
        assert not isin_is_valid("SA00078791211")

    def test_check_digit_is_computed_not_read(self):
        assert isin_check_digit("SA000787912X") == "1"


@pytest.fixture(scope="module")
def catalog():
    """The catalogue as it actually stands in the repo."""
    return load_catalog(strict=False)


class TestTheRealCatalogue:
    """The files in the repo, as ingestion will load them."""

    def test_it_loads_strict(self):
        assert load_catalog(strict=True) is not None

    def test_it_validates_without_errors(self, catalog):
        report = catalog.validate()
        assert report.errors == ()

    def test_every_manifest_document_exists_on_disk(self, catalog):
        for entry in catalog.documents:
            assert entry.path(catalog.documents_root).is_file(), entry.filename

    def test_every_document_resolves_to_at_least_one_isin(self, catalog):
        for entry in catalog.documents:
            assert catalog.isins_for(entry), entry.doc_id

    def test_every_instrument_is_covered_by_a_document(self, catalog):
        # Not required by the design -- an uncovered instrument is only a
        # warning -- but true today, and worth knowing if it stops being true.
        for instrument in catalog.instruments:
            assert catalog.documents_for_instrument(instrument.isin), instrument.label

    def test_every_isin_is_well_formed(self, catalog):
        for instrument in catalog.instruments:
            assert isin_is_valid(instrument.isin), instrument.isin

    def test_sector_notes_cover_more_than_one_instrument(self, catalog):
        # The reason the join has to come from the manifest: these documents
        # are many-to-many and mention no ISIN in their text at all.
        sector_notes = [e for e in catalog.documents if e.doc_type == "sector_note"]
        assert sector_notes
        for entry in sector_notes:
            assert len(catalog.isins_for(entry)) > 1, entry.doc_id

    def test_a_two_holding_portfolio_narrows_the_corpus(self, catalog):
        doc_ids = catalog.doc_ids_for_tickers(["2010", "1180"])
        assert 0 < len(doc_ids) < len(catalog.documents)
