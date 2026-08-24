"""Tests for ingestion.chunk.

Two layers, for the same reason as test_catalog. The splitting rules are pure
and are pinned against small synthetic documents, because the cases that matter
-- a section too big for the budget, a stub with no body, a run of text with no
sentence boundary -- are rare or absent in the healthy corpus. A final layer
then asserts the properties the real corpus has to satisfy before it is indexed.

The corpus tests share one chunking pass; nothing here mutates it.
"""

from __future__ import annotations

from datetime import date

import pytest

from ingestion.catalog import ManifestEntry, load_catalog
from ingestion.chunk import (
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
    PUBLISHER_HEADER,
    Section,
    _merge_short_sections,
    _split_to_budget,
    _split_words,
    boilerplate_paragraphs,
    chunk_corpus,
    chunk_document,
    count_tokens,
    is_heading,
    split_sections,
)
from ingestion.extract import ARABIC, ENGLISH, Document, Page, classify, extract_corpus

ARABIC_PROSE = (
    "تواصل أسواق البتروكيماويات العالمية المعاناة من فائض في المعروض مصحوب بضعف "
    "الطلب من الصين وأوروبا وشهد عام تراجعا واسعا في الأسعار."
)
ENGLISH_PROSE = (
    "The ten listed Saudi banks reported combined second quarter net profit of "
    "SAR 24.87 billion, an increase of 12.5 percent year on year."
)


def make_document(*pages: str, title: str = "A Note About Something") -> Document:
    """A Document with the given page texts and plausible metadata."""
    entry = ManifestEntry(
        filename="note.pdf",
        title=title,
        publisher="Meridian Gulf Research",
        published_at=date(2026, 8, 6),
        languages=("en",),
        doc_type="company_note",
        classification="internal",
        tickers=("2010",),
        is_synthetic=True,
    )
    return Document(
        entry=entry,
        isins=("SA0007879121",),
        pages=tuple(
            Page(number=i, text=text, language=classify(text))
            for i, text in enumerate(pages, start=1)
        ),
    )


class TestCountTokens:
    def test_it_counts_subwords_not_words(self):
        # One word, several tokens: the count is the model's, not a split().
        assert count_tokens("counterintuitively") > 1

    def test_arabic_costs_somewhat_more_tokens_per_character(self):
        # Real, but modest -- about 1.5x. Measuring this against cl100k_base
        # instead gave 3.2x and sent the first version of this module chasing a
        # size asymmetry that belonged to the tokeniser, not to Arabic. The
        # bound is loose on both sides on purpose: what matters is that the gap
        # exists and is nowhere near the threefold figure.
        arabic = len(ARABIC_PROSE) / count_tokens(ARABIC_PROSE)
        english = len(ENGLISH_PROSE) / count_tokens(ENGLISH_PROSE)
        assert 1.2 < english / arabic < 2.0


class TestIsHeading:
    @pytest.mark.parametrize(
        "line",
        ["Risks", "Margins and funding", "المخاطر", "النقد والتصنيف الائتماني"],
    )
    def test_real_headings_are_recognised(self, line):
        assert is_heading(line)

    @pytest.mark.parametrize(
        "line", [ENGLISH_PROSE, ARABIC_PROSE, "", "Risks.", "ما هي المخاطر؟"]
    )
    def test_prose_and_sentence_endings_are_not_headings(self, line):
        assert not is_heading(line)


class TestSplitSections:
    def test_a_heading_becomes_the_title_and_is_lifted_out_of_the_body(self):
        (section,) = split_sections(make_document(f"Risks\n{ENGLISH_PROSE}"))
        assert section.title == "Risks"
        assert section.body == ENGLISH_PROSE

    def test_a_block_with_no_heading_has_no_title(self):
        (section,) = split_sections(make_document(ENGLISH_PROSE))
        assert section.title is None
        assert section.body == ENGLISH_PROSE

    def test_the_publisher_header_is_dropped(self):
        sections = split_sections(make_document(f"{PUBLISHER_HEADER}\n\nRisks\nx y z."))
        assert len(sections) == 1
        assert sections[0].title == "Risks"

    def test_boilerplate_is_dropped(self):
        page = f"Risks\n{ENGLISH_PROSE}\n\n{ARABIC_PROSE}"
        sections = split_sections(make_document(page), drop=frozenset({ARABIC_PROSE}))
        assert [s.title for s in sections] == ["Risks"]

    def test_the_key_facts_block_is_labelled_and_kept_whole(self):
        block = "A Note\nCompany SABIC\nTadawul Code 2010\nRating Underweight"
        (section,) = split_sections(make_document(block))
        assert section.title == "Key facts"
        assert section.is_front_matter
        assert "Rating Underweight" in section.body

    def test_an_arabic_key_facts_block_gets_an_arabic_label(self):
        (section,) = split_sections(make_document("تخفيض الوزن: التوصية2010 سابك"))
        assert section.title == "بيانات أساسية"
        assert section.is_front_matter


class TestPageSpanning:
    def test_a_section_running_over_the_page_break_is_rejoined(self):
        # The first block of page 2 opens with prose, not a heading, so it is
        # the tail of the section that ran off the bottom of page 1.
        document = make_document(f"Risks\n{ENGLISH_PROSE}", "and a final clause.")
        (section,) = split_sections(document)
        assert section.body.endswith("and a final clause.")
        assert (section.first_page, section.last_page) == (1, 2)
        assert section.pages == "1-2"

    def test_a_page_that_starts_with_a_heading_starts_a_new_section(self):
        document = make_document(f"Risks\n{ENGLISH_PROSE}", f"Outlook\n{ENGLISH_PROSE}")
        assert [s.title for s in split_sections(document)] == ["Risks", "Outlook"]

    def test_boilerplate_is_dropped_before_the_rejoin(self):
        # The banking note's second page begins with the disclaimer. Rejoining
        # first would glue it to the end of the previous section, which is
        # worse than indexing it on its own.
        document = make_document(f"Risks\n{ENGLISH_PROSE}", ARABIC_PROSE)
        (section,) = split_sections(document, drop=frozenset({ARABIC_PROSE}))
        assert ARABIC_PROSE not in section.body
        assert section.last_page == 1


class TestSplitToBudget:
    def test_text_within_budget_is_left_alone(self):
        assert _split_to_budget(ENGLISH_PROSE, 384, 48) == [ENGLISH_PROSE]

    def test_an_oversized_section_splits_and_no_piece_exceeds_the_budget(self):
        pieces = _split_to_budget(" ".join([ENGLISH_PROSE] * 10), 100, 20)
        assert len(pieces) > 1
        for piece in pieces:
            assert count_tokens(piece) <= 100

    def test_it_prefers_to_cut_between_sentences(self):
        body = "First sentence here. Second sentence here. Third sentence here."
        pieces = _split_to_budget(body, 8, 0)
        assert all(piece.endswith(".") for piece in pieces)

    def test_consecutive_pieces_overlap(self):
        body = " ".join(f"Sentence number {i} of the section." for i in range(20))
        first, second = _split_to_budget(body, 60, 20)[:2]
        tail = first.split(". ")[-1]
        assert tail and tail in second

    def test_text_with_no_sentence_boundary_still_splits(self):
        # Arabic here often has no boundary the regex can find: pdftotext
        # displaces the full stops when it lays out right-to-left text.
        body = " ".join(["كلمة"] * 400)
        pieces = _split_to_budget(body, 100, 20)
        assert len(pieces) > 1
        for piece in pieces:
            assert count_tokens(piece) <= 100


class TestSplitWords:
    def test_pieces_are_evenly_sized_rather_than_greedy(self):
        # Greedy filling left a three-word remainder in the last piece, which
        # then occupied a slot in the top-k while carrying no answer. Asserted
        # as a property rather than an exact count, so it holds whatever
        # tokeniser settings.embedding_model brings.
        text = " ".join(["word"] * 300)
        pieces = _split_words(text, 200)
        sizes = [count_tokens(p) for p in pieces]
        assert len(pieces) == -(-count_tokens(text) // 200)
        assert max(sizes) - min(sizes) <= 2

    def test_every_word_survives(self):
        words = [f"w{i}" for i in range(300)]
        assert " ".join(_split_words(" ".join(words), 50)).split() == words


class TestMergeShortSections:
    def stub(self, title="Business mix", body="short"):
        return Section(title=title, body=body, first_page=1, last_page=1)

    def real(self, title="Risks"):
        return Section(title=title, body=ENGLISH_PROSE, first_page=1, last_page=1)

    def test_a_stub_is_folded_into_the_section_that_follows(self):
        merged = _merge_short_sections([self.stub(), self.real()])
        assert len(merged) == 1
        assert ENGLISH_PROSE in merged[0].body

    def test_the_merged_section_takes_the_following_heading(self):
        # The following section is the one with a body under it.
        (merged,) = _merge_short_sections([self.stub(), self.real()])
        assert merged.title == "Risks"

    def test_the_stubs_own_heading_survives_in_the_body(self):
        (merged,) = _merge_short_sections([self.stub(), self.real()])
        assert "Business mix" in merged.body

    def test_a_section_above_the_floor_is_left_alone(self):
        sections = [self.real("Risks"), self.real("Outlook")]
        assert [s.title for s in _merge_short_sections(sections)] == ["Risks", "Outlook"]

    def test_a_trailing_stub_is_kept_rather_than_dropped(self):
        merged = _merge_short_sections([self.real(), self.stub()])
        assert len(merged) == 2
        assert merged[1].body == "short"

    def test_front_matter_is_exempt_however_short(self):
        # The Arabic key-facts block is a single 15-token line, but that line
        # is the analyst's rating -- the densest text in the document.
        facts = Section(
            title="بيانات أساسية",
            body="تخفيض الوزن: التوصية2010 سابك",
            first_page=1,
            last_page=1,
            is_front_matter=True,
        )
        merged = _merge_short_sections([facts, self.real()])
        assert [s.title for s in merged] == ["بيانات أساسية", "Risks"]

    def test_the_merged_section_spans_both_page_ranges(self):
        first = Section("A", "short", first_page=1, last_page=1)
        second = Section("B", ENGLISH_PROSE, first_page=2, last_page=3)
        (merged,) = _merge_short_sections([first, second])
        assert merged.pages == "1-3"


class TestChunkDocument:
    def test_every_chunk_is_prefixed_with_the_document_title(self):
        document = make_document(f"Risks\n{ENGLISH_PROSE}", title="SABIC: A Note")
        (chunk,) = chunk_document(document)
        assert chunk.text.startswith("SABIC: A Note\nRisks\n")

    def test_the_body_stays_as_written(self):
        (chunk,) = chunk_document(make_document(f"Risks\n{ENGLISH_PROSE}"))
        assert chunk.body == ENGLISH_PROSE

    def test_language_comes_from_the_body_not_the_english_prefix(self):
        # Every document title in this corpus is English, so classifying the
        # prefixed text would relabel short Arabic sections as English.
        document = make_document(f"المخاطر\n{ARABIC_PROSE}", title="SABIC: A Note")
        (chunk,) = chunk_document(document)
        assert chunk.language == ARABIC

    def test_indexes_are_contiguous_from_zero(self):
        page = "\n\n".join(f"Section {i}\n{ENGLISH_PROSE}" for i in range(5))
        chunks = chunk_document(make_document(page))
        assert [c.index for c in chunks] == list(range(5))

    def test_the_chunk_id_carries_the_doc_id_and_position(self):
        (chunk,) = chunk_document(make_document(f"Risks\n{ENGLISH_PROSE}"))
        assert chunk.chunk_id == "note#00"

    def test_the_prefix_is_charged_against_the_budget(self):
        # What the ceiling has to govern is what gets embedded, and that
        # includes the title and heading, not only the body.
        long_title = " ".join(["Title"] * 40)
        body = " ".join([ENGLISH_PROSE] * 5)
        for chunk in chunk_document(
            make_document(f"Risks\n{body}", title=long_title), max_tokens=200
        ):
            assert chunk.token_count <= 200

    def test_a_split_section_keeps_its_heading_on_every_piece(self):
        body = " ".join([ENGLISH_PROSE] * 8)
        chunks = chunk_document(make_document(f"Risks\n{body}"), max_tokens=120)
        assert len(chunks) > 1
        assert all(c.section == "Risks" for c in chunks)
        assert all(c.text.startswith("A Note About Something\nRisks\n") for c in chunks)


@pytest.fixture(scope="module")
def documents():
    return extract_corpus(load_catalog())


@pytest.fixture(scope="module")
def chunks(documents):
    return chunk_corpus(documents)


class TestTheRealCorpus:
    def test_every_document_produces_chunks(self, documents, chunks):
        produced = {c.doc_id for c in chunks}
        assert produced == {d.doc_id for d in documents}

    def test_no_chunk_exceeds_the_ceiling(self, chunks):
        for chunk in chunks:
            assert chunk.token_count <= MAX_CHUNK_TOKENS, chunk.chunk_id

    def test_the_only_chunks_under_the_floor_are_key_facts(self, chunks):
        # Everything else that small was merged forward. The exemption is
        # deliberate: the Arabic key-facts line is 15 tokens of pure rating.
        labels = {"Key facts", "بيانات أساسية"}
        for chunk in chunks:
            if count_tokens(chunk.body) < MIN_CHUNK_TOKENS:
                assert chunk.section in labels, chunk.chunk_id

    def test_chunk_ids_are_unique(self, chunks):
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_both_languages_are_represented(self, chunks):
        assert {c.language for c in chunks} == {ARABIC, ENGLISH}

    def test_chunk_count_is_in_the_expected_band(self, chunks):
        # Loose on purpose: a wording change in the corpus should not fail the
        # suite, but a chunker that suddenly emits 20 or 2,000 should.
        assert 120 < len(chunks) < 260


class TestTheCorpusIsSelfIdentifying:
    def test_every_chunk_names_its_document(self, documents, chunks):
        # The claim the prefix exists for. Without it the twelve "Risks"
        # sections are twelve anonymous paragraphs about risk.
        titles = {d.doc_id: d.title for d in documents}
        for chunk in chunks:
            assert chunk.text.startswith(titles[chunk.doc_id]), chunk.chunk_id

    def test_the_risks_sections_are_distinguishable_from_one_another(self, chunks):
        risks = [c for c in chunks if c.section == "Risks"]
        assert len(risks) == 12
        # The bodies alone do not name the company; the embedded text does.
        assert sum("SABIC" in c.body for c in risks) < len(risks)
        assert len({c.text for c in risks}) == len(risks)

    def test_a_chunk_can_be_cited_down_to_the_page(self, documents, chunks):
        by_id = {d.doc_id: d for d in documents}
        for chunk in chunks:
            pages = len(by_id[chunk.doc_id].pages)
            assert 1 <= chunk.first_page <= chunk.last_page <= pages, chunk.chunk_id


class TestBoilerplateIsGone:
    def test_the_disclaimers_are_not_indexed(self, documents, chunks):
        for paragraph in boilerplate_paragraphs(documents):
            for chunk in chunks:
                assert paragraph not in chunk.text, chunk.chunk_id

    def test_the_publisher_header_is_not_indexed(self, chunks):
        for chunk in chunks:
            assert PUBLISHER_HEADER not in chunk.text, chunk.chunk_id


class TestFrontMatter:
    LABELS = {"Key facts", "بيانات أساسية"}

    def test_every_document_has_a_key_facts_chunk(self, documents, chunks):
        for document in documents:
            own = [c for c in chunks if c.doc_id == document.doc_id]
            assert [c for c in own if c.section in self.LABELS], document.doc_id

    def test_a_bilingual_note_has_one_in_each_language(self, documents, chunks):
        bilingual = [d for d in documents if set(d.entry.languages) == {"ar", "en"}]
        assert len(bilingual) == 12
        for document in bilingual:
            facts = [
                c for c in chunks
                if c.doc_id == document.doc_id and c.section in self.LABELS
            ]
            assert {c.language for c in facts} == {ARABIC, ENGLISH}, document.doc_id

    def test_the_rating_is_retrievable(self, chunks):
        # An advisor asking "what is the house view on SABIC" has to land on
        # the rating, and the rating lives only in the key-facts block.
        (facts,) = [
            c for c in chunks
            if c.doc_id.endswith("sabic_2010") and c.section == "Key facts"
        ]
        assert "Underweight" in facts.text
        assert "SA0007879121" in facts.text


class TestMetadata:
    def test_values_are_all_scalars(self, documents, chunks):
        by_id = {d.doc_id: d for d in documents}
        for chunk in chunks:
            for key, value in chunk.metadata(by_id[chunk.doc_id]).items():
                assert isinstance(value, (str, int, float, bool)), key

    def test_it_carries_what_a_citation_needs(self, documents, chunks):
        by_id = {d.doc_id: d for d in documents}
        for chunk in chunks:
            metadata = chunk.metadata(by_id[chunk.doc_id])
            assert metadata["doc_id"] and metadata["title"] and metadata["isins"]
            assert metadata["chunk_id"] == chunk.chunk_id
            assert metadata["pages"] == chunk.pages
            assert metadata["language"] in (ARABIC, ENGLISH)

    def test_a_missing_section_becomes_an_empty_string_not_none(self):
        # Chroma rejects None as a metadata value.
        document = make_document(ENGLISH_PROSE)
        (chunk,) = chunk_document(document)
        assert chunk.section is None
        assert chunk.metadata(document)["section"] == ""
