"""Split ``Document`` records into the units that actually get embedded.

These documents are already cut into sections by their author -- a short
heading ("Risks", "المخاطر") followed by one paragraph -- so sections are the
unit and MAX_CHUNK_TOKENS is only a ceiling for the first document that turns
up with a two-page section. Nothing in this corpus reaches it.

Two things here are easy to get wrong:

* Size is counted with ``settings.embedding_model``'s own tokeniser, not in
  characters and not with another model's. Switching from cl100k to bge-m3
  moved Arabic from a 3.2x token penalty to 1.5x and changed the ceiling.
* Each chunk is embedded as ``{title}\\n{heading}\\n\\n{body}``. The SABIC
  "Risks" section never says "SABIC", and twelve notes have one, so without the
  prefix their embeddings are nearly indistinguishable.

Boilerplate is dropped here rather than in extraction -- extracted text has to
correspond to the PDF -- and before page-spanning sections are stitched, or the
stitch reunites across the hole it leaves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from config import settings
from ingestion.extract import Document, classify, extract_corpus, repeated_paragraphs


class ChunkingError(RuntimeError):
    """Raised when the corpus cannot be turned into embeddable chunks."""

# Above the largest natural section (150 tokens) plus its prefix, so structure
# decides boundaries and this only catches oversized input. Well under bge-m3's
# 8,192 limit -- the risk is an unfocused vector, not truncation.
MAX_CHUNK_TOKENS = 256

# Carried between the pieces of a split section so a sentence that answers the
# question is not stranded with half its context on the other side of the cut.
OVERLAP_TOKENS = 48

# Shorter than this and a chunk is a fragment -- a stub heading, a stray line --
# with too little content to embed distinctly. Merged forward into the next
# section of the same document instead of being indexed or discarded.
MIN_CHUNK_TOKENS = 24

# A heading is a short line with no sentence-ending punctuation. Prose lines in
# this corpus wrap at ~110 characters and end in a full stop, so the two do not
# overlap; validated across all 16 documents before this rule was relied on.
HEADING_MAX_CHARS = 60
HEADING_MAX_WORDS = 8
SENTENCE_END = ".!?:؟؛"

# Printed on every first page above the title. It is already in the metadata as
# `publisher`, and as a chunk it is 16 identical 38-character duplicates.
PUBLISHER_HEADER = "Meridian Gulf Research\nEQUITY RESEARCH"

# The title-and-key-facts block: no heading of its own, and the section that
# answers "what is the rating". Matched on field labels rather than shape, since
# a shape rule would also catch a heading orphaned at the foot of a page.
FRONT_MATTER_MARKERS = ("Tadawul Code", "Coverage ", "التغطية", "التوصية")
FRONT_MATTER_TITLE = {"ar": "بيانات أساسية", "en": "Key facts", "und": "Key facts"}

# Splits after sentence-ending punctuation, keeping the punctuation with the
# sentence it ends.
_SENTENCE = re.compile(r"(?<=[.!?؟؛])\s+")


@lru_cache(maxsize=2)
def _tokeniser(name: str):
    """The embedding model's tokeniser, loaded lazily and once.

    Only the tokeniser files are fetched, not the model weights, so this is a
    few megabytes rather than the full download.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ChunkingError(
            "transformers is not installed. Run `uv sync`."
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception as exc:
        raise ChunkingError(
            f"could not load the tokeniser for {name!r}.\n"
            "The first run needs network access to fetch it; after that it is "
            "cached under ~/.cache/huggingface."
        ) from exc


def count_tokens(text: str, model_name: str | None = None) -> int:
    """Tokens as the embedding model counts them, not a word or character estimate.

    Sizing a chunk against any other tokeniser measures the wrong thing -- see
    the note at the top of this module, where doing exactly that produced a
    threefold overestimate of what Arabic costs.

    Counts the text only. The model also prepends and appends one special token,
    but that is a constant two that the ceiling has thousands of tokens of room
    for, and excluding it keeps the count additive -- the splitter relies on the
    size of two joined pieces being the sum of their sizes.
    """
    tokeniser = _tokeniser(model_name or settings.embedding_model)
    return len(tokeniser.encode(text, add_special_tokens=False))


def is_heading(line: str) -> bool:
    """Whether a line looks like a section heading rather than prose."""
    line = line.strip()
    return (
        bool(line)
        and len(line) <= HEADING_MAX_CHARS
        and line[-1] not in SENTENCE_END
        and len(line.split()) <= HEADING_MAX_WORDS
    )


@dataclass(frozen=True)
class Section:
    """One authored section: a heading, its body, and the pages it covers."""

    title: str | None
    body: str
    first_page: int
    last_page: int
    # Exempt from the short-section merge. The floor exists to suppress
    # fragments, and a key-facts block is the opposite of a fragment: the
    # Arabic one is a single 15-token line, but that line is the rating.
    is_front_matter: bool = False

    @property
    def pages(self) -> str:
        """``"2"`` or ``"1-2"`` -- what a citation prints."""
        if self.first_page == self.last_page:
            return str(self.first_page)
        return f"{self.first_page}-{self.last_page}"


@dataclass(frozen=True)
class Chunk:
    """An embeddable unit of one document."""

    doc_id: str
    index: int  # position within the document, 0-based
    section: str | None
    body: str  # as written in the PDF
    text: str  # what is embedded: title + heading + body
    language: str
    first_page: int
    last_page: int
    token_count: int

    @property
    def chunk_id(self) -> str:
        """Stable and readable, so a bad retrieval can be traced to a page."""
        return f"{self.doc_id}#{self.index:02d}"

    @property
    def pages(self) -> str:
        if self.first_page == self.last_page:
            return str(self.first_page)
        return f"{self.first_page}-{self.last_page}"

    def metadata(self, document: Document) -> dict[str, str | int | bool]:
        """Document metadata plus this chunk's own, all scalars for Chroma.

        Carries the document's fields down to the chunk so a retrieved result
        can be cited -- title, publisher, date, ISINs -- without a second
        lookup. Filtering still happens on ``doc_id``; see extract.Document.
        """
        return {
            **document.metadata(),
            "chunk_id": self.chunk_id,
            "chunk_index": self.index,
            "section": self.section or "",
            "language": self.language,
            "pages": self.pages,
            "token_count": self.token_count,
        }


@dataclass(frozen=True)
class _Block:
    """A paragraph block with the page it started on. Internal to this module."""

    text: str
    first_page: int
    last_page: int


def boilerplate_paragraphs(
    documents: list[Document], min_documents: int = 3
) -> frozenset[str]:
    """The repeated paragraphs to drop, as a set for exact matching.

    Exact matching rather than fuzzy: the disclaimer is emitted from a template,
    so it is character-identical everywhere, and a fuzzy rule would eventually
    swallow a real paragraph that two analysts happened to phrase alike.
    """
    return frozenset(
        paragraph
        for _, paragraph in repeated_paragraphs(documents, min_documents=min_documents)
    )


def _blocks(document: Document, drop: frozenset[str]) -> list[_Block]:
    """Paragraph blocks in reading order, boilerplate removed, pages stitched.

    Order matters here. Boilerplate is dropped *before* page-spanning sections
    are rejoined, because the banking note's second page begins with the English
    disclaimer: stitch first and that disclaimer is glued onto the end of the
    "Outlook" section, which is worse than indexing it separately.
    """
    blocks: list[_Block] = []
    for page in document.pages:
        first_on_page = True
        for text in page.text.split("\n\n"):
            text = text.strip()
            if not text or text == PUBLISHER_HEADER or text in drop:
                continue

            # No heading on the first surviving block of a page means the
            # section ran off the bottom of the previous one. Eight do here;
            # unstitched, their tails index with no heading at all.
            continues = first_on_page and blocks and not is_heading(text.split("\n")[0])
            if continues:
                previous = blocks[-1]
                blocks[-1] = _Block(
                    text=f"{previous.text}\n{text}",
                    first_page=previous.first_page,
                    last_page=page.number,
                )
            else:
                blocks.append(
                    _Block(text=text, first_page=page.number, last_page=page.number)
                )
            first_on_page = False
    return blocks


def split_sections(document: Document, drop: frozenset[str] = frozenset()) -> list[Section]:
    """The document as authored sections, in reading order."""
    sections = []
    for block in _blocks(document, drop):
        lines = block.text.split("\n")
        front_matter = any(marker in block.text for marker in FRONT_MATTER_MARKERS)
        if front_matter:
            # Title and key facts. Kept whole, including the title lines: this
            # is the block that carries the rating, the ticker and the ISIN.
            title = FRONT_MATTER_TITLE[classify(block.text)]
            body = block.text
        elif len(lines) > 1 and is_heading(lines[0]):
            title, body = lines[0].strip(), "\n".join(lines[1:]).strip()
        else:
            title, body = None, block.text
        sections.append(
            Section(
                title=title,
                body=body,
                first_page=block.first_page,
                last_page=block.last_page,
                is_front_matter=front_matter,
            )
        )
    return sections


def _split_words(text: str, max_tokens: int) -> list[str]:
    """Last-resort split of a run of text with no usable sentence boundary.

    Fills to an even target size rather than greedily to the ceiling. Greedy
    filling leaves the remainder in the final piece, which in practice produced
    a three-word chunk carrying nothing but a clause ending -- a fragment that
    still occupies a slot in the top-k and displaces a real answer.
    """
    total = count_tokens(text)
    pieces_needed = max(1, -(-total // max_tokens))  # ceiling division
    target = -(-total // pieces_needed)

    pieces: list[str] = []
    run: list[str] = []
    for word in text.split():
        if run and count_tokens(" ".join([*run, word])) > target:
            pieces.append(" ".join(run))
            run = []
        run.append(word)
    if run:
        pieces.append(" ".join(run))
    return pieces


def _split_to_budget(body: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Cut an oversized section into pieces, preferring sentence boundaries.

    Falls back to splitting between words when a single sentence is itself over
    budget. That fallback is not hypothetical for this corpus: pdftotext's
    handling of right-to-left text attaches numerals to the following word and
    displaces full stops, so an Arabic section can present as one long sentence
    with no boundary the regex can find.
    """
    if count_tokens(body) <= max_tokens:
        return [body]

    pieces: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            pieces.append(" ".join(current).strip())

    for sentence in _SENTENCE.split(body):
        sentence = sentence.strip()
        if not sentence:
            continue

        if count_tokens(sentence) > max_tokens:
            flush()
            # Keep the last run open so a following short sentence can join it
            # rather than becoming a chunk of its own.
            runs = _split_words(sentence, max_tokens)
            pieces.extend(runs[:-1])
            current = [runs[-1]]
            continue

        if current and count_tokens(" ".join([*current, sentence])) > max_tokens:
            flush()
            # Re-open with the tail of what was just emitted, so a sentence and
            # the figure it refers to are not separated by the cut.
            carry: list[str] = []
            for previous in reversed(current):
                if count_tokens(" ".join([previous, *carry])) > overlap_tokens:
                    break
                carry.insert(0, previous)
            current = carry
        current.append(sentence)

    flush()
    return [piece for piece in pieces if piece]


def _merge_short_sections(sections: list[Section]) -> list[Section]:
    """Fold a section too small to stand alone into the one that follows it.

    A stub -- a heading whose body did not survive, a one-line note -- embeds to
    a vector dominated by its heading and competes with the real section on the
    same topic. Merging forward rather than backward keeps the heading attached
    to the text it introduces. A trailing stub has nothing to merge into and is
    kept as it is; dropping it would lose text.

    The merged section takes the *following* section's heading, since that is
    the one with the body under it. The stub's own heading is not lost -- it is
    carried into the merged body, where it stays searchable.
    """
    merged: list[Section] = []
    pending: Section | None = None
    for section in sections:
        if pending is not None:
            head = f"{pending.title}\n{pending.body}" if pending.title else pending.body
            tail = f"{section.title}\n{section.body}" if section.title else section.body
            section = Section(
                title=section.title or pending.title,
                body=f"{head}\n\n{tail}",
                first_page=pending.first_page,
                last_page=section.last_page,
                is_front_matter=pending.is_front_matter or section.is_front_matter,
            )
            pending = None
        if not section.is_front_matter and count_tokens(section.body) < MIN_CHUNK_TOKENS:
            pending = section
            continue
        merged.append(section)
    if pending is not None:
        merged.append(pending)
    return merged


def chunk_document(
    document: Document,
    drop: frozenset[str] = frozenset(),
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[Chunk]:
    """One document's chunks, in reading order."""
    chunks: list[Chunk] = []
    for section in _merge_short_sections(split_sections(document, drop)):
        heading = f"{section.title}\n" if section.title else ""
        prefix = f"{document.title}\n{heading}\n"

        # The prefix is embedded too, so it is charged against the ceiling. An
        # Arabic sector title alone runs ~50 tokens.
        budget = max(max_tokens - count_tokens(prefix), MIN_CHUNK_TOKENS)

        for body in _split_to_budget(section.body, budget, overlap_tokens):
            text = f"{prefix}{body}"
            chunks.append(
                Chunk(
                    doc_id=document.doc_id,
                    index=len(chunks),
                    section=section.title,
                    body=body,
                    text=text,
                    # From the body: the title prefix is English on every
                    # document, and classifying `text` would relabel short
                    # Arabic sections as English.
                    language=classify(body),
                    first_page=section.first_page,
                    last_page=section.last_page,
                    token_count=count_tokens(text),
                )
            )
    return chunks


def chunk_corpus(
    documents: list[Document] | None = None,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[Chunk]:
    """Every chunk of every document, in publication then reading order.

    Boilerplate is detected across the whole corpus, so this is the entry point
    that produces indexable chunks -- chunking a document on its own cannot know
    which of its paragraphs are repeated elsewhere.
    """
    documents = extract_corpus() if documents is None else documents
    drop = boilerplate_paragraphs(documents)
    return [
        chunk
        for document in documents
        for chunk in chunk_document(document, drop, max_tokens, overlap_tokens)
    ]


def main(argv: list[str] | None = None) -> int:
    """``uv run python -m ingestion.chunk`` -- chunk everything and report."""
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="ingestion.chunk",
        description="Split the corpus into chunks and report their shape.",
    )
    parser.add_argument("--doc", default=None, help="print one document's chunks")
    parser.add_argument("--full", action="store_true", help="print every chunk in full")
    args = parser.parse_args(argv)

    documents = extract_corpus()
    chunks = chunk_corpus(documents)

    if args.doc or args.full:
        selected = [c for c in chunks if not args.doc or c.doc_id == args.doc]
        if not selected:
            raise SystemExit(f"no chunks for doc_id {args.doc!r}")
        for chunk in selected:
            print(f"\n{'-' * 70}")
            print(
                f"{chunk.chunk_id}  [{chunk.language}]  page {chunk.pages}  "
                f"{chunk.token_count} tokens  section={chunk.section!r}"
            )
            print(f"{'-' * 70}\n{chunk.text}")
        return 0

    print(f"{'document':40} {'chunks':>6} {'tokens':>7}  sections")
    for document in documents:
        own = [c for c in chunks if c.doc_id == document.doc_id]
        sections = ", ".join(c.section or "-" for c in own)
        print(
            f"{document.doc_id:40} {len(own):>6} "
            f"{sum(c.token_count for c in own):>7}  {sections[:60]}"
        )

    sizes = sorted(c.token_count for c in chunks)
    by_language: dict[str, list[int]] = {}
    for chunk in chunks:
        by_language.setdefault(chunk.language, []).append(chunk.token_count)

    print(f"\n{len(chunks)} chunks, {sum(sizes):,} tokens total")
    print(
        f"  size: min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]} "
        f"(ceiling {MAX_CHUNK_TOKENS})"
    )
    for language, counts in sorted(by_language.items()):
        counts.sort()
        print(
            f"  {language}: {len(counts):>3} chunks, median {counts[len(counts) // 2]} "
            f"tokens, max {counts[-1]}"
        )
    over = [c for c in chunks if c.token_count > MAX_CHUNK_TOKENS]
    for chunk in over:
        print(f"  warning: {chunk.chunk_id} is {chunk.token_count} tokens")
    return 1 if over else 0


if __name__ == "__main__":
    raise SystemExit(main())
