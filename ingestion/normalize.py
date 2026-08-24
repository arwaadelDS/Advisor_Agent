"""Normalise text extracted from PDFs so that Arabic embeds correctly.

pdftotext returns Arabic as presentation forms (``مصرف`` -> ``ﻣﺼﺮﻑ``: same
word on screen, different codepoints), wraps every RTL run in bidi controls,
and drops the space at Arabic/Latin boundaries so figures fuse to the word
beside them (``435مليون``). None of it errors and all of it wrecks retrieval,
because the query and the indexed text end up sharing almost no tokens.

``normalize`` fixes those three, in that order, and nothing else. Alef folding,
diacritic stripping and pdftotext's displaced punctuation are all left alone:
the first two are lossy, and repairing punctuation means asserting where a mark
belonged, which the obvious rule gets wrong (moving the stop left is right for
``.وشهد`` and wrong for ``.2025``).

Pure string functions, no I/O: use ingestion/inspect_pdf.py to see the effect
on a real file. ``normalize`` turns form feeds into paragraph breaks, so
callers needing per-page provenance must split the raw text on ``\\f`` first.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# The full set, not just the three this corpus uses -- other typesetters emit
# others. Escapes rather than literals: these are invisible characters.
BIDI_CONTROLS = frozenset(
    "\u200e"  # LEFT-TO-RIGHT MARK
    "\u200f"  # RIGHT-TO-LEFT MARK
    "\u202a"  # LEFT-TO-RIGHT EMBEDDING
    "\u202b"  # RIGHT-TO-LEFT EMBEDDING
    "\u202c"  # POP DIRECTIONAL FORMATTING
    "\u202d"  # LEFT-TO-RIGHT OVERRIDE
    "\u202e"  # RIGHT-TO-LEFT OVERRIDE
    "\u2066"  # LEFT-TO-RIGHT ISOLATE
    "\u2067"  # RIGHT-TO-LEFT ISOLATE
    "\u2068"  # FIRST STRONG ISOLATE
    "\u2069"  # POP DIRECTIONAL ISOLATE
)

# Kashida: decorative letter-stretching, no phonetic value. Stripped *after*
# NFKC, since some presentation forms decompose into tatweel plus a mark.
TATWEEL = "\u0640"

# A detached glyph tail. The one codepoint in Presentation Forms-B with no
# compatibility decomposition and no meaning, so NFKC leaves it behind.
TAIL_FRAGMENT = "\ufe73"

# Visible, but carrying no meaning: safe to drop before embedding.
DECORATIVE = frozenset({TATWEEL, TAIL_FRAGMENT})

# Zero-width characters that carry no meaning in Arabic but do tokenise.
ZERO_WIDTH = frozenset(
    "\u200b"  # ZERO WIDTH SPACE
    "\u200c"  # ZERO WIDTH NON-JOINER
    "\u200d"  # ZERO WIDTH JOINER
    "\ufeff"  # ZERO WIDTH NO-BREAK SPACE / BOM
    "\u00ad"  # SOFT HYPHEN
)

ARABIC_BASE = (0x0600, 0x06FF)
PRESENTATION_FORMS_A = (0xFB50, 0xFDFF)
PRESENTATION_FORMS_B = (0xFE70, 0xFEFF)

# Letters only, as ranges rather than the whole 0600-06FF block: that block also
# holds the Arabic-Indic digits and the combining marks, and neither of those
# beside a numeral is the missing word boundary this repairs.
_ARABIC_LETTERS = (
    "ء-غ"  # hamza .. ghain
    "ف-ي"  # feh .. yeh (skips U+0640 tatweel)
    "ٮ-ٯ"  # dotless beh, dotless qaf
    "ٱ-ۓ"  # alef wasla .. yeh barree
    "ە"  # aeh
    "ۺ-ۿ"  # sheen/dad/ghain dotless variants, heh with inverted v
)

# The seam between an Arabic letter and an ASCII digit, either direction. Zero
# width -- it matches the position, so substituting consumes nothing.
ARABIC_DIGIT_BOUNDARY = re.compile(
    f"(?<=[{_ARABIC_LETTERS}])(?=[0-9])|(?<=[0-9])(?=[{_ARABIC_LETTERS}])"
)

_HORIZONTAL_WS = re.compile(r"[^\S\n]+")
_BLANK_RUN = re.compile(r"\n{3,}")


def _in(ch: str, span: tuple[int, int]) -> bool:
    return span[0] <= ord(ch) <= span[1]


_NOISE = BIDI_CONTROLS | ZERO_WIDTH | DECORATIVE


def strip_noise(text: str) -> str:
    """Remove characters that carry no linguistic content.

    Bidi controls and zero-width characters (invisible), plus tatweel and the
    tail fragment (visible but purely decorative). Deliberately does *not* touch
    Arabic symbols or honorific ligatures -- those have no base-letter form
    because they are content, not corruption.
    """
    return "".join(ch for ch in text if ch not in _NOISE)


def space_arabic_digits(text: str) -> str:
    """Restore the word boundary pdftotext drops between Arabic and a numeral.

    ``435مليون`` -> ``435 مليون``, ``الربع1.2`` -> ``الربع 1.2``. Only inserts;
    never removes, reorders or rewrites a character, so the original text is
    recoverable by deleting the space again.

    Must run after ``strip_noise``: a bidi control sits at precisely this
    boundary, and while it is still there the digit and the letter are not
    adjacent for the pattern to match.
    """
    return ARABIC_DIGIT_BOUNDARY.sub(" ", text)


def collapse_whitespace(text: str) -> str:
    """Tidy pdftotext's ``-layout`` spacing without losing paragraph structure.

    ``-layout`` pads columns with runs of spaces to preserve visual position, so
    a two-column table row arrives as ``Company        SABIC``. Collapsing those
    runs joins the cells into a readable line. Blank lines are kept, because
    paragraph boundaries are what section-aware chunking splits on.
    """
    text = text.replace("\f", "\n\n")
    lines = (_HORIZONTAL_WS.sub(" ", line).strip() for line in text.split("\n"))
    return _BLANK_RUN.sub("\n\n", "\n".join(lines)).strip()


def normalize(text: str) -> str:
    """Make extracted PDF text safe to chunk and embed.

    NFKC -> strip noise -> space Arabic/digit seams -> collapse whitespace. Every
    step depends on the one before it: NFKC must run first so presentation forms
    become base letters; tatweel stripping must run after it because NFKC can
    *produce* tatweel; digit spacing must run after the bidi controls are gone,
    because a control sits on the boundary it looks for; and whitespace
    collapsing must run last because NFKC can produce spaces (U+FDFA expands to
    a four-word phrase).

    Idempotent: ``normalize(normalize(t)) == normalize(t)``.
    """
    text = unicodedata.normalize("NFKC", text)
    return collapse_whitespace(space_arabic_digits(strip_noise(text)))


@dataclass(frozen=True)
class Census:
    """Counts of the character classes that matter for Arabic retrieval."""

    total: int
    arabic_base: int
    arabic_symbols: int
    presentation_a: int
    presentation_b: int
    bidi_controls: int
    decorative: int
    zero_width: int
    other_non_ascii: int

    @property
    def presentation_forms(self) -> int:
        """Forms NFKC is able to resolve, and so must have resolved already."""
        return self.presentation_a + self.presentation_b

    @property
    def is_clean(self) -> bool:
        """True when nothing is left that would corrupt an embedding."""
        return (
            self.presentation_forms == 0
            and self.bidi_controls == 0
            and self.decorative == 0
            and self.zero_width == 0
        )


def census(text: str) -> Census:
    """Count the character classes that decide whether this text will embed well.

    A character in the presentation blocks counts as a *presentation form* only
    if it has a compatibility decomposition -- that is, only if NFKC was
    supposed to fold it into base letters. The 40-odd codepoints that have no
    decomposition (Quranic annotation symbols, ornate parentheses, honorific
    ligatures such as U+FDFD BISMILLAH) are counted as ``arabic_symbols``
    instead. They stay in the text because they are content: there is no base
    form to fold them into, and dropping them would lose meaning.
    """
    counts = dict.fromkeys(
        ("arabic_base", "arabic_symbols", "presentation_a", "presentation_b",
         "bidi_controls", "decorative", "zero_width", "other_non_ascii"),
        0,
    )
    for ch in text:
        if ch in BIDI_CONTROLS:
            counts["bidi_controls"] += 1
        elif ch in DECORATIVE:
            counts["decorative"] += 1
        elif ch in ZERO_WIDTH:
            counts["zero_width"] += 1
        elif _in(ch, PRESENTATION_FORMS_A) or _in(ch, PRESENTATION_FORMS_B):
            block = "presentation_a" if _in(ch, PRESENTATION_FORMS_A) else "presentation_b"
            counts[block if unicodedata.decomposition(ch) else "arabic_symbols"] += 1
        elif _in(ch, ARABIC_BASE):
            counts["arabic_base"] += 1
        elif ord(ch) > 127:
            counts["other_non_ascii"] += 1
    return Census(total=len(text), **counts)


def is_clean(text: str) -> bool:
    """Post-condition for ingestion: nothing left that would corrupt an embedding."""
    return census(text).is_clean


@dataclass(frozen=True)
class CharChange:
    """What normalisation did to one source character."""

    source: str
    result: str
    reason: str

    @property
    def changed(self) -> bool:
        return self.source != self.result


def explain(text: str) -> list[CharChange]:
    """Per-character account of what ``normalize`` does and why.

    Used by the ``--inspect`` CLI to make the transformation auditable rather
    than something to take on trust. Digit spacing and whitespace collapsing act
    on the seams *between* characters rather than on characters, so neither is
    represented here.
    """
    changes: list[CharChange] = []
    for ch in text:
        if ch in BIDI_CONTROLS:
            changes.append(CharChange(ch, "", "removed: bidi control"))
        elif ch == TATWEEL:
            changes.append(CharChange(ch, "", "removed: tatweel"))
        elif ch == TAIL_FRAGMENT:
            changes.append(CharChange(ch, "", "removed: tail fragment"))
        elif ch in ZERO_WIDTH:
            changes.append(CharChange(ch, "", "removed: zero-width"))
        else:
            folded = strip_noise(unicodedata.normalize("NFKC", ch))
            if folded == ch:
                changes.append(CharChange(ch, ch, "unchanged"))
            elif len(folded) > 1:
                changes.append(CharChange(ch, folded, "NFKC: ligature split"))
            else:
                changes.append(CharChange(ch, folded, "NFKC: presentation form"))
    return changes
