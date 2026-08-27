"""Check an answer's citations against the extracts it was given.

``format_context`` numbers the extracts ``[1]``, ``[2]`` ... in list order and
``render_citations`` enumerates the same list, so the numbering lines up by
construction. What does not is the model staying inside the range: nothing stops
it writing ``[7]`` after being shown five extracts, and on screen that reads
like any other citation, with nothing behind it to open.

Two things are checkable without a second model:

* every ``[n]`` has an extract behind it. A number outside ``1..len(chunks)``
  refers to research that was never retrieved, and fails.
* how much of the answer carries a citation at all. Counted, not failed -- a
  refusal has no citations and is the answer we want.

Whether ``[2]`` supports the sentence citing it is not checked here. That needs
an entailment model or a judge LLM.

Sentence splitting is approximate: "e.g." splits one sentence into two. That
only moves the uncited count, which is a signal to read rather than a number to
report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# "[2]", plus the grouped forms models drift into: "[1, 2]", and the same with
# the Arabic comma, which is what an Arabic answer usually comes back with.
_MARKER = re.compile(r"\[\s*(\d+(?:\s*[,،]\s*\d+)*)\s*\]")

_SEPARATOR = re.compile(r"[,،]")

# Arabic ends a question with "؟" and a sentence with "." as English does.
# "؛" is a semicolon and does not end one.
_SENTENCE_END = re.compile(r"(?<=[.!?؟])\s+|\n+")

# A fragment made only of citations belongs to the sentence before it: models
# write both "... margin pressure [1]." and "... margin pressure. [1]".
_ONLY_MARKERS = re.compile(r"^(?:\s*\[[\d\s,،]+\]\s*)+$")

# Any letter in any script. Used to drop fragments that are pure punctuation or
# a bare number, which are not sentences anyone failed to cite.
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


@dataclass(frozen=True)
class CitationReport:
    """What an answer's citations line up with, and what they do not.

    ``extracts`` is how many the model was shown; ``cited`` the numbers it used,
    deduplicated and in the order they appear in the prose.
    """

    extracts: int
    cited: tuple[int, ...]
    out_of_range: tuple[int, ...]
    unused: tuple[int, ...]
    sentences: int
    uncited: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True when every reference points at an extract that exists."""
        # Uncited sentences and unused extracts are both normal, so neither
        # counts against this.
        return not self.out_of_range

    def summary(self) -> str:
        """One line, for a log or an eval row."""
        if self.extracts:
            head = f"{len(self.cited)}/{self.extracts} extracts cited"
        else:
            head = "no extracts retrieved"
        if self.out_of_range:
            numbers = ", ".join(f"[{n}]" for n in self.out_of_range)
            return f"{head}; out of range: {numbers}"
        if self.uncited:
            return f"{head}; {len(self.uncited)}/{self.sentences} sentences uncited"
        return head


def markers(text: str) -> list[int]:
    """Every extract number referenced in ``text``, in the order written.

    Groups expand, so "[1, 2]" is two references. Brackets holding anything but
    digits ("[see note]") are not citations.
    """
    found: list[int] = []
    for match in _MARKER.finditer(text or ""):
        found.extend(int(part) for part in _SEPARATOR.split(match.group(1)))
    return found


def sentences(text: str) -> list[str]:
    """The answer split into the units a citation is expected to sit on.

    A markdown bullet counts as one. A citation-only fragment is folded back
    into the sentence before it, since models write both "... pressure [1]."
    and "... pressure. [1]".
    """
    parts: list[str] = []
    for raw in _SENTENCE_END.split(text or ""):
        part = raw.strip()
        if not part:
            continue
        if _ONLY_MARKERS.match(part) and parts:
            parts[-1] = f"{parts[-1]} {part}"
            continue
        if not _LETTER.search(part):
            continue
        parts.append(part)
    return parts


def check_citations(answer: str, result: Any) -> CitationReport:
    """Compare an answer's citations against the extracts it was composed from.

    Only the length of ``result.chunks`` is read, and a plain dict works too, so
    an eval can call this without building the schema. No network, no model.
    """
    chunks = getattr(result, "chunks", None)
    if chunks is None and isinstance(result, dict):
        chunks = result.get("chunks")
    extracts = len(chunks or [])

    # fromkeys rather than set(): keeps first-appearance order.
    cited = tuple(dict.fromkeys(markers(answer)))
    lines = sentences(answer)

    return CitationReport(
        extracts=extracts,
        cited=cited,
        out_of_range=tuple(n for n in cited if not 1 <= n <= extracts),
        unused=tuple(n for n in range(1, extracts + 1) if n not in cited),
        sentences=len(lines),
        uncited=tuple(line for line in lines if not _MARKER.search(line)),
    )
