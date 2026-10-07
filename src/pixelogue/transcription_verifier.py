"""Complete-answer comparison against answer-blind visible transcriptions.

Comparison is by words and punctuation after Unicode, dash and quotation-mark normalization, so
line breaks and spacing do not decide the result, except for indented text such as code, which
is compared line by line. A source line break may be written as a comma or semicolon. An answer
may open with one short label clause ending in a colon and may end with one extra period; any
missing, added or changed word still fails. Readers that disagree, including only in letter case
or diacritics, leave the result undecided.
"""

from __future__ import annotations

import re
import unicodedata

from pixelogue.contracts import GateVerdict
from pixelogue.task_evidence import ImageRegion, TranscriptSource

_DASHES = dict.fromkeys(
    map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"), "-"
)
_QUOTES = {
    **dict.fromkeys(map(ord, "\u201c\u201d\u201e\u201f\u00ab\u00bb"), '"'),
    **dict.fromkeys(map(ord, "\u2018\u2019\u201a\u201b"), "'"),
}
_TOKEN = re.compile(r"\w+|[^\w\s]")
_LABEL = re.compile(r"([^\n:]{1,120}):[ \t]*(?:\n|(?=\S))")
# A label such as "The text reads:" may precede the transcript; longer clauses are content.
MAX_LABEL_WORDS = 8
_LINE_BREAK_MARKS = frozenset({",", ";"})


def _unify(text: str) -> str:
    """Apply compatibility normalization and one form for dashes and quotation marks."""
    return unicodedata.normalize("NFKC", text).translate(_DASHES).translate(_QUOTES)


def _lines(text: str) -> list[str]:
    """Return normalized lines without trailing spaces or surrounding blank lines."""
    lines = [line.rstrip() for line in _unify(text).split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _tokens(text: str) -> list[tuple[str, bool]]:
    """Return words and punctuation marks, each flagged when it ends its source line."""
    result: list[tuple[str, bool]] = []
    for line in _unify(text).split("\n"):
        tokens = _TOKEN.findall(line)
        result.extend((token, index == len(tokens) - 1) for index, token in enumerate(tokens))
    return result


def _indented(text: str) -> bool:
    return any(line[:1] in {" ", "\t"} and line.strip() for line in _lines(text))


def readings_agree(first: str, second: str) -> bool:
    """Accept two blind readings only when their words and punctuation are identical."""
    if _indented(first) or _indented(second):
        return _lines(first) == _lines(second)
    return [token for token, _ in _tokens(first)] == [token for token, _ in _tokens(second)]


def transcript_matches(reference: str, candidate: str) -> bool:
    """Compare a candidate with an agreed reading, allowing only formatting differences."""
    if _indented(reference):
        return _lines(reference) == _lines(candidate)
    expected = _tokens(reference)
    observed = [token for token, _ in _tokens(candidate)]
    i = j = 0
    while i < len(expected) or j < len(observed):
        if i < len(expected) and j < len(observed) and expected[i][0] == observed[j]:
            i += 1
            j += 1
        elif (
            j < len(observed) and observed[j] in _LINE_BREAK_MARKS and i > 0 and expected[i - 1][1]
        ):
            j += 1
        elif i < len(expected) and expected[i][0] in _LINE_BREAK_MARKS and expected[i][1]:
            i += 1
        else:
            return False
    return True


def answer_variants(answer: str) -> tuple[str, ...]:
    """Return the answer with an optional short label, outer wrapper and final period removed."""
    text = answer.strip()
    bodies = [text]
    label = _LABEL.match(text)
    if label is not None and len(label.group(1).split()) <= MAX_LABEL_WORDS:
        bodies.append(text[label.end() :].strip())
    variants: list[str] = []
    for body in bodies:
        for item in (body, transcript_content(body)):
            variants.append(item)
            if item.endswith(".") and not item.endswith(".."):
                variants.append(item[:-1])
    return tuple(dict.fromkeys(variants))


def transcript_content(answer: str) -> str:
    """Remove only a surrounding quotation or one complete code fence.

    No matching substring, explanatory sentence or extra trailing content is
    discarded. Interior punctuation, indentation and line breaks remain exact.
    """
    fence = re.fullmatch(r"```[A-Za-z0-9_+-]*\n([\s\S]*)\n```", answer)
    if fence is not None:
        return fence.group(1)
    for opening, closing in (('"', '"'), ("“", "”")):
        if len(answer) >= 2 and answer.startswith(opening) and answer.endswith(closing):
            return answer[1:-1]
    return answer


def extractive_content(text: str) -> str:
    """Normalize spacing, dash and quote forms, sentence-initial case and a final prose period.

    Interior spelling, punctuation, numbers, qualifiers and order stay literal.
    This comparison is for extractive QA, never verbatim transcription.
    """
    text = re.sub(r"\s+", " ", _unify(transcript_content(text.strip()))).strip()
    if re.search(r"[A-Za-z]\.$", text):
        text = text[:-1]
    return text[:1].casefold() + text[1:]


def verify_extractive(
    sources: tuple[TranscriptSource, TranscriptSource],
    candidate_answer: str,
    region: ImageRegion | None,
) -> GateVerdict:
    """Compare the whole answer with two independently read answer-bearing spans."""
    for source in sources:
        if verify_transcription((source, source), source.expected_text, region) != GateVerdict.MET:
            return GateVerdict.UNKNOWN
    expected = extractive_content(sources[0].expected_text)
    if not expected or expected != extractive_content(sources[1].expected_text):
        return GateVerdict.UNKNOWN
    return (
        GateVerdict.MET if expected == extractive_content(candidate_answer) else GateVerdict.NOT_MET
    )


def verify_transcription(
    sources: tuple[TranscriptSource, TranscriptSource],
    candidate_answer: str,
    region: ImageRegion | None,
) -> GateVerdict:
    """Require agreement, full-unit containment and a whole answer that differs only in format."""
    for source in sources:
        if source.coverage != "MET" or not source.requested_unit_complete:
            return GateVerdict.UNKNOWN
        source_region = source.source_region
        if source_region is None:
            return GateVerdict.UNKNOWN
        if region is not None and not (
            region.left <= source_region.left < source_region.right <= region.right
            and region.top <= source_region.top < source_region.bottom <= region.bottom
        ):
            return GateVerdict.UNKNOWN
    reference = sources[0].expected_text
    if not readings_agree(reference, sources[1].expected_text):
        return GateVerdict.UNKNOWN
    if any(transcript_matches(reference, variant) for variant in answer_variants(candidate_answer)):
        return GateVerdict.MET
    if not transcript_content(candidate_answer).strip():
        return GateVerdict.UNKNOWN
    return GateVerdict.NOT_MET
