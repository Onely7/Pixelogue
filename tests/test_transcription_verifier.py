"""Whole-answer transcription and public-region boundary failures."""

import pytest

from pixelogue.contracts import GateVerdict
from pixelogue.task_evidence import ImageRegion, TranscriptSource
from pixelogue.transcription_verifier import verify_transcription

REGION = ImageRegion(left=0, top=0, right=1, bottom=1)


def source(text: str = "Reproduce\nissue\nyes\nResolve\nno") -> TranscriptSource:
    return TranscriptSource(
        coverage="MET",
        expected_lines=tuple(text.split("\n")),
        source_region=REGION,
        requested_unit_complete=True,
        reason="Entire requested unit is readable",
    )


@pytest.mark.parametrize("wrapper", ["{}", '"{}"', "“{}”", "```\n{}\n```", "```python\n{}\n```"])
def test_exact_content_passes_with_only_an_outer_wrapper(wrapper: str) -> None:
    reading = source("if x:\n  print(x)")
    assert (
        verify_transcription((reading, reading), wrapper.format(reading.expected_text), REGION)
        is GateVerdict.MET
    )


@pytest.mark.parametrize(
    "answer",
    [
        "Reproduce\nissue\nResolve",  # Missing branch labels in the observed chart.
        "Reproduce\nissue\nyes\nResolve\nno\nextra",  # Matching substring is insufficient.
        # A lead-in longer than a short label is content, not a label.
        "The chart shows these labels from top to bottom in order: Reproduce issue yes Resolve no",
        "Reproduce, issue, yes, and Resolve, no",  # An added word is a change.
        "reproduce issue yes resolve no",  # Letter case is content.
    ],
)
def test_omissions_and_unrequested_trailing_content_are_rejected(answer: str) -> None:
    reading = source()
    assert verify_transcription((reading, reading), answer, REGION) is GateVerdict.NOT_MET


def test_disagreement_empty_answers_and_partially_bound_units_abstain() -> None:
    reading = source()
    assert (
        verify_transcription((reading, source("Different")), reading.expected_text, REGION)
        is GateVerdict.UNKNOWN
    )
    assert verify_transcription((reading, reading), "", REGION) is GateVerdict.UNKNOWN
    truncated = ImageRegion(left=0, top=0, right=1, bottom=0.93)
    assert (
        verify_transcription((reading, reading), reading.expected_text, truncated)
        is GateVerdict.UNKNOWN
    )


def test_a_partial_source_cannot_be_labeled_complete() -> None:
    with pytest.raises(ValueError):
        TranscriptSource(
            coverage="MET",
            expected_lines=("partial",),
            source_region=REGION,
            requested_unit_complete=False,
            reason="Bottom of requested box is cut off",
        )
    with pytest.raises(ValueError):
        TranscriptSource(
            coverage="UNKNOWN",
            expected_lines=("partial",),
            source_region=REGION,
            requested_unit_complete=False,
            reason="Cannot read whole unit",
        )


def test_quotes_in_the_original_text_are_not_mistaken_for_a_response_wrapper() -> None:
    reading = source('"Hello"')
    assert verify_transcription((reading, reading), '"Hello"', REGION) is GateVerdict.MET
    assert verify_transcription((reading, reading), "Hello", REGION) is GateVerdict.NOT_MET


@pytest.mark.parametrize(
    "answer",
    [
        "The text is: Reproduce\nissue\nyes\nResolve\nno",
        "Reproduce issue yes Resolve no",
        "Reproduce, issue; yes, Resolve, no.",
        "“Reproduce issue yes Resolve no”",
    ],
)
def test_short_labels_spacing_and_line_separators_are_formatting(answer: str) -> None:
    reading = source()
    assert verify_transcription((reading, reading), answer, REGION) is GateVerdict.MET


# Rejected in the 2026-10-07 diverse-100 run with both judges MET; the first five differ from
# the agreed reading only in formatting, the last three add words or labels.
BR_CASES = [
    (
        "Relativistic Outflow\nof Charged Particles\non Open Field Lines",
        "Relativistic Outflow of Charged Particles on Open Field Lines",
        GateVerdict.MET,
    ),
    ("Participants above age\n40 (n=15)", "Participants above age 40 (n=15)", GateVerdict.MET),
    (
        "2015\n2016\n2017\n2018\n2019\n2020",
        "2015, 2016, 2017, 2018, 2019, 2020",
        GateVerdict.MET,
    ),
    (
        "BETA 1: 2013-07-23_aqua-dream.png [modified] - Krita",
        "BETA 1: 2013-07-23_aqua-dream.png [modified] \u2013 Krita",
        GateVerdict.MET,
    ),
    ("Chi-square\nFisher's Exact\nTest", "Chi-square Fisher\u2019s Exact Test", GateVerdict.MET),
    (
        "1901\n1950\n2000\n2021",
        "The years are listed in chronological order from left to right: 1901, 1950, 2000, "
        "and 2021.",
        GateVerdict.NOT_MET,
    ),
    (
        "Highly democratic countries tend to have\nlower levels of corruption\nElectoral "
        "democracy index",
        "Title: Highly democratic countries tend to have lower levels of corruption\n"
        "Subtitle: Electoral democracy index",
        GateVerdict.NOT_MET,
    ),
    (
        "Female Prof. Salary ($1k/yr)\n36\n35",
        'The order is: 1. Y-axis title: "Female Prof. Salary ($1k/yr)"; 2. ticks: 36, 35',
        GateVerdict.NOT_MET,
    ),
]


@pytest.mark.parametrize(("reading", "answer", "verdict"), BR_CASES)
def test_measured_transcription_failures(reading: str, answer: str, verdict: GateVerdict) -> None:
    sources = (source(reading), source(reading))
    assert verify_transcription(sources, answer, REGION) is verdict


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("PANNECOT", "PANNE\u00c7OT"),
        ("are Payable in Advance.", "are payable in Advance."),
        ("Display 1 0 0\n1920 1080", "Display 1 0 800 600"),
    ],
)
def test_readers_that_differ_in_case_diacritics_or_words_abstain(first: str, second: str) -> None:
    assert (
        verify_transcription((source(first), source(second)), first, REGION) is GateVerdict.UNKNOWN
    )


def test_readers_that_differ_only_in_line_breaks_agree() -> None:
    sources = (source("Hello World"), source("Hello\nWorld"))
    assert verify_transcription(sources, "Hello World", REGION) is GateVerdict.MET


def test_indented_text_keeps_its_line_structure() -> None:
    reading = source("def f(x):\n    return x")
    sources = (reading, reading)
    assert verify_transcription(sources, "```python\ndef f(x):\n    return x\n```", REGION) is (
        GateVerdict.MET
    )
    assert verify_transcription(sources, "def f(x): return x", REGION) is GateVerdict.NOT_MET
    assert verify_transcription(sources, "def f(x):\nreturn x", REGION) is GateVerdict.NOT_MET
