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
        "The text is: Reproduce\nissue\nyes\nResolve\nno",  # No inferred span stripping.
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
