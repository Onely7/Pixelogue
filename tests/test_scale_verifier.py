"""Linear and clock calibration, precision and missing evidence."""

from __future__ import annotations

from pixelogue.contracts import GateVerdict
from pixelogue.rules import NumericValue
from pixelogue.scale_verifier import ScaleAnswer, ScaleMark, ScaleSource, verify_scale
from pixelogue.task_evidence import ImageRegion

REGION = ImageRegion(left=0, top=0, right=1, bottom=1)


def _linear() -> ScaleSource:
    return ScaleSource(
        coverage="MET",
        kind="linear",
        scope_id="scope",
        view_id="view",
        scope_region=REGION,
        pointer_region=REGION,
        pointer_position="0.45",
        marks=(
            ScaleMark(
                position="0",
                value=NumericValue(value="0", unit="kg"),
                printed_label="0 kg",
                region=REGION,
            ),
            ScaleMark(
                position="1",
                value=NumericValue(value="10", unit="kg"),
                printed_label="10 kg",
                region=REGION,
            ),
        ),
        unit="kg",
        precision_statement="nearest 1 kg",
        precision_digits=0,
        value_resolution="1",
        reason="Both ticks and pointer visible",
    )


def _answer(value: str, unit: str = "kg") -> ScaleAnswer:
    return ScaleAnswer(
        coverage="MET",
        answer_quote=f"{value} {unit}",
        value=NumericValue(value=value, unit=unit),
        reason="literal",
    )


def _check(
    source: ScaleSource, answer: ScaleAnswer, precision: str = "nearest 1 kg"
) -> GateVerdict:
    return verify_scale(
        (source, source),
        (answer, answer),
        precision,
        source.scope_id,
        source.view_id,
        answer.answer_quote,
    )


def test_linear_interpolation_rounds_only_to_public_resolution() -> None:
    source = _linear()
    assert _check(source, _answer("5")) is GateVerdict.MET
    assert _check(source, _answer("4")) is GateVerdict.NOT_MET
    assert _check(source, _answer("4.5")) is GateVerdict.UNKNOWN
    assert _check(source, _answer("5", "g")) is GateVerdict.NOT_MET


def test_non_linear_or_incompatible_calibration_abstains() -> None:
    source = _linear()
    inconsistent = source.model_copy(
        update={
            "marks": (
                *source.marks,
                ScaleMark(
                    position="0.5",
                    value=NumericValue(value="6", unit="kg"),
                    printed_label="6 kg",
                    region=REGION,
                ),
            )
        }
    )
    assert _check(inconsistent, _answer("5")) is GateVerdict.UNKNOWN
    assert _check(source, _answer("5"), precision="nearest 0.1 kg") is GateVerdict.UNKNOWN


def test_clock_checks_both_hands_and_uses_twelve_hour_time() -> None:
    clock = ScaleSource(
        coverage="MET",
        kind="clock",
        scope_id="scope",
        view_id="view",
        scope_region=REGION,
        pointer_region=REGION,
        pointer_position="1/4",
        hour_position="13/48",
        unit=None,
        precision_statement="minute",
        reason="Both hands visible",
    )
    answer = ScaleAnswer(coverage="MET", answer_quote="3:15", time="3:15", reason="literal")
    assert _check(clock, answer, precision="minute") is GateVerdict.MET
    wrong = answer.model_copy(update={"answer_quote": "4:15", "time": "4:15"})
    assert _check(clock, wrong, precision="minute") is GateVerdict.NOT_MET
    ambiguous = clock.model_copy(update={"hour_position": "1/2"})
    assert _check(ambiguous, answer, precision="minute") is GateVerdict.UNKNOWN


def test_disagreeing_extractions_and_other_view_abstain() -> None:
    source = _linear()
    answer = _answer("5")
    other = source.model_copy(update={"pointer_position": "0.6"})
    assert (
        verify_scale(
            (source, other),
            (answer, answer),
            "nearest 1 kg",
            source.scope_id,
            source.view_id,
            answer.answer_quote,
        )
        is GateVerdict.UNKNOWN
    )
    assert (
        verify_scale(
            (source, source),
            (answer, answer),
            "nearest 1 kg",
            source.scope_id,
            "other",
            answer.answer_quote,
        )
        is GateVerdict.UNKNOWN
    )
