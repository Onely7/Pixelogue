"""Calibrated ruler, gauge and clock readings with bounded precision."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Annotated, Literal

from pydantic import Field, model_validator

from pixelogue.config import StrictModel
from pixelogue.contracts import GateVerdict
from pixelogue.errors import ExecutionError
from pixelogue.rules import NumericValue, parse_numeric_lexeme
from pixelogue.task_evidence import ImageRegion


def _fraction(text: str) -> Fraction:
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            result = Fraction(int(numerator), int(denominator))
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("Invalid rational scale coordinate") from exc
        return result
    try:
        return parse_numeric_lexeme(text)
    except ExecutionError as exc:
        raise ValueError("Invalid scale coordinate") from exc


class ScaleMark(StrictModel):
    """A labeled calibration tick and its position on the declared axis."""

    position: str
    value: NumericValue
    printed_label: str = Field(min_length=1)
    region: ImageRegion

    @model_validator(mode="after")
    def check_label(self) -> ScaleMark:
        """A calibration value must be printed in its tick label."""
        if self.value.value not in self.printed_label:
            raise ValueError("Calibration value absent from printed tick label")
        coordinate = _fraction(self.position)
        if not 0 <= coordinate <= 1:
            raise ValueError("Scale mark outside normalized axis")
        return self


class ScaleSource(StrictModel):
    """Blind pointer and calibration evidence in a bound image region."""

    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    kind: Literal["linear", "clock"]
    scope_id: str
    view_id: str
    scope_region: ImageRegion
    pointer_region: ImageRegion | None = None
    pointer_position: str | None = None
    hour_position: str | None = None
    marks: Annotated[tuple[ScaleMark, ...], Field(max_length=32)] = ()
    unit: str | None = None
    precision_statement: str
    precision_digits: Annotated[int, Field(ge=0, le=6)] = 0
    value_resolution: str | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_source(self) -> ScaleSource:
        """Require an actual local pointer and type-specific calibration."""
        if self.coverage != "MET":
            if self.pointer_region is not None or self.pointer_position is not None or self.marks:
                raise ValueError("Incomplete scale extraction cannot certify readings")
            return self
        if self.pointer_region is None or self.pointer_position is None:
            raise ValueError("Scale reading needs visible pointer and position")
        region = self.pointer_region
        scope = self.scope_region
        if not (
            scope.left <= region.left < region.right <= scope.right
            and scope.top <= region.top < region.bottom <= scope.bottom
        ):
            raise ValueError("Pointer lies outside image scope")
        for mark in self.marks:
            region = mark.region
            if not (
                scope.left <= region.left < region.right <= scope.right
                and scope.top <= region.top < region.bottom <= scope.bottom
            ):
                raise ValueError("Scale mark lies outside image scope")
        position = _fraction(self.pointer_position)
        if not 0 <= position <= 1:
            raise ValueError("Pointer position outside normalized scale")
        if self.kind == "linear":
            if (
                len(self.marks) < 2
                or self.value_resolution is None
                or self.hour_position is not None
            ):
                raise ValueError("Linear scale needs two marks and declared resolution")
            if _fraction(self.value_resolution) <= 0:
                raise ValueError("Scale resolution must be positive")
        elif self.hour_position is None or self.marks:
            raise ValueError("Clock needs hour and minute hands, not linear tick values")
        return self


class ScaleAnswer(StrictModel):
    """Answer-only typed numeric value or clock time."""

    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    answer_quote: str = ""
    value: NumericValue | None = None
    time: str | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_shape(self) -> ScaleAnswer:
        """Require one quoted reading for a complete parse."""
        populated = sum(item is not None for item in (self.value, self.time))
        if self.coverage == "MET" and (populated != 1 or not self.answer_quote):
            raise ValueError("Complete scale answer needs one quoted reading")
        if self.coverage != "MET" and (populated or self.answer_quote):
            raise ValueError("Incomplete scale answer cannot provide a reading")
        if self.value is not None and self.value.value not in self.answer_quote:
            raise ValueError("Parsed number absent from answer quote")
        if self.time is not None and self.time not in self.answer_quote:
            raise ValueError("Parsed time absent from answer quote")
        return self


def _canonical(source: ScaleSource) -> dict[str, object]:
    result = source.model_dump(mode="json", exclude={"reason", "pointer_region"})
    for mark in result["marks"]:
        mark.pop("region", None)
    result["marks"].sort(key=lambda item: _fraction(item["position"]))
    return result


def _linear(source: ScaleSource) -> Fraction | None:
    marks = sorted(source.marks, key=lambda item: _fraction(item.position))
    positions = [_fraction(mark.position) for mark in marks]
    values = [parse_numeric_lexeme(mark.value.value) for mark in marks]
    if len(set(positions)) != len(positions) or len({mark.value.unit for mark in marks}) != 1:
        return None
    if marks[0].value.unit != source.unit:
        return None
    slope = (values[1] - values[0]) / (positions[1] - positions[0])
    if slope == 0 or any(
        value != values[0] + slope * (position - positions[0])
        for position, value in zip(positions, values, strict=True)
    ):
        return None
    assert source.pointer_position is not None
    pointer = _fraction(source.pointer_position)
    if not positions[0] <= pointer <= positions[-1]:
        return None
    return values[0] + slope * (pointer - positions[0])


def _clock(source: ScaleSource) -> str | None:
    if source.hour_position is None or source.pointer_position is None:
        return None
    minute_position = _fraction(source.pointer_position)
    minute_exact = minute_position * 60
    minute = int(minute_exact + Fraction(1, 2)) % 60
    if abs(minute_exact - minute) > Fraction(1, 4):
        return None
    hour_position = _fraction(source.hour_position)
    if not 0 <= hour_position < 1:
        return None
    hour = int(hour_position * 12)
    expected_hour_position = Fraction(hour, 12) + Fraction(minute, 720)
    if abs(hour_position - expected_hour_position) > Fraction(1, 240):
        return None
    return f"{12 if hour == 0 else hour}:{minute:02d}"


def _round_to_resolution(value: Fraction, resolution: Fraction) -> Fraction:
    scaled = abs(value / resolution)
    quotient, remainder = divmod(scaled.numerator, scaled.denominator)
    rounded = quotient + (2 * remainder >= scaled.denominator)
    return (1 if value >= 0 else -1) * rounded * resolution


def verify_scale(
    sources: tuple[ScaleSource, ScaleSource],
    answers: tuple[ScaleAnswer, ScaleAnswer],
    precision: object,
    scope_id: str,
    view_id: str,
    candidate_answer: str,
) -> GateVerdict:
    """Recompute a reading from agreed marks and reject excessive numerical precision."""
    if any(
        source.coverage != "MET"
        or source.scope_id != scope_id
        or source.view_id != view_id
        or source.precision_statement != precision
        for source in sources
    ) or _canonical(sources[0]) != _canonical(sources[1]):
        return GateVerdict.UNKNOWN
    if any(
        answer.coverage != "MET" or answer.answer_quote not in candidate_answer
        for answer in answers
    ):
        return GateVerdict.UNKNOWN
    if answers[0].model_dump(mode="json", exclude={"reason"}) != answers[1].model_dump(
        mode="json", exclude={"reason"}
    ):
        return GateVerdict.UNKNOWN
    source, answer = sources[0], answers[0]
    if source.kind == "clock":
        expected = _clock(source)
        if (
            expected is None
            or answer.time is None
            or not re.fullmatch(r"(?:[1-9]|1[0-2]):[0-5][0-9]", answer.time)
        ):
            return GateVerdict.UNKNOWN
        return GateVerdict.MET if answer.time == expected else GateVerdict.NOT_MET
    try:
        expected = _linear(source)
        if expected is None or answer.value is None or source.value_resolution is None:
            return GateVerdict.UNKNOWN
        if answer.value.unit != source.unit:
            return GateVerdict.NOT_MET
        observed = parse_numeric_lexeme(answer.value.value)
        resolution = _fraction(source.value_resolution)
    except (ExecutionError, ValueError, ZeroDivisionError):
        return GateVerdict.UNKNOWN
    decimals = answer.value.value.split(".")[-1] if "." in answer.value.value else ""
    if len(decimals) > source.precision_digits:
        return GateVerdict.UNKNOWN
    return (
        GateVerdict.MET
        if observed == _round_to_resolution(expected, resolution)
        else GateVerdict.NOT_MET
    )
