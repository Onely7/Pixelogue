"""Finite rule search with explicit uniqueness for visual pattern tasks."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import Annotated, Literal

from pydantic import Field, model_validator

from pixelogue.config import StrictModel
from pixelogue.contracts import GateVerdict
from pixelogue.errors import ExecutionError
from pixelogue.rules import parse_numeric_lexeme
from pixelogue.task_evidence import ImageRegion

PATTERN_TASKS = frozenset({"pattern_rule", "pattern_completion", "pattern_exception"})
RuleKind = Literal[
    "constant",
    "translation",
    "rotation",
    "reflection",
    "count_progression",
    "attribute_cycle",
    "set_composition",
]


class PatternFrame(StrictModel):
    """A finite visible panel represented by registered observable features."""

    frame_id: str = Field(min_length=1)
    x: str | None = None
    y: str | None = None
    quarter_turn: Annotated[int, Field(ge=0, le=3)] | None = None
    mirrored: bool | None = None
    count: Annotated[int, Field(ge=0)] | None = None
    attribute: str | None = None
    members: tuple[str, ...] | None = None
    region: ImageRegion

    @model_validator(mode="after")
    def check_frame(self) -> PatternFrame:
        """Reject nonfinite positions and repeated set members."""
        for coordinate in (self.x, self.y):
            if coordinate is not None:
                _number(coordinate)
        if self.members is not None and len(self.members) != len(set(self.members)):
            raise ValueError("Repeated member in one pattern panel")
        return self


class PatternOption(StrictModel):
    """One visible answer option with a stable public label."""

    option_id: str = Field(min_length=1)
    frame: PatternFrame


class PatternSource(StrictModel):
    """Blind finite sequence and visible completion options."""

    task_id: str
    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    closed: bool
    scope_id: str
    view_id: str
    scope_region: ImageRegion
    frames: Annotated[tuple[PatternFrame, ...], Field(max_length=24)]
    options: Annotated[tuple[PatternOption, ...], Field(max_length=16)] = ()
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_source(self) -> PatternSource:
        """Require closed, unique and local finite pattern evidence."""
        ids = [frame.frame_id for frame in self.frames]
        option_ids = [option.option_id for option in self.options]
        if len(ids) != len(set(ids)) or len(option_ids) != len(set(option_ids)):
            raise ValueError("Duplicate pattern panel or option ID")
        if self.coverage != "MET" and (self.closed or self.frames or self.options):
            raise ValueError("Partial pattern extraction cannot certify panels")
        if self.coverage == "MET" and (
            len(self.frames) < 3 or (self.task_id == "pattern_completion" and not self.options)
        ):
            raise ValueError("Pattern needs enough panels and visible options")
        for frame in [*self.frames, *(option.frame for option in self.options)]:
            region = frame.region
            scope = self.scope_region
            if not (
                scope.left <= region.left < region.right <= scope.right
                and scope.top <= region.top < region.bottom <= scope.bottom
            ):
                raise ValueError("Pattern panel outside image scope")
        return self


class PatternAnswer(StrictModel):
    """Answer-only rule name, option label or exception panel."""

    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    answer_quote: str = ""
    rule: RuleKind | None = None
    option_id: str | None = None
    frame_id: str | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_shape(self) -> PatternAnswer:
        """Reject multi-form or unquoted pattern answers."""
        populated = sum(item is not None for item in (self.rule, self.option_id, self.frame_id))
        if self.coverage == "MET" and (populated != 1 or not self.answer_quote):
            raise ValueError("Complete pattern answer needs one quoted result")
        if self.coverage != "MET" and (populated or self.answer_quote):
            raise ValueError("Incomplete pattern answer cannot supply a result")
        return self


def _number(text: str) -> Fraction:
    try:
        return parse_numeric_lexeme(text)
    except ExecutionError as exc:
        raise ValueError("Invalid pattern coordinate") from exc


def _features(frame: PatternFrame, exclude: frozenset[str] = frozenset()) -> tuple[object, ...]:
    return tuple(
        value
        for key, value in (
            ("x", _number(frame.x) if frame.x is not None else None),
            ("y", _number(frame.y) if frame.y is not None else None),
            ("quarter_turn", frame.quarter_turn),
            ("mirrored", frame.mirrored),
            ("count", frame.count),
            ("attribute", frame.attribute),
            ("members", tuple(sorted(frame.members)) if frame.members is not None else None),
        )
        if key not in exclude
    )


def _same_other(frames: Sequence[PatternFrame], exclude: frozenset[str]) -> bool:
    return all(_features(frame, exclude) == _features(frames[0], exclude) for frame in frames[1:])


def _candidates(frames: Sequence[PatternFrame], gap: int | None = None) -> set[tuple[str, str]]:
    """Enumerate only the seven registered rule families and finite parameters."""
    indexed = [(index, frame) for index, frame in enumerate(frames) if index != gap]
    if len(indexed) < 3:
        return set()
    observed = [frame for _, frame in indexed]
    results: set[tuple[str, str]] = set()
    if _same_other(observed, frozenset()) and any(
        value is not None for value in _features(observed[0])
    ):
        results.add(("constant", ""))
    for feature, kind in (("count", "count_progression"), ("quarter_turn", "rotation")):
        if all(getattr(frame, feature) is not None for frame in observed) and _same_other(
            observed, frozenset({feature})
        ):
            first_index, first_frame = indexed[0]
            first = getattr(first_frame, feature)
            assert isinstance(first, int)
            for step in range(1, 4) if kind == "rotation" else range(-24, 25):
                if step == 0:
                    continue
                if all(
                    getattr(frame, feature)
                    == (
                        (first + step * (index - first_index)) % 4
                        if kind == "rotation"
                        else first + step * (index - first_index)
                    )
                    for index, frame in indexed
                ):
                    results.add((kind, str(step)))
    if all(frame.x is not None and frame.y is not None for frame in observed) and _same_other(
        observed, frozenset({"x", "y"})
    ):
        i0, first = indexed[0]
        i1, second = indexed[1]
        assert (
            first.x is not None
            and first.y is not None
            and second.x is not None
            and second.y is not None
        )
        dx = (_number(second.x) - _number(first.x)) / (i1 - i0)
        dy = (_number(second.y) - _number(first.y)) / (i1 - i0)
        if (
            (dx or dy)
            and abs(dx) <= 10
            and abs(dy) <= 10
            and all(
                _number(frame.x or "") == _number(first.x) + dx * (index - i0)
                and _number(frame.y or "") == _number(first.y) + dy * (index - i0)
                for index, frame in indexed
            )
        ):
            results.add(("translation", f"{dx};{dy}"))
    if all(frame.mirrored is not None for frame in observed) and _same_other(
        observed, frozenset({"mirrored"})
    ):
        i0, first = indexed[0]
        assert first.mirrored is not None
        if all(
            frame.mirrored == (first.mirrored ^ bool((index - i0) % 2)) for index, frame in indexed
        ):
            results.add(("reflection", "toggle"))
    if all(frame.attribute is not None for frame in observed) and _same_other(
        observed, frozenset({"attribute"})
    ):
        for period in range(2, 5):
            if len(frames) < 2 * period or len({frame.attribute for frame in observed}) < 2:
                continue
            palette: dict[int, str] = {}
            valid = True
            for index, frame in indexed:
                value = frame.attribute
                assert value is not None
                if index % period in palette and palette[index % period] != value:
                    valid = False
                    break
                palette[index % period] = value
            if valid and len(palette) == period and len(set(palette.values())) == period:
                results.add(("attribute_cycle", "|".join(palette[i] for i in range(period))))
    if all(frame.members is not None for frame in observed) and _same_other(
        observed, frozenset({"members"})
    ):
        for operator in ("union", "intersection", "difference"):
            valid_triples = 0
            valid = True
            for start in range(0, len(frames) - 2, 3):
                if gap is not None and gap in {start, start + 1, start + 2}:
                    continue
                first, second, third = (
                    set(frames[start + offset].members or ()) for offset in range(3)
                )
                expected = (
                    first | second
                    if operator == "union"
                    else first & second
                    if operator == "intersection"
                    else first - second
                )
                valid_triples += 1
                if expected != third:
                    valid = False
                    break
            if valid and valid_triples:
                results.add(("set_composition", operator))
    return results


def _predict(
    frames: Sequence[PatternFrame], rule: tuple[str, str], index: int
) -> tuple[object, ...] | None:
    kind, parameter = rule
    observed = [(i, frame) for i, frame in enumerate(frames) if i != index]
    if not observed:
        return None
    first_index, first = observed[0]
    target = list(_features(first))
    if kind == "constant":
        pass
    elif kind == "count_progression":
        assert first.count is not None
        target[4] = first.count + int(parameter) * (index - first_index)
    elif kind == "rotation":
        assert first.quarter_turn is not None
        target[2] = (first.quarter_turn + int(parameter) * (index - first_index)) % 4
    elif kind == "translation":
        dx_text, dy_text = parameter.split(";", 1)
        assert first.x is not None and first.y is not None
        dx, dy = Fraction(dx_text), Fraction(dy_text)
        target[0] = _number(first.x) + dx * (index - first_index)
        target[1] = _number(first.y) + dy * (index - first_index)
    elif kind == "reflection":
        assert first.mirrored is not None
        target[3] = first.mirrored ^ bool((index - first_index) % 2)
    elif kind == "attribute_cycle":
        palette = parameter.split("|")
        target[5] = palette[index % len(palette)]
    elif kind == "set_composition":
        start = index - index % 3
        if index % 3 != 2 or start + 1 >= len(frames):
            return None
        left = set(frames[start].members or ())
        right = set(frames[start + 1].members or ())
        target[6] = tuple(
            sorted(
                left | right
                if parameter == "union"
                else left & right
                if parameter == "intersection"
                else left - right
            )
        )
    else:
        return None
    return tuple(target)


def _canonical(source: PatternSource) -> dict[str, object]:
    result = source.model_dump(mode="json", exclude={"reason"})
    for frame in result["frames"]:
        frame.pop("region", None)
    for option in result["options"]:
        option["frame"].pop("region", None)
    return result


def verify_pattern(
    task_id: str,
    sources: tuple[PatternSource, PatternSource],
    answers: tuple[PatternAnswer, PatternAnswer],
    scope_id: str,
    view_id: str,
    candidate_answer: str,
) -> GateVerdict:
    """Accept only one consistent rule and, where needed, one solution."""
    if task_id not in PATTERN_TASKS:
        raise ValueError("No pattern verifier for task")
    if any(
        source.coverage != "MET"
        or not source.closed
        or source.task_id != task_id
        or source.scope_id != scope_id
        or source.view_id != view_id
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
    if task_id == "pattern_rule":
        rules = _candidates(source.frames)
        if len(rules) != 1 or answer.rule is None:
            return GateVerdict.UNKNOWN
        expected = next(iter(rules))[0]
        observed = answer.rule
    elif task_id == "pattern_completion":
        extended = (*source.frames, source.options[0].frame)
        rules = _candidates(extended, gap=len(source.frames))
        if len(rules) != 1 or answer.option_id is None:
            return GateVerdict.UNKNOWN
        prediction = _predict(extended, next(iter(rules)), len(source.frames))
        matches = [
            option.option_id for option in source.options if _features(option.frame) == prediction
        ]
        if len(matches) != 1:
            return GateVerdict.UNKNOWN
        expected, observed = matches[0], answer.option_id
    else:
        candidates: set[tuple[str, tuple[str, str]]] = set()
        for index, frame in enumerate(source.frames):
            for rule in _candidates(source.frames, gap=index):
                if _predict(source.frames, rule, index) != _features(frame):
                    candidates.add((frame.frame_id, rule))
        if len(candidates) != 1 or answer.frame_id is None:
            return GateVerdict.UNKNOWN
        expected, observed = next(iter(candidates))[0], answer.frame_id
    if observed not in answer.answer_quote:
        return GateVerdict.UNKNOWN
    return GateVerdict.MET if observed == expected else GateVerdict.NOT_MET
