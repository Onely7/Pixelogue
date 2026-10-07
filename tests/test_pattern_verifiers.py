"""Finite pattern grammar, uniqueness, completion and exception cases."""

from __future__ import annotations

from pixelogue.contracts import GateVerdict
from pixelogue.pattern_verifiers import (
    PatternAnswer,
    PatternFrame,
    PatternOption,
    PatternSource,
    verify_pattern,
)
from pixelogue.task_evidence import ImageRegion

REGION = ImageRegion(left=0, top=0, right=1, bottom=1)


def _frame(key: str, **features: object) -> PatternFrame:
    return PatternFrame.model_validate({"frame_id": key, "region": REGION, **features})


def _source(
    task: str, frames: tuple[PatternFrame, ...], options: tuple[PatternOption, ...] = ()
) -> PatternSource:
    return PatternSource(
        task_id=task,
        coverage="MET",
        closed=True,
        scope_id="scope",
        view_id="view",
        scope_region=REGION,
        frames=frames,
        options=options,
        reason="Complete visible sequence and options",
    )


def _answer(quote: str, **result: object) -> PatternAnswer:
    return PatternAnswer.model_validate(
        {"coverage": "MET", "answer_quote": quote, "reason": "literal", **result}
    )


def _check(source: PatternSource, answer: PatternAnswer) -> GateVerdict:
    return verify_pattern(
        source.task_id,
        (source, source),
        (answer, answer),
        source.scope_id,
        source.view_id,
        answer.answer_quote,
    )


def test_unique_progression_rule_and_competing_rules_abstain() -> None:
    frames = (_frame("a", count=1), _frame("b", count=2), _frame("c", count=3))
    source = _source("pattern_rule", frames)
    assert _check(source, _answer("count_progression", rule="count_progression")) is GateVerdict.MET
    assert _check(source, _answer("rotation", rule="rotation")) is GateVerdict.NOT_MET
    ambiguous = _source(
        "pattern_rule",
        (
            _frame("a", count=1, x="1", y="0"),
            _frame("b", count=2, x="2", y="0"),
            _frame("c", count=3, x="3", y="0"),
        ),
    )
    assert (
        _check(ambiguous, _answer("count_progression", rule="count_progression"))
        is GateVerdict.UNKNOWN
    )


def test_completion_requires_one_rule_and_one_visible_option() -> None:
    frames = (_frame("a", count=1), _frame("b", count=2), _frame("c", count=3))
    options = (
        PatternOption(option_id="D", frame=_frame("d", count=4)),
        PatternOption(option_id="E", frame=_frame("e", count=5)),
    )
    source = _source("pattern_completion", frames, options)
    assert _check(source, _answer("D", option_id="D")) is GateVerdict.MET
    assert _check(source, _answer("E", option_id="E")) is GateVerdict.NOT_MET
    duplicate = source.model_copy(
        update={"options": (*options, PatternOption(option_id="F", frame=_frame("f", count=4)))}
    )
    assert _check(duplicate, _answer("D", option_id="D")) is GateVerdict.UNKNOWN


def test_exception_search_preserves_original_indices() -> None:
    source = _source(
        "pattern_exception",
        (
            _frame("a", count=1),
            _frame("b", count=2),
            _frame("bad", count=9),
            _frame("d", count=4),
            _frame("e", count=5),
        ),
    )
    assert _check(source, _answer("bad", frame_id="bad")) is GateVerdict.MET
    assert _check(source, _answer("d", frame_id="d")) is GateVerdict.NOT_MET


def test_cycle_and_set_composition_are_bounded_rule_families() -> None:
    cycle = _source(
        "pattern_rule",
        (
            _frame("a", attribute="red"),
            _frame("b", attribute="blue"),
            _frame("c", attribute="red"),
            _frame("d", attribute="blue"),
        ),
    )
    assert _check(cycle, _answer("attribute_cycle", rule="attribute_cycle")) is GateVerdict.MET
    composition = _source(
        "pattern_rule",
        (
            _frame("a", members=("a",)),
            _frame("b", members=("b",)),
            _frame("c", members=("a", "b")),
        ),
    )
    assert (
        _check(composition, _answer("set_composition", rule="set_composition")) is GateVerdict.MET
    )


def test_disagreement_and_wrong_image_view_abstain() -> None:
    source = _source(
        "pattern_rule",
        (
            _frame("a", count=1),
            _frame("b", count=2),
            _frame("c", count=3),
        ),
    )
    answer = _answer("count_progression", rule="count_progression")
    other = source.model_copy(update={"frames": source.frames[:-1] + (_frame("c", count=4),)})
    assert (
        verify_pattern(
            source.task_id,
            (source, other),
            (answer, answer),
            source.scope_id,
            source.view_id,
            answer.answer_quote,
        )
        is GateVerdict.UNKNOWN
    )
    assert (
        verify_pattern(
            source.task_id,
            (source, source),
            (answer, answer),
            source.scope_id,
            "other",
            answer.answer_quote,
        )
        is GateVerdict.UNKNOWN
    )
