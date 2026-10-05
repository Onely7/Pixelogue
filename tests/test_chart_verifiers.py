"""Chart interval, scale, completeness and export contract cases."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from pixelogue.chart_verifiers import (
    ChartAnswer,
    ChartAxis,
    ChartMark,
    ChartSource,
    verify_chart,
)
from pixelogue.contracts import GateVerdict
from pixelogue.rules import NumericValue
from pixelogue.task_evidence import ImageRegion

REGION = ImageRegion(left=0, top=0, right=1, bottom=1)


def _mark(
    series: str,
    category: str,
    lower: str,
    upper: str | None = None,
    *,
    precision: str = "explicit_label",
) -> ChartMark:
    upper = upper or lower
    return ChartMark.model_validate(
        {
            "series": series,
            "category": category,
            "lower": lower,
            "upper": upper,
            "precision": precision,
            "decimal_places": len(lower.partition(".")[2]),
            "visible_label": lower if precision == "explicit_label" else None,
            "region": REGION,
        }
    )


def _source(
    task: str,
    operation: str,
    marks: tuple[ChartMark, ...],
    *,
    series: tuple[str, ...] = ("A",),
    categories: tuple[str, ...] = ("Jan",),
    **options: object,
) -> ChartSource:
    return ChartSource.model_validate(
        {
            "task_id": task,
            "coverage": "MET",
            "scope_id": "scope",
            "view_id": "view",
            "scope_region": REGION,
            "axis": ChartAxis(scale="linear", unit="kg", ticks=("0", "100")),
            "legend_complete": True,
            "closed": True,
            "encoding": "line",
            "marks": marks,
            "query": {
                "operation": operation,
                "series": series,
                "categories": categories,
                **options,
            },
            "reason": "All marks and tick labels visible",
        }
    )


def _answer(**values: object) -> ChartAnswer:
    return ChartAnswer.model_validate({"coverage": "MET", "reason": "literal", **values})


def _check(
    source: ChartSource, answer: ChartAnswer | None, text: str, **parameters: object
) -> tuple[GateVerdict, GateVerdict | None]:
    return verify_chart(
        source.task_id,
        (source, source),
        (answer, answer) if answer else None,
        parameters,
        source.scope_id,
        source.view_id,
        text,
    )


def test_rank_ignores_answer_digit_limit_and_infers_only_printed_percent_unit():
    marks = (
        _mark("A", "high", "27.8").model_copy(update={"visible_label": "27.8%"}),
        _mark("A", "low", "0.57").model_copy(update={"visible_label": "0.57%"}),
    )
    source = _source(
        "chart_extremum_ranking", "rank", marks, categories=("high", "low"), rank_mode="max_min"
    ).model_copy(update={"axis": ChartAxis(scale="unmarked", unit=None, ticks=())})
    other = source.model_copy(
        update={
            "axis": ChartAxis(scale="unmarked", unit="%", ticks=()),
            "marks": (marks[0].model_copy(update={"decimal_places": 2}), marks[1]),
        }
    )
    answer = _answer(answer_quote="high low", rank_groups=(("high",), ("low",)))
    args = (
        "chart_extremum_ranking",
        (source, other),
        (answer, answer),
        {"rank_mode": "max_min", "rank_order": "descending"},
        "scope",
        "view",
        "high low",
    )
    assert verify_chart(*args)[0] is GateVerdict.MET
    wrong = _answer(answer_quote="low high", rank_groups=(("low",), ("high",)))
    assert (
        verify_chart(*args[:2], (wrong, wrong), *args[3:-1], "low high")[0] is GateVerdict.NOT_MET
    )


@pytest.mark.parametrize("label,unit", [("0.57", "%"), ("0.57%", "kg"), ("10.57%", "%")])
def test_rank_does_not_guess_missing_or_conflicting_percent_evidence(label, unit):
    marks = (
        _mark("A", "high", "27.8").model_copy(update={"visible_label": "27.8%"}),
        _mark("A", "low", "0.57").model_copy(update={"visible_label": label}),
    )
    source = _source("chart_extremum_ranking", "rank", marks, categories=("high", "low"))
    source = source.model_copy(update={"axis": ChartAxis(scale="unmarked", unit=None, ticks=())})
    other = source.model_copy(update={"axis": ChartAxis(scale="unmarked", unit=unit, ticks=())})
    answer = _answer(answer_quote="high low", rank_groups=(("high",), ("low",)))
    assert (
        verify_chart(
            source.task_id, (source, other), (answer, answer), {}, "scope", "view", "high low"
        )[0]
        is GateVerdict.UNKNOWN
    )


def test_rank_rejects_a_label_that_omits_the_extracted_numeric_lexeme():
    mark = _mark("A", "low", "0.57").model_copy(update={"visible_label": ".57%"})
    with pytest.raises(ValidationError, match="visible printed label"):
        _source("chart_extremum_ranking", "rank", (mark,))


def test_value_lookup_keeps_unit_and_digit_limit_disagreements_unknown():
    mark = _mark("A", "Jan", "27.8").model_copy(update={"visible_label": "27.8%"})
    source = _source("chart_value_lookup", "value", (mark,)).model_copy(
        update={"axis": ChartAxis(scale="unmarked", unit=None, ticks=())}
    )
    other = source.model_copy(update={"axis": ChartAxis(scale="unmarked", unit="%", ticks=())})
    answer = _answer(answer_quote="27.8%", value=NumericValue(value="27.8", unit="%"))
    assert (
        verify_chart(
            source.task_id, (source, other), (answer, answer), {}, "scope", "view", "27.8%"
        )[0]
        is GateVerdict.UNKNOWN
    )
    other = source.model_copy(update={"marks": (mark.model_copy(update={"decimal_places": 2}),)})
    assert (
        verify_chart(
            source.task_id, (source, other), (answer, answer), {}, "scope", "view", "27.8%"
        )[0]
        is GateVerdict.UNKNOWN
    )


def test_exact_lookup_and_coarse_interval_do_not_invent_precision() -> None:
    exact = _source("chart_value_lookup", "value", (_mark("A", "Jan", "20"),))
    correct = _answer(answer_quote="20 kg", value=NumericValue(value="20", unit="kg"))
    assert _check(exact, correct, "20 kg", precision="explicit_label")[0] is GateVerdict.MET
    approximate = _source(
        "chart_value_lookup",
        "value",
        (_mark("A", "Jan", "19", "21", precision="calibrated_estimate"),),
    )
    assert (
        _check(approximate, correct, "20 kg", precision="calibrated_estimate")[0] is GateVerdict.MET
    )
    false_precision = _answer(answer_quote="20.00 kg", value=NumericValue(value="20.00", unit="kg"))
    assert (
        _check(approximate, false_precision, "20.00 kg", precision="calibrated_estimate")[0]
        is GateVerdict.UNKNOWN
    )


def test_comparison_abstains_for_overlapping_intervals() -> None:
    marks = (
        _mark("A", "Jan", "19", "21", precision="interval"),
        _mark("B", "Jan", "20", "22", precision="interval"),
    )
    source = _source("chart_comparison", "compare", marks, series=("A", "B"))
    answer = _answer(answer_quote="less", relation="less")
    assert _check(source, answer, "less")[0] is GateVerdict.UNKNOWN
    separated = source.model_copy(update={"marks": (marks[0], _mark("B", "Jan", "30"))})
    assert _check(separated, answer, "less")[0] is GateVerdict.MET


def test_complete_ranking_preserves_ties_and_trend_order() -> None:
    marks = (_mark("A", "Jan", "10"), _mark("A", "Feb", "20"), _mark("A", "Mar", "20"))
    ranking = _source("chart_extremum_ranking", "rank", marks, categories=("Jan", "Feb", "Mar"))
    ranks = _answer(answer_quote="Feb, Mar, Jan", rank_groups=(("Feb", "Mar"), ("Jan",)))
    assert _check(ranking, ranks, "Feb, Mar, Jan")[0] is GateVerdict.MET
    partial = ranking.model_copy(
        update={"query": ranking.query.model_copy(update={"categories": ("Jan", "Feb")})}
    )
    assert _check(partial, ranks, "Feb, Mar, Jan")[0] is GateVerdict.UNKNOWN
    trend = _source("chart_trend_summary", "trend", marks, categories=("Jan", "Feb", "Mar"))
    mixed = _answer(
        answer_quote="rising then flat", trend="mixed", trend_segments=("rising", "flat")
    )
    assert _check(trend, mixed, "rising then flat")[0] is GateVerdict.MET


def test_public_extrema_mode_cannot_drift_into_a_full_ranking() -> None:
    marks = (_mark("A", "Jan", "10"), _mark("A", "Feb", "20"), _mark("A", "Mar", "15"))
    source = _source(
        "chart_extremum_ranking",
        "rank",
        marks,
        categories=("Jan", "Feb", "Mar"),
        rank_mode="max_min",
    )
    answer = _answer(answer_quote="Feb and Jan", rank_groups=(("Feb",), ("Jan",)))
    assert (
        _check(source, answer, "Feb and Jan", rank_mode="max_min", rank_order="descending")[0]
        is GateVerdict.MET
    )
    wrong = _answer(answer_quote="Feb and Mar", rank_groups=(("Feb",), ("Mar",)))
    assert _check(source, wrong, "Feb and Mar", rank_mode="max_min")[0] is GateVerdict.NOT_MET
    assert _check(source, answer, "Feb and Jan", rank_mode="all")[0] is GateVerdict.UNKNOWN


def test_paired_extrema_retains_every_tie_and_requires_closed_scope() -> None:
    marks = (_mark("A", "Jan", "10"), _mark("A", "Feb", "20"), _mark("A", "Mar", "20"))
    source = _source(
        "chart_extremum_ranking",
        "rank",
        marks,
        categories=("Jan", "Feb", "Mar"),
        rank_mode="max_min",
    )
    answer = _answer(answer_quote="Feb, Mar; Jan", rank_groups=(("Feb", "Mar"), ("Jan",)))
    assert _check(source, answer, "Feb, Mar; Jan", rank_mode="max_min")[0] is GateVerdict.MET
    assert (
        _check(source.model_copy(update={"closed": False}), answer, "Feb, Mar; Jan")[0]
        is GateVerdict.UNKNOWN
    )


def test_paired_extrema_preserves_both_roles_when_every_value_is_tied() -> None:
    source = _source(
        "chart_extremum_ranking",
        "rank",
        (_mark("A", "Jan", "10"), _mark("A", "Feb", "10")),
        categories=("Jan", "Feb"),
        rank_mode="max_min",
    )
    answer = _answer(
        answer_quote="Feb, Jan are both highest and lowest",
        rank_groups=(("Feb", "Jan"), ("Feb", "Jan")),
    )
    assert _check(source, answer, answer.answer_quote, rank_mode="max_min")[0] is GateVerdict.MET


def test_series_crossing_requires_line_encoding() -> None:
    marks = (
        _mark("A", "Jan", "10"),
        _mark("B", "Jan", "20"),
        _mark("A", "Feb", "30"),
        _mark("B", "Feb", "20"),
    )
    source = _source(
        "chart_series_relation",
        "relation",
        marks,
        series=("A", "B"),
        categories=("Jan", "Feb"),
        relation="intersection",
    )
    crossing = _answer(answer_quote="crossing", relation="crossing")
    assert _check(source, crossing, "crossing", relation="intersection")[0] is GateVerdict.MET
    bars = source.model_copy(update={"encoding": "bar"})
    assert _check(bars, crossing, "crossing", relation="intersection")[0] is GateVerdict.UNKNOWN


def test_extremum_uses_separated_intervals_without_claiming_exact_values() -> None:
    marks = (
        _mark("A", "Jan", "9", "11", precision="interval"),
        _mark("A", "Feb", "19", "21", precision="interval"),
    )
    source = _source(
        "chart_extremum_ranking", "rank", marks, categories=("Jan", "Feb"), rank_mode="max"
    )
    answer = _answer(answer_quote="Feb", rank_groups=(("Feb",),))
    assert _check(source, answer, "Feb")[0] is GateVerdict.MET
    overlapping = source.model_copy(
        update={"marks": (marks[0], _mark("A", "Feb", "10", "21", precision="interval"))}
    )
    assert _check(overlapping, answer, "Feb")[0] is GateVerdict.UNKNOWN


def test_position_aliases_require_closed_separated_and_matching_marks() -> None:
    names = ("Leftmost marker", "Middle marker", "Rightmost marker")
    marks = tuple(
        _mark("A", name, str(10 + index * 10)).model_copy(
            update={
                "region": ImageRegion(
                    left=index * 0.3, top=0.2, right=index * 0.3 + 0.1, bottom=0.3
                )
            }
        )
        for index, name in enumerate(names)
    )
    source = _source("chart_extremum_ranking", "rank", marks, categories=names, rank_mode="max")
    other = source.model_copy(
        update={
            "marks": (
                marks[0],
                marks[1].model_copy(update={"category": "Second marker from left"}),
                marks[2],
            ),
            "query": source.query.model_copy(
                update={"categories": (names[0], "Second marker from left", names[2])}
            ),
        }
    )
    answer = _answer(answer_quote="Rightmost marker", rank_groups=(("Rightmost marker",),))
    assert (
        verify_chart(
            source.task_id,
            (source, other),
            (answer, answer),
            {},
            "scope",
            "view",
            answer.answer_quote,
        )[0]
        is GateVerdict.MET
    )
    displaced = other.model_copy(
        update={
            "marks": (
                marks[0],
                marks[1],
                marks[2].model_copy(
                    update={"region": ImageRegion(left=0.8, top=0.5, right=0.9, bottom=0.6)}
                ),
            )
        }
    )
    assert (
        verify_chart(
            source.task_id,
            (source, displaced),
            (answer, answer),
            {},
            "scope",
            "view",
            answer.answer_quote,
        )[0]
        is GateVerdict.UNKNOWN
    )


def test_log_axis_rejects_nonpositive_intervals() -> None:
    source = _source("chart_value_lookup", "value", (_mark("A", "Jan", "0"),))
    log = source.model_copy(update={"axis": ChartAxis(scale="log", unit="kg", ticks=("1", "100"))})
    answer = _answer(answer_quote="0 kg", value=NumericValue(value="0", unit="kg"))
    assert _check(log, answer, "0 kg", precision="explicit_label")[0] is GateVerdict.UNKNOWN


def test_directly_labeled_bar_chart_without_numeric_ticks() -> None:
    source = _source(
        "chart_value_lookup",
        "value",
        (_mark("Domains", "reddit.com", "27.8"),),
        series=("Domains",),
        categories=("reddit.com",),
    ).model_copy(
        update={"axis": ChartAxis(scale="unmarked", unit="%", ticks=()), "encoding": "bar"}
    )
    correct = _answer(answer_quote="27.8%", value=NumericValue(value="27.8", unit="%"))
    wrong = _answer(answer_quote="28%", value=NumericValue(value="28", unit="%"))
    assert _check(source, correct, "27.8%", precision="explicit_label")[0] is GateVerdict.MET
    assert _check(source, wrong, "28%", precision="explicit_label")[0] is GateVerdict.NOT_MET


def test_unmarked_chart_rejects_guessed_ticks_and_unlabeled_estimates() -> None:
    with pytest.raises(ValidationError):
        ChartAxis(scale="unmarked", unit="%", ticks=("0", "100"))
    with pytest.raises(ValidationError):
        ChartAxis(scale="linear", unit="%", ticks=())
    source = _source(
        "chart_value_lookup",
        "value",
        (_mark("Domains", "reddit.com", "27", "29", precision="interval"),),
        series=("Domains",),
        categories=("reddit.com",),
    )
    with pytest.raises(ValidationError):
        ChartSource.model_validate(
            {
                **source.model_dump(mode="json"),
                "axis": ChartAxis(scale="unmarked", unit="%", ticks=()).model_dump(mode="json"),
            }
        )


def test_reconstruction_schema_and_content_are_independent() -> None:
    source = _source(
        "chart_data_reconstruction",
        "reconstruct",
        (_mark("A", "Jan", "20"),),
        format="structured_json",
    )
    valid = json.dumps(
        {
            "unit": "kg",
            "marks": [
                {
                    "series": "A",
                    "category": "Jan",
                    "lower": "20",
                    "upper": "20",
                    "precision": "explicit_label",
                }
            ],
        }
    )
    assert _check(source, None, valid, format="structured_json") == (
        GateVerdict.MET,
        GateVerdict.MET,
    )
    wrong = valid.replace('"20"', '"21"')
    assert _check(source, None, wrong, format="structured_json") == (
        GateVerdict.NOT_MET,
        GateVerdict.MET,
    )
    assert _check(source, None, "{bad", format="structured_json") == (
        GateVerdict.UNKNOWN,
        GateVerdict.NOT_MET,
    )
    duplicated = valid.replace('"lower": "20"', '"lower": "21", "lower": "20"')
    assert _check(source, None, duplicated, format="structured_json") == (
        GateVerdict.UNKNOWN,
        GateVerdict.NOT_MET,
    )
