"""Chart interval, scale, completeness and export contract cases."""

from __future__ import annotations

import json

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
            "decimal_places": 0,
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


def test_log_axis_rejects_nonpositive_intervals() -> None:
    source = _source("chart_value_lookup", "value", (_mark("A", "Jan", "0"),))
    log = source.model_copy(update={"axis": ChartAxis(scale="log", unit="kg", ticks=("1", "100"))})
    answer = _answer(answer_quote="0 kg", value=NumericValue(value="0", unit="kg"))
    assert _check(log, answer, "0 kg", precision="explicit_label")[0] is GateVerdict.UNKNOWN


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
