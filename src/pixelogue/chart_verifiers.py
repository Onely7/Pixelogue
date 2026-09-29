"""Conservative chart checks over calibrated intervals and closed series."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Annotated, Literal

from pydantic import Field, model_validator

from pixelogue.config import StrictModel
from pixelogue.contracts import GateVerdict
from pixelogue.errors import ExecutionError, ExternalInputError
from pixelogue.rules import NumericValue, parse_numeric_lexeme
from pixelogue.serialization import strict_json_object
from pixelogue.task_evidence import ImageRegion

CHART_TASKS = frozenset(
    {
        "chart_value_lookup",
        "chart_comparison",
        "chart_extremum_ranking",
        "chart_trend_summary",
        "chart_series_relation",
        "chart_data_reconstruction",
    }
)
Precision = Literal["explicit_label", "calibrated_estimate", "interval"]


class ChartAxis(StrictModel):
    """Y-axis semantics and labeled calibration marks."""

    scale: Literal["linear", "log", "unmarked"]
    unit: str | None = None
    ticks: Annotated[tuple[str, ...], Field(max_length=32)]

    @model_validator(mode="after")
    def check_ticks(self) -> ChartAxis:
        """Reject ambiguous or nonmonotonic axis calibration."""
        if self.scale == "unmarked":
            if self.ticks:
                raise ValueError("Unmarked axis cannot claim visible calibration ticks")
            return self
        if len(self.ticks) < 2:
            raise ValueError("Calibrated axis requires at least two labeled ticks")
        try:
            values = tuple(parse_numeric_lexeme(item) for item in self.ticks)
        except ExecutionError as exc:
            raise ValueError("Invalid axis tick") from exc
        if any(left >= right for left, right in zip(values, values[1:], strict=False)):
            raise ValueError("Axis ticks must increase")
        if self.scale == "log" and any(value <= 0 for value in values):
            raise ValueError("Log axis requires positive ticks")
        return self


class ChartMark(StrictModel):
    """One legend-bound mark with an honest value interval."""

    series: str = Field(min_length=1)
    category: str = Field(min_length=1)
    lower: str
    upper: str
    precision: Precision
    decimal_places: Annotated[int, Field(ge=0, le=8)]
    visible_label: str | None = None
    region: ImageRegion

    @model_validator(mode="after")
    def check_interval(self) -> ChartMark:
        """Distinguish exact labels from pixel-derived intervals."""
        try:
            lower = parse_numeric_lexeme(self.lower)
            upper = parse_numeric_lexeme(self.upper)
        except ExecutionError as exc:
            raise ValueError("Invalid chart mark value") from exc
        if lower > upper or (self.precision == "explicit_label" and lower != upper):
            raise ValueError("Invalid mark interval or exactness claim")
        if self.precision == "explicit_label" and (
            self.visible_label is None or self.lower not in self.visible_label
        ):
            raise ValueError("Exact chart values require a visible printed label")
        return self


class ChartQuery(StrictModel):
    """Public lookup, comparison or series operation."""

    operation: Literal["value", "compare", "rank", "trend", "relation", "reconstruct"]
    series: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    rank_order: Literal["ascending", "descending"] = "descending"
    rank_mode: Literal["all", "max", "min"] = "all"
    relation: Literal["intersection", "dominance", "variation"] | None = None
    format: Literal["structured_json"] | None = None


class ChartSource(StrictModel):
    """Answer-blind axis, legend and mark extraction in one image view."""

    task_id: str
    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    scope_id: str
    view_id: str
    scope_region: ImageRegion
    axis: ChartAxis | None
    legend_complete: bool
    closed: bool
    encoding: Literal["line", "bar", "map"]
    marks: Annotated[tuple[ChartMark, ...], Field(max_length=256)]
    query: ChartQuery
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_source(self) -> ChartSource:
        """Reject duplicate points, partial MET sources and off-scope marks."""
        keys = [(mark.series, mark.category) for mark in self.marks]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate chart mark binding")
        if self.coverage != "MET" and (self.axis is not None or self.marks or self.closed):
            raise ValueError("Incomplete chart extraction cannot certify an axis or marks")
        if (
            self.coverage == "MET"
            and self.axis is not None
            and self.axis.scale == "unmarked"
            and any(mark.precision != "explicit_label" for mark in self.marks)
        ):
            raise ValueError("An unmarked axis supports only directly printed exact values")
        for mark in self.marks:
            region = mark.region
            scope = self.scope_region
            if not (
                scope.left <= region.left < region.right <= scope.right
                and scope.top <= region.top < region.bottom <= scope.bottom
            ):
                raise ValueError("Chart mark outside image scope")
        return self


class ChartAnswer(StrictModel):
    """Answer-only parse of one numeric or categorical chart result."""

    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    answer_quote: str = ""
    value: NumericValue | None = None
    relation: Literal["less", "equal", "greater", "above", "below", "crossing"] | None = None
    rank_groups: tuple[tuple[str, ...], ...] | None = None
    trend: Literal["rising", "falling", "flat", "mixed"] | None = None
    trend_segments: tuple[Literal["rising", "falling", "flat"], ...] | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_shape(self) -> ChartAnswer:
        """Never treat an ambiguous multi-form parse as a result."""
        populated = sum(
            item is not None for item in (self.value, self.relation, self.rank_groups, self.trend)
        )
        if self.coverage == "MET" and (populated != 1 or not self.answer_quote):
            raise ValueError("Complete chart answer requires one quoted form")
        if self.coverage != "MET" and (populated or self.answer_quote):
            raise ValueError("Incomplete chart answer cannot supply a result")
        if self.trend == "mixed" and not self.trend_segments:
            raise ValueError("Mixed trend needs ordered segment directions")
        if self.trend != "mixed" and self.trend_segments is not None:
            raise ValueError("Segment list is only for mixed trends")
        return self


class ChartDataRow(StrictModel):
    """One value interval in a public structured chart export."""

    series: str
    category: str
    lower: str
    upper: str
    precision: Precision

    @model_validator(mode="after")
    def check_interval(self) -> ChartDataRow:
        """Require numerical bounds and exact labels where declared."""
        try:
            low = parse_numeric_lexeme(self.lower)
            high = parse_numeric_lexeme(self.upper)
        except ExecutionError as exc:
            raise ValueError("Invalid chart export number") from exc
        if low > high or (self.precision == "explicit_label" and low != high):
            raise ValueError("Invalid chart export interval")
        return self


class ChartDataAnswer(StrictModel):
    """Strict public chart-data JSON schema."""

    unit: str | None
    marks: tuple[ChartDataRow, ...]


def _canonical(source: ChartSource) -> dict[str, object]:
    result = source.model_dump(mode="json", exclude={"reason"})
    for mark in result["marks"]:
        mark.pop("region", None)
    result["marks"].sort(key=lambda item: (item["series"], item["category"]))
    return result


def _interval(mark: ChartMark) -> tuple[Fraction, Fraction]:
    return parse_numeric_lexeme(mark.lower), parse_numeric_lexeme(mark.upper)


def _relation(left: ChartMark, right: ChartMark) -> str | None:
    a, b = _interval(left), _interval(right)
    if a[1] < b[0]:
        return "less"
    if a[0] > b[1]:
        return "greater"
    if a[0] == a[1] == b[0] == b[1]:
        return "equal"
    return None


def _rank(marks: tuple[ChartMark, ...], descending: bool) -> tuple[tuple[str, ...], ...] | None:
    if not marks or any(low != high for mark in marks for low, high in (_interval(mark),)):
        return None
    values: dict[Fraction, list[str]] = {}
    for mark in marks:
        value, _ = _interval(mark)
        values.setdefault(value, []).append(mark.category)
    return tuple(tuple(sorted(values[key])) for key in sorted(values, reverse=descending))


def _trend(marks: tuple[ChartMark, ...]) -> tuple[str, tuple[str, ...]] | None:
    if len(marks) < 2:
        return None
    changes = [_relation(a, b) for a, b in zip(marks, marks[1:], strict=False)]
    if any(change is None for change in changes):
        return None
    segments = tuple(
        "rising" if change == "less" else "falling" if change == "greater" else "flat"
        for change in changes
    )
    return (segments[0] if len(set(segments)) == 1 else "mixed"), segments


def _series_relation(source: ChartSource) -> str | None:
    if len(source.query.series) != 2 or not source.closed:
        return None
    first, second = source.query.series
    categories = source.query.categories
    by_key = {(mark.series, mark.category): mark for mark in source.marks}
    if not categories or any(
        (series, category) not in by_key for series in (first, second) for category in categories
    ):
        return None
    relations = [
        _relation(by_key[(first, category)], by_key[(second, category)]) for category in categories
    ]
    if any(relation is None for relation in relations):
        return None
    if all(relation == "greater" for relation in relations):
        return "above"
    if all(relation == "less" for relation in relations):
        return "below"
    if source.encoding == "line" and any(
        a != b for a, b in zip(relations, relations[1:], strict=False)
    ):
        return "crossing"
    if all(relation == "equal" for relation in relations):
        return "equal"
    return None


def verify_chart(
    task_id: str,
    sources: tuple[ChartSource, ChartSource],
    answers: tuple[ChartAnswer, ChartAnswer] | None,
    public_parameters: Mapping[str, object],
    scope_id: str,
    view_id: str,
    candidate_answer: str,
) -> tuple[GateVerdict, GateVerdict | None]:
    """Check visual encoding and arithmetic, with a distinct export-schema result."""
    if task_id not in CHART_TASKS:
        raise ValueError("No chart verifier for this task")
    unknown = (
        GateVerdict.UNKNOWN,
        GateVerdict.UNKNOWN if task_id == "chart_data_reconstruction" else None,
    )
    if any(
        source.coverage != "MET"
        or source.task_id != task_id
        or source.scope_id != scope_id
        or source.view_id != view_id
        or source.axis is None
        or not source.legend_complete
        for source in sources
    ) or _canonical(sources[0]) != _canonical(sources[1]):
        return unknown
    source = sources[0]
    assert source.axis is not None
    if source.axis.scale != "unmarked":
        axis_min = parse_numeric_lexeme(source.axis.ticks[0])
        axis_max = parse_numeric_lexeme(source.axis.ticks[-1])
        if any(
            (low < axis_min or high > axis_max or (source.axis.scale == "log" and low <= 0))
            for mark in source.marks
            for low, high in (_interval(mark),)
        ):
            return unknown
    query = source.query
    by_key = {(mark.series, mark.category): mark for mark in source.marks}
    if task_id == "chart_data_reconstruction":
        if (
            query.operation != "reconstruct"
            or query.format != "structured_json"
            or not source.closed
        ):
            return unknown
        if public_parameters.get("format") != query.format:
            return unknown
        try:
            raw = strict_json_object(candidate_answer)
            if set(raw) != {"unit", "marks"}:
                raise ValueError("Invalid public chart JSON shape")
            decoded = ChartDataAnswer.model_validate_json(candidate_answer)
        except (ExternalInputError, ValueError, TypeError):
            return GateVerdict.UNKNOWN, GateVerdict.NOT_MET
        actual = sorted(
            (m.series, m.category, m.lower, m.upper, m.precision) for m in decoded.marks
        )
        expected = sorted(
            (m.series, m.category, m.lower, m.upper, m.precision) for m in source.marks
        )
        return (
            GateVerdict.MET
            if decoded.unit == source.axis.unit and actual == expected
            else GateVerdict.NOT_MET
        ), GateVerdict.MET
    if answers is None or any(
        answer.coverage != "MET" or answer.answer_quote not in candidate_answer
        for answer in answers
    ):
        return unknown
    if answers[0].model_dump(mode="json", exclude={"reason"}) != answers[1].model_dump(
        mode="json", exclude={"reason"}
    ):
        return unknown
    answer = answers[0]
    if answer.value is not None and answer.value.value not in answer.answer_quote:
        return unknown
    if answer.rank_groups is not None and any(
        label not in answer.answer_quote for group in answer.rank_groups for label in group
    ):
        return unknown
    expected: object | None = None
    observed: object | None = None
    if task_id == "chart_value_lookup" and query.operation == "value":
        if len(query.series) != 1 or len(query.categories) != 1:
            return unknown
        mark = by_key.get((query.series[0], query.categories[0]))
        if mark is None or answer.value is None or answer.value.unit != source.axis.unit:
            return unknown
        if public_parameters.get("precision") != mark.precision:
            return unknown
        try:
            value = parse_numeric_lexeme(answer.value.value)
        except ExecutionError:
            return unknown
        decimals = answer.value.value.split(".")[-1] if "." in answer.value.value else ""
        if len(decimals) > mark.decimal_places:
            return unknown
        low, high = _interval(mark)
        expected = True
        observed = low <= value <= high
    elif task_id == "chart_comparison" and query.operation == "compare":
        if len(query.series) != 2 or len(query.categories) != 1:
            return unknown
        a = by_key.get((query.series[0], query.categories[0]))
        b = by_key.get((query.series[1], query.categories[0]))
        if a is None or b is None:
            return unknown
        expected, observed = _relation(a, b), answer.relation
    elif task_id == "chart_extremum_ranking" and query.operation == "rank":
        if len(query.series) != 1 or not source.closed:
            return unknown
        all_categories = {mark.category for mark in source.marks if mark.series == query.series[0]}
        if set(query.categories) != all_categories:
            return unknown
        marks = tuple(
            by_key[(query.series[0], category)]
            for category in query.categories
            if (query.series[0], category) in by_key
        )
        if len(marks) != len(query.categories):
            return unknown
        ranked = _rank(marks, query.rank_order == "descending")
        if ranked is None:
            return unknown
        if query.rank_mode == "max":
            expected = (ranked[0] if query.rank_order == "descending" else ranked[-1],)
        elif query.rank_mode == "min":
            expected = (ranked[-1] if query.rank_order == "descending" else ranked[0],)
        else:
            expected = ranked
        observed = answer.rank_groups
    elif task_id == "chart_trend_summary" and query.operation == "trend":
        if len(query.series) != 1 or not source.closed:
            return unknown
        marks = tuple(
            by_key[(query.series[0], category)]
            for category in query.categories
            if (query.series[0], category) in by_key
        )
        if len(marks) != len(query.categories):
            return unknown
        computed = _trend(marks)
        if computed is None:
            return unknown
        expected, segments = computed
        observed = answer.trend
        if expected == "mixed" and answer.trend_segments != segments:
            return GateVerdict.NOT_MET, None
    elif task_id == "chart_series_relation" and query.operation == "relation":
        declared_relation = public_parameters.get("relation")
        if (
            declared_relation not in {"dominance", "intersection"}
            or declared_relation != query.relation
        ):
            return unknown
        expected, observed = _series_relation(source), answer.relation
        if declared_relation == "intersection" and expected not in {"crossing", "equal"}:
            return unknown
        if declared_relation == "dominance" and expected not in {"above", "below"}:
            return unknown
    if expected is None or observed is None:
        return unknown
    return (GateVerdict.MET if expected == observed else GateVerdict.NOT_MET), None
