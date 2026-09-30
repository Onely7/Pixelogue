"""Explicit geometry-premise extraction and isolated symbolic solving."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from pixelogue.config import StrictModel
from pixelogue.contracts import GateVerdict
from pixelogue.errors import ExecutionError
from pixelogue.specialist_env import call_worker
from pixelogue.task_evidence import ImageRegion

GeometryVariable = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,15}$")]


class GeometryPremise(StrictModel):
    """One printed or explicitly marked theorem premise."""

    rule: Literal[
        "given",
        "right_angle",
        "triangle_angle_sum",
        "parallel_equal_angle",
        "similar_ratio",
        "pythagorean",
    ]
    variables: tuple[GeometryVariable, ...]
    constants: tuple[str, ...] = ()
    evidence_text: str = Field(min_length=1)
    region: ImageRegion

    @model_validator(mode="after")
    def check_visible_constant(self) -> GeometryPremise:
        """Require the registered rule's arguments and anchor each numerical assumption."""
        required = {
            "given": (1, 1),
            "right_angle": (1, 0),
            "triangle_angle_sum": (3, 0),
            "parallel_equal_angle": (2, 0),
            "similar_ratio": (2, 2),
            "pythagorean": (3, 0),
        }[self.rule]
        if (len(self.variables), len(self.constants)) != required:
            raise ValueError(
                f"Geometry rule {self.rule} requires {required[0]} variables "
                f"and {required[1]} constants"
            )
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("Geometry rule variables must be distinct")
        if any(value not in self.evidence_text for value in self.constants):
            raise ValueError("Geometry constant is not present in its source evidence")
        return self


class GeometryProblem(StrictModel):
    """Complete visual and public geometry problem without proposed answer."""

    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    domain: str = Field(min_length=1)
    scope_id: str
    view_id: str
    scope_region: ImageRegion
    premises: Annotated[tuple[GeometryPremise, ...], Field(max_length=32)]
    target: GeometryVariable
    target_domain: Literal["length", "angle"]
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_evidence(self) -> GeometryProblem:
        """Reject partial or nonlocal proof inputs."""
        if self.coverage != "MET" and self.premises:
            raise ValueError("Incomplete geometry extraction cannot supply premises")
        variables = {name for premise in self.premises for name in premise.variables}
        if len(variables) > 16:
            raise ValueError("Geometry extraction exceeds sixteen registered variables")
        if self.coverage == "MET" and self.target not in variables:
            raise ValueError("Complete geometry extraction must bind the target to a premise")
        fact_keys = [(item.rule, item.variables, item.constants) for item in self.premises]
        if len(fact_keys) != len(set(fact_keys)):
            raise ValueError("Repeated geometry premise")
        for premise in self.premises:
            region = premise.region
            scope = self.scope_region
            if not (
                scope.left <= region.left < region.right <= scope.right
                and scope.top <= region.top < region.bottom <= scope.bottom
            ):
                raise ValueError("Geometry premise lies outside the requested scope")
        return self


class GeometryNumericAnswer(StrictModel):
    """One literal rational or decimal value parsed without the image."""

    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    answer_quote: str
    reported: str | None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_answer(self) -> GeometryNumericAnswer:
        """Ensure a complete answer is quoted exactly."""
        if self.coverage == "MET" and (
            self.reported is None or self.reported not in self.answer_quote
        ):
            raise ValueError("Complete numeric answer needs a literal reported value")
        if self.coverage != "MET" and (self.reported is not None or self.answer_quote):
            raise ValueError("Incomplete numeric answer cannot present a result")
        return self


def verify_geometry_problem(
    sources: tuple[GeometryProblem, GeometryProblem],
    answers: tuple[GeometryNumericAnswer, GeometryNumericAnswer],
    domain: str,
    scope_id: str,
    view_id: str,
    candidate_answer: str,
) -> tuple[GateVerdict, dict[str, object]]:
    """Require blind premise consensus and an exact isolated SymPy proof."""
    if any(
        source.coverage != "MET"
        or source.domain != domain
        or source.scope_id != scope_id
        or source.view_id != view_id
        for source in sources
    ):
        return GateVerdict.UNKNOWN, {"reason": "Incomplete or mismatched visual premise domain"}

    def comparable(source: GeometryProblem) -> tuple[object, ...]:
        """Compare proof facts without requiring identical prose or list order."""
        return (
            source.domain,
            source.scope_id,
            source.view_id,
            source.target,
            source.target_domain,
        )

    if comparable(sources[0]) != comparable(sources[1]):
        return GateVerdict.UNKNOWN, {"reason": "Independent premise extractions disagree"}
    premise_maps = [
        {(item.rule, item.variables, item.constants): item.region for item in source.premises}
        for source in sources
    ]
    if premise_maps[0].keys() != premise_maps[1].keys():
        return GateVerdict.UNKNOWN, {"reason": "Independent premise extractions disagree"}
    for key, first in premise_maps[0].items():
        second = premise_maps[1][key]
        width = max(0.0, min(first.right, second.right) - max(first.left, second.left))
        height = max(0.0, min(first.bottom, second.bottom) - max(first.top, second.top))
        intersection = width * height
        area_first = (first.right - first.left) * (first.bottom - first.top)
        area_second = (second.right - second.left) * (second.bottom - second.top)
        if intersection / (area_first + area_second - intersection) < 0.1:
            return GateVerdict.UNKNOWN, {"reason": "Independent premise regions disagree"}
    if any(
        answer.coverage != "MET" or answer.answer_quote not in candidate_answer
        for answer in answers
    ):
        return GateVerdict.UNKNOWN, {"reason": "Candidate answer is incomplete"}
    if answers[0].reported != answers[1].reported or answers[0].reported is None:
        return GateVerdict.UNKNOWN, {"reason": "Answer parses disagree"}
    try:
        response = call_worker(
            "symbolic",
            "sympy",
            {
                "operation": "geometry",
                "premises": [
                    premise.model_dump(mode="json", exclude={"region", "evidence_text"})
                    for premise in sources[0].premises
                ],
                "target": sources[0].target,
                "target_domain": sources[0].target_domain,
                "reported": answers[0].reported,
            },
        )
    except ExecutionError as exc:
        return GateVerdict.UNKNOWN, {"reason": str(exc)}
    return GateVerdict(response["verdict"]), response
