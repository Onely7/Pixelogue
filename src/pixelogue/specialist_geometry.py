"""Explicit geometry-premise extraction and isolated symbolic solving."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from pixelogue.config import StrictModel
from pixelogue.contracts import GateVerdict
from pixelogue.errors import ExecutionError
from pixelogue.specialist_env import call_worker
from pixelogue.task_evidence import ImageRegion


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
    variables: tuple[str, ...]
    constants: tuple[str, ...] = ()
    evidence_text: str = Field(min_length=1)
    region: ImageRegion

    @model_validator(mode="after")
    def check_visible_constant(self) -> GeometryPremise:
        """Anchor every numerical assumption to explicit visible evidence."""
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
    target: str = Field(min_length=1)
    target_domain: Literal["length", "angle"]
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_evidence(self) -> GeometryProblem:
        """Reject partial or nonlocal proof inputs."""
        if self.coverage != "MET" and self.premises:
            raise ValueError("Incomplete geometry extraction cannot supply premises")
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
    answer_quote: str = ""
    reported: str | None = None
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
    canonical = [
        source.model_dump(mode="json", exclude={"reason", "scope_region"}) for source in sources
    ]
    for item in canonical:
        for premise in item["premises"]:
            premise.pop("region", None)
    if canonical[0] != canonical[1]:
        return GateVerdict.UNKNOWN, {"reason": "Independent premise extractions disagree"}
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
