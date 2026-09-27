"""Single visible UI action specification without executing the action."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, model_validator

from pixelogue.config import StrictModel
from pixelogue.contracts import GateVerdict
from pixelogue.task_evidence import ImageRegion


class UIControl(StrictModel):
    """One unique visible control and its actionable bounds."""

    control_id: str = Field(min_length=1)
    kind: Literal["button", "field", "link", "checkbox", "menu"]
    enabled: bool
    region: ImageRegion


class UIActionSource(StrictModel):
    """Blind control inventory for the exact delivered raster."""

    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    domain: str = Field(min_length=1)
    scope_id: str
    view_id: str
    scope_region: ImageRegion
    controls: Annotated[tuple[UIControl, ...], Field(max_length=64)]
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_controls(self) -> UIActionSource:
        """Require unique, local controls and no inventory from partial evidence."""
        if self.coverage != "MET" and self.controls:
            raise ValueError("Incomplete screen cannot certify controls")
        ids = [control.control_id for control in self.controls]
        if len(ids) != len(set(ids)):
            raise ValueError("Repeated visible control ID")
        for control in self.controls:
            region = control.region
            scope = self.scope_region
            if not (
                scope.left <= region.left < region.right <= scope.right
                and scope.top <= region.top < region.bottom <= scope.bottom
            ):
                raise ValueError("Control lies outside scoped screen")
        return self


class UIActionAnswer(StrictModel):
    """Declarative action on a target point in one image view."""

    action: Literal["click", "focus", "input"]
    target_id: str = Field(min_length=1)
    view_id: str = Field(min_length=1)
    x: Annotated[float, Field(ge=0, le=1)]
    y: Annotated[float, Field(ge=0, le=1)]
    pixel_x: Annotated[int, Field(ge=0)]
    pixel_y: Annotated[int, Field(ge=0)]
    text: str | None = None


def verify_ui_action(
    sources: tuple[UIActionSource, UIActionSource],
    domain: str,
    parameters: Mapping[str, object],
    scope_id: str,
    view_id: str,
    image_views: object,
    candidate_answer: str,
) -> tuple[GateVerdict, dict[str, object]]:
    """Check action, target, bounds and coordinate conversion on the delivered view."""
    if any(
        source.coverage != "MET"
        or source.domain != domain
        or source.scope_id != scope_id
        or source.view_id != view_id
        for source in sources
    ):
        return GateVerdict.UNKNOWN, {"reason": "Incomplete or mismatched screen evidence"}
    canonical = [source.model_dump(mode="json", exclude={"reason"}) for source in sources]
    if canonical[0] != canonical[1]:
        return GateVerdict.UNKNOWN, {"reason": "Independent control inventories disagree"}
    if not isinstance(image_views, list) or len(image_views) != 1:
        return GateVerdict.UNKNOWN, {"reason": "Delivered image view is unresolved"}
    view = image_views[0]
    if not isinstance(view, dict) or view.get("view_id") != view_id:
        return GateVerdict.UNKNOWN, {"reason": "Delivered image view ID differs"}
    try:
        width, height = int(view["width"]), int(view["height"])
    except (KeyError, TypeError, ValueError):
        return GateVerdict.UNKNOWN, {"reason": "Delivered image dimensions are missing"}
    if not 1 <= width <= 20_000 or not 1 <= height <= 20_000:
        return GateVerdict.UNKNOWN, {"reason": "Delivered image dimensions are invalid"}
    try:
        raw = json.loads(candidate_answer)
        if not isinstance(raw, dict) or set(raw) != set(UIActionAnswer.model_fields):
            raise ValueError("Wrong UI action output schema")
        answer = UIActionAnswer.model_validate_json(candidate_answer)
    except (ValueError, TypeError):
        return GateVerdict.UNKNOWN, {"reason": "Malformed UI action output"}
    target = parameters.get("target")
    action = parameters.get("action")
    controls = [item for item in sources[0].controls if item.control_id == target]
    if action not in {"click", "focus", "input"} or len(controls) != 1:
        return GateVerdict.UNKNOWN, {"reason": "Public action or unique target is unresolved"}
    control = controls[0]
    if not control.enabled or (action in {"focus", "input"} and control.kind != "field"):
        return GateVerdict.UNKNOWN, {"reason": "Target cannot perform public action"}
    input_text = parameters.get("input_text")
    if action == "input" and not isinstance(input_text, str):
        return GateVerdict.UNKNOWN, {"reason": "Public input text is missing"}
    if action != "input" and input_text is not None:
        return GateVerdict.UNKNOWN, {"reason": "Text supplied for non-input action"}
    region = control.region
    point_inside = (
        region.left <= answer.x <= region.right and region.top <= answer.y <= region.bottom
    )
    converted = (min(width - 1, int(answer.x * width)), min(height - 1, int(answer.y * height)))
    correct = (
        answer.action == action
        and answer.target_id == target
        and answer.view_id == view_id
        and point_inside
        and (answer.pixel_x, answer.pixel_y) == converted
        and answer.text == input_text
    )
    return (GateVerdict.MET if correct else GateVerdict.NOT_MET), {
        "target_region": region.model_dump(mode="json"),
        "converted_pixel": converted,
        "view_dimensions": (width, height),
    }
