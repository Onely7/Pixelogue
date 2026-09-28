"""Static code reconstruction checked against an isolated render of the actual view."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Annotated, Literal

from PIL import Image, ImageChops, ImageStat
from pydantic import Field

from pixelogue.config import StrictModel
from pixelogue.contracts import GateVerdict
from pixelogue.errors import ExecutionError, ExternalInputError
from pixelogue.serialization import strict_json_object
from pixelogue.specialist_env import call_renderer
from pixelogue.task_evidence import ImageRegion


class RenderSource(StrictModel):
    """Blind complete visual reading of the requested view and visible labels."""

    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    domain: str = Field(min_length=1)
    scope_id: str
    view_id: str
    scope_region: ImageRegion
    visible_text: Annotated[tuple[str, ...], Field(max_length=128)]
    reason: str = Field(min_length=1)


class RenderAnswer(StrictModel):
    """Strict public static code result."""

    format: Literal["svg", "tikz_subset", "html_css"]
    code: str | dict[str, str]


def _pixel_metrics(
    reference: Image.Image, rendered: Image.Image, region: ImageRegion
) -> tuple[float, float]:
    """Calculate average RGB error and foreground overlap in the bound scope."""
    if reference.size != rendered.size:
        raise ValueError("Rendered viewport differs from reference")
    width, height = reference.size
    box = (
        int(region.left * width),
        int(region.top * height),
        max(1, int(region.right * width)),
        max(1, int(region.bottom * height)),
    )
    reference = reference.convert("RGB").crop(box)
    rendered = rendered.convert("RGB").crop(box)
    difference = ImageChops.difference(reference, rendered)
    mean_error = sum(ImageStat.Stat(difference).mean) / (3 * 255)
    ref_gray = reference.convert("L")
    ren_gray = rendered.convert("L")
    ref_mask = ref_gray.point(lambda value: 255 if value < 245 else 0)
    ren_mask = ren_gray.point(lambda value: 255 if value < 245 else 0)
    intersection = ImageChops.multiply(ref_mask, ren_mask)
    union = ImageChops.lighter(ref_mask, ren_mask)
    union_count = ImageStat.Stat(union).sum[0]
    overlap = ImageStat.Stat(intersection).sum[0] / union_count if union_count else 1.0
    return mean_error, overlap


def verify_render(
    sources: tuple[RenderSource, RenderSource],
    domain: str,
    expected_format: object,
    scope_id: str,
    view_id: str,
    image_views: object,
    reference_image: Path | None,
    candidate_answer: str,
) -> tuple[GateVerdict, dict[str, object]]:
    """Require blind text agreement, static syntax and isolated image similarity."""
    if any(
        source.coverage != "MET"
        or source.domain != domain
        or source.scope_id != scope_id
        or source.view_id != view_id
        for source in sources
    ):
        return GateVerdict.UNKNOWN, {"reason": "Incomplete visual reconstruction evidence"}
    if sources[0].scope_region != sources[1].scope_region or sorted(
        sources[0].visible_text
    ) != sorted(sources[1].visible_text):
        return GateVerdict.UNKNOWN, {"reason": "Independent visual readings disagree"}
    if reference_image is None or not isinstance(image_views, list) or len(image_views) != 1:
        return GateVerdict.UNKNOWN, {"reason": "Reference view is unavailable"}
    view = image_views[0]
    if not isinstance(view, dict) or view.get("view_id") != view_id:
        return GateVerdict.UNKNOWN, {"reason": "Reference view identity mismatch"}
    try:
        width, height = int(view["width"]), int(view["height"])
        raw = strict_json_object(candidate_answer)
        if set(raw) != {"format", "code"}:
            raise ValueError("Wrong code output schema")
        answer = RenderAnswer.model_validate_json(candidate_answer)
    except (ExternalInputError, KeyError, TypeError, ValueError):
        return GateVerdict.UNKNOWN, {"reason": "Malformed code answer or viewport"}
    if answer.format != expected_format or not 1 <= width <= 2048 or not 1 <= height <= 2048:
        return GateVerdict.UNKNOWN, {"reason": "Unsupported format or viewport"}
    if answer.format == "html_css" and not isinstance(answer.code, dict):
        return GateVerdict.UNKNOWN, {"reason": "HTML/CSS code needs separate static fields"}
    if answer.format != "html_css" and not isinstance(answer.code, str):
        return GateVerdict.UNKNOWN, {"reason": "Vector code must be a string"}
    try:
        result = call_renderer(
            {"format": answer.format, "code": answer.code, "width": width, "height": height}
        )
    except ExecutionError as exc:
        return GateVerdict.UNKNOWN, {"reason": str(exc)}
    if result["verdict"] != "MET":
        return GateVerdict.UNKNOWN, {"reason": result.get("reason", "Renderer abstained")}
    labels = result.get("texts")
    if not isinstance(labels, list) or sorted(labels) != sorted(sources[0].visible_text):
        return GateVerdict.NOT_MET, {"reason": "Rendered labels differ from blind image reading"}
    try:
        image_bytes = base64.b64decode(result["png_base64"], validate=True)
        with (
            Image.open(reference_image) as reference,
            Image.open(io.BytesIO(image_bytes)) as rendered,
        ):
            mean_error, foreground_overlap = _pixel_metrics(
                reference, rendered, sources[0].scope_region
            )
    except (OSError, ValueError, KeyError, TypeError):
        return GateVerdict.UNKNOWN, {"reason": "Rendered or reference image cannot be compared"}
    verdict = (
        GateVerdict.MET
        if mean_error <= 0.12 and foreground_overlap >= 0.80
        else GateVerdict.NOT_MET
    )
    return verdict, {
        "mean_rgb_error": mean_error,
        "foreground_overlap": foreground_overlap,
        "thresholds": {"max_mean_rgb_error": 0.12, "min_foreground_overlap": 0.80},
    }
