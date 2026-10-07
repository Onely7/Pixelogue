"""Bounding-box answers checked against two blind box readings."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, ValidationError

from pixelogue.config import StrictModel
from pixelogue.contracts import GateVerdict
from pixelogue.errors import ExternalInputError
from pixelogue.serialization import strict_json_object
from pixelogue.task_evidence import ImageRegion, region_iou

BOX_TASKS = frozenset({"object_box_grounding"})
MAX_BOXES = 20
# Overlap needed for two boxes to mark the same object, as in common detection benchmarks.
IOU_THRESHOLD = 0.5


class BoxSource(StrictModel):
    """One reader's answer-blind boxes for every target the question describes."""

    coverage: Literal["MET", "UNKNOWN"]
    boxes: Annotated[tuple[ImageRegion, ...], Field(max_length=MAX_BOXES)]
    reason: str = Field(min_length=1)


def parse_box_answer(candidate_answer: str) -> tuple[ImageRegion, ...] | None:
    """Read the public answer format ``{"boxes": [[left, top, right, bottom], ...]}``.

    Returns:
        The answer's boxes in order, or ``None`` when the answer breaks the format.
    """
    try:
        raw = strict_json_object(candidate_answer.strip())
    except ExternalInputError:
        return None
    boxes = raw.get("boxes")
    if set(raw) != {"boxes"} or not isinstance(boxes, list) or not 0 < len(boxes) <= MAX_BOXES:
        return None
    parsed: list[ImageRegion] = []
    for box in boxes:
        if (
            not isinstance(box, list)
            or len(box) != 4
            or not all(type(value) in {int, float} for value in box)
        ):
            return None
        try:
            parsed.append(
                ImageRegion(
                    left=float(box[0]), top=float(box[1]), right=float(box[2]), bottom=float(box[3])
                )
            )
        except ValidationError:
            return None
    return tuple(parsed)


def matched_pairs(
    first: tuple[ImageRegion, ...], second: tuple[ImageRegion, ...]
) -> tuple[tuple[int, int, float], ...] | None:
    """Pair boxes one-to-one by greatest overlap.

    Returns:
        ``(first index, second index, IoU)`` for every box, or ``None`` when the counts differ
        or some box has no partner at the required overlap.
    """
    if len(first) != len(second):
        return None
    candidates = sorted(
        (
            (region_iou(left, right), i, j)
            for i, left in enumerate(first)
            for j, right in enumerate(second)
        ),
        reverse=True,
    )
    used_first: set[int] = set()
    used_second: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for overlap, i, j in candidates:
        if overlap < IOU_THRESHOLD:
            break
        if i in used_first or j in used_second:
            continue
        used_first.add(i)
        used_second.add(j)
        pairs.append((i, j, overlap))
    return tuple(sorted(pairs)) if len(pairs) == len(first) else None


def verify_boxes(
    sources: tuple[BoxSource, BoxSource], candidate_answer: str
) -> tuple[GateVerdict, dict[str, Any]]:
    """Accept boxes only when both blind readers agree and each answer box matches theirs.

    Readers that abstain or disagree with each other leave the result UNKNOWN; an answer that
    breaks the box format or misses either agreed reading is NOT_MET.
    """
    details: dict[str, Any] = {"threshold": IOU_THRESHOLD}
    if any(source.coverage != "MET" or not source.boxes for source in sources):
        return GateVerdict.UNKNOWN, details
    readers = matched_pairs(sources[0].boxes, sources[1].boxes)
    details["reader_pairs"] = readers
    if readers is None:
        return GateVerdict.UNKNOWN, details
    answer = parse_box_answer(candidate_answer)
    if answer is None:
        details["answer"] = "invalid box format"
        return GateVerdict.NOT_MET, details
    against = tuple(matched_pairs(answer, source.boxes) for source in sources)
    details["answer_pairs"] = against
    return (
        GateVerdict.MET if all(pairs is not None for pairs in against) else GateVerdict.NOT_MET
    ), (details)
