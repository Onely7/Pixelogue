"""Bounding-box answers must match two blind box readings one-to-one."""

from __future__ import annotations

import json

import pytest

from pixelogue.box_verifier import BoxSource, matched_pairs, parse_box_answer, verify_boxes
from pixelogue.catalog import task_catalog
from pixelogue.contracts import GateVerdict, InstructionCandidate
from pixelogue.task_evidence import ImageRegion
from pixelogue.task_verification import verify_operation

LEFT = ImageRegion(left=0.1, top=0.2, right=0.3, bottom=0.5)
RIGHT = ImageRegion(left=0.6, top=0.2, right=0.8, bottom=0.5)


def _reader(*boxes: ImageRegion, coverage: str = "MET") -> BoxSource:
    return BoxSource.model_validate({"coverage": coverage, "boxes": boxes, "reason": "drawn blind"})


def _answer(*boxes: list[float]) -> str:
    return json.dumps({"boxes": list(boxes)})


def test_answer_boxes_must_match_both_blind_readers() -> None:
    readers = (_reader(LEFT, RIGHT), _reader(RIGHT, LEFT))
    close = _answer([0.11, 0.21, 0.31, 0.5], [0.6, 0.2, 0.79, 0.52])
    assert verify_boxes(readers, close)[0] is GateVerdict.MET
    shifted = _answer([0.2, 0.2, 0.4, 0.5], [0.6, 0.2, 0.8, 0.5])
    assert verify_boxes(readers, shifted)[0] is GateVerdict.NOT_MET
    missing = _answer([0.1, 0.2, 0.3, 0.5])
    assert verify_boxes(readers, missing)[0] is GateVerdict.NOT_MET
    assert verify_boxes(readers, "The car is on the left.")[0] is GateVerdict.NOT_MET
    disagreeing = (_reader(LEFT, RIGHT), _reader(LEFT))
    assert verify_boxes(disagreeing, close)[0] is GateVerdict.UNKNOWN
    abstaining = (_reader(LEFT, RIGHT), _reader(coverage="UNKNOWN"))
    assert verify_boxes(abstaining, close)[0] is GateVerdict.UNKNOWN


@pytest.mark.parametrize(
    "text",
    [
        '{"boxes": [[0.3, 0.2, 0.1, 0.5]]}',
        '{"boxes": [[0.1, 0.2, 0.3, 1.5]]}',
        '{"boxes": [[0.1, 0.2, 0.3]]}',
        '{"boxes": [[true, 0.2, 0.3, 0.5]]}',
        '{"boxes": []}',
        '{"boxes": [[0.1, 0.2, 0.3, 0.5]], "label": "car"}',
        '```json\n{"boxes": [[0.1, 0.2, 0.3, 0.5]]}\n```',
    ],
)
def test_box_answers_reject_inverted_out_of_range_and_extra_content(text: str) -> None:
    assert parse_box_answer(text) is None


def test_matching_is_one_to_one() -> None:
    twin = ImageRegion(left=0.12, top=0.2, right=0.3, bottom=0.5)
    assert matched_pairs((LEFT, twin), (LEFT, RIGHT)) is None
    assert matched_pairs((LEFT, RIGHT), (RIGHT, LEFT)) is not None


def test_box_readers_never_see_the_candidate_answer() -> None:
    task = next(item for item in task_catalog().tasks if item.id == "object_box_grounding")
    candidate = InstructionCandidate(
        candidate_id="boxes",
        task_id=task.id,
        family=task.family,
        visible_scope="the cars",
        instruction_summary=task.definition,
        required_capabilities=task.required_capabilities,
        catalog_version="8.0",
        scope_id="canvas",
        view_id="view",
        evidence_refs=("scope-evidence",),
        verification_contracts=task.verification_contracts,
    )

    def invoke(stage, payload, model, judge):
        assert stage == "box_source"
        assert "candidate_answer" not in payload
        return _reader(LEFT, RIGHT)

    answer = _answer([0.1, 0.2, 0.3, 0.5], [0.6, 0.2, 0.8, 0.5])
    result = verify_operation(
        candidate, {"question": "Box both cars.", "candidate_answer": answer}, invoke
    )
    assert [(check.name, check.verdict) for check in result] == [("box_iou_check", GateVerdict.MET)]
