"""Freeze blind UI-click calibration cases from ScreenSpot target annotations."""

from __future__ import annotations

import json
from pathlib import Path

from pixelogue.contracts import GateVerdict
from pixelogue.specialist_evaluation import SpecialistEvaluationCase
from pixelogue.specialist_ui import UIActionSource, verify_ui_action

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/screenspot-ui-eval"
PREPARED = ROOT / "artifacts/prepared-screenspot-ui-eval"
OUTPUT = ROOT / "artifacts/plan-followup/screenspot-ui-cases.jsonl"
DOMAIN = "screenspot-text-click-v1"


def _point(box: list[float], positive: bool) -> tuple[float, float]:
    """Choose the annotated center or a fixed off-target corner."""
    if positive:
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    corners = ((0.04, 0.04), (0.96, 0.04), (0.04, 0.96), (0.96, 0.96))
    x, y = max(
        corners,
        key=lambda point: (
            (point[0] - (box[0] + box[2]) / 2) ** 2 + (point[1] - (box[1] + box[3]) / 2) ** 2
        ),
    )
    if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
        raise ValueError("No independent wrong-click point outside target")
    return x, y


def main() -> None:
    """Build disjoint development and confirmation cases without model-visible labels."""
    images = {
        item["source_id"]: item
        for item in map(json.loads, (PREPARED / "images.jsonl").read_text().splitlines())
    }
    labels = [
        json.loads(line) for line in (SOURCE / "private_labels.jsonl").read_text().splitlines()
    ]
    if len(labels) != 89 or len(images) != 89:
        raise ValueError("ScreenSpot preparation is incomplete")
    if len({images[item["source_id"]]["visual_group_id"] for item in labels}) != 89:
        raise ValueError("ScreenSpot confirmation images share a visual group")
    cases = []
    for index, label in enumerate(labels):
        image = images[label["source_id"]]
        split = label["split"]
        positive = split == "development" or index < 30
        x, y = _point(label["bbox"], positive)
        view = image["full_view"]
        answer = {
            "action": "click",
            "target_id": label["instruction"],
            "view_id": view["view_id"],
            "x": x,
            "y": y,
            "pixel_x": min(view["width"] - 1, int(x * view["width"])),
            "pixel_y": min(view["height"] - 1, int(y * view["height"])),
            "text": None,
        }
        case = {
            "case_id": "screenspot-row-" + label["source_id"].split(":", 1)[1],
            "image": image,
            "task_id": "ui_action_specification",
            "domain": DOMAIN,
            "scope_id": "whole-screenshot",
            "view_id": view["view_id"],
            "visible_scope": "the complete visible application screenshot",
            "public_parameters": [
                {"name": "action", "value": "click", "origin": "instruction", "evidence_refs": []},
                {
                    "name": "target",
                    "value": label["instruction"],
                    "origin": "instruction",
                    "evidence_refs": [],
                },
            ],
            "target_language": "en",
            "public_history": [],
            "question": (
                "Specify one click on the visible control described as '"
                + label["instruction"]
                + "' using the view ID and normalized and pixel coordinates; do not execute it."
            ),
            "candidate_answer": json.dumps(answer, sort_keys=True),
            "split": split,
            "gold_accept": positive if split == "confirmation" else None,
            "label_provenance": label["label_provenance"] if split == "confirmation" else None,
        }
        SpecialistEvaluationCase.model_validate_json(json.dumps(case))
        # Controller-side label consistency only; actual evaluation still needs two blind readers.
        source = UIActionSource.model_validate_json(
            json.dumps(
                {
                    "coverage": "MET",
                    "domain": DOMAIN,
                    "scope_id": "whole-screenshot",
                    "view_id": view["view_id"],
                    "scope_region": {"left": 0, "top": 0, "right": 1, "bottom": 1},
                    "controls": [
                        {
                            "control_id": label["instruction"],
                            "kind": "button",
                            "enabled": True,
                            "region": {
                                "left": label["bbox"][0],
                                "top": label["bbox"][1],
                                "right": label["bbox"][2],
                                "bottom": label["bbox"][3],
                            },
                        }
                    ],
                    "reason": "ScreenSpot independent target annotation",
                }
            )
        )
        verdict, _ = verify_ui_action(
            (source, source),
            DOMAIN,
            {"action": "click", "target": label["instruction"]},
            "whole-screenshot",
            view["view_id"],
            [{"view_id": view["view_id"], "width": view["width"], "height": view["height"]}],
            case["candidate_answer"],
        )
        if verdict != (GateVerdict.MET if positive else GateVerdict.NOT_MET):
            raise ValueError(f"ScreenSpot case has an inconsistent gold label: {case['case_id']}")
        cases.append(case)
    if (
        sum(case["gold_accept"] is True for case in cases) != 20
        or sum(case["gold_accept"] is False for case in cases) != 59
    ):
        raise ValueError("ScreenSpot confirmation class balance changed")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("".join(json.dumps(case, sort_keys=True) + "\n" for case in cases))
    for split in ("development", "confirmation"):
        OUTPUT.with_name(f"screenspot-ui-{split}.jsonl").write_text(
            "".join(
                json.dumps(case, sort_keys=True) + "\n" for case in cases if case["split"] == split
            )
        )
    print(json.dumps({"cases": len(cases), "development": 10, "confirmation": 79}))


if __name__ == "__main__":
    main()
