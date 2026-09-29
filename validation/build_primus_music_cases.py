"""Freeze blind score-reading cases from the official PrIMuS MEI labels."""

from __future__ import annotations

import json
from pathlib import Path

from pixelogue.contracts import GateVerdict
from pixelogue.specialist_evaluation import SpecialistEvaluationCase
from pixelogue.specialist_music import MusicSource, verify_music

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/primus-music-eval"
PREPARED = ROOT / "artifacts/prepared-primus-music-eval"
OUTPUT = ROOT / "artifacts/plan-followup/primus-music-cases.jsonl"
DOMAIN = "primus-printed-gf-key0-firstbar-v1"


def _wrong_pitch(pitch: str) -> str:
    """Use an adjacent written note as a plausible incorrect transcription."""
    letters = "CDEFGAB"
    index = letters.index(pitch[0])
    octave = int(pitch[1:])
    return letters[(index + 1) % 7] + str(octave + (index == 6))


def main() -> None:
    """Build disjoint cases and check their labels against the local score worker."""
    images = {
        item["source_id"]: item
        for item in map(json.loads, (PREPARED / "images.jsonl").read_text().splitlines())
    }
    labels = [
        json.loads(line) for line in (SOURCE / "private_labels.jsonl").read_text().splitlines()
    ]
    if len(labels) != 89 or len(images) != 89:
        raise ValueError("PrIMuS preparation is incomplete")
    if len({images[item["source_id"]]["visual_group_id"] for item in labels}) != 89:
        raise ValueError("PrIMuS score groups are not independent")
    cases = []
    for index, item in enumerate(labels):
        image = images[item["source_id"]]
        view = image["full_view"]
        label = item["label"]
        split = item["split"]
        positive = split == "development" or index < 30
        events = [dict(event) for event in label["answer_events"]]
        if not positive:
            first_note = next((event for event in events if event["pitch"] is not None), None)
            if first_note is None:
                raise ValueError("PrIMuS score lacks a note for negative answer")
            first_note["pitch"] = _wrong_pitch(first_note["pitch"])
        answer = json.dumps({"events": events}, sort_keys=True)
        sample = item["source_id"].split(":", 1)[1]
        case = {
            "case_id": "primus-score-" + sample,
            "image": image,
            "task_id": "music_notation_reading",
            "domain": DOMAIN,
            "scope_id": "first-complete-measure",
            "view_id": view["view_id"],
            "visible_scope": "the first complete measure of the printed single-voice score",
            "public_parameters": [
                {"name": "bar_range", "value": "1-1", "origin": "instruction", "evidence_refs": []}
            ],
            "target_language": "en",
            "public_history": [],
            "question": (
                "Transcribe each note and rest in the first complete measure (bar 1) "
                "as ordered pitch, quarter-length duration and tie events."
            ),
            "candidate_answer": answer,
            "split": split,
            "gold_accept": positive if split == "confirmation" else None,
            "label_provenance": item["label_provenance"] if split == "confirmation" else None,
        }
        SpecialistEvaluationCase.model_validate_json(json.dumps(case))
        # This checks labels only; actual calibration still uses two independent image readings.
        source = MusicSource.model_validate_json(
            json.dumps(
                {
                    "coverage": "MET",
                    "domain": DOMAIN,
                    "scope_id": "first-complete-measure",
                    "view_id": view["view_id"],
                    "scope_region": {"left": 0, "top": 0, "right": 1, "bottom": 1},
                    "clef": label["clef"],
                    "key_sharps": label["key_sharps"],
                    "meter_top": label["meter_top"],
                    "meter_bottom": label["meter_bottom"],
                    "bar_range": "1-1",
                    "measures": [
                        {
                            "number": 1,
                            "events": [
                                {
                                    **event,
                                    "region": {"left": 0, "top": 0, "right": 1, "bottom": 1},
                                }
                                for event in label["source_events"]
                            ],
                        }
                    ],
                    "reason": "Independent PrIMuS MEI label",
                }
            )
        )
        verdict, _ = verify_music(
            (source, source),
            DOMAIN,
            "1-1",
            "first-complete-measure",
            view["view_id"],
            answer,
        )
        if verdict != (GateVerdict.MET if positive else GateVerdict.NOT_MET):
            raise ValueError(f"PrIMuS label conflicts with score validator: {sample}")
        cases.append(case)
    if (
        sum(case["gold_accept"] is True for case in cases) != 20
        or sum(case["gold_accept"] is False for case in cases) != 59
    ):
        raise ValueError("PrIMuS confirmation class balance changed")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("".join(json.dumps(case, sort_keys=True) + "\n" for case in cases))
    for split in ("development", "confirmation"):
        OUTPUT.with_name(f"primus-music-{split}.jsonl").write_text(
            "".join(
                json.dumps(case, sort_keys=True) + "\n" for case in cases if case["split"] == split
            )
        )
    print(json.dumps({"cases": len(cases), "development": 10, "confirmation": 79}))


if __name__ == "__main__":
    main()
