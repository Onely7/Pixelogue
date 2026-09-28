"""Freeze blind positive and negative PubChem structure evaluation cases."""

from __future__ import annotations

import json
from pathlib import Path

from pixelogue.specialist_evaluation import SpecialistEvaluationCase

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/pubchem-2d-simple-eval"
PREPARED = ROOT / "artifacts/prepared-pubchem-2d-simple-eval"
OUTPUT = ROOT / "artifacts/plan-followup/pubchem-2d-chemical-cases.jsonl"
DOMAIN = "pubchem-2d-acyclic-cnohalogen-3-5-v1"


def main() -> None:
    """Build a disjoint 10/79 split using only local verified source labels."""
    images = {
        item["source_id"]: item
        for item in map(json.loads, (PREPARED / "images.jsonl").read_text().splitlines())
    }
    labels = [
        json.loads(line) for line in (SOURCE / "private_labels.jsonl").read_text().splitlines()
    ]
    assert len(labels) == len(images) == 89
    assert len({item["canonical_smiles"] for item in labels}) == 89
    assert len({images[item["source_id"]]["visual_group_id"] for item in labels}) == 89
    confirmation = labels[10:]
    cases = []
    for index, label in enumerate(labels):
        image = images[label["source_id"]]
        split = label["split"]
        positive = split == "development" or index < 30
        alternative = confirmation[(index + 19) % len(confirmation)]
        assert alternative["canonical_smiles"] != label["canonical_smiles"]
        candidate = label["canonical_smiles"] if positive else alternative["canonical_smiles"]
        cases.append(
            {
                "case_id": "pubchem-cid-" + str(label["cid"]),
                "image": image,
                "task_id": "chemical_structure_reading",
                "domain": DOMAIN,
                "scope_id": "whole-molecule",
                "view_id": image["full_view"]["view_id"],
                "visible_scope": "the complete PubChem 2D molecular depiction",
                "public_parameters": [
                    {
                        "name": "notation",
                        "value": "nonstereo_smiles",
                        "origin": "instruction",
                        "evidence_refs": [],
                    }
                ],
                "target_language": "en",
                "public_history": [],
                "question": "Give the nonstereochemical SMILES for the complete depicted molecule.",
                "candidate_answer": json.dumps({"smiles": candidate}),
                "split": split,
                "gold_accept": positive if split == "confirmation" else None,
                "label_provenance": (
                    label["label_provenance"]
                    + "; negative answer from PubChem CID "
                    + str(alternative["cid"])
                    if split == "confirmation" and not positive
                    else label["label_provenance"]
                    if split == "confirmation"
                    else None
                ),
            }
        )
    assert sum(case["gold_accept"] is True for case in cases) == 20
    assert sum(case["gold_accept"] is False for case in cases) == 59
    for case in cases:
        SpecialistEvaluationCase.model_validate_json(json.dumps(case))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("".join(json.dumps(case, sort_keys=True) + "\n" for case in cases))
    for split in ("development", "confirmation"):
        OUTPUT.with_name(f"pubchem-2d-chemical-{split}.jsonl").write_text(
            "".join(
                json.dumps(case, sort_keys=True) + "\n" for case in cases if case["split"] == split
            )
        )
    print(json.dumps({"cases": len(cases), "development": 10, "confirmation": 79}))


if __name__ == "__main__":
    main()
