"""Freeze a new blind PubChem confirmation set after independent image ingestion."""

from __future__ import annotations

import json
from pathlib import Path

from pixelogue.specialist_evaluation import SpecialistEvaluationCase

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/pubchem-2d-v2-eval"
PREPARED = ROOT / "artifacts/prepared-pubchem-2d-v2-eval"
OUTPUT = ROOT / "artifacts/plan-followup/pubchem-2d-chemical-v2-confirmation.jsonl"
SELECTION = ROOT / "validation/pubchem_2d_eval_v2_selection.json"
DOMAIN = "pubchem-2d-acyclic-cnohalogen-3-5-v1"


def main() -> None:
    """Fix thirty correct and sixty incorrect answers without exposing labels to judges."""
    selection = json.loads(SELECTION.read_text())
    positive_count = selection["confirmation_positive"]
    expected_count = selection["selected"]
    images = {
        item["source_id"]: item
        for item in map(json.loads, (PREPARED / "images.jsonl").read_text().splitlines())
    }
    labels = [
        json.loads(line) for line in (SOURCE / "private_labels.jsonl").read_text().splitlines()
    ]
    if len(labels) != len(images) or len(labels) != expected_count:
        raise ValueError("PubChem v2 image or label count changed")
    if len({item["canonical_smiles"] for item in labels}) != expected_count:
        raise ValueError("PubChem v2 structures are not independent")
    if len({images[item["source_id"]]["visual_group_id"] for item in labels}) != expected_count:
        raise ValueError("PubChem v2 image groups are not independent")
    cases = []
    for index, label in enumerate(labels):
        image = images[label["source_id"]]
        positive = index < positive_count
        alternative = labels[(index + positive_count + 1) % expected_count]
        if alternative["canonical_smiles"] == label["canonical_smiles"]:
            raise ValueError("A negative answer repeats the source structure")
        candidate = label["canonical_smiles"] if positive else alternative["canonical_smiles"]
        provenance = label["label_provenance"]
        if not positive:
            provenance += "; negative answer from PubChem CID " + str(alternative["cid"])
        case = {
            "case_id": "pubchem-v2-cid-" + str(label["cid"]),
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
            "split": "confirmation",
            "gold_accept": positive,
            "label_provenance": provenance,
        }
        SpecialistEvaluationCase.model_validate_json(json.dumps(case))
        cases.append(case)
    if sum(case["gold_accept"] is True for case in cases) != positive_count:
        raise ValueError("PubChem v2 positive count changed")
    payload = "".join(json.dumps(case, sort_keys=True) + "\n" for case in cases)
    if OUTPUT.exists() and OUTPUT.read_text() != payload:
        raise ValueError("Frozen PubChem v2 confirmation cases changed")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(payload)
    print(json.dumps({"cases": len(cases), "positive": positive_count}))


if __name__ == "__main__":
    main()
