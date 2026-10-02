"""Order-independent routing evidence and strict capability boundaries."""

import copy
import json

import pytest

from pixelogue.config import ModelEndpoint, RuntimeConfig, TaskRuntimeConfig
from pixelogue.errors import ExecutionError
from pixelogue.planner import instruction_candidates
from pixelogue.serving import VllmClient
from pixelogue.task_evidence import ArrayScopedEvidenceReport
from pixelogue.task_runtime import validate_evidence

REGION = {"left": 0.0, "top": 0.0, "right": 1.0, "bottom": 1.0}


def report():
    return {
        "image_id": "image",
        "reason": "Direct visual observations.",
        "scopes": [
            {
                "scope_id": "subject",
                "view_id": "view",
                "public_description": "Visible dog",
                "object_label": "dog",
                "region": REGION,
                "observations": [
                    {
                        "evidence_id": "attr",
                        "capability": "visible_attribute",
                        "verdict": "MET",
                        "region": REGION,
                        "detail": "The dog's fur is brown.",
                    },
                    {
                        "evidence_id": "entity",
                        "capability": "visible_entity",
                        "verdict": "MET",
                        "region": REGION,
                        "detail": "One dog is directly visible.",
                    },
                ],
            }
        ],
    }


def decode(data):
    return ArrayScopedEvidenceReport.model_validate_json(json.dumps(data)).to_inventory()


def test_attribute_before_entity_and_entity_before_attribute_have_same_candidates():
    data = report()
    reverse = copy.deepcopy(data)
    reverse["scopes"][0]["observations"].reverse()
    inventories = [decode(item) for item in (data, reverse)]
    settings = TaskRuntimeConfig()
    for inventory in inventories:
        validate_evidence(inventory, "view", settings)
    task_sets = [
        {
            candidate.task_id
            for candidate in instruction_candidates(
                inventory, seed=1, turn_index=1, settings=settings
            )
        }
        for inventory in inventories
    ]
    assert task_sets[0] == task_sets[1]
    assert "attribute_lookup" in task_sets[0] and "object_identification" in task_sets[0]


@pytest.mark.parametrize(
    "violation",
    [
        "capability_duplicate",
        "evidence_id_duplicate",
        "outside_parent",
        "unknown_capability",
        "wrong_view",
    ],
)
def test_invalid_observations_remain_rejected(violation):
    data = copy.deepcopy(report())
    scope = data["scopes"][0]
    observations = scope["observations"]
    if violation == "capability_duplicate":
        observations.append({**observations[0], "evidence_id": "other"})
    elif violation == "evidence_id_duplicate":
        observations[0]["evidence_id"] = observations[1]["evidence_id"]
    elif violation == "outside_parent":
        scope["region"] = {"left": 0.2, "top": 0.2, "right": 0.8, "bottom": 0.8}
    elif violation == "unknown_capability":
        observations[0]["capability"] = "invented"
    else:
        scope["view_id"] = "different"
    with pytest.raises(ExecutionError):
        inventory = decode(data)
        validate_evidence(inventory, "view", TaskRuntimeConfig())


def test_decoder_limits_array_capabilities_and_preserves_description_contract():
    client = VllmClient(ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"), RuntimeConfig(), run_id="array")
    try:
        body = client._build_body(
            "evidence_extraction",
            {
                "image_id": "image",
                "image_views": [],
                "capability_vocabulary": {
                    "visible_entity": "Entity",
                    "visible_attribute": "Attribute",
                },
                "max_scopes": 2,
                "max_observations_per_scope": 3,
            },
            (),
            ArrayScopedEvidenceReport,
            max_tokens=4096,
            temperature=0.0,
            seed=1,
        )
        schema = body["response_format"]["json_schema"]["schema"]
        assert schema["$defs"]["CapabilityObservation"]["properties"]["capability"]["enum"] == [
            "visible_attribute",
            "visible_entity",
        ]
        assert (
            schema["$defs"]["ArrayScopeEvidenceReport"]["properties"]["observations"]["maxItems"]
            == 3
        )
        assert schema["properties"]["scopes"]["maxItems"] == 2
        text = json.loads(body["messages"][1]["content"][0]["text"])["instruction"]
        assert "field is an array" in text
        assert "Name independent properties separately" in text
        assert "object keyed by" not in text
        assert (
            "enum"
            not in ArrayScopedEvidenceReport.model_json_schema()["$defs"]["CapabilityObservation"][
                "properties"
            ]["capability"]
        )
    finally:
        client.client.close()
