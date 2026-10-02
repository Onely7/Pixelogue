"""Compact routing removes fixed IDs while preserving strict visual evidence contracts."""

import copy
import json

import pytest
from pydantic import ValidationError

from pixelogue.config import ModelEndpoint, RuntimeConfig, TaskRuntimeConfig
from pixelogue.errors import ExecutionError
from pixelogue.planner import instruction_candidates
from pixelogue.serving import VllmClient
from pixelogue.task_evidence import CompactScopedEvidenceReport, bind_evidence_identity
from pixelogue.task_runtime import validate_evidence

REGION = {"left": 0, "top": 0, "right": 1, "bottom": 1}


def report():
    return {
        "scopes": [
            {
                "public_description": "the dog",
                "object_label": "dog",
                "region": REGION,
                "observations": [
                    {
                        "capability": "visible_attribute",
                        "verdict": "MET",
                        "region": REGION,
                        "detail": "White fur and a black nose are visible.",
                    },
                    {
                        "capability": "visible_entity",
                        "verdict": "MET",
                        "region": REGION,
                        "detail": "A dog is visible.",
                    },
                ],
            }
        ],
        "reason": "The same subject has multiple visible properties.",
    }


def test_compact_ids_are_local_stable_and_order_does_not_change_eligible_tasks():
    data = report()
    reversed_data = copy.deepcopy(data)
    reversed_data["scopes"][0]["observations"].reverse()
    inventories = [
        bind_evidence_identity(
            CompactScopedEvidenceReport.model_validate_json(json.dumps(item)), "image", "view"
        )
        for item in (data, reversed_data)
    ]
    for inventory in inventories:
        validate_evidence(inventory, "view", TaskRuntimeConfig())
        assert inventory.image_id == "image" and inventory.scopes[0].view_id == "view"
        assert inventory.scopes[0].scope_id == "scope_1"
        assert {obs.evidence_id for obs in inventory.scopes[0].observations} == {
            "obs_1_1",
            "obs_1_2",
        }
        assert all(
            obs.verdict == "MET" and obs.region.model_dump() == REGION
            for obs in inventory.scopes[0].observations
        )
    sets = [
        {candidate.task_id for candidate in instruction_candidates(item, seed=1, turn_index=1)}
        for item in inventories
    ]
    assert sets[0] == sets[1] and {"attribute_lookup", "object_identification"} <= sets[0]
    assert (
        bind_evidence_identity(
            CompactScopedEvidenceReport.model_validate_json(json.dumps(data)), "image", "view"
        )
        == inventories[0]
    )


@pytest.mark.parametrize(
    "invalid",
    [
        "duplicate_capability",
        "unknown_capability",
        "outside_scope",
        "outside_subject",
        "foreign_image_id",
        "foreign_view_id",
        "supplied_evidence_id",
    ],
)
def test_compact_cannot_supply_foreign_ids_or_pass_invalid_observations(invalid):
    data = copy.deepcopy(report())
    scope = data["scopes"][0]
    if invalid == "duplicate_capability":
        scope["observations"].append(scope["observations"][0].copy())
    elif invalid == "unknown_capability":
        scope["observations"][0]["capability"] = "invented"
    elif invalid == "outside_scope":
        scope["region"] = {"left": 0.1, "top": 0.1, "right": 0.9, "bottom": 0.9}
    elif invalid == "outside_subject":
        scope["observations"][1]["region"] = {"left": 0.1, "top": 0.1, "right": 0.3, "bottom": 0.3}
    elif invalid == "foreign_image_id":
        data["image_id"] = "other"
    elif invalid == "foreign_view_id":
        scope["view_id"] = "other"
    else:
        scope["observations"][0]["evidence_id"] = "other-scope"
    with pytest.raises((ExecutionError, ValidationError)):
        inventory = bind_evidence_identity(
            CompactScopedEvidenceReport.model_validate_json(json.dumps(data)), "image", "view"
        )
        validate_evidence(inventory, "view", TaskRuntimeConfig())


def test_missing_attribute_stays_unknown_and_does_not_get_filled():
    data = report()
    data["scopes"][0]["observations"].pop(0)
    inventory = bind_evidence_identity(
        CompactScopedEvidenceReport.model_validate_json(json.dumps(data)), "image", "view"
    )
    assert not any(
        candidate.task_id == "attribute_lookup"
        for candidate in instruction_candidates(inventory, seed=1, turn_index=1)
    )
    assert len(inventory.scopes[0].observations) == 1


def test_compact_decoder_keeps_details_regions_vocabulary_and_bounds():
    client = VllmClient(ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"), RuntimeConfig(), run_id="compact")
    try:
        body = client._build_body(
            "evidence_extraction",
            {
                "image_id": "image",
                "image_views": [],
                "capability_vocabulary": {
                    "visible_entity": "entity",
                    "visible_attribute": "property",
                },
                "max_scopes": 2,
                "max_observations_per_scope": 3,
            },
            (),
            CompactScopedEvidenceReport,
            max_tokens=4096,
            temperature=0.0,
            seed=1,
        )
    finally:
        client.client.close()
    schema = body["response_format"]["json_schema"]["schema"]
    scope = schema["$defs"]["CompactScopeEvidenceReport"]["properties"]
    observation = schema["$defs"]["CompactCapabilityObservation"]["properties"]
    assert (
        "image_id" not in schema["properties"]
        and "view_id" not in scope
        and "scope_id" not in scope
    )
    assert "evidence_id" not in observation and {
        "region",
        "detail",
        "verdict",
        "capability",
    } <= set(observation)
    assert observation["capability"]["enum"] == ["visible_attribute", "visible_entity"]
    assert (
        scope["observations"]["maxItems"] == 3 and schema["properties"]["scopes"]["maxItems"] == 2
    )
    prompt = json.loads(body["messages"][1]["content"][0]["text"])["instruction"]
    assert "Do not output image_id" in prompt and "fur color and nose color" in prompt
    assert "Copy image_id and each view_id exactly" not in prompt
    assert "Each observation has a unique evidence_id" not in prompt
