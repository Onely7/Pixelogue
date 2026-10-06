from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from pixelogue.catalog import load_legacy_migration, load_task_catalog, task_catalog
from pixelogue.config import load_config
from pixelogue.contracts import InstructionCandidate
from pixelogue.drafting import FactKey, QuestionDraft, draft_task_contract, draft_to_candidate
from pixelogue.errors import ExecutionError, ExternalInputError
from pixelogue.operations import compile_configuration
from pixelogue.prompts import validate_stage_payload
from pixelogue.serialization import canonical_json
from pixelogue.store import RunStore
from pixelogue.task_catalog import TaskCatalog
from pixelogue.task_evidence import ImageRegion
from pixelogue.task_runtime import (
    fingerprint,
    operation_contract,
)

REGION = ImageRegion(left=0.0, top=0.0, right=1.0, bottom=1.0)


def test_catalog_counts_and_legacy_mapping_cover_the_specification():
    catalog = task_catalog()
    assert len(catalog.tasks) == 72
    assert sum(task.status == "core_candidate" for task in catalog.tasks) == 65
    assert sum(task.status == "validator_gated_extension" for task in catalog.tasks) == 7
    assert len(catalog.families) == 14
    migration = {row["old_id"]: row for row in load_legacy_migration()}
    assert len(migration) == 24
    assert migration["chart_lookup"]["new_ids"] == ["chart_encoding_lookup", "chart_value_lookup"]
    assert migration["grounded_sum"]["new_ids"] == ["grounded_arithmetic"]
    assert migration["region_description"]["new_ids"] == ["grounded_description"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update(version="8.0"),
        lambda c: c.update(extra_field=True),
        lambda c: c["counts"].update(core_candidates=66),
        lambda c: c["tasks"][0].update(id=c["tasks"][1]["id"]),
        lambda c: c["tasks"][0].update(required_capabilities=["imaginary_capability"]),
        lambda c: c["tasks"][0].update(eligibility_checks=["imaginary_check"]),
        lambda c: c["tasks"][0].update(verification_contracts=["imaginary_verifier"]),
        lambda c: c["tasks"][0].update(family="text_reading"),
        lambda c: c["tasks"][65]["parameters"].update(enabled_by_default=True),
        lambda c: c["families"]["visual_description"]["task_ids"].append("unknown_task"),
        lambda c: c["profile_contracts"]["limitation"]["eligible_task_ids"].append("unknown_task"),
    ],
)
def test_catalog_rejects_malformed_counts_versions_and_references(mutation):
    catalog = deepcopy(load_task_catalog())
    mutation(catalog)
    with pytest.raises(ValidationError):
        TaskCatalog.model_validate_json(canonical_json(catalog))


def test_compile_exposes_admission_migration_and_versioned_run_identity(tmp_path):
    config = load_config(Path("configs/pilot.yaml"))
    compiled = compile_configuration(config)
    assert compiled["task_catalog"]["counts"]["tasks"] == 72
    assert not compiled["task_admission"]["screen_to_code"]["normal_profile_available"]
    assert {"ImageProfile", "QuestionDraftBatch", "QuestionGateVote"} <= set(compiled["schemas"])
    assert "ScopedEvidenceInventory" not in compiled["schemas"]
    assert len(compiled["legacy_task_migration"]) == 24
    # An old configuration-only identity cannot resume under a new catalog contract.
    from pixelogue.serialization import canonical_hash

    old_hash = canonical_hash(config.model_dump(mode="json", exclude={"tasks"}))
    with RunStore(tmp_path, "legacy", require_local_wal=False) as store:
        store.initialize_run("legacy", old_hash, "pilot")
        with pytest.raises(ExternalInputError, match="another configuration"):
            store.initialize_run("legacy", config.config_hash, "pilot")


def test_legacy_candidate_labels_remain_immutable_when_reading_old_records():
    record = {
        "candidate_id": "legacy",
        "task_id": "region_description",
        "family": "observation_attribute",
        "visible_scope": "left half",
        "instruction_summary": "Describe the region.",
        "required_capabilities": ["visible_region"],
    }
    candidate = InstructionCandidate.model_validate_json(json.dumps(record))
    assert candidate.catalog_version is None
    assert candidate.task_id == "region_description"
    assert candidate.family == "observation_attribute"


def test_specialist_cannot_bypass_its_validator_by_claiming_a_limitation():
    task = next(task for task in task_catalog().tasks if task.id == "screen_to_code")
    with pytest.raises(ValidationError, match="does not support this answerability profile"):
        InstructionCandidate(
            candidate_id="fake",
            task_id=task.id,
            family=task.family,
            profile="limitation",
            visible_scope="screen",
            instruction_summary=task.definition_en,
            required_capabilities=task.required_capabilities,
            catalog_version="7.0",
            scope_id="screen",
            view_id="view",
            evidence_refs=("evidence",),
            verification_contracts=("dual_visual_review",),
        )


def test_generic_person_category_is_distinct_from_individual_identity():
    task = next(item for item in task_catalog().tasks if item.id == "object_identification")
    assert "generic person categories" in task.definition_en
    assert "individual's identity" in task.definition_en
    assert "named identity" in task.do_not_infer


def _drafted(task_id: str, target: str) -> InstructionCandidate:
    draft = QuestionDraft(
        task_id=task_id,
        question="What is the object on the left side of the image?",
        target=target,
        public_parameters=(),
        scope_region=REGION,
        target_region=REGION,
        fact_key=FactKey(subject=target, dimension="category"),
    )
    return draft_to_candidate(
        draft,
        image_id="image",
        view_id="view",
        turn_index=1,
        draft_index=0,
        allowed_task_ids=frozenset({task_id}),
    )


def test_identification_fingerprint_collapses_label_format_and_plural() -> None:
    identities = {
        fingerprint(_drafted("object_identification", label), "image")
        for label in ("qr_code", "QR code", "Three QR codes")
    }
    assert len(identities) == 1


def test_model_payloads_exclude_catalog_provenance() -> None:
    task = next(task for task in task_catalog().tasks if task.id == "object_identification")
    for payload in (
        operation_contract(_drafted("object_identification", "blue object")),
        draft_task_contract(task),
    ):
        text = json.dumps(payload)
        assert "inspiration_subsets" not in text
        assert "source_urls" not in text
    with pytest.raises(ExecutionError, match="Forbidden fields"):
        validate_stage_payload("question_gate", {"selected_instruction": {"provenance": "labels"}})
