from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from pixelogue.catalog import load_legacy_migration, load_task_catalog, task_catalog
from pixelogue.config import TaskRuntimeConfig, load_config
from pixelogue.contracts import InstructionCandidate, PublicMessage
from pixelogue.errors import ExecutionError, ExternalInputError
from pixelogue.operations import compile_configuration
from pixelogue.planner import instruction_candidates
from pixelogue.prompts import validate_stage_payload
from pixelogue.serialization import canonical_json
from pixelogue.store import RunStore
from pixelogue.task_catalog import TaskCatalog
from pixelogue.task_evidence import (
    CandidateBinding,
    CandidateBindings,
    CapabilityObservation,
    EligibilityObservation,
    ImageRegion,
    ObservationVerdict,
    PublicParameter,
    ScopedEvidenceInventory,
    ScopeEvidence,
)
from pixelogue.task_runtime import (
    admission_report,
    bind_candidates,
    operation_contract,
    selector_candidate,
    validate_evidence,
)

REGION = ImageRegion(left=0.0, top=0.0, right=1.0, bottom=1.0)


def scope(scope_id: str, **capabilities: ObservationVerdict) -> ScopeEvidence:
    return ScopeEvidence(
        scope_id=scope_id,
        view_id="view",
        public_description=f"the {scope_id} region",
        region=REGION,
        observations=tuple(
            CapabilityObservation(
                evidence_id=f"{scope_id}:{key}",
                capability=key,
                verdict=value,
                region=REGION,
                detail=f"Observation for {key} in {scope_id}",
            )
            for key, value in capabilities.items()
        ),
    )


def inventory(*scopes: ScopeEvidence) -> ScopedEvidenceInventory:
    return ScopedEvidenceInventory(image_id="image", scopes=scopes, reason="Local observations")


def candidates(data: ScopedEvidenceInventory, **kwargs) -> tuple[InstructionCandidate, ...]:
    return instruction_candidates(data, seed=13, turn_index=1, **kwargs)


def binding(candidate: InstructionCandidate, **kwargs) -> CandidateBinding:
    values = {
        "candidate_id": candidate.candidate_id,
        "public_parameters": (
            PublicParameter(name="target", value="the cup", origin="instruction"),
        ),
        "checks": tuple(
            EligibilityObservation(check_id=key, verdict="MET", reason="Supported")
            for key in operation_contract(candidate)["eligibility_checks"]
        ),
        "evidence_refs": candidate.evidence_refs,
        "estimated_answer_tokens": 100,
    }
    return CandidateBinding.model_validate(values | kwargs)


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


def test_capabilities_cannot_be_combined_across_scopes():
    data = inventory(scope("left", visible_entity="MET"), scope("right", visible_attribute="MET"))
    result = candidates(data)
    assert {(item.task_id, item.scope_id) for item in result} == {("object_identification", "left")}
    complete = inventory(scope("left", visible_entity="MET", visible_attribute="MET"))
    assert "attribute_lookup" in {item.task_id for item in candidates(complete)}


@pytest.mark.parametrize("verdict", ["NOT_MET", "UNKNOWN"])
def test_unknown_or_failed_capabilities_do_not_enable_normal_tasks(verdict):
    assert not candidates(inventory(scope("left", visible_entity=verdict)))


def test_eligibility_observations_are_bound_to_the_delivered_view_and_bounds():
    data = inventory(scope("left", visible_entity="MET"))
    with pytest.raises(ExecutionError, match="different image view"):
        validate_evidence(data, "other-view", TaskRuntimeConfig())
    with pytest.raises(ValidationError, match="outside its scope"):
        scope("left", visible_entity="MET").model_copy().model_validate(
            {
                **scope("left", visible_entity="MET").model_dump(),
                "region": ImageRegion(left=0.0, top=0.0, right=0.5, bottom=1.0),
            }
        )
    with pytest.raises(ExecutionError, match="Unknown evidence capability"):
        validate_evidence(inventory(scope("left", invented="MET")), "view", TaskRuntimeConfig())


def test_supported_tables_enter_candidates_while_missing_verifiers_stay_blocked():
    observations: dict[str, ObservationVerdict] = {
        key: "MET" for key in task_catalog().capabilities
    }
    data = inventory(scope("all", **observations))
    selected = {item.task_id for item in candidates(data, limit=100)}
    assert "object_identification" in selected
    assert "formula_transcription" in selected
    assert "table_structure_reconstruction" in selected
    assert "graph_path_tracing" in selected
    assert not (
        selected
        & {task.id for task in task_catalog().tasks if task.status == "validator_gated_extension"}
    )
    assert TaskRuntimeConfig(enabled_extensions=("screen_to_code",)).enabled_extensions == (
        "screen_to_code",
    )
    with pytest.raises(ValidationError, match="Unknown specialized extension"):
        TaskRuntimeConfig(enabled_extensions=("invented",))
    assert admission_report()["screen_to_code"]["blocked_reasons"]


def test_family_rotation_is_deterministic_and_does_not_sample_only_large_families():
    observations: dict[str, ObservationVerdict] = {
        key: "MET" for key in task_catalog().capabilities
    }
    data = inventory(scope("all", **observations))
    all_candidates = candidates(data, limit=100)
    families = {item.family for item in all_candidates}
    selected = [
        instruction_candidates(data, seed=13, turn_index=turn, limit=1)[0]
        for turn in range(1, len(families) + 1)
    ]
    assert {item.family for item in selected} == families
    assert candidates(data, limit=8) == candidates(data, limit=8)
    assert len({item.family for item in candidates(data, limit=8)}) == 8


def test_binding_cannot_borrow_refs_or_ignore_unknown_eligibility():
    data = inventory(scope("left", visible_entity="MET"), scope("right", visible_entity="MET"))
    left = next(c for c in candidates(data) if c.scope_id == "left")
    with pytest.raises(ExecutionError, match="another scope"):
        bind_candidates(
            (left,),
            CandidateBindings(bindings=(binding(left, evidence_refs=("right:visible_entity",)),)),
            data,
            (),
            TaskRuntimeConfig(),
        )
    uncertain = binding(
        left,
        checks=tuple(
            check.model_copy(update={"verdict": "UNKNOWN"}) for check in binding(left).checks
        ),
    )
    assert not bind_candidates(
        (left,), CandidateBindings(bindings=(uncertain,)), data, (), TaskRuntimeConfig()
    )
    with pytest.raises(ExecutionError, match="Missing or unknown"):
        bind_candidates(
            (left,),
            CandidateBindings(bindings=(binding(left, checks=()),)),
            data,
            (),
            TaskRuntimeConfig(),
        )


def test_budget_and_semantic_fingerprints_precede_selector_calls():
    data = inventory(scope("left", visible_entity="MET"))
    template = candidates(data)[0]
    over_budget = binding(template, estimated_answer_tokens=1025)
    assert not bind_candidates(
        (template,), CandidateBindings(bindings=(over_budget,)), data, (), TaskRuntimeConfig()
    )
    result = bind_candidates(
        (template,), CandidateBindings(bindings=(binding(template),)), data, (), TaskRuntimeConfig()
    )
    assert len(result) == 1
    same_request = binding(
        template,
        public_parameters=(
            PublicParameter(name="target", value=" THE  CUP ", origin="instruction"),
        ),
    )
    assert not bind_candidates(
        (template,),
        CandidateBindings(bindings=(same_request,)),
        data,
        (),
        TaskRuntimeConfig(),
        frozenset({result[0].candidate_id}),
    )
    changed = binding(
        template,
        public_parameters=(
            PublicParameter(name="target", value="the plate", origin="instruction"),
        ),
    )
    other = bind_candidates(
        (template,), CandidateBindings(bindings=(changed,)), data, (), TaskRuntimeConfig()
    )
    assert other[0].candidate_id != result[0].candidate_id


def test_factual_parameters_cannot_reference_future_history():
    data = inventory(scope("left", visible_entity="MET"))
    template = candidates(data)[0]
    proposed = binding(
        template,
        public_parameters=(
            PublicParameter(
                name="target", value="the cup", origin="history", evidence_refs=("future-answer",)
            ),
        ),
    )
    history = (
        PublicMessage(message_id="previous", turn_index=1, role="user", content="What is visible?"),
    )
    with pytest.raises(ExecutionError, match="uncommitted history"):
        bind_candidates(
            (template,), CandidateBindings(bindings=(proposed,)), data, history, TaskRuntimeConfig()
        )


def test_limitation_uses_a_local_guard_instead_of_readable_text():
    data = inventory(scope("sign", resolvable_region="MET", readable_text="UNKNOWN"))
    assert "text_transcription" not in {c.task_id for c in candidates(data)}
    settings = TaskRuntimeConfig(profiles=("limitation",))
    result = candidates(data, settings=settings)
    transcription = next(c for c in result if c.task_id == "text_transcription")
    assert transcription.profile == "limitation"
    assert transcription.verification_contracts == ("dual_visual_review",)
    assert set(operation_contract(transcription)["eligibility_checks"]) == {
        "scope_resolved",
        "profile_guard",
    }
    assert not candidates(inventory(scope("sign", readable_text="UNKNOWN")), settings=settings)


def test_false_premise_cannot_use_failure_to_find_an_object():
    settings = TaskRuntimeConfig(profiles=("false_premise",))
    assert not candidates(
        inventory(scope("left", visible_entity="MET", closed_scope="UNKNOWN")), settings=settings
    )
    assert candidates(
        inventory(scope("left", visible_entity="MET", closed_scope="MET")), settings=settings
    )


def test_model_payloads_exclude_source_mapping_and_private_evidence():
    candidate = candidates(inventory(scope("left", visible_entity="MET")))[0]
    payload = selector_candidate(candidate)
    text = json.dumps(payload)
    assert "left:visible_entity" not in text
    assert "inspiration_subsets" not in text
    assert "source_urls" not in text
    validate_stage_payload("instruction_selection", {"candidates": [payload]})
    with pytest.raises(ExecutionError, match="Forbidden fields"):
        validate_stage_payload("instruction_selection", {"candidates": [{"provenance": "labels"}]})
    with pytest.raises(ExecutionError, match="Answer-independent"):
        validate_stage_payload(
            "candidate_binding", {"candidates": [{"candidate_answer": "hidden"}]}
        )


def test_compile_exposes_admission_migration_and_versioned_run_identity(tmp_path):
    config = load_config(Path("configs/pilot.yaml"))
    compiled = compile_configuration(config)
    assert compiled["task_catalog"]["counts"]["tasks"] == 72
    assert not compiled["task_admission"]["screen_to_code"]["normal_profile_available"]
    assert "ScopedEvidenceInventory" in compiled["schemas"]
    assert len(compiled["legacy_task_migration"]) == 24
    # An old configuration-only identity cannot resume under a new catalog contract.
    from pixelogue.serialization import canonical_hash

    old_hash = canonical_hash(config.model_dump(mode="json", exclude={"tasks"}))
    with RunStore(tmp_path, "legacy", require_local_wal=False) as store:
        store.initialize_run("legacy", old_hash, "pilot")
        with pytest.raises(ExternalInputError, match="another configuration"):
            store.initialize_run("legacy", config.config_hash, "pilot")


def test_output_verdict_vocabulary_cannot_be_bound_as_a_desired_answer():
    data = inventory(scope("left", resolvable_region="MET"))
    template = next(c for c in candidates(data) if c.task_id == "visual_claim_verification")
    proposed = binding(
        template,
        public_parameters=(
            PublicParameter(name="target", value="the cup", origin="instruction"),
            PublicParameter(name="claim", value="The cup is red.", origin="instruction"),
            PublicParameter(name="verdicts", value="supported", origin="instruction"),
        ),
    )
    with pytest.raises(ExecutionError, match="Unknown public parameter"):
        bind_candidates(
            (template,), CandidateBindings(bindings=(proposed,)), data, (), TaskRuntimeConfig()
        )


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
