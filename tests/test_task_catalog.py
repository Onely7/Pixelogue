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
    CapabilityReport,
    EligibilityObservation,
    ImageRegion,
    ObservationVerdict,
    PublicParameter,
    ScopedEvidenceInventory,
    ScopedEvidenceReport,
    ScopeEvidence,
    ScopeEvidenceReport,
    alias_evidence_ids,
)
from pixelogue.task_runtime import (
    admission_report,
    bind_candidates,
    bind_candidates_individually,
    bindable_parameter_names,
    binding_candidate,
    fingerprint,
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


def test_later_turn_prefers_unused_tasks_within_candidate_limit() -> None:
    observations: dict[str, ObservationVerdict] = {
        key: "MET" for key in task_catalog().capabilities
    }
    data = inventory(scope("all", **observations))
    first = instruction_candidates(data, seed=13, turn_index=1, limit=8)
    used = frozenset(candidate.task_id for candidate in first)
    later = instruction_candidates(data, seed=13, turn_index=2, limit=8, used_task_ids=used)
    assert len(later) == 8
    assert not used & {candidate.task_id for candidate in later}


def test_binding_cannot_borrow_refs_or_ignore_unknown_eligibility():
    data = inventory(scope("left", visible_entity="MET"), scope("right", visible_entity="MET"))
    left = next(c for c in candidates(data) if c.scope_id == "left")
    with pytest.raises(ExecutionError, match="another scope") as caught:
        bind_candidates(
            (left,),
            CandidateBindings(bindings=(binding(left, evidence_refs=("right:visible_entity",)),)),
            data,
            (),
            TaskRuntimeConfig(),
        )
    assert "right:visible_entity" in str(caught.value)
    assert "left:visible_entity" in str(caught.value)
    uncertain = binding(
        left,
        checks=tuple(
            check.model_copy(update={"verdict": "UNKNOWN"}) for check in binding(left).checks
        ),
    )
    assert not bind_candidates(
        (left,), CandidateBindings(bindings=(uncertain,)), data, (), TaskRuntimeConfig()
    )
    with pytest.raises(ExecutionError, match=f"Candidate {left.candidate_id}: missing check IDs"):
        bind_candidates(
            (left,),
            CandidateBindings(bindings=(binding(left, checks=()),)),
            data,
            (),
            TaskRuntimeConfig(),
        )


def test_binding_reports_exact_unknown_parameter_without_admitting_it() -> None:
    data = inventory(scope("left", visible_entity="MET"))
    template = candidates(data)[0]
    proposed = binding(
        template,
        public_parameters=(
            PublicParameter(name="target", value="cup", origin="instruction"),
            PublicParameter(name="scope_resolved", value="MET", origin="instruction"),
        ),
    )
    with pytest.raises(ExecutionError) as caught:
        bind_candidates(
            (template,), CandidateBindings(bindings=(proposed,)), data, (), TaskRuntimeConfig()
        )
    assert caught.value.reason == "CANDIDATE_PARAMETER_UNKNOWN"
    assert "scope_resolved" in str(caught.value)
    assert "allowed names" in str(caught.value)


def test_candidate_input_separates_check_ids_from_parameter_names() -> None:
    data = inventory(
        scope("left", visible_entity="MET", visible_attribute="UNKNOWN"),
        scope("right", visible_entity="MET"),
    )
    template = next(item for item in candidates(data) if item.scope_id == "left")
    exposed = selector_candidate(template)
    bound_input = binding_candidate(template, data)
    assert bound_input["local_evidence_ids"] == ["left:visible_attribute", "left:visible_entity"]
    assert set(bound_input["required_evidence_ids"]) <= set(bound_input["local_evidence_ids"])
    assert bound_input["required_evidence_ids"] == ["left:visible_entity"]
    assert set(exposed["required_check_ids"]) == set(exposed["eligibility_checks"])
    assert "target" in exposed["bindable_parameter_names"]
    assert not set(exposed["required_check_ids"]) & set(exposed["bindable_parameter_names"])
    assert set(exposed["required_parameter_names"]) <= set(exposed["bindable_parameter_names"])
    assert "scope" not in exposed["bindable_parameter_names"]
    description = next(task for task in task_catalog().tasks if task.id == "grounded_description")
    assert "scope" in bindable_parameter_names(description)


def test_compact_evidence_ids_preserve_scope_and_reject_old_references() -> None:
    original = inventory(
        scope("left", visible_entity="MET"),
        scope("right", visible_attribute="MET"),
    )
    compact, mapping = alias_evidence_ids(original)
    assert mapping == {"left:visible_entity": "e1", "right:visible_attribute": "e2"}
    assert original.scopes[0].observations[0].evidence_id == "left:visible_entity"
    assert compact.scopes[0].observations[0].detail == original.scopes[0].observations[0].detail
    assert compact.scopes[1].observations[0].evidence_id == "e2"
    ScopedEvidenceInventory.model_validate(compact.model_dump())
    left = next(item for item in candidates(compact) if item.scope_id == "left")
    assert binding_candidate(left, compact)["local_evidence_ids"] == ["e1"]
    with pytest.raises(ExecutionError) as caught:
        bind_candidates(
            (left,),
            CandidateBindings(bindings=(binding(left, evidence_refs=("left:visible_entity",)),)),
            compact,
            (),
            TaskRuntimeConfig(),
        )
    assert caught.value.reason == "CANDIDATE_EVIDENCE_SCOPE"


def test_bad_identification_binding_does_not_hide_valid_sibling() -> None:
    """A local source error excludes its entry while a grounded sibling survives."""
    regions = (scope("left", visible_entity="MET"), scope("right", visible_entity="MET"))
    named_scopes = tuple(
        region.model_copy(
            update={
                "observations": (
                    region.observations[0].model_copy(update={"detail": f"A {name} is visible"}),
                )
            }
        )
        for region, name in zip(regions, ("horse", "dog"), strict=True)
    )
    data = inventory(*named_scopes)
    options = candidates(data, limit=64)
    left = next(
        item
        for item in options
        if item.task_id == "object_identification" and item.scope_id == "left"
    )
    right = next(
        item
        for item in options
        if item.task_id == "object_identification" and item.scope_id == "right"
    )
    valid = binding(
        left,
        public_parameters=(
            PublicParameter(
                name="target",
                value="horse",
                origin="image",
                evidence_refs=("left:visible_entity",),
            ),
        ),
    )
    invalid = binding(
        right,
        public_parameters=(
            PublicParameter(
                name="target",
                value="mallet",
                origin="image",
                evidence_refs=("right:visible_entity",),
            ),
        ),
    )
    result = bind_candidates_individually(
        (left, right), CandidateBindings(bindings=(valid, invalid)), data, (), TaskRuntimeConfig()
    )
    assert len(result.admitted) == 1
    assert result.admitted[0].scope_id == "left"
    assert [(item.candidate_id, item.reason) for item in result.rejected] == [
        (right.candidate_id, "CANDIDATE_PARAMETER_SOURCE")
    ]

    with pytest.raises(ExecutionError, match="Unknown or repeated candidate binding"):
        bind_candidates_individually(
            (left, right),
            CandidateBindings(bindings=(valid, valid)),
            data,
            (),
            TaskRuntimeConfig(),
        )
    with pytest.raises(ExecutionError, match="Unknown or repeated candidate binding"):
        bind_candidates_individually(
            (left, right),
            CandidateBindings(
                bindings=(valid, invalid.model_copy(update={"candidate_id": "unknown"}))
            ),
            data,
            (),
            TaskRuntimeConfig(),
        )


def test_met_binding_missing_required_public_choice_is_retryable_contract_error() -> None:
    data = inventory(scope("left", scene_context="MET"))
    template = next(item for item in candidates(data) if item.task_id == "scene_categorization")
    exposed = selector_candidate(template)
    assert exposed["required_parameter_names"] == ["category_set"]
    with pytest.raises(ExecutionError) as caught:
        bind_candidates(
            (template,),
            CandidateBindings(bindings=(binding(template),)),
            data,
            (),
            TaskRuntimeConfig(),
        )
    assert caught.value.reason == "CANDIDATE_PARAMETER_MISSING"
    assert "category_set" in str(caught.value)


def test_attribute_binding_names_distinct_public_properties_across_turns() -> None:
    data = inventory(scope("dog", visible_entity="MET", visible_attribute="MET"))
    template = next(item for item in candidates(data) if item.task_id == "attribute_lookup")
    assert selector_candidate(template)["required_parameter_names"] == ["attribute"]
    with pytest.raises(ExecutionError) as missing:
        bind_candidates(
            (template,),
            CandidateBindings(bindings=(binding(template),)),
            data,
            (),
            TaskRuntimeConfig(),
        )
    assert missing.value.reason == "CANDIDATE_PARAMETER_MISSING"

    target = PublicParameter(name="target", value="the dog", origin="instruction")

    def bind_property(property_name: str, used: frozenset[str] = frozenset()):
        proposal = binding(
            template,
            public_parameters=(
                target,
                PublicParameter(name="attribute", value=property_name, origin="instruction"),
            ),
        )
        return bind_candidates(
            (template,),
            CandidateBindings(bindings=(proposal,)),
            data,
            (),
            TaskRuntimeConfig(),
            used,
        )

    fur = bind_property("fur color")[0]
    assert not bind_property("fur color", frozenset({fur.candidate_id}))
    nose = bind_property("nose color", frozenset({fur.candidate_id}))[0]
    assert nose.candidate_id != fur.candidate_id


def test_scene_categories_require_distinct_public_choices() -> None:
    data = inventory(scope("left", scene_context="MET"))
    template = next(item for item in candidates(data) if item.task_id == "scene_categorization")
    target = PublicParameter(
        name="target",
        value="residential street",
        origin="image",
        evidence_refs=("left:scene_context",),
    )
    singleton = PublicParameter(
        name="category_set",
        value="residential street",
        origin="image",
        evidence_refs=("left:scene_context",),
    )
    with pytest.raises(ExecutionError) as caught:
        bind_candidates(
            (template,),
            CandidateBindings(bindings=(binding(template, public_parameters=(target, singleton)),)),
            data,
            (),
            TaskRuntimeConfig(),
        )
    assert caught.value.reason == "CANDIDATE_PARAMETER_VALUE"

    alternatives = PublicParameter(
        name="category_set",
        value=("residential street", "rural road"),
        origin="instruction",
    )
    admitted = bind_candidates(
        (template,),
        CandidateBindings(bindings=(binding(template, public_parameters=(target, alternatives)),)),
        data,
        (),
        TaskRuntimeConfig(),
    )
    assert len(admitted) == 1

    generic_target = PublicParameter(
        name="target", value="residential street", origin="instruction"
    )
    with pytest.raises(ExecutionError) as uncited:
        bind_candidates(
            (template,),
            CandidateBindings(
                bindings=(
                    binding(
                        template,
                        public_parameters=(generic_target, alternatives),
                    ),
                )
            ),
            data,
            (),
            TaskRuntimeConfig(),
        )
    assert uncited.value.reason == "CANDIDATE_PARAMETER_VALUE"


def test_answer_labels_are_hidden_from_model_operation_contract() -> None:
    data = inventory(
        scope(
            "left",
            visible_entity="MET",
            visible_attribute="MET",
            scene_context="MET",
            visible_interaction="MET",
        )
    )
    for task_id, label, visible_scope in (
        ("attribute_lookup", "black", "A black primate sitting on a railing."),
        ("object_identification", "gibbon", "A gibbon sitting on a metal railing."),
        ("scene_categorization", "collapsed bridge", "A collapsed bridge scene."),
        ("visible_action_relation", "sitting", "A black primate sitting on a railing."),
    ):
        template = next(item for item in candidates(data) if item.task_id == task_id)
        candidate = template.model_copy(
            update={
                "visible_scope": visible_scope,
                "public_parameters": (
                    PublicParameter(
                        name="target",
                        value=label,
                        origin="image",
                        evidence_refs=(f"left:{template.required_capabilities[0]}",),
                    ),
                ),
            }
        )
        contract = operation_contract(candidate)
        assert label not in json.dumps(contract).lower()
        assert all(parameter["name"] != "target" for parameter in contract["public_parameters"])
        assert label not in json.dumps(selector_candidate(candidate)).lower()
        assert contract["scope"] != visible_scope


def test_bound_target_region_uses_only_target_evidence_without_disclosing_answer() -> None:
    left = ImageRegion(left=0.1, top=0.2, right=0.4, bottom=0.8)
    right = ImageRegion(left=0.6, top=0.2, right=0.9, bottom=0.8)
    data = inventory(
        ScopeEvidence(
            scope_id="pair",
            view_id="view",
            public_description="Two nearby objects",
            region=REGION,
            observations=(
                CapabilityObservation(
                    evidence_id="left",
                    capability="visible_entity",
                    verdict="MET",
                    region=left,
                    detail="Several brown horses are visible",
                ),
                CapabilityObservation(
                    evidence_id="right",
                    capability="scene_context",
                    verdict="MET",
                    region=right,
                    detail="Adjacent visible object",
                ),
            ),
        )
    )
    template = next(item for item in candidates(data) if item.task_id == "object_identification")
    proposal = binding(
        template,
        public_parameters=(
            PublicParameter(name="target", value="horse", origin="image", evidence_refs=("left",)),
        ),
        evidence_refs=("left", "right"),
    )
    (bound,) = bind_candidates(
        (template,), CandidateBindings(bindings=(proposal,)), data, (), TaskRuntimeConfig()
    )
    assert bound.target_region == left
    assert InstructionCandidate.model_validate(bound.model_dump()).target_region == left
    contract = operation_contract(bound)
    assert contract["target_region"] == left.model_dump()
    assert selector_candidate(bound)["target_region"] == left.model_dump()
    assert "horse" not in json.dumps(contract).lower()
    assert "horse" not in json.dumps(selector_candidate(bound)).lower()

    unsupported = binding(
        template,
        public_parameters=(
            PublicParameter(name="target", value="mallet", origin="image", evidence_refs=("left",)),
        ),
    )
    with pytest.raises(ExecutionError) as caught:
        bind_candidates(
            (template,), CandidateBindings(bindings=(unsupported,)), data, (), TaskRuntimeConfig()
        )
    assert caught.value.reason == "CANDIDATE_PARAMETER_SOURCE"
    assert "mallet" in str(caught.value)


def test_identification_reference_resolves_only_a_labeled_local_entity() -> None:
    data = inventory(
        ScopeEvidence(
            scope_id="headwear",
            view_id="view",
            public_description="A wrapped head covering",
            object_label="turban",
            region=REGION,
            observations=(
                CapabilityObservation(
                    evidence_id="headwear:entity",
                    capability="visible_entity",
                    verdict="MET",
                    region=REGION,
                    detail="A wrapped head covering is visible.",
                ),
            ),
        )
    )
    template = next(item for item in candidates(data) if item.task_id == "object_identification")
    proposal = binding(
        template,
        public_parameters=(
            PublicParameter(
                name="target",
                value="ref:headwear:entity",
                origin="image",
                evidence_refs=("headwear:entity",),
            ),
        ),
    )
    (bound,) = bind_candidates(
        (template,), CandidateBindings(bindings=(proposal,)), data, (), TaskRuntimeConfig()
    )
    assert bound.public_parameters[0].value == "turban"
    assert "turban" not in json.dumps(selector_candidate(bound)).lower()

    for changed in (
        proposal.model_copy(
            update={
                "public_parameters": (
                    proposal.public_parameters[0].model_copy(update={"value": "ref:unknown"}),
                )
            }
        ),
        proposal.model_copy(
            update={
                "public_parameters": (
                    proposal.public_parameters[0].model_copy(
                        update={"evidence_refs": ("unknown",)}
                    ),
                )
            }
        ),
    ):
        with pytest.raises(ExecutionError) as caught:
            bind_candidates(
                (template,), CandidateBindings(bindings=(changed,)), data, (), TaskRuntimeConfig()
            )
        assert caught.value.reason == "CANDIDATE_PARAMETER_SOURCE"

    unlabeled = data.model_copy(
        update={"scopes": (data.scopes[0].model_copy(update={"object_label": None}),)}
    )
    with pytest.raises(ExecutionError) as caught:
        bind_candidates(
            (template,), CandidateBindings(bindings=(proposal,)), unlabeled, (), TaskRuntimeConfig()
        )
    assert caught.value.reason == "CANDIDATE_PARAMETER_SOURCE"


def test_object_label_requires_a_visible_met_entity() -> None:
    report = ScopedEvidenceReport(
        image_id="image",
        reason="A plausible label without a visible entity is insufficient.",
        scopes=(
            ScopeEvidenceReport(
                scope_id="scope",
                view_id="view",
                public_description="unknown image content",
                object_label="turban",
                region=REGION,
                observations={
                    "visible_entity": CapabilityReport(
                        evidence_id="entity",
                        verdict="UNKNOWN",
                        region=REGION,
                        detail="The shape is not readable.",
                    )
                },
            ),
        ),
    )
    with pytest.raises(ExecutionError) as caught:
        report.to_inventory()
    assert caught.value.reason == "MODEL_SCHEMA_MISMATCH"


def test_identification_fingerprint_collapses_label_format_and_plural() -> None:
    data = inventory(scope("left", visible_entity="MET"))
    template = next(item for item in candidates(data) if item.task_id == "object_identification")
    identities = set()
    for label in ("qr_code", "QR code", "Three QR codes"):
        candidate = template.model_copy(
            update={
                "public_parameters": (
                    PublicParameter(
                        name="target",
                        value=label,
                        origin="image",
                        evidence_refs=("left:visible_entity",),
                    ),
                ),
            }
        )
        identities.add(fingerprint(candidate, data.image_id))
    assert len(identities) == 1


def test_missing_required_evidence_rejects_only_that_candidate() -> None:
    data = inventory(scope("left", visible_entity="MET", visible_attribute="MET"))
    available = candidates(data)
    object_candidate = next(item for item in available if item.task_id == "object_identification")
    attribute_candidate = next(item for item in available if item.task_id == "attribute_lookup")
    incomplete = binding(attribute_candidate, evidence_refs=("left:visible_attribute",))

    admitted = bind_candidates(
        (object_candidate, attribute_candidate),
        CandidateBindings(bindings=(binding(object_candidate), incomplete)),
        data,
        (),
        TaskRuntimeConfig(),
    )

    assert len(admitted) == 1
    assert admitted[0].task_id == "object_identification"


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
    with pytest.raises(ExecutionError, match="unknown public parameter"):
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
