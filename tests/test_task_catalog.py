from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from pixelogue.catalog import load_task_catalog, task_catalog
from pixelogue.cli import app
from pixelogue.config import load_config
from pixelogue.contracts import InstructionCandidate
from pixelogue.drafting import (
    DraftParameter,
    FactKey,
    QuestionDraft,
    draft_task_contract,
    draft_to_candidate,
)
from pixelogue.errors import ExecutionError, ExternalInputError
from pixelogue.operations import compile_configuration
from pixelogue.prompts import validate_stage_payload
from pixelogue.serialization import canonical_json
from pixelogue.store import RunStore
from pixelogue.task_catalog import TaskCatalog
from pixelogue.task_docs import REGENERATE_COMMAND, render_tasks_markdown
from pixelogue.task_evidence import ImageRegion
from pixelogue.task_runtime import (
    fingerprint,
    operation_contract,
    required_parameter_names,
)

REGION = ImageRegion(left=0.0, top=0.0, right=1.0, bottom=1.0)
# Jargon that made earlier definitions hard to read for drafting models and people.
JARGON = re.compile(
    r"\b(resolved|bound|public|canvas|closed scope|grounded|declared|versioned|profile|scope_id)\b",
    re.IGNORECASE,
)
# An eligibility check that asks the question to state something needs that public choice.
CHECK_PARAMETERS = {
    "count_unit_defined": {"count_unit"},
    "relation_frame_defined": {"frame"},
    "grouping_key_defined": {"group_key"},
    "hypothetical_public": {"update"},
    "fields_bound": {"fields"},
    "precision_declared": {"precision"},
    "public_rule_input_defined": {"input_values"},
    "claim_public_and_local": {"claim"},
    "local_question_supported": {"local_question"},
    "predicates_observable": {"conditions", "condition"},
    "music_context_complete": {"bar_range"},
    "chemical_notation_resolved": {"notation"},
    "circuit_notation_resolved": {"notation"},
}


def test_catalog_counts_cover_the_specification():
    catalog = task_catalog()
    assert catalog.version == "8.0"
    assert len(catalog.tasks) == 78
    assert sum(task.status == "core" for task in catalog.tasks) == 71
    assert sum(task.status == "extension" for task in catalog.tasks) == 7
    assert len(catalog.families) == 18


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update(version="7.0"),
        lambda c: c.update(extra_field=True),
        lambda c: c["tasks"][0].update(id=c["tasks"][1]["id"]),
        lambda c: c["tasks"][0].update(required_capabilities=["imaginary_capability"]),
        lambda c: c["tasks"][0].update(eligibility_checks=["imaginary_check"]),
        lambda c: c["tasks"][0].update(verification_contracts=["imaginary_verifier"]),
        lambda c: c["tasks"][0].update(family="text_reading"),
        lambda c: c["tasks"][0].update(status="extension"),
        lambda c: c["tasks"][0].update(number=1),
        lambda c: c["tasks"][0]["parameters"].update(
            target={"kind": "text", "description": "Locator.", "required": True}
        ),
        lambda c: c["tasks"][1]["parameters"]["attribute"].update(values=["color"]),
        lambda c: c["tasks"][1]["parameters"]["attribute"].update(min_items=1),
        lambda c: c["families"]["visual_description"]["task_ids"].append("unknown_task"),
        lambda c: c["capabilities"].update(unused_capability="Never referenced."),
    ],
)
def test_catalog_rejects_malformed_versions_references_and_parameters(mutation):
    catalog = deepcopy(load_task_catalog())
    mutation(catalog)
    with pytest.raises(ValidationError):
        TaskCatalog.model_validate_json(canonical_json(catalog))


def test_merged_operations_are_gone():
    ids = {task.id for task in task_catalog().tasks}
    merged = {
        "visual_summary",
        "screen_summary",
        "set_operation",
        "text_reading_order",
        "code_transcription",
        "evidence_localization",
    }
    assert not ids & merged
    assert {"grounded_description", "select_by_conditions", "text_transcription"} <= ids


def _model_facing_texts():
    catalog = task_catalog()
    for family_id, family in catalog.families.items():
        yield f"family {family_id}", family.label
    for check_id, text in catalog.eligibility_checks.items():
        yield f"check {check_id}", text
    for contract_id, contract in catalog.verification_contracts.items():
        yield f"contract {contract_id}", contract.contract
    for task in catalog.tasks:
        yield f"{task.id} label", task.label
        yield f"{task.id} definition", task.definition
        yield f"{task.id} do_not_infer", task.do_not_infer
        if task.answer_format is not None:
            yield f"{task.id} answer_format", task.answer_format
        for name, parameter in task.parameters.items():
            yield f"{task.id} parameter {name}", parameter.description


def test_model_facing_catalog_text_is_plain_english():
    for where, text in _model_facing_texts():
        assert text.isascii(), where
        assert not JARGON.search(text), (where, JARGON.search(text))


def test_definitions_say_what_is_asked_and_answered_briefly():
    for task in task_catalog().tasks:
        assert task.definition.startswith("The user "), task.id
        assert "the answer" in task.definition, task.id
        assert len(task.definition.split()) <= 65, task.id
        assert len(re.findall(r"[.!?](?:\s|$)", task.definition)) <= 3, task.id


def test_checks_that_need_a_stated_choice_have_an_explicit_required_parameter():
    for task in task_catalog().tasks:
        required = set(required_parameter_names(task))
        for check, names in CHECK_PARAMETERS.items():
            if check in task.eligibility_checks:
                assert required & names, (task.id, check)


def test_compile_exposes_admission_and_versioned_run_identity(tmp_path):
    config = load_config(Path("configs/pilot.yaml"))
    compiled = compile_configuration(config)
    assert len(compiled["task_catalog"]["tasks"]) == 78
    assert not compiled["task_admission"]["screen_to_code"]["available"]
    assert all(
        entry["available"]
        for entry in compiled["task_admission"].values()
        if entry["status"] == "core"
    )
    assert {"ImageProfile", "QuestionDraftBatch", "QuestionGateVote"} <= set(compiled["schemas"])
    assert "ScopedEvidenceInventory" not in compiled["schemas"]
    assert "legacy_task_migration" not in compiled
    # An old configuration-only identity cannot resume under a new catalog contract.
    from pixelogue.serialization import canonical_hash

    old_hash = canonical_hash(config.model_dump(mode="json", exclude={"tasks"}))
    with RunStore(tmp_path, "legacy", require_local_wal=False) as store:
        store.initialize_run("legacy", old_hash, "pilot")
        with pytest.raises(ExternalInputError, match="another configuration"):
            store.initialize_run("legacy", config.config_hash, "pilot")


def test_unversioned_candidate_labels_remain_immutable_when_reading_old_records():
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


def test_extension_candidates_cannot_drop_their_specialized_validator():
    task = next(task for task in task_catalog().tasks if task.id == "screen_to_code")
    with pytest.raises(ValidationError, match="cannot omit or replace required verifiers"):
        InstructionCandidate(
            candidate_id="fake",
            task_id=task.id,
            family=task.family,
            visible_scope="screen",
            instruction_summary=task.definition,
            required_capabilities=task.required_capabilities,
            catalog_version="8.0",
            scope_id="screen",
            view_id="view",
            evidence_refs=("evidence",),
            verification_contracts=("dual_visual_review",),
        )


def test_generic_person_category_is_distinct_from_individual_identity():
    task = next(item for item in task_catalog().tasks if item.id == "object_identification")
    assert "generic category" in task.definition
    assert "identity" in task.do_not_infer


def _drafted(
    task_id: str, target: str, parameters: tuple[DraftParameter, ...] = ()
) -> InstructionCandidate:
    draft = QuestionDraft(
        task_id=task_id,
        question="What is the object on the left side of the image?",
        target=target,
        public_parameters=parameters,
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
        assert "related_finevision_subsets" not in text
        assert "example_question" not in text
    with pytest.raises(ExecutionError, match="Forbidden fields"):
        validate_stage_payload("question_gate", {"selected_instruction": {"provenance": "labels"}})
    with pytest.raises(ExecutionError, match="Forbidden fields"):
        validate_stage_payload("question_gate", {"selected_instruction": {"example_question": "x"}})


def test_committed_task_reference_is_generated_from_the_catalog() -> None:
    rendered = render_tasks_markdown(task_catalog())
    committed = Path("docs/tasks/TASKS.md").read_text(encoding="utf-8")
    assert committed == rendered, f"Regenerate the task reference with: {REGENERATE_COMMAND}"
    for task in task_catalog().tasks:
        assert f"#### `{task.id}`: {task.label}" in rendered


def test_compile_writes_the_task_reference_on_request(tmp_path: Path) -> None:
    reference = tmp_path / "docs" / "TASKS.md"
    result = CliRunner().invoke(
        app,
        [
            "compile",
            "--output",
            str(tmp_path / "plan.json"),
            "--tasks-markdown",
            str(reference),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["tasks_markdown"] == str(reference)
    assert reference.read_text(encoding="utf-8") == render_tasks_markdown(task_catalog())


def test_knowledge_operations_never_identify_people_and_hide_their_answer_target() -> None:
    tasks = {task.id: task for task in task_catalog().tasks}
    assert "People are never identified" in tasks["named_entity_recognition"].definition
    choices = {
        "named_entity_recognition": ("entity_kind", "landmark"),
        "style_recognition": ("style_kind", "architectural_style"),
        "map_region_identification": ("region_level", "country"),
    }
    for task_id, (name, value) in choices.items():
        assert tasks[task_id].family == "knowledge_recognition"
        assert "answer_consensus_check" in tasks[task_id].verification_contracts
        parameter = DraftParameter(name=name, value=value)
        contract = operation_contract(_drafted(task_id, "the golden gate bridge", (parameter,)))
        assert contract["scope"] == "the selected image region"
        assert all(item["name"] != "target" for item in contract["public_parameters"])
    assert "person" in tasks["named_entity_recognition"].do_not_infer
    assert any(
        "never facts about a person" in item
        for item in task_catalog().input_contract.permitted_context
    )
