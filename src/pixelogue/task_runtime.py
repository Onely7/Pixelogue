"""Controller-owned operation admission and model-safe task contracts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pixelogue.calibration import eligible_domains, model_calibration_lock
from pixelogue.catalog import task_catalog
from pixelogue.config import ModelConfig, TaskRuntimeConfig
from pixelogue.contracts import InstructionCandidate, PublicMessage
from pixelogue.errors import ExecutionError
from pixelogue.evaluation import normalized_identification_words
from pixelogue.serialization import canonical_hash
from pixelogue.task_catalog import TaskDefinition
from pixelogue.task_evidence import CandidateBindings, ScopedEvidenceInventory
from pixelogue.task_registry import REGISTRATIONS, registration

# Keep this public view for existing catalog checks; registrations own implementation status.
IMPLEMENTED_VERIFIERS = frozenset(REGISTRATIONS)
# These describe output vocabularies or fixed policies, never a value to choose as a desired answer.
POLICY_PARAMETERS = frozenset(
    {
        "verdicts",
        "execution",
        "source_errors",
        "enabled_by_default",
        "division_by_zero",
        "numeric_area",
        "coordinate_output",
        "default_output",
        "javascript",
        "asset_policy",
        "canvas_policy",
        "dimensionless_values",
        "action_scope",
        "code_languages",
        "rule_grammar",
    }
)
GUARD_PARAMETERS = {
    "count_unit_defined": "count_unit",
    "predicates_observable": "predicate",
    "grouping_key_defined": "group_key",
    "claim_public_and_local": "claim",
    "local_question_supported": "local_question",
    "precision_declared": "precision",
    "relation_frame_defined": "frame",
    "fields_bound": "fields",
    "public_rule_input_defined": "input_values",
    "music_context_complete": "bar_range",
    "chemical_notation_resolved": "notation",
    "circuit_notation_resolved": "notation",
}


def admission_report(
    settings: TaskRuntimeConfig | None = None,
    models: ModelConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Explain catalog membership separately from currently executable normal operations."""
    return {
        task.id: {
            "status": task.status,
            "normal_profile_available": not unavailable_reasons(task, settings, models),
            "blocked_reasons": list(unavailable_reasons(task, settings, models)),
            "certified_domains": list(certified_domains(task, settings, models)),
            "required_verification_contracts": list(task.verification_contracts),
            "validators": [
                {
                    "name": name,
                    "version": entry.version if (entry := registration(name)) else None,
                    "environment": entry.environment if entry else None,
                    "dependency": entry.dependency if entry else None,
                    "environment_error": entry.environment_error() if entry else "not implemented",
                }
                for name in task.verification_contracts
            ],
        }
        for task in task_catalog().tasks
    }


def certified_domains(
    task: TaskDefinition,
    settings: TaskRuntimeConfig | None,
    models: ModelConfig | None,
) -> tuple[str, ...]:
    """Resolve extension certificates for this exact active model pair."""
    if (
        task.status != "validator_gated_extension"
        or settings is None
        or models is None
        or settings.calibration_manifest is None
        or task.id not in settings.enabled_extensions
    ):
        return ()
    locks = (
        model_calibration_lock(models.generator_a),
        model_calibration_lock(models.generator_b),
    )
    return eligible_domains(settings.calibration_manifest, task.id, locks, "1")


def unavailable_reasons(
    task: TaskDefinition,
    settings: TaskRuntimeConfig | None = None,
    models: ModelConfig | None = None,
) -> tuple[str, ...]:
    """Return actual missing implementations; configuration cannot invent a validator."""
    reasons: list[str] = []
    for name in task.verification_contracts:
        entry = registration(name)
        if entry is None:
            reasons.append(f"missing validator: {name}")
        elif not entry.supports(task.id):
            reasons.append(f"validator does not support {task.id}: {name}")
        elif error := entry.environment_error():
            reasons.append(error)
    if task.status == "validator_gated_extension":
        if settings is None or task.id not in settings.enabled_extensions:
            reasons.append("specialized extension is not enabled")
        elif not certified_domains(task, settings, models):
            reasons.append("no calibration for the active models and validator")
    return tuple(reasons)


def operation_contract(candidate: InstructionCandidate) -> dict[str, Any]:
    """Expose semantic constraints and public choices, never source metadata or private votes."""
    if candidate.catalog_version is None:
        return candidate.model_dump(mode="json", exclude_none=True)
    catalog = task_catalog()
    task = next(task for task in catalog.tasks if task.id == candidate.task_id)
    private_target_task = task.id in {
        "object_identification",
        "scene_categorization",
        "visible_action_relation",
    }
    scope = (
        "the selected image scene"
        if task.id == "scene_categorization"
        else "the selected image region"
        if task.id in {"object_identification", "visible_action_relation"}
        else candidate.visible_scope
    )
    checks = {name: catalog.eligibility_checks[name] for name in task.eligibility_checks}
    if candidate.calibrated_domain is not None:
        checks["calibrated_domain_supported"] = (
            "The visible source uses the exact certified notation and format domain; "
            "unsupported or unclear conventions are UNKNOWN."
        )
    if candidate.profile != "normal":
        checks = {
            "scope_resolved": catalog.eligibility_checks["scope_resolved"],
            "profile_guard": catalog.profile_contracts[candidate.profile].eligibility,
        }
    return {
        "catalog_version": catalog.version,
        "task_id": task.id,
        "family": task.family,
        "definition": task.definition_en,
        "scope": scope,
        "scope_id": candidate.scope_id,
        "profile": candidate.profile,
        "profile_contract": catalog.profile_contracts[candidate.profile].model_dump(
            mode="json", exclude={"eligible_task_ids"}
        ),
        "public_parameters": [
            p.model_dump(mode="json", exclude={"evidence_refs"})
            for p in candidate.public_parameters
            if not (private_target_task and p.name == "target")
        ],
        "parameter_contract": task.parameters,
        "eligibility_checks": checks,
        "do_not_infer": task.do_not_infer,
        "required_verification_contracts": list(candidate.verification_contracts),
        "calibrated_domain": candidate.calibrated_domain,
        "runtime_restrictions": (
            "One primitive add/subtract/multiply/divide expression, exact result, no rounding or "
            "derived percentages. Units must satisfy the primitive calculator contract."
            if task.id == "grounded_arithmetic"
            else None
        ),
    }


def selector_candidate(candidate: InstructionCandidate) -> dict[str, Any]:
    """Keep candidate identity while withholding private evidence observations."""
    task = next(task for task in task_catalog().tasks if task.id == candidate.task_id)
    contract = operation_contract(candidate)
    return {
        "candidate_id": candidate.candidate_id,
        "required_capabilities": list(candidate.required_capabilities),
        **contract,
        "required_check_ids": list(contract["eligibility_checks"]),
        "bindable_parameter_names": list(bindable_parameter_names(task)),
        "required_parameter_names": list(required_parameter_names(task, candidate.profile)),
    }


def bindable_parameter_names(task: TaskDefinition) -> tuple[str, ...]:
    """List public choices that a model may bind for one catalog task."""
    return tuple(
        sorted(
            (set(task.parameters) - POLICY_PARAMETERS)
            | {"target"}
            | {
                value
                for check_id, value in GUARD_PARAMETERS.items()
                if check_id in task.eligibility_checks
            }
        )
    )


def required_parameter_names(task: TaskDefinition, profile: str = "normal") -> tuple[str, ...]:
    """List public parameter names needed before a MET normal-profile binding can be used."""
    if profile != "normal":
        return ()
    required = (
        {key for key, value in task.parameters.items() if isinstance(value, tuple)}
        - {"derived_forms"}
        - POLICY_PARAMETERS
    )
    required |= {
        GUARD_PARAMETERS[check] for check in task.eligibility_checks if check in GUARD_PARAMETERS
    }
    if task.id == "scene_categorization":
        required.add("category_set")
    if task.id == "referring_expression_generation":
        required.add("target_binding")
    return tuple(sorted(required))


def fingerprint(candidate: InstructionCandidate, image_id: str) -> str:
    """Identify the operation and public choices independently of turn and question wording."""

    def normalize(value: Any) -> Any:
        if isinstance(value, str):
            return " ".join(value.split()).casefold()
        if isinstance(value, tuple):
            return [normalize(item) for item in value]
        return value

    return canonical_hash(
        {
            "version": candidate.catalog_version,
            "image": image_id,
            "scope": candidate.scope_id,
            "task": candidate.task_id,
            "profile": candidate.profile,
            "calibrated_domain": candidate.calibrated_domain,
            "parameters": {
                p.name: (
                    " ".join(normalized_identification_words(p.value))
                    if candidate.task_id == "object_identification"
                    and p.name == "target"
                    and isinstance(p.value, str)
                    else normalize(p.value)
                )
                for p in sorted(candidate.public_parameters, key=lambda item: item.name)
            },
        }
    )


def validate_evidence(
    inventory: ScopedEvidenceInventory,
    view_id: str,
    settings: TaskRuntimeConfig,
) -> None:
    """Reject unknown capabilities, mismatched views, and exceeded operational bounds."""
    vocabulary = task_catalog().capabilities
    if len(inventory.scopes) > settings.max_scopes:
        raise ExecutionError("EVIDENCE_SCOPE_LIMIT", "Too many evidence scopes")
    for scope in inventory.scopes:
        if scope.view_id != view_id:
            raise ExecutionError("EVIDENCE_VIEW_MISMATCH", "Evidence uses a different image view")
        if len(scope.observations) > settings.max_observations_per_scope:
            raise ExecutionError("EVIDENCE_OBSERVATION_LIMIT", "Too many scope observations")
        if any(item.capability not in vocabulary for item in scope.observations):
            raise ExecutionError("EVIDENCE_CAPABILITY_UNKNOWN", "Unknown evidence capability")


def bind_candidates(
    templates: Sequence[InstructionCandidate],
    response: CandidateBindings,
    inventory: ScopedEvidenceInventory,
    history: Sequence[PublicMessage],
    settings: TaskRuntimeConfig,
    used_fingerprints: frozenset[str] = frozenset(),
) -> tuple[InstructionCandidate, ...]:
    """Admit only complete local bindings within the reviewed answer budget.

    A model's missing or UNKNOWN check never becomes an affirmative capability. Unsupported
    parameter combinations are omitted before spending selector or answer calls.
    """
    by_id = {item.candidate_id: item for item in templates}
    scopes = {scope.scope_id: scope for scope in inventory.scopes}
    seen: set[str] = set()
    accepted: list[InstructionCandidate] = []
    for binding in response.bindings:
        if binding.candidate_id not in by_id or binding.candidate_id in seen:
            raise ExecutionError("CANDIDATE_BINDING_ID", "Unknown or repeated candidate binding")
        seen.add(binding.candidate_id)
        template = by_id[binding.candidate_id]
        if template.scope_id is None:
            raise ExecutionError("CANDIDATE_SCOPE_MISSING", "Candidate has no bound scope")
        scope = scopes[template.scope_id]
        local_refs = {item.evidence_id for item in scope.observations}
        if not set(binding.evidence_refs) <= local_refs:
            raise ExecutionError(
                "CANDIDATE_EVIDENCE_SCOPE", "Binding borrows evidence from another scope"
            )
        if template.profile == "normal":
            required_refs = {
                item.evidence_id
                for item in scope.observations
                if item.capability in template.required_capabilities and item.verdict == "MET"
            }
            if not required_refs <= set(binding.evidence_refs):
                continue
        expected_checks = set(operation_contract(template)["eligibility_checks"])
        supplied_checks = {check.check_id for check in binding.checks}
        if supplied_checks != expected_checks:
            raise ExecutionError(
                "CANDIDATE_CHECKS_MISMATCH",
                f"Candidate {binding.candidate_id}: missing check IDs "
                f"{sorted(expected_checks - supplied_checks)}; unknown check IDs "
                f"{sorted(supplied_checks - expected_checks)}; required check IDs "
                f"{sorted(expected_checks)}",
            )
        if any(check.verdict != "MET" for check in binding.checks):
            continue
        if binding.estimated_answer_tokens > settings.answer_max_tokens:
            continue
        parameters = {p.name: p for p in binding.public_parameters}
        # Every candidate needs a publicly explicit target. Other choices follow its operation.
        if "target" not in parameters:
            raise ExecutionError(
                "CANDIDATE_PARAMETER_MISSING",
                f"Candidate {binding.candidate_id}: missing separate target binding",
            )
        task = next(task for task in task_catalog().tasks if task.id == template.task_id)
        allowed = set(bindable_parameter_names(task))
        if not parameters.keys() <= allowed:
            raise ExecutionError(
                "CANDIDATE_PARAMETER_UNKNOWN",
                f"Candidate {binding.candidate_id}: unknown public parameter names "
                f"{sorted(parameters.keys() - allowed)}; allowed names {sorted(allowed)}",
            )
        history_ids = {message.message_id for message in history}
        for parameter in binding.public_parameters:
            refs = set(parameter.evidence_refs)
            if parameter.origin == "image" and not refs <= local_refs:
                raise ExecutionError(
                    "CANDIDATE_PARAMETER_SOURCE", "Parameter uses nonlocal image evidence"
                )
            if parameter.origin == "history" and not refs <= history_ids:
                raise ExecutionError(
                    "CANDIDATE_PARAMETER_SOURCE", "Parameter uses uncommitted history"
                )
            choices = task.parameters.get(parameter.name)
            if isinstance(choices, tuple) and parameter.value not in choices:
                raise ExecutionError("CANDIDATE_PARAMETER_VALUE", "Unsupported parameter choice")
        if template.profile == "normal":
            required_choices = set(required_parameter_names(task))
            if missing := required_choices - parameters.keys():
                raise ExecutionError(
                    "CANDIDATE_PARAMETER_MISSING",
                    f"Candidate {binding.candidate_id}: missing public parameter names "
                    f"{sorted(missing)}; required names {sorted(required_choices)}",
                )
            if task.id == "scene_categorization":
                choices = parameters["category_set"]
                if (
                    choices.origin != "instruction"
                    or not isinstance(choices.value, tuple)
                    or len(choices.value) < 2
                    or len({value.casefold() for value in choices.value}) != len(choices.value)
                    or parameters["target"].origin != "image"
                    or parameters["target"].value not in choices.value
                ):
                    raise ExecutionError(
                        "CANDIDATE_PARAMETER_VALUE",
                        "Scene categories require at least two distinct public alternatives "
                        "including the exact image-origin target label with local evidence refs",
                    )
            if task.id == "ui_action_specification":
                if parameters["action"].value == "input" and "input_text" not in parameters:
                    continue
                if parameters["action"].value != "input" and "input_text" in parameters:
                    continue
            if task.id == "grounded_arithmetic" and "derived_forms" in parameters:
                continue
            if (
                task.id == "document_structure_reconstruction"
                and parameters["format"].value != "structured_json"
            ):
                continue
        candidate = template.model_copy(
            update={
                "public_parameters": binding.public_parameters,
                "evidence_refs": binding.evidence_refs,
            }
        )
        identity = fingerprint(candidate, inventory.image_id)
        if identity not in used_fingerprints:
            accepted.append(candidate.model_copy(update={"candidate_id": identity}))
    return tuple(accepted)
