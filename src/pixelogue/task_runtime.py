"""Controller-owned operation admission and model-safe task contracts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pixelogue.catalog import task_catalog
from pixelogue.config import TaskRuntimeConfig
from pixelogue.contracts import InstructionCandidate, PublicMessage
from pixelogue.errors import ExecutionError
from pixelogue.serialization import canonical_hash
from pixelogue.task_catalog import TaskDefinition
from pixelogue.task_evidence import CandidateBindings, ScopedEvidenceInventory

# Every entry has an actual dispatch path in the coordinator. Specialized validators are absent.
IMPLEMENTED_VERIFIERS = frozenset(
    {
        "dual_visual_review",
        "closed_set_check",
        "exact_arithmetic_check",
        "transcript_alignment",
        "evidence_binding_check",
        "ui_grounding_check",
        "panel_comparison_check",
    }
)
# The current member/count comparator and primitive arithmetic engine cannot certify these modes.
UNSUPPORTED_OPERATIONS = {
    "spatial_ordering": "ordered-member verification is unavailable",
    "set_cardinality_comparison": "cardinality comparison verification is unavailable",
    "quantified_statement_verification": "quantifier verification is unavailable",
    "grounded_hypothetical_update": "hypothetical-set verification is unavailable",
    "quantity_comparison": "typed numeric ordering verification is unavailable",
    "grounded_aggregation": "aggregate-expression verification is unavailable",
    "unit_conversion": "versioned unit-conversion rules are unavailable",
    "cross_region_consistency_check": "conditional arithmetic applicability is unavailable",
}


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
}


def admission_report() -> dict[str, dict[str, Any]]:
    """Explain catalog membership separately from currently executable normal operations."""
    return {
        task.id: {
            "status": task.status,
            "normal_profile_available": not unavailable_reasons(task),
            "blocked_reasons": list(unavailable_reasons(task)),
            "required_verification_contracts": list(task.verification_contracts),
        }
        for task in task_catalog().tasks
    }


def unavailable_reasons(task: TaskDefinition) -> tuple[str, ...]:
    """Return actual missing implementations; configuration cannot invent a validator."""
    reasons = [
        f"missing validator: {name}"
        for name in task.verification_contracts
        if name not in IMPLEMENTED_VERIFIERS
    ]
    if task.status == "validator_gated_extension":
        reasons.append("specialized extension requires explicit enablement and domain calibration")
    if task.id in UNSUPPORTED_OPERATIONS:
        reasons.append(UNSUPPORTED_OPERATIONS[task.id])
    return tuple(reasons)


def operation_contract(candidate: InstructionCandidate) -> dict[str, Any]:
    """Expose semantic constraints and public choices, never source metadata or private votes."""
    if candidate.catalog_version is None:
        return candidate.model_dump(mode="json", exclude_none=True)
    catalog = task_catalog()
    task = next(task for task in catalog.tasks if task.id == candidate.task_id)
    checks = {name: catalog.eligibility_checks[name] for name in task.eligibility_checks}
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
        "scope": candidate.visible_scope,
        "scope_id": candidate.scope_id,
        "profile": candidate.profile,
        "profile_contract": catalog.profile_contracts[candidate.profile].model_dump(
            mode="json", exclude={"eligible_task_ids"}
        ),
        "public_parameters": [
            p.model_dump(mode="json", exclude={"evidence_refs"})
            for p in candidate.public_parameters
        ],
        "parameter_contract": task.parameters,
        "eligibility_checks": checks,
        "do_not_infer": task.do_not_infer,
        "required_verification_contracts": list(candidate.verification_contracts),
        "runtime_restrictions": (
            "One primitive add/subtract/multiply/divide expression, exact result, no rounding or "
            "derived percentages. Units must satisfy the primitive calculator contract."
            if task.id == "grounded_arithmetic"
            else None
        ),
    }


def selector_candidate(candidate: InstructionCandidate) -> dict[str, Any]:
    """Keep candidate identity while withholding private evidence observations."""
    return {"candidate_id": candidate.candidate_id, **operation_contract(candidate)}


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
            "parameters": {
                p.name: normalize(p.value)
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
                raise ExecutionError(
                    "CANDIDATE_EVIDENCE_MISSING", "Required capability evidence is omitted"
                )
        expected_checks = set(operation_contract(template)["eligibility_checks"])
        if {check.check_id for check in binding.checks} != expected_checks:
            raise ExecutionError(
                "CANDIDATE_CHECKS_MISMATCH", "Missing or unknown eligibility check"
            )
        if any(check.verdict != "MET" for check in binding.checks):
            continue
        if binding.estimated_answer_tokens > settings.answer_max_tokens:
            continue
        parameters = {p.name: p for p in binding.public_parameters}
        # Every candidate needs a publicly explicit target. Other choices follow its operation.
        if "target" not in parameters:
            continue
        task = next(task for task in task_catalog().tasks if task.id == template.task_id)
        allowed = (set(task.parameters) - POLICY_PARAMETERS) | {
            "target",
            "scope",
            *GUARD_PARAMETERS.values(),
        }
        if not parameters.keys() <= allowed:
            raise ExecutionError("CANDIDATE_PARAMETER_UNKNOWN", "Unknown public parameter")
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
            required_choices = (
                {key for key, value in task.parameters.items() if isinstance(value, tuple)}
                - {"derived_forms"}
                - POLICY_PARAMETERS
            )
            required_choices |= {
                GUARD_PARAMETERS[check]
                for check in task.eligibility_checks
                if check in GUARD_PARAMETERS
            }
            if task.id == "scene_categorization":
                required_choices.add("category_set")
            if task.id == "referring_expression_generation":
                required_choices.add("target_binding")
            if not required_choices <= parameters.keys():
                continue
            if task.id == "grounded_arithmetic" and "derived_forms" in parameters:
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
