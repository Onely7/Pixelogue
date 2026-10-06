"""Controller-owned operation admission and model-safe task contracts."""

from __future__ import annotations

from typing import Any

from pixelogue.calibration import eligible_domains, model_calibration_lock
from pixelogue.catalog import task_catalog
from pixelogue.config import ModelConfig, TaskRuntimeConfig
from pixelogue.contracts import InstructionCandidate
from pixelogue.evaluation import normalized_identification_words
from pixelogue.serialization import canonical_hash
from pixelogue.specialist_chemistry import chemical_domain_conditions
from pixelogue.task_catalog import TaskDefinition
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


# Fixed public operation text shared by drafts, answers and judges.
BOUNDARY_OPERATION_TEXT = {
    "object_identification": "Request a visible category at supported granularity. "
    "Generic person categories are permitted; individual identity is not.",
    "attribute_lookup": "Request only the public attribute or object part of the "
    "bound subject. A neighboring subject's property does not answer this request.",
    "visible_action_relation": "Request a directly visible action or interaction. "
    "Body posture alone is an attribute; static placement is a spatial relation. "
    "Contact can support an interaction, but does not establish intent or motion.",
    "referring_object_resolution": "Resolve the unique entity satisfying the public "
    "description; do not replace this operation with naming its category.",
    "entity_count": "Count the publicly defined units in a closed scope. "
    "Unclear membership, occlusion, or an incomplete enumeration means UNKNOWN.",
    "spatial_relation": "State a relation in the public reference frame between "
    "resolved visible subjects; do not infer an action from static placement.",
}
OUTPUT_CONTRACT_TEXT = {
    "chart_extremum_ranking": "Use the public rank_mode and rank_order. Include all ties. "
    "max_min asks for maximum and minimum only; all asks for the complete ordering.",
    "document_extractive_qa": "Return the complete minimal answer-bearing span in the "
    "source language, preserving qualifiers, negation and exceptions. Only "
    "sentence-initial case and a final prose period may differ. Translation, "
    "paraphrase and extra claims are not extractive output.",
    "formula_transcription": "Use the registered literal mathematical grammar: symbols, "
    "numbers, grouping, arithmetic, equality, scripts, fractions and square roots. "
    "Whole LaTeX display delimiters are allowed. Chemical bond diagrams, arrays, "
    "matrices and unregistered commands are unsupported; choose another eligible "
    "operation rather than promising their transcription.",
    "table_structure_reconstruction": "For html_table return exactly one static table "
    "using balanced table, thead/tbody/tfoot, tr, th and td tags, with br for cell "
    "line breaks. Cells allow rowspan/colspan; th allows scope=row/col/rowgroup/colgroup. "
    "table allows numeric border/cellpadding/cellspacing. Optional style supports only "
    "border-collapse=collapse/separate, text-align=left/right/center/start/end/justify "
    "and vertical-align=top/middle/bottom. No other CSS, scripts, event handlers, "
    "external assets or tags. "
    "Preserve blank cells, header/data roles and merged-cell spans. For structured_json "
    "use table_id, rows, cols, cells, data_rows, closed; each cell has row, col, "
    "rowspan, colspan, text, kind (header or data), without private image regions. "
    "markdown_simple_only supports unmerged tables with one header row.",
}
RUNTIME_RESTRICTIONS = {
    "grounded_arithmetic": "One primitive add/subtract/multiply/divide expression, exact result, no rounding or "
    "derived percentages. Units must satisfy the primitive calculator contract.",
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
    specialists = [
        registration(name) for name in task.verification_contracts if name != "dual_visual_review"
    ]
    if len(specialists) != 1 or specialists[0] is None:
        return ()
    return eligible_domains(settings.calibration_manifest, task.id, locks, specialists[0].version)


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
        "attribute_lookup",
        "object_identification",
        "scene_categorization",
        "visible_action_relation",
    }
    scope = (
        "the selected image scene"
        if task.id == "scene_categorization"
        else "the selected image region"
        if task.id in {"attribute_lookup", "object_identification", "visible_action_relation"}
        else candidate.visible_scope
    )
    checks = {name: catalog.eligibility_checks[name] for name in task.eligibility_checks}
    if candidate.calibrated_domain is not None:
        checks["calibrated_domain_supported"] = (
            "The visible source uses the exact certified notation and format domain; "
            "unsupported or unclear conventions are UNKNOWN."
        )
        if candidate.task_id == "chemical_structure_reading":
            if conditions := chemical_domain_conditions(candidate.calibrated_domain):
                checks["calibrated_domain_supported"] += " " + conditions
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
        "view_id": candidate.view_id,
        **(
            {"scope_region": candidate.scope_region.model_dump(mode="json")}
            if candidate.scope_region is not None
            else {}
        ),
        "boundary_contract": {
            "region": "Coordinates are normalized to the exact delivered view. Keep the bound "
            "subject and scope; do not expand the region or borrow another subject's evidence.",
            "operation": BOUNDARY_OPERATION_TEXT.get(
                task.id, "Preserve the declared operation and its public conditions."
            ),
        },
        **(
            {"target_region": candidate.target_region.model_dump(mode="json")}
            if candidate.target_region is not None
            else {}
        ),
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
        "output_contract": OUTPUT_CONTRACT_TEXT.get(task.id),
        "eligibility_checks": checks,
        "do_not_infer": task.do_not_infer,
        "required_verification_contracts": list(candidate.verification_contracts),
        "calibrated_domain": candidate.calibrated_domain,
        "runtime_restrictions": RUNTIME_RESTRICTIONS.get(task.id),
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
    if task.id == "attribute_lookup":
        required.add("attribute")
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
