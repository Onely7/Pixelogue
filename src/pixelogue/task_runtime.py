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
# Operations whose public target would name the answer; models see only a generic scope.
PRIVATE_TARGET_TASKS = frozenset(
    {
        "attribute_lookup",
        "object_identification",
        "scene_categorization",
        "visible_action",
        "named_entity_recognition",
        "style_recognition",
        "map_region_identification",
    }
)
REGION_BOUNDARY = (
    "Coordinates are normalized to the exact delivered view. Keep the requested subject and "
    "area; do not expand the region or borrow another subject's evidence."
)


def admission_report(
    settings: TaskRuntimeConfig | None = None,
    models: ModelConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Explain catalog membership separately from currently executable operations."""
    return {
        task.id: {
            "status": task.status,
            "available": not unavailable_reasons(task, settings, models),
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
        task.status != "extension"
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
    if task.status == "extension":
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
    private_target = task.id in PRIVATE_TARGET_TASKS
    scope = (
        "the selected image scene"
        if task.id == "scene_categorization"
        else "the selected image region"
        if private_target
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
    return {
        "catalog_version": catalog.version,
        "task_id": task.id,
        "family": task.family,
        "definition": task.definition,
        "scope": scope,
        "scope_id": candidate.scope_id,
        "view_id": candidate.view_id,
        **(
            {"scope_region": candidate.scope_region.model_dump(mode="json")}
            if candidate.scope_region is not None
            else {}
        ),
        "boundary_contract": {"region": REGION_BOUNDARY},
        **(
            {"target_region": candidate.target_region.model_dump(mode="json")}
            if candidate.target_region is not None
            else {}
        ),
        "public_parameters": [
            p.model_dump(mode="json", exclude={"evidence_refs"})
            for p in candidate.public_parameters
            if not (private_target and p.name == "target")
        ],
        "parameter_contract": parameter_contract(task),
        "answer_format": task.answer_format,
        "eligibility_checks": checks,
        "do_not_infer": task.do_not_infer,
        "required_verification_contracts": list(candidate.verification_contracts),
        "calibrated_domain": candidate.calibrated_domain,
    }


def parameter_contract(task: TaskDefinition) -> dict[str, dict[str, Any]]:
    """Describe every public choice of an operation for drafters and judges."""
    return {
        name: parameter.model_dump(mode="json", exclude_none=True)
        for name, parameter in task.parameters.items()
    }


def bindable_parameter_names(task: TaskDefinition) -> tuple[str, ...]:
    """List public choices that a model may bind for one catalog task."""
    return tuple(sorted({*task.parameters, "target"}))


def required_parameter_names(task: TaskDefinition) -> tuple[str, ...]:
    """List public parameter names that every draft of this operation must bind."""
    return tuple(sorted(name for name, item in task.parameters.items() if item.required))


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
