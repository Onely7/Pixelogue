"""CP-SAT selection and independent integer audit."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import Field

from pixelogue.config import StrictModel
from pixelogue.contracts import SelectionCandidate, SelectionManifest, SourcePurpose
from pixelogue.errors import AuditError, ExternalInputError
from pixelogue.serialization import canonical_hash, canonical_json


class SelectionPolicy(StrictModel):
    """Integer constraints applied to one frozen quality pool."""

    target_count: int = Field(ge=1)
    language_quotas: dict[str, int]
    family_minimums: dict[str, int] = {}
    max_per_image: int = Field(default=3, ge=1)
    max_per_visual_group: int = Field(default=6, ge=1)
    max_per_semantic_family: int = Field(default=1, ge=1)
    allow_evaluation_sources: bool = False
    time_limit_seconds: float = Field(default=120, gt=0)
    seed: int = 20260915


class AuditReport(StrictModel):
    """Counts recomputed from selected candidates without solver auxiliaries."""

    status: Literal["MET", "NOT_MET"]
    selected_count: int
    language_counts: dict[str, int]
    family_counts: dict[str, int]
    image_maximum: int
    visual_group_maximum: int
    violations: tuple[str, ...]
    audit_hash: str


def freeze_pool(candidates: Sequence[SelectionCandidate]) -> tuple[list[SelectionCandidate], str]:
    """Return a stable unique quality pool and its immutable identity."""
    by_id: dict[str, SelectionCandidate] = {}
    for candidate in candidates:
        if candidate.conversation_id in by_id:
            raise ExternalInputError("DUPLICATE_POOL_ID", candidate.conversation_id)
        by_id[candidate.conversation_id] = candidate
    ordered = [by_id[candidate_id] for candidate_id in sorted(by_id)]
    pool_hash = canonical_hash([candidate.model_dump(mode="json") for candidate in ordered])
    return ordered, pool_hash


def select_candidates(
    candidates: Sequence[SelectionCandidate],
    policy: SelectionPolicy,
) -> SelectionManifest:
    """Find a deterministic feasible subset with OR-Tools CP-SAT.

    Raises:
        ExternalInputError: If OR-Tools is not installed or the pool is structurally invalid.
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError as error:
        raise ExternalInputError(
            "ORTOOLS_MISSING",
            "Install the CPU extra with `uv sync --extra cpu`",
        ) from error
    pool, pool_hash = freeze_pool(candidates)
    if not policy.allow_evaluation_sources:
        pool = [candidate for candidate in pool if candidate.purpose is SourcePurpose.TRAINING]
    model = cp_model.CpModel()
    variables = [model.new_bool_var(f"selected_{index}") for index in range(len(pool))]
    model.add(sum(variables) == policy.target_count)
    _add_exact_constraints(model, variables, pool, "language", policy.language_quotas)
    _add_minimum_constraints(model, variables, pool, "task_family", policy.family_minimums)
    _add_maximum_constraints(model, variables, pool, "image_id", policy.max_per_image)
    _add_maximum_constraints(
        model,
        variables,
        pool,
        "visual_group_id",
        policy.max_per_visual_group,
    )
    _add_maximum_constraints(
        model,
        variables,
        pool,
        "semantic_family",
        policy.max_per_semantic_family,
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = policy.time_limit_seconds
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = policy.seed
    status = solver.solve(model)
    solver_status: Literal["FEASIBLE", "OPTIMAL", "INFEASIBLE", "UNKNOWN"]
    if status == cp_model.OPTIMAL:
        solver_status = "OPTIMAL"
    elif status == cp_model.FEASIBLE:
        solver_status = "FEASIBLE"
    elif status == cp_model.INFEASIBLE:
        solver_status = "INFEASIBLE"
    else:
        solver_status = "UNKNOWN"
    selected_ids: tuple[str, ...] = ()
    if solver_status in {"OPTIMAL", "FEASIBLE"}:
        selected_ids = tuple(
            candidate.conversation_id
            for candidate, variable in zip(pool, variables, strict=True)
            if solver.value(variable)
        )
    return SelectionManifest(
        pool_hash=pool_hash,
        selected_ids=selected_ids,
        solver_status=solver_status,
    )


def _group_indices(candidates: Sequence[SelectionCandidate], field: str) -> dict[str, list[int]]:
    output: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        output[str(getattr(candidate, field))].append(index)
    return output


def _add_exact_constraints(
    model: Any,
    variables: Sequence[Any],
    candidates: Sequence[SelectionCandidate],
    field: str,
    values: Mapping[str, int],
) -> None:
    groups = _group_indices(candidates, field)
    for key, expected in values.items():
        model.add(sum(variables[index] for index in groups.get(key, [])) == expected)


def _add_minimum_constraints(
    model: Any,
    variables: Sequence[Any],
    candidates: Sequence[SelectionCandidate],
    field: str,
    values: Mapping[str, int],
) -> None:
    groups = _group_indices(candidates, field)
    for key, expected in values.items():
        model.add(sum(variables[index] for index in groups.get(key, [])) >= expected)


def _add_maximum_constraints(
    model: Any,
    variables: Sequence[Any],
    candidates: Sequence[SelectionCandidate],
    field: str,
    maximum: int,
) -> None:
    for indices in _group_indices(candidates, field).values():
        model.add(sum(variables[index] for index in indices) <= maximum)


def audit_selection(
    candidates: Sequence[SelectionCandidate],
    manifest: SelectionManifest,
    policy: SelectionPolicy,
) -> AuditReport:
    """Recompute every implemented selection condition from selected IDs."""
    pool, pool_hash = freeze_pool(candidates)
    violations: list[str] = []
    if pool_hash != manifest.pool_hash:
        violations.append("POOL_HASH_MISMATCH")
    if len(manifest.selected_ids) != len(set(manifest.selected_ids)):
        violations.append("DUPLICATE_SELECTED_ID")
    by_id = {candidate.conversation_id: candidate for candidate in pool}
    missing = sorted(set(manifest.selected_ids) - set(by_id))
    if missing:
        violations.append("SELECTED_ID_OUTSIDE_POOL")
    selected = [by_id[selected_id] for selected_id in manifest.selected_ids if selected_id in by_id]
    language_counts = Counter(candidate.language for candidate in selected)
    family_counts = Counter(candidate.task_family for candidate in selected)
    image_counts = Counter(candidate.image_id for candidate in selected)
    group_counts = Counter(candidate.visual_group_id for candidate in selected)
    semantic_counts = Counter(candidate.semantic_family for candidate in selected)
    if len(selected) != policy.target_count:
        violations.append("TARGET_COUNT_MISMATCH")
    if dict(language_counts) != policy.language_quotas:
        violations.append("LANGUAGE_QUOTA_MISMATCH")
    if any(family_counts[family] < minimum for family, minimum in policy.family_minimums.items()):
        violations.append("FAMILY_MINIMUM_MISMATCH")
    if image_counts and max(image_counts.values()) > policy.max_per_image:
        violations.append("IMAGE_MAXIMUM_EXCEEDED")
    if group_counts and max(group_counts.values()) > policy.max_per_visual_group:
        violations.append("VISUAL_GROUP_MAXIMUM_EXCEEDED")
    if semantic_counts and max(semantic_counts.values()) > policy.max_per_semantic_family:
        violations.append("SEMANTIC_FAMILY_MAXIMUM_EXCEEDED")
    if not policy.allow_evaluation_sources and any(
        candidate.purpose is SourcePurpose.EVALUATION for candidate in selected
    ):
        violations.append("EVALUATION_SOURCE_SELECTED")
    selected_count = len(selected)
    sorted_language_counts = dict(sorted(language_counts.items()))
    sorted_family_counts = dict(sorted(family_counts.items()))
    image_maximum = max(image_counts.values(), default=0)
    visual_group_maximum = max(group_counts.values(), default=0)
    sorted_violations = tuple(sorted(violations))
    report_body = {
        "pool_hash": pool_hash,
        "selected_ids": manifest.selected_ids,
        "policy": policy.model_dump(mode="json"),
        "selected_count": selected_count,
        "language_counts": sorted_language_counts,
        "family_counts": sorted_family_counts,
        "image_maximum": image_maximum,
        "visual_group_maximum": visual_group_maximum,
        "violations": sorted_violations,
    }
    return AuditReport(
        status="MET" if not violations else "NOT_MET",
        selected_count=selected_count,
        language_counts=sorted_language_counts,
        family_counts=sorted_family_counts,
        image_maximum=image_maximum,
        visual_group_maximum=visual_group_maximum,
        violations=sorted_violations,
        audit_hash=canonical_hash(report_body),
    )


def bind_audit(manifest: SelectionManifest, report: AuditReport) -> SelectionManifest:
    """Bind a successful independent audit to a selection manifest.

    Raises:
        AuditError: If any selection condition failed.
    """
    if report.status != "MET":
        raise AuditError("SELECTION_AUDIT_FAILED", ", ".join(report.violations))
    selection = manifest.model_dump(
        mode="json",
        exclude={"audit_hash", "audit_report_hash"},
    )
    binding_hash = canonical_hash({"selection": selection, "audit_report_hash": report.audit_hash})
    return SelectionManifest.model_validate_json(
        canonical_json(
            {
                **selection,
                "audit_report_hash": report.audit_hash,
                "audit_hash": binding_hash,
            }
        )
    )
