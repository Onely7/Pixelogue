"""Deterministic grounded candidate and generation-model planning."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from pixelogue.catalog import task_catalog
from pixelogue.config import ModelConfig, TaskRuntimeConfig, allocate_quotas
from pixelogue.contracts import InstructionCandidate
from pixelogue.task_evidence import ScopedEvidenceInventory
from pixelogue.task_runtime import certified_domains, fingerprint, unavailable_reasons


def instruction_candidates(
    inventory: ScopedEvidenceInventory,
    *,
    seed: int,
    turn_index: int,
    limit: int = 8,
    settings: TaskRuntimeConfig | None = None,
    models: ModelConfig | None = None,
    used_task_ids: frozenset[str] = frozenset(),
) -> tuple[InstructionCandidate, ...]:
    """Route local evidence into bounded templates, rotating eligible families.

    These templates still require focused parameter/eligibility binding before selection. Missing
    capabilities are UNKNOWN; evidence from different scopes is never combined implicitly.
    """
    if limit < 1:
        raise ValueError("Candidate limit must be positive")
    settings = settings or TaskRuntimeConfig()
    catalog = task_catalog()
    families: dict[str, list[InstructionCandidate]] = {}
    for scope in inventory.scopes:
        supported = {item.capability for item in scope.observations if item.verdict == "MET"}
        missing = {item.capability for item in scope.observations if item.verdict != "MET"}
        for task in catalog.tasks:
            if (
                task.status == "validator_gated_extension"
                and task.id not in settings.enabled_extensions
            ):
                continue
            for profile in settings.profiles:
                if profile == "normal":
                    if (
                        unavailable_reasons(task, settings, models)
                        or not set(task.required_capabilities) <= supported
                    ):
                        continue
                    verifiers = task.verification_contracts
                else:
                    if task.id not in catalog.profile_contracts[profile].eligible_task_ids:
                        continue
                    if not ({"resolvable_region", "visible_entity"} & supported):
                        continue
                    if profile == "limitation" and not (set(task.required_capabilities) & missing):
                        continue
                    if profile == "false_premise" and "closed_scope" not in supported:
                        continue
                    verifiers = ("dual_visual_review",)
                refs = tuple(item.evidence_id for item in scope.observations)
                candidate = InstructionCandidate(
                    candidate_id="unbound",
                    task_id=task.id,
                    family=task.family,
                    profile=profile,
                    visible_scope=scope.public_description,
                    instruction_summary=task.definition_en,
                    required_capabilities=task.required_capabilities,
                    catalog_version=catalog.version,
                    scope_id=scope.scope_id,
                    view_id=scope.view_id,
                    scope_region=scope.region,
                    evidence_refs=refs,
                    verification_contracts=verifiers,
                    calibrated_domain=(
                        certified_domains(task, settings, models)[0]
                        if task.status == "validator_gated_extension"
                        else None
                    ),
                )
                candidate = candidate.model_copy(
                    update={"candidate_id": fingerprint(candidate, inventory.image_id)}
                )
                families.setdefault(task.family, []).append(candidate)

    def rank(value: str) -> bytes:
        return hashlib.sha256(f"{seed}:{inventory.image_id}:{value}".encode()).digest()

    family_order = sorted(families, key=rank)
    if not family_order:
        return ()
    offset = (turn_index - 1) % len(family_order)
    family_order = family_order[offset:] + family_order[:offset]
    for candidates in families.values():
        candidates.sort(key=lambda c: (c.task_id in used_task_ids, rank(c.candidate_id)))
    ranked: list[InstructionCandidate] = []
    depth = 0
    while True:
        round_candidates = [
            families[name][depth] for name in family_order if len(families[name]) > depth
        ]
        if not round_candidates:
            break
        ranked.extend(round_candidates)
        depth += 1
    fresh = [candidate for candidate in ranked if candidate.task_id not in used_task_ids]
    used = [candidate for candidate in ranked if candidate.task_id in used_task_ids]
    if turn_index > 1 and len(fresh) >= limit and limit > 1:
        attribute_retry = next(
            (candidate for candidate in used if candidate.task_id == "attribute_lookup"), None
        )
        if attribute_retry is not None:
            return tuple((*fresh[: limit - 1], attribute_retry))
    return tuple((fresh + used)[:limit])


def allocated_role(
    conversation_id: str,
    allocation: Mapping[str, int],
) -> str:
    """Choose one weighted role deterministically for an entire conversation."""
    total = sum(allocation.values())
    if total <= 0 or any(weight <= 0 for weight in allocation.values()):
        raise ValueError("generation allocation weights must be positive")
    position = int.from_bytes(hashlib.sha256(conversation_id.encode()).digest()[:8], "big") % total
    cursor = 0
    for role, weight in sorted(allocation.items()):
        cursor += weight
        if position < cursor:
            return role
    raise AssertionError("weighted allocation failed")


def exact_schedule(
    total: int,
    weights: Mapping[str, int | str],
    *,
    seed: int,
    namespace: str,
) -> tuple[str, ...]:
    """Build a deterministic schedule whose counts exactly match rational quotas."""
    quotas = allocate_quotas(total, weights)
    ranked = [
        (
            hashlib.sha256(f"{seed}:{namespace}:{value}:{occurrence}".encode()).digest(),
            value,
        )
        for value in sorted(quotas)
        for occurrence in range(quotas[value])
    ]
    return tuple(value for _, value in sorted(ranked))


def planned_turn_count(image_id: str, seed: int, minimum: int = 2, maximum: int = 6) -> int:
    """Choose a stable 2-to-6 turn count with reference weights 4/4/1/1/1."""
    if (minimum, maximum) != (2, 6):
        raise ValueError("Pixelogue currently supports exactly 2 through 6 turns")
    choices: Sequence[int] = (2, 2, 2, 2, 3, 3, 3, 3, 4, 5, 6)
    position = int.from_bytes(
        hashlib.sha256(f"{seed}:{image_id}:turn-count".encode()).digest()[:8], "big"
    ) % len(choices)
    return choices[position]
