"""Deterministic grounded candidate and generation-model planning."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from pixelogue.catalog import load_task_catalog
from pixelogue.config import allocate_quotas
from pixelogue.contracts import EvidenceInventory, InstructionCandidate


def instruction_candidates(
    inventory: EvidenceInventory,
    *,
    seed: int,
    turn_index: int,
    limit: int = 8,
) -> tuple[InstructionCandidate, ...]:
    """Build a bounded stable list of tasks supported by observed capabilities."""
    capabilities = set(inventory.capabilities)
    scopes = inventory.visible_scopes or ("the visible image",)
    eligible: list[InstructionCandidate] = []
    for task in load_task_catalog()["tasks"]:
        required = tuple(task["required_capabilities"])
        if not set(required).issubset(capabilities):
            continue
        scope = scopes[len(eligible) % len(scopes)]
        candidate_id = hashlib.sha256(
            f"{seed}:{inventory.image_id}:{turn_index}:{task['id']}:{scope}".encode()
        ).hexdigest()
        eligible.append(
            InstructionCandidate(
                candidate_id=candidate_id,
                task_id=task["id"],
                family=task["family"],
                visible_scope=scope,
                instruction_summary=task["definition_en"],
                required_capabilities=required,
            )
        )
    eligible.sort(
        key=lambda candidate: hashlib.sha256(
            f"{seed}:{inventory.image_id}:{turn_index}:{candidate.candidate_id}".encode()
        ).digest()
    )
    return tuple(eligible[:limit])


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
