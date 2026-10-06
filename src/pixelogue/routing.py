"""Image profiles and corpus-level task-family routing for direct question drafting.

The router model returns only a coarse image profile and per-family feasibility. The
controller chooses the family with the largest deficit against the configured target
distribution among feasible families, preferring families not yet used in the
conversation. Ties use a seeded hash, so a route is reproducible from the same ledger.
"""

from __future__ import annotations

import hashlib
import threading
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from pixelogue.catalog import task_catalog
from pixelogue.config import StrictModel
from pixelogue.task_catalog import TaskDefinition

# Used when the profile is unusable: broadly applicable families, most general first.
FALLBACK_FAMILIES = (
    "visual_description",
    "text_reading",
    "reference_spatial",
    "set_logic",
    "evidence_verification",
)
PRIMARY_TASKS = 4
SECONDARY_TASKS = 2
# Operations verified only by the two blind reviews, evidence binding or transcript alignment.
# Structured extraction verifiers (sets, tables, charts, graphs, scales, geometry, formulas)
# abstain far more often, so early turns that decide the two-turn minimum avoid them.
LIGHT_CONTRACTS = frozenset(
    {"dual_visual_review", "evidence_binding_check", "transcript_alignment"}
)


def lightly_verified(task: TaskDefinition) -> bool:
    """Return whether an operation needs no structured extraction verifier."""
    return set(task.verification_contracts) <= LIGHT_CONTRACTS


class ImageProfile(StrictModel):
    """Coarse router output; it never contains a question, answer or category label set."""

    image_kind: Literal[
        "photo",
        "document",
        "table",
        "chart",
        "diagram",
        "map",
        "screen",
        "math",
        "illustration",
        "mixed",
        "other",
    ]
    readable_text: Literal["none", "some", "dense"]
    supported_families: Annotated[
        tuple[Annotated[str, Field(min_length=1, max_length=64)], ...], Field(max_length=16)
    ]
    reason: Annotated[str, Field(min_length=1, max_length=160)]

    @model_validator(mode="after")
    def validate_families(self) -> ImageProfile:
        """Reject repeated family names."""
        if len(self.supported_families) != len(set(self.supported_families)):
            raise ValueError("Each family may be listed once")
        return self


@dataclass(frozen=True)
class Route:
    """The families and preferred operations offered to one turn's drafter."""

    primary_family: str
    primary_task_ids: tuple[str, ...]
    secondary_family: str | None
    secondary_task_ids: tuple[str, ...]
    basis: Literal["profile", "fallback"]

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Return offered operations in drafting priority order."""
        return self.primary_task_ids + self.secondary_task_ids

    def to_json(self) -> dict[str, Any]:
        """Serialize for durable route records."""
        value = asdict(self)
        value["primary_task_ids"] = list(self.primary_task_ids)
        value["secondary_task_ids"] = list(self.secondary_task_ids)
        return value

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Route:
        """Restore a saved route exactly."""
        return cls(
            primary_family=value["primary_family"],
            primary_task_ids=tuple(value["primary_task_ids"]),
            secondary_family=value["secondary_family"],
            secondary_task_ids=tuple(value["secondary_task_ids"]),
            basis=value["basis"],
        )


class FamilyLedger:
    """Thread-safe counts of committed turns by family and task in this run."""

    def __init__(self, task_ids: Iterable[str] = ()) -> None:
        """Start from already committed task IDs, for example after a resume."""
        self._lock = threading.Lock()
        self._tasks: Counter[str] = Counter()
        self._families: Counter[str] = Counter()
        families = {task.id: task.family for task in task_catalog().tasks}
        self._family_of = families
        for task_id in task_ids:
            self.record(task_id)

    def record(self, task_id: str) -> None:
        """Count one committed turn."""
        with self._lock:
            self._tasks[task_id] += 1
            self._families[self._family_of[task_id]] += 1

    def snapshot(self) -> tuple[dict[str, int], dict[str, int]]:
        """Return copies of the family and task counts."""
        with self._lock:
            return dict(self._families), dict(self._tasks)


def family_definitions(tasks: Mapping[str, TaskDefinition]) -> list[dict[str, Any]]:
    """Describe each family that has at least one available operation, for the router."""
    catalog = task_catalog()
    rows = []
    for family, definition in catalog.families.items():
        members = [task for task in tasks.values() if task.family == family]
        if members:
            rows.append(
                {
                    "family": family,
                    "label": definition.label_en,
                    "operations": [task.label_en for task in members],
                }
            )
    return rows


def _rank(seed: int, image_id: str, turn_index: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{image_id}:{turn_index}:{value}".encode()).digest()


def choose_route(
    profile: ImageProfile | None,
    ledger: FamilyLedger,
    *,
    tasks: Mapping[str, TaskDefinition],
    family_targets: Literal["uniform"] | Mapping[str, float],
    task_weights: Mapping[str, float],
    used_families: frozenset[str],
    used_task_ids: frozenset[str],
    seed: int,
    image_id: str,
    turn_index: int,
    light_only: bool = False,
) -> Route | None:
    """Choose primary and secondary families and their preferred operations.

    ``light_only`` restricts the offer to lightly verified operations, for the turns that
    decide whether a conversation reaches its minimum length. Returns ``None`` only when no
    available family exists at all.
    """
    if light_only:
        tasks = {task_id: task for task_id, task in tasks.items() if lightly_verified(task)}
    available = {task.family for task in tasks.values()}
    feasible = set(profile.supported_families) & available if profile is not None else set()
    if profile is not None and profile.image_kind != "screen":
        # Screen operations need an actual interface; annotated figures were routed here.
        feasible.discard("screen_ui")
    basis: Literal["profile", "fallback"] = "profile"
    if not feasible:
        feasible = set(FALLBACK_FAMILIES) & available
        basis = "fallback"
    if not feasible:
        return None
    families, task_counts = ledger.snapshot()
    total = sum(families.values())
    if family_targets == "uniform":
        targets = {family: 1.0 / len(available) for family in available}
    else:
        weight = sum(value for key, value in family_targets.items() if key in available)
        targets = {
            family: family_targets.get(family, 0.0) / weight if weight else 0.0
            for family in available
        }

    def deficit(family: str) -> float:
        share = families.get(family, 0) / total if total else 0.0
        return targets.get(family, 0.0) - share

    fresh = feasible - used_families
    pool = fresh or feasible
    ordered = sorted(
        pool,
        key=lambda family: (-deficit(family), _rank(seed, image_id, turn_index, family)),
    )
    remaining = sorted(
        feasible - {ordered[0]},
        key=lambda family: (
            family in used_families,
            -deficit(family),
            _rank(seed, image_id, turn_index, family),
        ),
    )

    def operations(family: str, limit: int) -> tuple[str, ...]:
        members = [task for task in tasks.values() if task.family == family]
        members.sort(
            key=lambda task: (
                task.id in used_task_ids,
                task_counts.get(task.id, 0) / task_weights.get(task.id, 1.0),
                _rank(seed, image_id, turn_index, task.id),
            )
        )
        return tuple(task.id for task in members[:limit])

    secondary = remaining[0] if remaining else None
    return Route(
        primary_family=ordered[0],
        primary_task_ids=operations(ordered[0], PRIMARY_TASKS),
        secondary_family=secondary,
        secondary_task_ids=operations(secondary, SECONDARY_TASKS) if secondary else (),
        basis=basis,
    )
