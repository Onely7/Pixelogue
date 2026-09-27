"""Versioned registry of operation validators and their runtime requirements."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Literal

ValidatorEnvironment = Literal["core", "symbolic", "notation", "chemistry", "renderer"]


@dataclass(frozen=True)
class ValidatorRegistration:
    """An implemented dispatch path with explicit optional dependencies."""

    name: str
    version: str
    environment: ValidatorEnvironment = "core"
    dependency: str | None = None
    supported_tasks: frozenset[str] | None = None

    def supports(self, task_id: str) -> bool:
        """Report whether this dispatch implements the task's actual operation."""
        return self.supported_tasks is None or task_id in self.supported_tasks

    def environment_error(self) -> str | None:
        """Report a missing Python dependency without importing specialist code."""
        if self.dependency is not None and find_spec(self.dependency) is None:
            return f"missing {self.environment} dependency: {self.dependency}"
        return None


REGISTRATIONS = {
    entry.name: entry
    for entry in (
        ValidatorRegistration("dual_visual_review", "1"),
        ValidatorRegistration("closed_set_check", "1"),
        ValidatorRegistration("exact_arithmetic_check", "2"),
        ValidatorRegistration("transcript_alignment", "1"),
        ValidatorRegistration("evidence_binding_check", "1"),
        ValidatorRegistration("ui_grounding_check", "1"),
        ValidatorRegistration("panel_comparison_check", "1"),
        ValidatorRegistration(
            "table_structure_check",
            "1",
            supported_tasks=frozenset(
                {
                    "table_cell_lookup",
                    "table_predicate_selection",
                    "table_structure_reconstruction",
                    "table_cross_reference",
                }
            ),
        ),
        ValidatorRegistration(
            "schema_check",
            "1",
            supported_tasks=frozenset(
                {
                    "table_structure_reconstruction",
                    "chart_data_reconstruction",
                    "text_field_extraction",
                    "document_structure_reconstruction",
                }
            ),
        ),
        ValidatorRegistration(
            "chart_encoding_check",
            "1",
            supported_tasks=frozenset(
                {
                    "chart_value_lookup",
                    "chart_comparison",
                    "chart_extremum_ranking",
                    "chart_trend_summary",
                    "chart_series_relation",
                    "chart_data_reconstruction",
                }
            ),
        ),
    )
}


def registration(name: str) -> ValidatorRegistration | None:
    """Return the implemented validator, or no registration for a design-only contract."""
    return REGISTRATIONS.get(name)
