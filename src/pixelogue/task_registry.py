"""Versioned registry of operation validators and their runtime requirements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pixelogue.specialist_env import environment_error

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
        if self.environment != "core":
            return environment_error(self.environment, self.dependency)
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
        ValidatorRegistration(
            "formula_structure_check",
            "1",
            supported_tasks=frozenset({"formula_transcription"}),
        ),
        ValidatorRegistration(
            "graph_check",
            "1",
            supported_tasks=frozenset(
                {
                    "diagram_element_lookup",
                    "graph_connectivity",
                    "graph_path_tracing",
                    "diagram_process_description",
                    "diagram_branch_evaluation",
                }
            ),
        ),
        ValidatorRegistration(
            "scale_check",
            "1",
            supported_tasks=frozenset({"measurement_reading"}),
        ),
        ValidatorRegistration(
            "pattern_check",
            "1",
            supported_tasks=frozenset(
                {
                    "pattern_rule_identification",
                    "pattern_completion",
                    "rule_based_exception",
                }
            ),
        ),
        ValidatorRegistration(
            "geometry_check",
            "1",
            supported_tasks=frozenset({"geometric_relation_analysis"}),
        ),
        ValidatorRegistration(
            "formal_geometry_validator",
            "3",
            "symbolic",
            "sympy",
            frozenset({"geometric_constraint_solving"}),
        ),
        ValidatorRegistration(
            "music_notation_validator",
            "2",
            "notation",
            "music21",
            frozenset({"music_notation_reading"}),
        ),
        ValidatorRegistration(
            "chemical_graph_validator",
            "4",
            "chemistry",
            "rdkit",
            frozenset({"chemical_structure_reading"}),
        ),
        ValidatorRegistration(
            "circuit_graph_validator",
            "2",
            supported_tasks=frozenset({"circuit_structure_reading"}),
        ),
        ValidatorRegistration(
            "ui_action_validator",
            "6",
            supported_tasks=frozenset({"ui_action_specification"}),
        ),
        ValidatorRegistration(
            "sandbox_render_validator",
            "2",
            "renderer",
            "playwright",
            frozenset({"diagram_to_code", "screen_to_code"}),
        ),
    )
}


def registration(name: str) -> ValidatorRegistration | None:
    """Return the implemented validator, or no registration for a design-only contract."""
    return REGISTRATIONS.get(name)
