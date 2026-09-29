"""Per-task implementation, CPU evidence, environment and calibration inventory."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from pixelogue.catalog import task_catalog
from pixelogue.chart_verifiers import CHART_TASKS
from pixelogue.config import PixelogueConfig
from pixelogue.contracts import ConversationArtifact
from pixelogue.document_verifiers import DOCUMENT_TASKS
from pixelogue.finite_verifiers import FINITE_TASKS
from pixelogue.graph_verifiers import GRAPH_TASKS
from pixelogue.io import read_jsonl, write_json
from pixelogue.pattern_verifiers import PATTERN_TASKS
from pixelogue.quantitative_verifiers import QUANTITATIVE_TASKS
from pixelogue.table_verifiers import TABLE_TASKS
from pixelogue.task_registry import registration
from pixelogue.task_runtime import admission_report

SPECIALIST_TESTS = {
    "geometric_constraint_solving": "tests/test_specialist_geometry.py",
    "diagram_to_code": "tests/test_specialist_render.py",
    "screen_to_code": "tests/test_specialist_render.py",
    "music_notation_reading": "tests/test_specialist_music.py",
    "chemical_structure_reading": "tests/test_specialist_chemistry.py",
    "circuit_structure_reading": "tests/test_specialist_circuit.py",
    "ui_action_specification": "tests/test_specialist_ui.py",
}

# Named operation cases are distinct from a passing shared test module.
TASK_BOUNDARY_TESTS = {
    "spatial_ordering": "test_order_requires_all_visible_members_in_requested_sequence",
    "set_cardinality_comparison": "test_cardinality_comparison_uses_closed_group_counts",
    "quantified_statement_verification": "test_quantifier_checks_public_choice_and_threshold",
    "grounded_hypothetical_update": "test_hypothetical_add_remove_and_relabel_follow_public_update",
    "cross_region_consistency_check": "test_comparison_and_cross_region_consistency_require_comparable_units",
    "quantity_comparison": "test_quantity_comparison_and_consistency_reject_wrong_relation_and_unreadable_source",
    "grounded_aggregation": "test_closed_aggregation_and_half_up_rounding",
    "unit_conversion": "test_weighted_mean_and_versioned_conversion",
    "text_field_extraction": "test_requested_field_values_missing_status_and_schema",
    "formula_transcription": "test_stacked_fraction_is_not_slash_or_algebraic_equivalence",
    "document_structure_reconstruction": "test_document_tree_preserves_content_hierarchy_and_order",
    "table_cell_lookup": "test_lookup_checks_cell_position_and_blank_value",
    "table_predicate_selection": "test_selection_checks_closed_rows_and_numeric_predicate",
    "table_structure_reconstruction": "test_reconstruction_separates_schema_from_content_and_preserves_spans",
    "table_cross_reference": "test_join_requires_unique_keys_and_complete_visible_tables",
    "chart_value_lookup": "test_exact_lookup_and_coarse_interval_do_not_invent_precision",
    "chart_comparison": "test_comparison_abstains_for_overlapping_intervals",
    "chart_extremum_ranking": "test_complete_ranking_preserves_ties_and_trend_order",
    "chart_trend_summary": "test_complete_ranking_preserves_ties_and_trend_order",
    "chart_series_relation": "test_series_crossing_requires_line_encoding",
    "chart_data_reconstruction": "test_reconstruction_schema_and_content_are_independent",
    "measurement_reading": "test_linear_interpolation_rounds_only_to_public_resolution",
    "diagram_element_lookup": "test_element_lookup_and_directed_neighbors",
    "graph_connectivity": "test_element_lookup_and_directed_neighbors",
    "graph_path_tracing": "test_all_paths_include_every_visible_alternative",
    "diagram_process_description": "test_process_description_checks_all_explicit_edges",
    "diagram_branch_evaluation": "test_public_numeric_branch_uses_printed_conditions",
    "geometric_relation_analysis": "test_approximate_parallelism_cannot_prove_exact_relation",
    "pattern_rule_identification": "test_unique_progression_rule_and_competing_rules_abstain",
    "pattern_completion": "test_completion_requires_one_rule_and_one_visible_option",
    "rule_based_exception": "test_exception_search_preserves_original_indices",
    "geometric_constraint_solving": "test_right_triangle_rule_derives_unique_positive_length",
    "diagram_to_code": "test_svg_and_html_reject_executable_syntax",
    "screen_to_code": "test_svg_and_html_reject_executable_syntax",
    "music_notation_reading": "test_music21_score_accepts_exact_pitches_durations_and_rests",
    "chemical_structure_reading": "test_rdkit_matches_graph_and_rejects_wrong_connectivity",
    "circuit_structure_reading": "test_parallel_pair_requires_both_shared_nets",
    "ui_action_specification": "test_click_point_checks_both_target_bounds_and_view_pixel_transform",
}


def _test_module(task_id: str) -> str:
    if task_id in SPECIALIST_TESTS:
        return SPECIALIST_TESTS[task_id]
    for tasks, path in (
        (FINITE_TASKS, "tests/test_finite_verifiers.py"),
        (QUANTITATIVE_TASKS, "tests/test_quantitative_verifiers.py"),
        (TABLE_TASKS, "tests/test_table_verifiers.py"),
        (CHART_TASKS, "tests/test_chart_verifiers.py"),
        (DOCUMENT_TASKS, "tests/test_document_verifiers.py"),
        (GRAPH_TASKS, "tests/test_graph_verifiers.py"),
        (PATTERN_TASKS, "tests/test_pattern_verifiers.py"),
    ):
        if task_id in tasks:
            return path
    return {
        "formula_transcription": "tests/test_formula_verifier.py",
        "geometric_relation_analysis": "tests/test_geometry_verifier.py",
        "measurement_reading": "tests/test_scale_verifier.py",
    }.get(task_id, "tests/test_task_verification.py")


def _junit_cases(path: Path | None) -> dict[str, dict[str, int]]:
    if path is None:
        return {}
    root = ElementTree.parse(path).getroot()
    counts: dict[str, dict[str, int]] = {}
    for case in root.iter("testcase"):
        class_name = case.get("classname", "")
        module = "tests/" + class_name.rsplit(".", 1)[-1] + ".py"
        values = counts.setdefault(module, {"passed": 0, "failed": 0, "skipped": 0})
        if case.find("failure") is not None or case.find("error") is not None:
            values["failed"] += 1
        elif case.find("skipped") is not None:
            values["skipped"] += 1
        else:
            values["passed"] += 1
    return counts


def _passed_test_names(path: Path | None) -> dict[str, set[str]]:
    """Index only successful named cases for task-specific CPU evidence."""
    if path is None:
        return {}
    root = ElementTree.parse(path).getroot()
    passed: dict[str, set[str]] = {}
    for case in root.iter("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            continue
        if case.find("skipped") is not None:
            continue
        module = "tests/" + case.get("classname", "").rsplit(".", 1)[-1] + ".py"
        passed.setdefault(module, set()).add(case.get("name", "").split("[", 1)[0])
    return passed


def task_status_report(
    config: PixelogueConfig,
    junit_path: Path | None = None,
    conversations_path: Path | None = None,
) -> dict[str, Any]:
    """Join actual admission with scoped test evidence without inferring image accuracy."""
    admissions = admission_report(config.tasks, config.models)
    tests = _junit_cases(junit_path)
    passed_names = _passed_test_names(junit_path)
    gpu_cases: dict[str, list[dict[str, Any]]] = {}
    if conversations_path is not None:
        for conversation in read_jsonl(conversations_path, ConversationArtifact):
            for turn in conversation.turns:
                gpu_cases.setdefault(turn.instruction.task_id, []).append(
                    {
                        "image_id": conversation.image.image_id,
                        "source_id": conversation.image.source_id,
                        "conversation_id": conversation.conversation_id,
                        "turn_index": turn.turn_index,
                        "status": turn.status,
                        "generator_model": turn.generation_model,
                    }
                )
    rows: list[dict[str, Any]] = []
    for task in task_catalog().tasks:
        validators = [registration(name) for name in task.verification_contracts]
        implemented = all(entry is not None and entry.supports(task.id) for entry in validators)
        environment_errors = sorted(
            {
                error
                for entry in validators
                if entry is not None
                if (error := entry.environment_error()) is not None
            }
        )
        path = _test_module(task.id)
        boundary_test = TASK_BOUNDARY_TESTS.get(task.id)
        boundary_status = (
            f"passed: {boundary_test}"
            if boundary_test is not None and boundary_test in passed_names.get(path, set())
            else "not_recorded"
        )
        test_counts = tests.get(path)
        if test_counts is None:
            cpu_status = "not_recorded"
        elif test_counts["failed"]:
            cpu_status = "shared_contract_tests_failed"
        elif test_counts["passed"]:
            cpu_status = "shared_contract_tests_passed"
        else:
            cpu_status = "only_skipped"
        extension = task.status == "validator_gated_extension"
        domains = admissions[task.id]["certified_domains"]
        rows.append(
            {
                "task_id": task.id,
                "number": task.number,
                "family": task.family,
                "status": task.status,
                "supported_parameters": task.parameters,
                "implemented": implemented,
                "validator_versions": {
                    entry.name: entry.version for entry in validators if entry is not None
                },
                "environment_ready": not environment_errors,
                "environment_errors": environment_errors,
                "calibration": "not_required"
                if not extension
                else ("certified" if domains else "pending"),
                "certified_domains": domains,
                "normal_selectable": admissions[task.id]["normal_profile_available"],
                "blocked_reasons": admissions[task.id]["blocked_reasons"],
                "cpu_test_module": path,
                "cpu_contract_status": cpu_status,
                "cpu_test_counts": test_counts,
                "cpu_task_specific_boundaries": boundary_status,
                "gpu_turn_cases": gpu_cases.get(task.id, []),
                "gpu_turn_case_count": len(gpu_cases.get(task.id, [])),
                "gpu_committed_turn_case_count": sum(
                    case["status"] == "COMMITTED" for case in gpu_cases.get(task.id, [])
                ),
            }
        )
    return {
        "config_hash": config.config_hash,
        "junit_path": str(junit_path) if junit_path is not None else None,
        "conversations_path": str(conversations_path) if conversations_path is not None else None,
        "summary": {
            "tasks": len(rows),
            "implemented": sum(row["implemented"] for row in rows),
            "normal_selectable": sum(row["normal_selectable"] for row in rows),
            "environment_ready": sum(row["environment_ready"] for row in rows),
            "cpu_shared_contract_passed": sum(
                row["cpu_contract_status"] == "shared_contract_tests_passed" for row in rows
            ),
            "cpu_named_boundary_passed": sum(
                row["cpu_task_specific_boundaries"] != "not_recorded" for row in rows
            ),
            "gpu_turn_cases": sum(row["gpu_turn_case_count"] for row in rows),
            "gpu_committed_turn_cases": sum(row["gpu_committed_turn_case_count"] for row in rows),
        },
        "tasks": rows,
        "cpu_evidence_limit": "Passing shared fixtures does not establish natural-image accuracy or all three verdict branches per task.",
        "gpu_evidence_limit": "These are attempted answer turns only. Candidate-only attempts and judge model identities are not inferred from public conversation records.",
    }


def write_task_status_reports(report: dict[str, Any], output_stem: Path) -> None:
    """Write the same per-task state as JSON, CSV and Markdown."""
    write_json(output_stem.with_suffix(".json"), report)
    with output_stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "number",
                "task_id",
                "implemented",
                "cpu_contract_status",
                "cpu_task_specific_boundaries",
                "gpu_turn_case_count",
                "gpu_committed_turn_case_count",
                "gpu_turn_cases",
                "environment_ready",
                "calibration",
                "normal_selectable",
                "blocked_reasons",
            )
        )
        for item in report["tasks"]:
            writer.writerow(
                (
                    item["number"],
                    item["task_id"],
                    item["implemented"],
                    item["cpu_contract_status"],
                    item["cpu_task_specific_boundaries"],
                    item["gpu_turn_case_count"],
                    item["gpu_committed_turn_case_count"],
                    json.dumps(item["gpu_turn_cases"], ensure_ascii=False),
                    item["environment_ready"],
                    item["calibration"],
                    item["normal_selectable"],
                    "; ".join(item["blocked_reasons"]),
                )
            )
    lines = [
        "# Pixelogue task status",
        "",
        f"Tasks: {report['summary']['tasks']}; normal selectable: {report['summary']['normal_selectable']}.",
        "",
        "| # | Task | Implemented | CPU contract | GPU committed / attempted turns | Environment | Calibration | Selectable |",
        "|---:|---|---|---|---:|---|---|---|",
    ]
    for item in report["tasks"]:
        lines.append(
            f"| {item['number']} | {item['task_id']} | {item['implemented']} | "
            f"{item['cpu_contract_status']} | {item['gpu_committed_turn_case_count']} / "
            f"{item['gpu_turn_case_count']} | "
            f"{item['environment_ready']} | "
            f"{item['calibration']} | {item['normal_selectable']} |"
        )
    lines += ["", report["cpu_evidence_limit"], report["gpu_evidence_limit"], ""]
    output_stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
