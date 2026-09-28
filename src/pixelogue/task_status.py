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


def task_status_report(
    config: PixelogueConfig,
    junit_path: Path | None = None,
    conversations_path: Path | None = None,
) -> dict[str, Any]:
    """Join actual admission with scoped test evidence without inferring image accuracy."""
    admissions = admission_report(config.tasks, config.models)
    tests = _junit_cases(junit_path)
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
                "cpu_task_specific_boundaries": "not_recorded",
                "gpu_turn_cases": gpu_cases.get(task.id, []),
                "gpu_turn_case_count": len(gpu_cases.get(task.id, [])),
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
            "gpu_turn_cases": sum(row["gpu_turn_case_count"] for row in rows),
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
        "| # | Task | Implemented | CPU contract | GPU turns | Environment | Calibration | Selectable |",
        "|---:|---|---|---|---:|---|---|---|",
    ]
    for item in report["tasks"]:
        lines.append(
            f"| {item['number']} | {item['task_id']} | {item['implemented']} | "
            f"{item['cpu_contract_status']} | {item['gpu_turn_case_count']} | "
            f"{item['environment_ready']} | "
            f"{item['calibration']} | {item['normal_selectable']} |"
        )
    lines += ["", report["cpu_evidence_limit"], report["gpu_evidence_limit"], ""]
    output_stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
