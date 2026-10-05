"""Applicable operation validators shared by holistic and detailed evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from pixelogue.catalog import task_catalog
from pixelogue.chart_verifiers import CHART_TASKS, ChartAnswer, ChartSource, verify_chart
from pixelogue.contracts import GateVerdict, InstructionCandidate
from pixelogue.document_verifiers import DOCUMENT_TASKS, DocumentSource, verify_document
from pixelogue.evaluation import consensus
from pixelogue.finite_verifiers import FINITE_TASKS, FiniteAnswer, FiniteSource, verify_finite
from pixelogue.formula_verifier import FormulaSource, verify_formula
from pixelogue.geometry_verifier import GeometryAnswer, GeometrySource, verify_geometry
from pixelogue.graph_verifiers import GRAPH_TASKS, GraphAnswer, GraphSource, verify_graph
from pixelogue.pattern_verifiers import PATTERN_TASKS, PatternAnswer, PatternSource, verify_pattern
from pixelogue.quantitative_verifiers import (
    QUANTITATIVE_TASKS,
    QuantityAnswer,
    QuantitySource,
    verify_quantitative,
)
from pixelogue.rules import (
    ComputationInventory,
    SetInventory,
    verify_computation_inventories,
    verify_set_inventories,
)
from pixelogue.scale_verifier import ScaleAnswer, ScaleSource, verify_scale
from pixelogue.specialist_chemistry import ChemicalSource, verify_chemistry
from pixelogue.specialist_circuit import CircuitSource, verify_circuit
from pixelogue.specialist_geometry import (
    GeometryNumericAnswer,
    GeometryProblem,
    verify_geometry_problem,
)
from pixelogue.specialist_music import MusicSource, verify_music
from pixelogue.specialist_render import RenderSource, verify_render
from pixelogue.specialist_ui import UIActionSource, verify_ui_action
from pixelogue.table_lookup import TableLookupSource, verify_table_lookup
from pixelogue.table_verifiers import TABLE_TASKS, TableAnswer, TableSource, verify_table
from pixelogue.task_evidence import (
    TranscriptInventory,
    TranscriptSource,
    VisualContractReview,
)
from pixelogue.task_runtime import operation_contract
from pixelogue.transcription_verifier import verify_extractive, verify_transcription

InvokeJudge = Callable[[str, dict[str, Any], type[BaseModel], int], BaseModel]


@dataclass(frozen=True)
class OperationCheck:
    """Controller result plus private independent extraction records."""

    name: str
    verdict: GateVerdict
    evidence: tuple[dict[str, Any], ...]
    reason: str


def verify_operation(
    instruction: InstructionCandidate,
    payload: dict[str, Any],
    invoke: InvokeJudge,
    reference_image: Path | None = None,
) -> tuple[OperationCheck, ...]:
    """Execute every declared supplement; no model controls applicability.

    Caller supplies the same allowed public inputs independently to each judge. Uncertain or
    inconsistent extraction cannot certify a turn. Generated action is never executed.
    """
    results: list[OperationCheck] = []
    public = {**payload, "expected_operation": operation_contract(instruction)}
    if instruction.task_id in {"diagram_to_code", "screen_to_code"}:
        source_payload = {key: value for key, value in public.items() if key != "candidate_answer"}
        sources = tuple(
            RenderSource.model_validate(
                invoke("specialist_render_source", source_payload, RenderSource, index)
            )
            for index in range(2)
        )
        assert instruction.scope_id is not None
        assert instruction.view_id is not None
        assert instruction.calibrated_domain is not None
        format_name = next(
            (item.value for item in instruction.public_parameters if item.name == "format"), None
        )
        verdict, calculation = verify_render(
            (sources[0], sources[1]),
            instruction.calibrated_domain,
            format_name,
            instruction.scope_id,
            instruction.view_id,
            payload.get("image_views"),
            reference_image,
            payload["candidate_answer"],
        )
        results.append(
            OperationCheck(
                "sandbox_render_validator",
                verdict,
                (*tuple(item.model_dump(mode="json") for item in sources), calculation),
                "Isolated code rendering check: " + verdict.value,
            )
        )
    if instruction.task_id == "ui_action_specification":
        source_payload = {key: value for key, value in public.items() if key != "candidate_answer"}
        sources = tuple(
            UIActionSource.model_validate(
                invoke("specialist_ui_source", source_payload, UIActionSource, index)
            )
            for index in range(2)
        )
        assert instruction.scope_id is not None
        assert instruction.view_id is not None
        assert instruction.calibrated_domain is not None
        parameters = {item.name: item.value for item in instruction.public_parameters}
        verdict, calculation = verify_ui_action(
            (sources[0], sources[1]),
            instruction.calibrated_domain,
            parameters,
            instruction.scope_id,
            instruction.view_id,
            payload.get("image_views"),
            payload["candidate_answer"],
        )
        results.append(
            OperationCheck(
                "ui_action_validator",
                verdict,
                (*tuple(item.model_dump(mode="json") for item in sources), calculation),
                "View-bound UI action check: " + verdict.value,
            )
        )
    if instruction.task_id == "circuit_structure_reading":
        source_payload = {key: value for key, value in public.items() if key != "candidate_answer"}
        sources = tuple(
            CircuitSource.model_validate(
                invoke("specialist_circuit_source", source_payload, CircuitSource, index)
            )
            for index in range(2)
        )
        assert instruction.scope_id is not None
        assert instruction.view_id is not None
        assert instruction.calibrated_domain is not None
        parameters = {item.name: item.value for item in instruction.public_parameters}
        verdict, calculation = verify_circuit(
            (sources[0], sources[1]),
            instruction.calibrated_domain,
            parameters.get("notation"),
            parameters.get("operation"),
            instruction.scope_id,
            instruction.view_id,
            payload["candidate_answer"],
        )
        results.append(
            OperationCheck(
                "circuit_graph_validator",
                verdict,
                (*tuple(item.model_dump(mode="json") for item in sources), calculation),
                "Circuit netlist check: " + verdict.value,
            )
        )
    if instruction.task_id == "chemical_structure_reading":
        source_payload = {key: value for key, value in public.items() if key != "candidate_answer"}
        sources = tuple(
            ChemicalSource.model_validate(
                invoke("specialist_chemistry_source", source_payload, ChemicalSource, index)
            )
            for index in range(2)
        )
        assert instruction.scope_id is not None
        assert instruction.view_id is not None
        assert instruction.calibrated_domain is not None
        notation = next(
            (item.value for item in instruction.public_parameters if item.name == "notation"), None
        )
        verdict, calculation = verify_chemistry(
            (sources[0], sources[1]),
            instruction.calibrated_domain,
            notation,
            instruction.scope_id,
            instruction.view_id,
            payload["candidate_answer"],
        )
        results.append(
            OperationCheck(
                "chemical_graph_validator",
                verdict,
                (*tuple(item.model_dump(mode="json") for item in sources), calculation),
                "Isolated chemical graph check: " + verdict.value,
            )
        )
    if instruction.task_id == "music_notation_reading":
        source_payload = {key: value for key, value in public.items() if key != "candidate_answer"}
        sources = tuple(
            MusicSource.model_validate(
                invoke("specialist_music_source", source_payload, MusicSource, index)
            )
            for index in range(2)
        )
        assert instruction.scope_id is not None
        assert instruction.view_id is not None
        assert instruction.calibrated_domain is not None
        bar_range = next(
            (item.value for item in instruction.public_parameters if item.name == "bar_range"),
            None,
        )
        verdict, calculation = verify_music(
            (sources[0], sources[1]),
            instruction.calibrated_domain,
            bar_range,
            instruction.scope_id,
            instruction.view_id,
            payload["candidate_answer"],
        )
        results.append(
            OperationCheck(
                "music_notation_validator",
                verdict,
                (*tuple(item.model_dump(mode="json") for item in sources), calculation),
                "Isolated notation check: " + verdict.value,
            )
        )
    if instruction.task_id == "geometric_constraint_solving":
        source_payload = {key: value for key, value in public.items() if key != "candidate_answer"}
        answer_payload = {
            key: public[key]
            for key in ("target_language", "question", "candidate_answer", "expected_operation")
            if key in public
        }
        geo_sources = tuple(
            GeometryProblem.model_validate(
                invoke("specialist_geometry_source", source_payload, GeometryProblem, index)
            )
            for index in range(2)
        )
        geo_answers = tuple(
            GeometryNumericAnswer.model_validate(
                invoke("specialist_geometry_answer", answer_payload, GeometryNumericAnswer, index)
            )
            for index in range(2)
        )
        assert instruction.scope_id is not None
        assert instruction.view_id is not None
        assert instruction.calibrated_domain is not None
        verdict, calculation = verify_geometry_problem(
            (geo_sources[0], geo_sources[1]),
            (geo_answers[0], geo_answers[1]),
            instruction.calibrated_domain,
            instruction.scope_id,
            instruction.view_id,
            payload["candidate_answer"],
        )
        results.append(
            OperationCheck(
                "formal_geometry_validator",
                verdict,
                (
                    *tuple(item.model_dump(mode="json") for item in (*geo_sources, *geo_answers)),
                    calculation,
                ),
                "Isolated symbolic geometry check: " + verdict.value,
            )
        )
    if instruction.task_id == "geometric_relation_analysis":
        geometry_source_payload = {
            key: value for key, value in public.items() if key != "candidate_answer"
        }
        geometry_answer_payload = {
            key: public[key]
            for key in ("target_language", "question", "candidate_answer", "expected_operation")
            if key in public
        }
        geometry_sources = tuple(
            GeometrySource.model_validate(
                invoke("geometry_source", geometry_source_payload, GeometrySource, index)
            )
            for index in range(2)
        )
        geometry_answers = tuple(
            GeometryAnswer.model_validate(
                invoke("geometry_answer", geometry_answer_payload, GeometryAnswer, index)
            )
            for index in range(2)
        )
        assert instruction.scope_id is not None
        assert instruction.view_id is not None
        verdict = verify_geometry(
            (geometry_sources[0], geometry_sources[1]),
            (geometry_answers[0], geometry_answers[1]),
            instruction.scope_id,
            instruction.view_id,
            payload["candidate_answer"],
        )
        results.append(
            OperationCheck(
                "geometry_check",
                verdict,
                tuple(
                    item.model_dump(mode="json") for item in (*geometry_sources, *geometry_answers)
                ),
                "Marked geometry controller check: " + verdict.value,
            )
        )
    if instruction.task_id in PATTERN_TASKS:
        pattern_source_payload = {
            key: value for key, value in public.items() if key != "candidate_answer"
        }
        pattern_answer_payload = {
            key: public[key]
            for key in ("target_language", "question", "candidate_answer", "expected_operation")
            if key in public
        }
        pattern_sources = tuple(
            PatternSource.model_validate(
                invoke("pattern_source", pattern_source_payload, PatternSource, index)
            )
            for index in range(2)
        )
        pattern_answers = tuple(
            PatternAnswer.model_validate(
                invoke("pattern_answer", pattern_answer_payload, PatternAnswer, index)
            )
            for index in range(2)
        )
        assert instruction.scope_id is not None
        assert instruction.view_id is not None
        verdict = verify_pattern(
            instruction.task_id,
            (pattern_sources[0], pattern_sources[1]),
            (pattern_answers[0], pattern_answers[1]),
            instruction.scope_id,
            instruction.view_id,
            payload["candidate_answer"],
        )
        results.append(
            OperationCheck(
                "pattern_check",
                verdict,
                tuple(
                    item.model_dump(mode="json") for item in (*pattern_sources, *pattern_answers)
                ),
                "Finite pattern rule check: " + verdict.value,
            )
        )
    if instruction.task_id == "measurement_reading":
        scale_source_payload = {
            key: value for key, value in public.items() if key != "candidate_answer"
        }
        scale_answer_payload = {
            key: public[key]
            for key in ("target_language", "question", "candidate_answer", "expected_operation")
            if key in public
        }
        scale_sources = tuple(
            ScaleSource.model_validate(
                invoke("scale_source", scale_source_payload, ScaleSource, index)
            )
            for index in range(2)
        )
        scale_answers = tuple(
            ScaleAnswer.model_validate(
                invoke("scale_answer", scale_answer_payload, ScaleAnswer, index)
            )
            for index in range(2)
        )
        assert instruction.scope_id is not None
        assert instruction.view_id is not None
        precision = next(
            (item.value for item in instruction.public_parameters if item.name == "precision"), None
        )
        verdict = verify_scale(
            (scale_sources[0], scale_sources[1]),
            (scale_answers[0], scale_answers[1]),
            precision,
            instruction.scope_id,
            instruction.view_id,
            payload["candidate_answer"],
        )
        results.append(
            OperationCheck(
                "scale_check",
                verdict,
                tuple(item.model_dump(mode="json") for item in (*scale_sources, *scale_answers)),
                "Calibrated scale controller check: " + verdict.value,
            )
        )
    if instruction.task_id in GRAPH_TASKS:
        graph_source_payload = {
            key: value for key, value in public.items() if key != "candidate_answer"
        }
        graph_answer_payload = {
            key: public[key]
            for key in ("target_language", "question", "candidate_answer", "expected_operation")
            if key in public
        }
        graph_sources = tuple(
            GraphSource.model_validate(
                invoke("graph_source", graph_source_payload, GraphSource, index)
            )
            for index in range(2)
        )
        graph_answers = tuple(
            GraphAnswer.model_validate(
                invoke("graph_answer", graph_answer_payload, GraphAnswer, index)
            )
            for index in range(2)
        )
        assert instruction.scope_id is not None
        assert instruction.view_id is not None
        verdict = verify_graph(
            instruction.task_id,
            (graph_sources[0], graph_sources[1]),
            (graph_answers[0], graph_answers[1]),
            {item.name: item.value for item in instruction.public_parameters},
            instruction.scope_id,
            instruction.view_id,
            payload["candidate_answer"],
        )
        graph_evidence = tuple(
            item.model_dump(mode="json") for item in (*graph_sources, *graph_answers)
        )
        for name in instruction.verification_contracts:
            if name in {"graph_check", "exact_arithmetic_check"}:
                results.append(
                    OperationCheck(
                        name, verdict, graph_evidence, "Graph controller check: " + verdict.value
                    )
                )
    if instruction.task_id == "formula_transcription":
        formula_payload = {key: value for key, value in public.items() if key != "candidate_answer"}
        formula_sources = tuple(
            FormulaSource.model_validate(
                invoke("formula_source", formula_payload, FormulaSource, index)
            )
            for index in range(2)
        )
        assert instruction.scope_id is not None
        assert instruction.view_id is not None
        notation = next(
            (item.value for item in instruction.public_parameters if item.name == "notation"), None
        )
        verdict = verify_formula(
            (formula_sources[0], formula_sources[1]),
            notation if isinstance(notation, str) else None,
            instruction.scope_id,
            instruction.view_id,
            payload["candidate_answer"],
        )
        results.append(
            OperationCheck(
                "formula_structure_check",
                verdict,
                tuple(item.model_dump(mode="json") for item in formula_sources),
                "Literal formula structure: " + verdict.value,
            )
        )
    if instruction.task_id in DOCUMENT_TASKS:
        document_source_payload = {
            key: value for key, value in public.items() if key != "candidate_answer"
        }
        document_sources = tuple(
            DocumentSource.model_validate(
                invoke("document_source", document_source_payload, DocumentSource, index)
            )
            for index in range(2)
        )
        assert instruction.scope_id is not None
        assert instruction.view_id is not None
        content, schema = verify_document(
            instruction.task_id,
            (document_sources[0], document_sources[1]),
            {item.name: item.value for item in instruction.public_parameters},
            instruction.scope_id,
            instruction.view_id,
            payload["candidate_answer"],
        )
        document_evidence = tuple(item.model_dump(mode="json") for item in document_sources)
        for name in instruction.verification_contracts:
            if name in {"transcript_alignment", "schema_check"}:
                verdict = schema if name == "schema_check" else content
                results.append(
                    OperationCheck(
                        name,
                        verdict,
                        document_evidence,
                        "Document controller check: " + verdict.value,
                    )
                )
    if instruction.task_id in CHART_TASKS:
        chart_source_payload = {
            key: value for key, value in public.items() if key != "candidate_answer"
        }
        chart_sources = tuple(
            ChartSource.model_validate(
                invoke("chart_source", chart_source_payload, ChartSource, index)
            )
            for index in range(2)
        )
        chart_answers: tuple[ChartAnswer, ChartAnswer] | None = None
        if instruction.task_id != "chart_data_reconstruction":
            chart_answer_payload = {
                key: public[key]
                for key in ("target_language", "question", "candidate_answer", "expected_operation")
                if key in public
            }
            extracted = tuple(
                ChartAnswer.model_validate(
                    invoke("chart_answer", chart_answer_payload, ChartAnswer, index)
                )
                for index in range(2)
            )
            chart_answers = (extracted[0], extracted[1])
        assert instruction.scope_id is not None
        assert instruction.view_id is not None
        content, schema = verify_chart(
            instruction.task_id,
            (chart_sources[0], chart_sources[1]),
            chart_answers,
            {item.name: item.value for item in instruction.public_parameters},
            instruction.scope_id,
            instruction.view_id,
            payload["candidate_answer"],
        )
        chart_evidence = tuple(
            item.model_dump(mode="json") for item in (*chart_sources, *(chart_answers or ()))
        )
        for name in instruction.verification_contracts:
            if name in {"chart_encoding_check", "closed_set_check", "schema_check"}:
                verdict = schema if name == "schema_check" else content
                results.append(
                    OperationCheck(
                        name,
                        verdict or GateVerdict.UNKNOWN,
                        chart_evidence,
                        "Chart controller check: " + (verdict or GateVerdict.UNKNOWN).value,
                    )
                )
    if instruction.task_id in TABLE_TASKS:
        table_source_payload = {
            key: value for key, value in public.items() if key != "candidate_answer"
        }
        lookup_sources: tuple[TableLookupSource, ...] = ()
        if instruction.task_id == "table_cell_lookup":
            lookup_sources = tuple(
                TableLookupSource.model_validate(
                    invoke("table_lookup_source", table_source_payload, TableLookupSource, index)
                )
                for index in range(2)
            )
        needs_full_grid = not lookup_sources or any(
            source.layout == "requires_full_grid" for source in lookup_sources
        )
        table_sources = (
            tuple(
                TableSource.model_validate(
                    invoke("table_source", table_source_payload, TableSource, index)
                )
                for index in range(2)
            )
            if needs_full_grid
            else ()
        )
        table_answers: tuple[TableAnswer, TableAnswer] | None = None
        if instruction.task_id != "table_structure_reconstruction":
            table_answer_payload = {
                key: public[key]
                for key in ("target_language", "question", "candidate_answer", "expected_operation")
                if key in public
            }
            extracted = tuple(
                TableAnswer.model_validate(
                    invoke("table_answer", table_answer_payload, TableAnswer, index)
                )
                for index in range(2)
            )
            table_answers = (extracted[0], extracted[1])
        assert instruction.scope_id is not None
        assert instruction.view_id is not None
        if needs_full_grid:
            content, schema = verify_table(
                instruction.task_id,
                (table_sources[0], table_sources[1]),
                table_answers,
                {item.name: item.value for item in instruction.public_parameters},
                instruction.scope_id,
                instruction.view_id,
                payload["candidate_answer"],
            )
        else:
            assert table_answers is not None
            content = verify_table_lookup(
                (lookup_sources[0], lookup_sources[1]),
                table_answers,
                instruction.scope_id,
                instruction.view_id,
                payload["candidate_answer"],
                scope_region=instruction.scope_region,
            )
            schema = None
        table_evidence = tuple(
            item.model_dump(mode="json")
            for item in (*lookup_sources, *table_sources, *(table_answers or ()))
        )
        for name in instruction.verification_contracts:
            if name in {"table_structure_check", "closed_set_check", "schema_check"}:
                verdict = schema if name == "schema_check" else content
                results.append(
                    OperationCheck(
                        name,
                        verdict or GateVerdict.UNKNOWN,
                        table_evidence,
                        "Closed table controller check: " + (verdict or GateVerdict.UNKNOWN).value,
                    )
                )
    if instruction.task_id in QUANTITATIVE_TASKS:
        source_payload = {key: value for key, value in public.items() if key != "candidate_answer"}
        answer_payload = {
            key: public[key]
            for key in ("target_language", "question", "candidate_answer", "expected_operation")
            if key in public
        }
        quantity_sources = tuple(
            QuantitySource.model_validate(
                invoke("quantity_source", source_payload, QuantitySource, index)
            )
            for index in range(2)
        )
        quantity_answers = tuple(
            QuantityAnswer.model_validate(
                invoke("quantity_answer", answer_payload, QuantityAnswer, index)
            )
            for index in range(2)
        )
        assert instruction.scope_id is not None
        assert instruction.view_id is not None
        verdict = verify_quantitative(
            instruction.task_id,
            (quantity_sources[0], quantity_sources[1]),
            (quantity_answers[0], quantity_answers[1]),
            {item.name: item.value for item in instruction.public_parameters},
            instruction.scope_id,
            instruction.view_id,
            payload["candidate_answer"],
        )
        evidence = tuple(
            item.model_dump(mode="json") for item in (*quantity_sources, *quantity_answers)
        )
        for name in instruction.verification_contracts:
            if name in {"closed_set_check", "exact_arithmetic_check"}:
                results.append(
                    OperationCheck(
                        name,
                        verdict,
                        evidence,
                        "Exact quantitative controller check: " + verdict.value,
                    )
                )
    if instruction.task_id in FINITE_TASKS:
        source_payload = {key: value for key, value in public.items() if key != "candidate_answer"}
        answer_payload = {
            key: public[key]
            for key in ("target_language", "question", "candidate_answer", "expected_operation")
            if key in public
        }
        sources = tuple(
            FiniteSource.model_validate(
                invoke("finite_source", source_payload, FiniteSource, index)
            )
            for index in range(2)
        )
        answers = tuple(
            FiniteAnswer.model_validate(
                invoke("finite_answer", answer_payload, FiniteAnswer, index)
            )
            for index in range(2)
        )
        parameters = {item.name: item.value for item in instruction.public_parameters}
        assert instruction.scope_id is not None
        assert instruction.view_id is not None
        verdict = verify_finite(
            instruction.task_id,
            (sources[0], sources[1]),
            (answers[0], answers[1]),
            parameters,
            instruction.scope_id,
            instruction.view_id,
        )
        evidence = tuple(item.model_dump(mode="json") for item in (*sources, *answers))
        for name in instruction.verification_contracts:
            if name in {"closed_set_check", "exact_arithmetic_check"}:
                results.append(
                    OperationCheck(
                        name, verdict, evidence, "Finite-source controller check: " + verdict.value
                    )
                )
    for name in instruction.verification_contracts:
        if instruction.task_id == "formula_transcription" and name == "formula_structure_check":
            continue
        if instruction.task_id == "measurement_reading" and name == "scale_check":
            continue
        if instruction.task_id in PATTERN_TASKS and name == "pattern_check":
            continue
        if instruction.task_id == "geometric_relation_analysis" and name == "geometry_check":
            continue
        if (
            instruction.task_id == "geometric_constraint_solving"
            and name == "formal_geometry_validator"
        ):
            continue
        if instruction.task_id == "music_notation_reading" and name == "music_notation_validator":
            continue
        if (
            instruction.task_id == "chemical_structure_reading"
            and name == "chemical_graph_validator"
        ):
            continue
        if instruction.task_id == "circuit_structure_reading" and name == "circuit_graph_validator":
            continue
        if instruction.task_id == "ui_action_specification" and name == "ui_action_validator":
            continue
        if (
            instruction.task_id in {"diagram_to_code", "screen_to_code"}
            and name == "sandbox_render_validator"
        ):
            continue
        if (
            instruction.task_id
            in FINITE_TASKS
            | QUANTITATIVE_TASKS
            | TABLE_TASKS
            | CHART_TASKS
            | DOCUMENT_TASKS
            | GRAPH_TASKS
            and name
            in {
                "closed_set_check",
                "exact_arithmetic_check",
                "table_structure_check",
                "schema_check",
                "chart_encoding_check",
                "transcript_alignment",
                "formula_structure_check",
                "graph_check",
            }
        ):
            continue
        if name == "dual_visual_review":
            continue
        models: list[BaseModel]
        verdict = GateVerdict.UNKNOWN
        if name == "closed_set_check":
            models = [invoke("set_inventory", public, SetInventory, index) for index in range(2)]
            inventories = [SetInventory.model_validate(item) for item in models]
            result = verify_set_inventories(inventories)
            if result is not None:
                verdict = GateVerdict(result.verdict)
        elif name == "exact_arithmetic_check":
            models = [
                invoke("computation_inventory", public, ComputationInventory, index)
                for index in range(2)
            ]
            computations = [ComputationInventory.model_validate(item) for item in models]
            operator = next(
                (
                    parameter.value
                    for parameter in instruction.public_parameters
                    if parameter.name == "operators"
                ),
                None,
            )
            if operator is not None and all(item.operation == operator for item in computations):
                computation = verify_computation_inventories(computations)
                verdict = GateVerdict(computation.verdict)
        elif name == "transcript_alignment":
            if instruction.task_id == "document_extractive_qa":
                source_payload = {
                    key: value for key, value in public.items() if key != "candidate_answer"
                }
                models = [
                    invoke("extractive_source", source_payload, TranscriptSource, index)
                    for index in range(2)
                ]
                sources = [TranscriptSource.model_validate(item) for item in models]
                verdict = verify_extractive(
                    (sources[0], sources[1]),
                    payload["candidate_answer"],
                    instruction.target_region or instruction.scope_region,
                )
            elif instruction.task_id in {
                "text_transcription",
                "code_transcription",
                "text_reading_order",
            }:
                source_payload = {
                    key: value for key, value in public.items() if key != "candidate_answer"
                }
                models = [
                    invoke("transcript_source", source_payload, TranscriptSource, index)
                    for index in range(2)
                ]
                sources = [TranscriptSource.model_validate(item) for item in models]
                verdict = verify_transcription(
                    (sources[0], sources[1]),
                    payload["candidate_answer"],
                    instruction.target_region or instruction.scope_region,
                )
            else:
                models = [
                    invoke("transcript_alignment", public, TranscriptInventory, index)
                    for index in range(2)
                ]
                transcripts = [TranscriptInventory.model_validate(item) for item in models]
                if (
                    all(t.coverage == "MET" for t in transcripts)
                    and transcripts[0].expected_text == transcripts[1].expected_text
                    and all(
                        t.answer_text and t.answer_text in payload["candidate_answer"]
                        for t in transcripts
                    )
                ):
                    verdict = consensus(
                        [
                            GateVerdict.MET
                            if t.expected_text == t.answer_text
                            else GateVerdict.NOT_MET
                            for t in transcripts
                        ]
                    )
        elif name in {"evidence_binding_check", "ui_grounding_check", "panel_comparison_check"}:
            contract_payload = {
                **public,
                "verification_contract": {
                    "id": name,
                    **task_catalog().verification_contracts[name].model_dump(mode="json"),
                },
            }
            models = [
                invoke("visual_contract_review", contract_payload, VisualContractReview, index)
                for index in range(2)
            ]
            reviews = [VisualContractReview.model_validate(item) for item in models]
            votes: list[GateVerdict] = []
            for review in reviews:
                valid = (
                    review.coverage == "MET"
                    and bool(review.bindings)
                    and all(
                        binding.answer_quote in payload["candidate_answer"]
                        for binding in review.bindings
                    )
                )
                if name == "panel_comparison_check":
                    regions = {
                        tuple(binding.region.model_dump().values()) for binding in review.bindings
                    }
                    valid = valid and len(regions) >= 2
                votes.append(GateVerdict(review.verdict) if valid else GateVerdict.UNKNOWN)
            verdict = consensus(votes)
        else:
            # Defensive protection when reading a saved candidate or extending the registry.
            results.append(OperationCheck(name, GateVerdict.UNKNOWN, (), "Validator unavailable"))
            continue
        results.append(
            OperationCheck(
                name,
                verdict,
                tuple(model.model_dump(mode="json") for model in models),
                "Independent extraction and controller verification: " + verdict.value,
            )
        )
    return tuple(results)
