"""Applicable operation validators shared by holistic and detailed evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from pixelogue.catalog import task_catalog
from pixelogue.contracts import GateVerdict, InstructionCandidate
from pixelogue.evaluation import consensus
from pixelogue.rules import (
    ComputationInventory,
    SetInventory,
    verify_computation_inventories,
    verify_set_inventories,
)
from pixelogue.task_evidence import TranscriptInventory, VisualContractReview
from pixelogue.task_runtime import operation_contract

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
) -> tuple[OperationCheck, ...]:
    """Execute every declared supplement; no model controls applicability.

    Caller supplies the same allowed public inputs independently to each judge. Uncertain or
    inconsistent extraction cannot certify a turn. No source code or generated action is executed.
    """
    results: list[OperationCheck] = []
    public = {**payload, "expected_operation": operation_contract(instruction)}
    for name in instruction.verification_contracts:
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
            models = [
                invoke("transcript_alignment", public, TranscriptInventory, index)
                for index in range(2)
            ]
            transcripts = [TranscriptInventory.model_validate(item) for item in models]
            if all(t.coverage == "MET" for t in transcripts):
                if transcripts[0].expected_text == transcripts[1].expected_text and all(
                    t.answer_text and t.answer_text in payload["candidate_answer"]
                    for t in transcripts
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
