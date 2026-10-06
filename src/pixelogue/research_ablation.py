"""Fixed-question factorial generation with isolated research-only outputs."""

from __future__ import annotations

import csv
import json
from itertools import product
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from pixelogue.config import PixelogueConfig, StrictModel
from pixelogue.contracts import (
    ImageArtifact,
    InstructionCandidate,
    PublicMessage,
    TextPayload,
    build_history_snapshot,
)
from pixelogue.errors import ExecutionError, ExternalInputError
from pixelogue.evaluation import repeated_public_question
from pixelogue.io import read_json, write_json
from pixelogue.pipeline import SynthesisCoordinator
from pixelogue.prompts import is_private_prompt_echo
from pixelogue.research_exposure import FiveQuestionRating
from pixelogue.serialization import canonical_hash
from pixelogue.serving import ModelImage, VllmClient
from pixelogue.store import RunStore
from pixelogue.task_runtime import operation_contract


class AblationTurn(StrictModel):
    """Frozen question and admitted operation for one depth."""

    question: str = Field(min_length=1)
    instruction: InstructionCandidate
    human_question_admissible: bool | None = None


class AblationCase(StrictModel):
    """One image and fixed questions; a role generates its full conversation."""

    case_id: str = Field(min_length=1)
    image: ImageArtifact
    target_language: Literal["en", "ja", "zh-Hans"]
    generator_role: Literal["generator_a", "generator_b"]
    turns: Annotated[tuple[AblationTurn, ...], Field(min_length=1, max_length=6)]

    @model_validator(mode="after")
    def check_views(self) -> AblationCase:
        """Keep all frozen operations on one exact delivered view."""
        if any(turn.instruction.view_id != self.image.full_view.view_id for turn in self.turns):
            raise ValueError("Ablation turn references a different view")
        return self


class AblationCell(StrictModel):
    """Only the three agreed comparison factors; answer repair is always off."""

    question_gate: Literal["pre_answer", "post_answer", "omitted"]
    question_evaluator: Literal["generator_a", "generator_b", "both"]
    history: Literal["verified_only", "generated"]


class AblationTrial(StrictModel):
    """One independent cell and image run with fixed identity."""

    trial_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_id: str
    cell: AblationCell
    order_index: int


class AblationPlan(StrictModel):
    """Frozen full-factorial schedule and code/config identity."""

    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    cases: tuple[AblationCase, ...]
    trials: tuple[AblationTrial, ...]
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def check_identity(self) -> AblationPlan:
        """Reject changed trials or input cases on resume."""
        body = self.model_dump(mode="json", exclude={"plan_hash"})
        if self.plan_hash != canonical_hash(body):
            # Older frozen plans predate InstructionCandidate.target_region. Preserve only
            # that omitted optional field when reconstructing their original identity.
            for case_model, case_data in zip(self.cases, body["cases"], strict=True):
                for turn_model, turn_data in zip(case_model.turns, case_data["turns"], strict=True):
                    instruction = turn_model.instruction
                    if "target_region" not in instruction.model_fields_set:
                        turn_data["instruction"].pop("target_region", None)
            if self.plan_hash != canonical_hash(body):
                raise ValueError("Ablation plan hash differs from contents")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("Ablation case IDs must be unique")
        if len({trial.trial_id for trial in self.trials}) != len(self.trials):
            raise ValueError("Ablation trial IDs must be unique")
        return self


class AblationTurnResult(StrictModel):
    """Gate and post-answer review at one depth, including unverified history."""

    depth: int
    gate_verdict: Literal["MET", "NOT_MET", "UNKNOWN", "SKIPPED"]
    gate_votes: tuple[FiveQuestionRating, ...]
    answer: str | None
    review: Literal["PASS", "FAIL", "ABSTAIN", "ERROR", "NOT_RUN"]
    status: Literal["COMMITTED", "REJECTED", "ABSTAINED", "ERROR"]
    history_includes_unverified: bool
    human_question_admissible: bool | None


class AblationResult(StrictModel):
    """Research-only conversation trace; never a training-export contract."""

    trial_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_id: str
    cell: AblationCell
    status: Literal["COMPLETE", "FAILED"]
    turns: tuple[AblationTurnResult, ...]
    error: str | None
    study_only: Literal[True] = True
    cost_usd: float | None = None


def build_ablation_plan(
    cases: tuple[AblationCase, ...], config: PixelogueConfig, seed: int
) -> AblationPlan:
    """Enumerate all 18 agreed cells with independent trial IDs and fixed order."""
    if not cases:
        raise ValueError("Ablation needs at least one case")
    cells = tuple(
        AblationCell(question_gate=gate, question_evaluator=evaluator, history=history)
        for gate, evaluator, history in product(
            ("pre_answer", "post_answer", "omitted"),
            ("generator_a", "generator_b", "both"),
            ("verified_only", "generated"),
        )
    )
    case_hash = canonical_hash([case.model_dump(mode="json") for case in cases])
    trials = [
        AblationTrial(
            trial_id=canonical_hash(
                {
                    "config": config.config_hash,
                    "cases": case_hash,
                    "case": case.case_id,
                    "cell": cell.model_dump(mode="json"),
                }
            ),
            case_id=case.case_id,
            cell=cell,
            order_index=0,
        )
        for case in cases
        for cell in cells
    ]
    trials.sort(key=lambda trial: canonical_hash({"seed": seed, "trial": trial.trial_id}))
    ordered = tuple(
        trial.model_copy(update={"order_index": index}) for index, trial in enumerate(trials)
    )
    body = {
        "config_hash": config.config_hash,
        "seed": seed,
        "cases": [case.model_dump(mode="json") for case in cases],
        "trials": [trial.model_dump(mode="json") for trial in ordered],
    }
    return AblationPlan.model_validate_json(json.dumps({**body, "plan_hash": canonical_hash(body)}))


def _question_gate(
    trial: AblationTrial,
    case: AblationCase,
    turn: AblationTurn,
    history: tuple[PublicMessage, ...],
    image: ModelImage,
    clients: dict[str, VllmClient],
    config: PixelogueConfig,
    shown_answer: str | None,
) -> tuple[Literal["MET", "NOT_MET", "UNKNOWN"], tuple[FiveQuestionRating, ...]]:
    roles = (
        ("generator_a", "generator_b")
        if trial.cell.question_evaluator == "both"
        else (trial.cell.question_evaluator,)
    )
    view = case.image.full_view
    payload: dict[str, object] = {
        "target_language": case.target_language,
        "public_history": [item.model_dump(mode="json") for item in history],
        "selected_instruction": operation_contract(turn.instruction),
        "question": turn.question,
        "image_views": [
            {
                "view_id": view.view_id,
                "encoded_sha256": view.encoded_sha256,
                "width": str(view.width),
                "height": str(view.height),
            }
        ],
    }
    if shown_answer is not None:
        payload["shown_answer"] = shown_answer
    votes = tuple(
        FiveQuestionRating.model_validate(
            clients[role]
            .invoke(
                "research_question_exposure",
                payload,
                (image,),
                FiveQuestionRating,
                max_tokens=384,
                temperature=0.0,
                seed=config.seed,
            )
            .value
        )
        for role in roles
    )
    if all(vote.accepted for vote in votes):
        return "MET", votes
    if any(
        "NOT_MET"
        in (
            vote.grounding,
            vote.operation_match,
            vote.answerability,
            vote.nonredundancy,
            vote.naturalness,
        )
        for vote in votes
    ):
        return "NOT_MET", votes
    return "UNKNOWN", votes


def _run_one(
    trial: AblationTrial,
    case: AblationCase,
    config: PixelogueConfig,
    artifact_root: Path,
    coordinator: SynthesisCoordinator,
    clients: dict[str, VllmClient],
) -> AblationResult:
    view = case.image.full_view
    image = ModelImage(
        view_id=view.view_id,
        path=artifact_root / view.relative_path,
        encoded_sha256=view.encoded_sha256,
        media_type=view.media_type,
    )
    image_views = [
        {
            "view_id": view.view_id,
            "encoded_sha256": view.encoded_sha256,
            "width": str(view.width),
            "height": str(view.height),
        }
    ]
    history: tuple[PublicMessage, ...] = ()
    unverified = False
    results: list[AblationTurnResult] = []
    try:
        for depth, turn in enumerate(case.turns, 1):
            question = PublicMessage(
                message_id=f"{trial.trial_id}:q:{depth}",
                turn_index=depth,
                role="user",
                content=turn.question,
            )
            if is_private_prompt_echo(turn.question) or repeated_public_question(
                turn.question, history
            ):
                results.append(
                    AblationTurnResult(
                        depth=depth,
                        gate_verdict="NOT_MET",
                        gate_votes=(),
                        answer=None,
                        review="NOT_RUN",
                        status="REJECTED",
                        history_includes_unverified=unverified,
                        human_question_admissible=turn.human_question_admissible,
                    )
                )
                break
            gate: Literal["MET", "NOT_MET", "UNKNOWN", "SKIPPED"] = "SKIPPED"
            votes: tuple[FiveQuestionRating, ...] = ()
            if trial.cell.question_gate == "pre_answer":
                gate, votes = _question_gate(
                    trial, case, turn, history, image, clients, config, None
                )
                if gate != "MET":
                    results.append(
                        AblationTurnResult(
                            depth=depth,
                            gate_verdict=gate,
                            gate_votes=votes,
                            answer=None,
                            review="NOT_RUN",
                            status="ABSTAINED" if gate == "UNKNOWN" else "REJECTED",
                            history_includes_unverified=unverified,
                            human_question_admissible=turn.human_question_admissible,
                        )
                    )
                    break
            payload = {
                "target_language": case.target_language,
                "public_history": [item.model_dump(mode="json") for item in history],
                "question": question.content,
                "expected_operation": operation_contract(turn.instruction),
                "active_requirements": [],
                "image_views": image_views,
            }
            answer_payload = TextPayload.model_validate(
                clients[case.generator_role]
                .invoke(
                    "answer_generation",
                    payload,
                    (image,),
                    TextPayload,
                    max_tokens=config.tasks.answer_max_tokens,
                    temperature=0.0,
                    seed=config.seed + depth,
                )
                .value
            )
            if answer_payload.text is None or is_private_prompt_echo(answer_payload.text):
                results.append(
                    AblationTurnResult(
                        depth=depth,
                        gate_verdict=gate,
                        gate_votes=votes,
                        answer=None,
                        review="NOT_RUN",
                        status="REJECTED",
                        history_includes_unverified=unverified,
                        human_question_admissible=turn.human_question_admissible,
                    )
                )
                break
            answer = PublicMessage(
                message_id=f"{trial.trial_id}:a:{depth}",
                turn_index=depth,
                role="assistant",
                content=answer_payload.text,
            )
            if trial.cell.question_gate == "post_answer":
                gate, votes = _question_gate(
                    trial, case, turn, history, image, clients, config, answer.content
                )
            snapshot = build_history_snapshot(trial.trial_id, depth, history)
            review = coordinator._rate_turn(  # noqa: SLF001 - internal research runner
                trial.trial_id,
                snapshot.history_hash,
                history,
                question,
                answer,
                turn.instruction,
                case.target_language,
                image_views,
                image,
                depth,
            )
            status: Literal["COMMITTED", "REJECTED", "ABSTAINED", "ERROR"]
            if gate == "NOT_MET" or review.aggregate == "FAIL":
                status = "REJECTED"
            elif gate == "UNKNOWN" or review.aggregate == "ABSTAIN":
                status = "ABSTAINED"
            elif review.aggregate == "ERROR":
                status = "ERROR"
            else:
                status = "COMMITTED"
            results.append(
                AblationTurnResult(
                    depth=depth,
                    gate_verdict=gate,
                    gate_votes=votes,
                    answer=answer.content,
                    review=review.aggregate,
                    status=status,
                    history_includes_unverified=unverified,
                    human_question_admissible=turn.human_question_admissible,
                )
            )
            if status != "COMMITTED":
                unverified = True
                if trial.cell.history == "verified_only":
                    break
            history = (*history, question, answer)
    except ExecutionError as exc:
        return AblationResult(
            trial_id=trial.trial_id,
            case_id=case.case_id,
            cell=trial.cell,
            status="FAILED",
            turns=tuple(results),
            error=f"{exc.reason}: {exc}",
        )
    return AblationResult(
        trial_id=trial.trial_id,
        case_id=case.case_id,
        cell=trial.cell,
        status="COMPLETE",
        turns=tuple(results),
        error=None,
    )


def run_ablation_plan(
    plan: AblationPlan,
    config: PixelogueConfig,
    artifact_root: Path,
    output_dir: Path,
    *,
    run_id: str,
    retry_failed: bool = False,
) -> dict[str, int]:
    """Run cells in frozen order, resume complete trials and isolate outputs from export."""
    if config.config_hash != plan.config_hash:
        raise ExternalInputError("ABLATION_CONFIG_CHANGED", "Plan needs its frozen configuration")
    path = output_dir / "plan.json"
    if path.exists() and read_json(path, AblationPlan) != plan:
        raise ExternalInputError("ABLATION_PLAN_CHANGED", "Frozen ablation plan differs")
    write_json(path, plan)
    cases = {case.case_id: case for case in plan.cases}
    stats = {"completed": 0, "failed": 0, "reused": 0}
    with RunStore(
        config.storage.run_root, run_id, require_local_wal=config.storage.require_local_wal
    ) as store:
        store.initialize_run(
            run_id, canonical_hash({"research_plan": plan.plan_hash}), config.profile
        )
        selector = VllmClient(
            config.models.active_selector_endpoint, config.runtime, run_id=run_id, store=None
        )
        clients = {
            "generator_a": VllmClient(
                config.models.generator_a, config.runtime, run_id=run_id, store=None
            ),
            "generator_b": VllmClient(
                config.models.generator_b, config.runtime, run_id=run_id, store=None
            ),
        }
        coordinator = SynthesisCoordinator(
            config, run_id, store, selector, clients["generator_a"], clients["generator_b"]
        )
        try:
            for trial in plan.trials:
                result_path = output_dir / "results" / f"{trial.trial_id}.json"
                if result_path.exists():
                    saved = read_json(result_path, AblationResult)
                    if saved.trial_id != trial.trial_id:
                        raise ExternalInputError("ABLATION_RESULT_CHANGED", "Saved trial differs")
                    if saved.status == "COMPLETE" or not retry_failed:
                        stats["reused"] += 1
                        continue
                result = _run_one(
                    trial, cases[trial.case_id], config, artifact_root, coordinator, clients
                )
                write_json(result_path, result)
                stats["completed" if result.status == "COMPLETE" else "failed"] += 1
        finally:
            selector.client.close()
            for client in clients.values():
                client.client.close()
    return stats


def ablation_report(plan: AblationPlan, output_dir: Path) -> dict[str, Any]:
    """Report all starting cases and depth reach, including rejected histories."""
    rows: list[dict[str, object]] = []
    for cell_key in sorted(
        {canonical_hash(trial.cell.model_dump(mode="json")) for trial in plan.trials}
    ):
        trials = [
            trial
            for trial in plan.trials
            if canonical_hash(trial.cell.model_dump(mode="json")) == cell_key
        ]
        results = [
            read_json(path, AblationResult) if path.exists() else None
            for trial in trials
            for path in [output_dir / "results" / f"{trial.trial_id}.json"]
        ]
        depth: dict[int, dict[str, int | None]] = {
            index: {
                "started": len(trials),
                "reached": sum(
                    result is not None and len(result.turns) >= index for result in results
                ),
                "committed": sum(
                    result is not None
                    and len(result.turns) >= index
                    and result.turns[index - 1].status == "COMMITTED"
                    for result in results
                ),
                "rejected": sum(
                    result is not None
                    and len(result.turns) >= index
                    and result.turns[index - 1].status == "REJECTED"
                    for result in results
                ),
                "abstained": sum(
                    result is not None
                    and len(result.turns) >= index
                    and result.turns[index - 1].status == "ABSTAINED"
                    for result in results
                ),
                "human_labeled": sum(
                    result is not None
                    and len(result.turns) >= index
                    and result.turns[index - 1].human_question_admissible is not None
                    for result in results
                ),
                "false_accept": sum(
                    result is not None
                    and len(result.turns) >= index
                    and result.turns[index - 1].human_question_admissible is False
                    and result.turns[index - 1].gate_verdict == "MET"
                    for result in results
                ),
                "false_reject": sum(
                    result is not None
                    and len(result.turns) >= index
                    and result.turns[index - 1].human_question_admissible is True
                    and result.turns[index - 1].gate_verdict == "NOT_MET"
                    for result in results
                ),
                "both_evaluator_errors": sum(
                    result is not None
                    and len(result.turns) >= index
                    and (turn := result.turns[index - 1]).human_question_admissible is not None
                    and len(turn.gate_votes) == 2
                    and all(
                        vote.accepted != turn.human_question_admissible for vote in turn.gate_votes
                    )
                    for result in results
                ),
            }
            for index in range(1, max(len(case.turns) for case in plan.cases) + 1)
        }
        for counts in depth.values():
            if counts["human_labeled"] == 0:
                counts["false_accept"] = None
                counts["false_reject"] = None
                counts["both_evaluator_errors"] = None
        rows.append(
            {
                "cell": trials[0].cell.model_dump(mode="json"),
                "planned": len(trials),
                "completed": sum(
                    result is not None and result.status == "COMPLETE" for result in results
                ),
                "failed": sum(
                    result is not None and result.status == "FAILED" for result in results
                ),
                "missing": sum(result is None for result in results),
                "depth": depth,
                "cost_usd": None,
            }
        )
    return {
        "plan_hash": plan.plan_hash,
        "cells": rows,
        "training_export_allowed": False,
        "cost_usd": None,
    }


def write_ablation_reports(plan: AblationPlan, output_dir: Path) -> dict[str, Any]:
    """Save JSON, compact CSV and Markdown for all factorial cells."""
    report = ablation_report(plan, output_dir)
    write_json(output_dir / "report.json", report)
    with (output_dir / "report.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "question_gate",
                "question_evaluator",
                "history",
                "planned",
                "completed",
                "failed",
                "missing",
                "cost_usd",
            )
        )
        for row in report["cells"]:
            cell = row["cell"]
            writer.writerow(
                (
                    cell["question_gate"],
                    cell["question_evaluator"],
                    cell["history"],
                    row["planned"],
                    row["completed"],
                    row["failed"],
                    row["missing"],
                    "",
                )
            )
    with (output_dir / "depth.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        columns = (
            "started",
            "reached",
            "committed",
            "rejected",
            "abstained",
            "human_labeled",
            "false_accept",
            "false_reject",
            "both_evaluator_errors",
        )
        writer.writerow(("question_gate", "question_evaluator", "history", "depth", *columns))
        for row in report["cells"]:
            cell = row["cell"]
            for depth, counts in row["depth"].items():
                writer.writerow(
                    (
                        cell["question_gate"],
                        cell["question_evaluator"],
                        cell["history"],
                        depth,
                        *(counts[key] for key in columns),
                    )
                )
    lines = [
        "# Fixed-question ablation",
        "",
        f"Plan: `{plan.plan_hash}`",
        "",
        "| Gate | Evaluator | History | Planned | Completed | Failed | Missing |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in report["cells"]:
        cell = row["cell"]
        lines.append(
            f"| {cell['question_gate']} | {cell['question_evaluator']} | "
            f"{cell['history']} | {row['planned']} | {row['completed']} | "
            f"{row['failed']} | {row['missing']} |"
        )
    lines += [
        "",
        "Costs are unknown without a price schedule. Depth counts include every starting case.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report
