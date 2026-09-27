"""Frozen answer-exposure trials with separate model calls and resumable records."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from pixelogue.calibration import model_calibration_lock
from pixelogue.config import PixelogueConfig, StrictModel
from pixelogue.contracts import ImageArtifact, PublicMessage
from pixelogue.errors import ExecutionError, ExternalInputError
from pixelogue.io import read_json, write_json
from pixelogue.serialization import canonical_hash
from pixelogue.serving import ModelImage, VllmClient

CONDITIONS = ("hidden", "plausible", "incorrect")
QUESTION_AXES = ("grounding", "operation_match", "answerability", "nonredundancy", "naturalness")


class FiveQuestionRating(StrictModel):
    """Five independent research-only question judgments."""

    grounding: Literal["MET", "NOT_MET", "UNKNOWN"]
    operation_match: Literal["MET", "NOT_MET", "UNKNOWN"]
    answerability: Literal["MET", "NOT_MET", "UNKNOWN"]
    nonredundancy: Literal["MET", "NOT_MET", "UNKNOWN"]
    naturalness: Literal["MET", "NOT_MET", "UNKNOWN"]
    reason: str = Field(min_length=1, max_length=500)

    @property
    def accepted(self) -> bool:
        """Treat any unknown or failed axis as not accepted."""
        return all(getattr(self, axis) == "MET" for axis in QUESTION_AXES)


class ExposureCase(StrictModel):
    """Frozen public question and prewritten candidate answers for one image."""

    case_id: str = Field(min_length=1)
    image: ImageArtifact
    target_language: Literal["en", "ja", "zh-Hans"]
    public_history: tuple[PublicMessage, ...]
    selected_instruction: dict[str, object]
    question: str = Field(min_length=1)
    plausible_answer: str = Field(min_length=1)
    incorrect_answer: str = Field(min_length=1)
    gold_question_admissible: bool | None = None
    gold_label_provenance: str | None = None

    @model_validator(mode="after")
    def distinct_answers(self) -> ExposureCase:
        """Keep the two exposures independently prewritten and distinguishable."""
        if self.plausible_answer.strip() == self.incorrect_answer.strip():
            raise ValueError("Plausible and incorrect answers must differ")
        if self.image.full_view.view_id == "":
            raise ValueError("Image view must be resolved")
        if self.gold_question_admissible is not None and not self.gold_label_provenance:
            raise ValueError("Gold question labels need independent provenance")
        return self


class ExposureTrial(StrictModel):
    """One exact evaluator invocation or saved-result reuse."""

    trial_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_id: str
    condition: Literal["hidden", "plausible", "incorrect"]
    repetition: Annotated[int, Field(ge=0)]
    order_index: Annotated[int, Field(ge=0)]


class ExposurePlan(StrictModel):
    """Immutable cases, model lock, random order and trial identities."""

    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_lock: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator: Literal["generator_a", "generator_b"]
    seed: int
    cases: tuple[ExposureCase, ...]
    trials: tuple[ExposureTrial, ...]
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def verify_plan(self) -> ExposurePlan:
        """Reject edits to frozen inputs or trial order after evaluation starts."""
        expected = canonical_hash(self.model_dump(mode="json", exclude={"plan_hash"}))
        if expected != self.plan_hash:
            raise ValueError("Exposure plan hash differs from frozen inputs")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("Exposure case IDs must be unique")
        if len({trial.trial_id for trial in self.trials}) != len(self.trials):
            raise ValueError("Exposure trial IDs must be unique")
        if {trial.case_id for trial in self.trials} != {case.case_id for case in self.cases}:
            raise ValueError("Exposure trials do not cover frozen cases")
        return self


class ExposureResult(StrictModel):
    """One completed or failed trial with nonzero missing-cost semantics."""

    trial_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["COMPLETE", "FAILED"]
    rating: FiveQuestionRating | None
    error: str | None
    request_hash: str | None
    response_hash: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    duration_ms: int
    cost_usd: float | None = None

    @model_validator(mode="after")
    def check_status(self) -> ExposureResult:
        """Keep failures and absent token or cost values explicit."""
        if (self.status == "COMPLETE") != (self.rating is not None):
            raise ValueError("Only completed trials can have a rating")
        if self.status == "FAILED" and not self.error:
            raise ValueError("Failed trials need a reason")
        return self


def build_exposure_plan(
    cases: tuple[ExposureCase, ...],
    config: PixelogueConfig,
    evaluator: Literal["generator_a", "generator_b"],
    repetitions: int,
    seed: int,
) -> ExposurePlan:
    """Freeze all candidate answers and randomized trial order before model calls."""
    if not cases or not 1 <= repetitions <= 20:
        raise ValueError("Exposure plan needs cases and 1-20 repetitions")
    endpoint = getattr(config.models, evaluator)
    model_lock = model_calibration_lock(endpoint)
    case_hash = canonical_hash([case.model_dump(mode="json") for case in cases])
    trials = [
        ExposureTrial(
            trial_id=canonical_hash(
                {
                    "config": config.config_hash,
                    "cases": case_hash,
                    "case": case.case_id,
                    "condition": condition,
                    "evaluator": model_lock,
                    "repetition": repetition,
                }
            ),
            case_id=case.case_id,
            condition=condition,
            repetition=repetition,
            order_index=0,
        )
        for case in cases
        for repetition in range(repetitions)
        for condition in CONDITIONS
    ]
    trials.sort(key=lambda trial: canonical_hash({"seed": seed, "trial": trial.trial_id}))
    ordered = tuple(
        trial.model_copy(update={"order_index": index}) for index, trial in enumerate(trials)
    )
    body = {
        "config_hash": config.config_hash,
        "model_lock": model_lock,
        "evaluator": evaluator,
        "seed": seed,
        "cases": tuple(case.model_dump(mode="json") for case in cases),
        "trials": tuple(trial.model_dump(mode="json") for trial in ordered),
    }
    return ExposurePlan.model_validate_json(
        json.dumps({**body, "plan_hash": canonical_hash(body)}, ensure_ascii=False)
    )


def freeze_exposure_plan(path: Path, plan: ExposurePlan) -> None:
    """Refuse to change a plan after any trial may have been launched."""
    if path.exists():
        if read_json(path, ExposurePlan) != plan:
            raise ExternalInputError("EXPOSURE_PLAN_CHANGED", "Existing frozen plan differs")
    else:
        write_json(path, plan)


def run_exposure_trials(
    plan: ExposurePlan,
    config: PixelogueConfig,
    artifact_root: Path,
    output_dir: Path,
    *,
    retry_failed: bool = False,
) -> dict[str, int]:
    """Call one evaluator for each missing trial; reuse exact saved trials on resume."""
    if (
        config.config_hash != plan.config_hash
        or model_calibration_lock(getattr(config.models, plan.evaluator)) != plan.model_lock
    ):
        raise ExternalInputError("EXPOSURE_MODEL_CHANGED", "Plan model or code identity changed")
    freeze_exposure_plan(output_dir / "plan.json", plan)
    cases = {case.case_id: case for case in plan.cases}
    endpoint = getattr(config.models, plan.evaluator)
    client = VllmClient(endpoint, config.runtime, run_id=plan.plan_hash, store=None)
    stats = {"completed": 0, "failed": 0, "reused": 0}
    try:
        for trial in plan.trials:
            result_path = output_dir / "results" / f"{trial.trial_id}.json"
            if result_path.exists():
                saved = read_json(result_path, ExposureResult)
                if saved.trial_id != trial.trial_id:
                    raise ExternalInputError("EXPOSURE_RESULT_CHANGED", "Saved trial ID differs")
                if saved.status == "COMPLETE" or not retry_failed:
                    stats["reused"] += 1
                    continue
            case = cases[trial.case_id]
            view = case.image.full_view
            image = ModelImage(
                view_id=view.view_id,
                path=artifact_root / view.relative_path,
                encoded_sha256=view.encoded_sha256,
                media_type=view.media_type,
            )
            payload: dict[str, object] = {
                "target_language": case.target_language,
                "public_history": [item.model_dump(mode="json") for item in case.public_history],
                "selected_instruction": case.selected_instruction,
                "question": case.question,
                "image_views": [
                    {
                        "view_id": view.view_id,
                        "encoded_sha256": view.encoded_sha256,
                        "width": str(view.width),
                        "height": str(view.height),
                    }
                ],
            }
            if trial.condition != "hidden":
                payload["shown_answer"] = (
                    case.plausible_answer
                    if trial.condition == "plausible"
                    else case.incorrect_answer
                )
            started = time.monotonic()
            try:
                response = client.invoke(
                    "research_question_exposure",
                    payload,
                    (image,),
                    FiveQuestionRating,
                    max_tokens=384,
                    temperature=0.0,
                    seed=plan.seed + trial.repetition,
                )
                result = ExposureResult(
                    trial_id=trial.trial_id,
                    status="COMPLETE",
                    rating=FiveQuestionRating.model_validate(response.value),
                    error=None,
                    request_hash=response.request_hash,
                    response_hash=response.response_hash,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    duration_ms=round((time.monotonic() - started) * 1000),
                    cost_usd=None,
                )
                stats["completed"] += 1
            except ExecutionError as exc:
                result = ExposureResult(
                    trial_id=trial.trial_id,
                    status="FAILED",
                    rating=None,
                    error=f"{exc.reason}: {exc}",
                    request_hash=None,
                    response_hash=None,
                    prompt_tokens=None,
                    completion_tokens=None,
                    duration_ms=round((time.monotonic() - started) * 1000),
                    cost_usd=None,
                )
                stats["failed"] += 1
            write_json(result_path, result)
    finally:
        client.client.close()
    return stats


def exposure_report(plan: ExposurePlan, output_dir: Path) -> dict[str, Any]:
    """Report paired plausible-hidden differences and explicit missing trials."""
    results: dict[str, ExposureResult] = {}
    for trial in plan.trials:
        path = output_dir / "results" / f"{trial.trial_id}.json"
        if path.exists():
            results[trial.trial_id] = read_json(path, ExposureResult)
    by_condition: dict[str, dict[str, int | float | None]] = {}
    for condition in CONDITIONS:
        selected = [
            results.get(trial.trial_id) for trial in plan.trials if trial.condition == condition
        ]
        completed = [item for item in selected if item is not None and item.rating is not None]
        accepted = sum(item.rating.accepted for item in completed if item.rating is not None)
        by_condition[condition] = {
            "planned": len(selected),
            "completed": len(completed),
            "failed": sum(item is not None and item.status == "FAILED" for item in selected),
            "missing": sum(item is None for item in selected),
            "accepted": accepted,
            "accept_rate": accepted / len(completed) if completed else None,
            "abstained": sum(
                any(getattr(item.rating, axis) == "UNKNOWN" for axis in QUESTION_AXES)
                for item in completed
                if item.rating is not None
            ),
        }
    paired: dict[tuple[str, int], dict[str, ExposureResult]] = {}
    for trial in plan.trials:
        result = results.get(trial.trial_id)
        if result is not None and result.rating is not None:
            paired.setdefault((trial.case_id, trial.repetition), {})[trial.condition] = result
    differences = [
        int(group["plausible"].rating.accepted) - int(group["hidden"].rating.accepted)
        for group in paired.values()
        if "plausible" in group
        and "hidden" in group
        and group["plausible"].rating is not None
        and group["hidden"].rating is not None
    ]
    gold_by_case = {case.case_id: case.gold_question_admissible for case in plan.cases}
    invalid_differences = [
        int(group["plausible"].rating.accepted) - int(group["hidden"].rating.accepted)
        for (case_id, _), group in paired.items()
        if gold_by_case[case_id] is False
        and "plausible" in group
        and "hidden" in group
        and group["plausible"].rating is not None
        and group["hidden"].rating is not None
    ]
    errors: dict[str, dict[str, int]] = {}
    for condition in CONDITIONS:
        false_accept = false_reject = labeled = 0
        for trial in plan.trials:
            if trial.condition != condition:
                continue
            gold = gold_by_case[trial.case_id]
            result = results.get(trial.trial_id)
            if gold is None or result is None or result.rating is None:
                continue
            labeled += 1
            false_accept += int(not gold and result.rating.accepted)
            false_reject += int(gold and not result.rating.accepted)
        errors[condition] = {
            "labeled": labeled,
            "false_accept": false_accept,
            "false_reject": false_reject,
        }
    return {
        "plan_hash": plan.plan_hash,
        "evaluator": plan.evaluator,
        "conditions": by_condition,
        "paired_plausible_minus_hidden": {
            "pairs": len(differences),
            "mean": sum(differences) / len(differences) if differences else None,
        },
        "aias_invalid_questions": {
            "pairs": len(invalid_differences),
            "mean": sum(invalid_differences) / len(invalid_differences)
            if invalid_differences
            else None,
        },
        "human_labeled_errors": errors,
        "cost_usd": None,
        "cost_status": "missing_price_schedule",
    }


def write_exposure_reports(plan: ExposurePlan, output_dir: Path) -> dict[str, Any]:
    """Write JSON, CSV and Markdown summaries from saved trial evidence."""
    report = exposure_report(plan, output_dir)
    write_json(output_dir / "report.json", report)
    csv_path = output_dir / "report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "condition",
                "planned",
                "completed",
                "failed",
                "missing",
                "accepted",
                "accept_rate",
                "abstained",
            )
        )
        for condition, item in report["conditions"].items():
            writer.writerow(
                (
                    condition,
                    *(
                        item[key]
                        for key in (
                            "planned",
                            "completed",
                            "failed",
                            "missing",
                            "accepted",
                            "accept_rate",
                            "abstained",
                        )
                    ),
                )
            )
    paired = report["paired_plausible_minus_hidden"]
    lines = [
        "# Answer exposure experiment",
        "",
        f"Plan: `{plan.plan_hash}`",
        "",
        "| Condition | Planned | Completed | Failed | Missing | Accepted |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, item in report["conditions"].items():
        lines.append(
            f"| {condition} | {item['planned']} | {item['completed']} | "
            f"{item['failed']} | {item['missing']} | {item['accepted']} |"
        )
    lines += [
        "",
        f"Paired plausible − hidden: {paired['mean']} over {paired['pairs']} pairs.",
        "Cost: unknown (no price schedule).",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report
