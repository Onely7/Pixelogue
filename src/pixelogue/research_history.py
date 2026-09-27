"""Exact public-history binding and two-stage counterfactual witness study."""

from __future__ import annotations

import csv
import time
import unicodedata
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from pixelogue.calibration import model_calibration_lock
from pixelogue.config import PixelogueConfig, StrictModel
from pixelogue.contracts import ImageArtifact, PublicMessage
from pixelogue.errors import ExecutionError, ExternalInputError
from pixelogue.io import read_json, write_json
from pixelogue.serialization import canonical_hash
from pixelogue.serving import ModelImage, VllmClient


class HistoryBinding(StrictModel):
    """Exact committed message span and a proposed condition-changing replacement."""

    relation: Literal["coreference", "public_constraint_change", "independent"]
    source_message_id: str | None = None
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, gt=0)
    source_quote: str | None = None
    alternative_quote: str | None = None

    @model_validator(mode="after")
    def check_shape(self) -> HistoryBinding:
        """Independent turns cannot pretend to carry a witness binding."""
        values = (
            self.source_message_id,
            self.start,
            self.end,
            self.source_quote,
            self.alternative_quote,
        )
        if self.relation == "independent" and any(value is not None for value in values):
            raise ValueError("Independent relation cannot bind a history span")
        if self.relation != "independent" and any(value is None for value in values):
            raise ValueError("Dependent relation needs a complete exact span")
        return self


class HistoryStudyCase(StrictModel):
    """Frozen research case; answer is withheld from the preanswer evaluator call."""

    case_id: str = Field(min_length=1)
    image: ImageArtifact
    target_language: Literal["en", "ja", "zh-Hans"]
    public_history: tuple[PublicMessage, ...]
    selected_instruction: dict[str, object]
    question: str = Field(min_length=1)
    candidate_answer: str = Field(min_length=1)
    binding: HistoryBinding

    @model_validator(mode="after")
    def validate_exact_binding(self) -> HistoryStudyCase:
        """Reject noncanonical prefixes and inaccurate source spans before model calls."""
        if len(self.public_history) % 2 or any(
            message.turn_index != index // 2 + 1
            or message.role != ("user" if index % 2 == 0 else "assistant")
            for index, message in enumerate(self.public_history)
        ):
            raise ValueError("History study requires a canonical committed dialogue prefix")
        resolve_alternative_history(self.public_history, self.binding)
        return self


class HistoryPreRating(StrictModel):
    """Blind counterfactual plausibility and condition-change judgments."""

    plausible_alternative: Literal["MET", "NOT_MET", "UNKNOWN"]
    changed_condition: Literal["MET", "NOT_MET", "UNKNOWN"]
    reason: str = Field(min_length=1)


class HistoryPostRating(StrictModel):
    """Actual answer's validity under original and altered public history."""

    valid_original: Literal["MET", "NOT_MET", "UNKNOWN"]
    valid_alternative: Literal["MET", "NOT_MET", "UNKNOWN"]
    reason: str = Field(min_length=1)


class HistoryTrialResult(StrictModel):
    """One separately saved model call, including failures and unknown cost."""

    trial_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_id: str
    evaluator: Literal["generator_a", "generator_b"]
    stage: Literal["pre", "post"]
    status: Literal["COMPLETE", "FAILED"]
    rating: HistoryPreRating | HistoryPostRating | None
    error: str | None
    request_hash: str | None
    response_hash: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    duration_ms: int
    cost_usd: float | None = None


def resolve_alternative_history(
    history: tuple[PublicMessage, ...], binding: HistoryBinding
) -> tuple[PublicMessage, ...]:
    """Substitute exactly one bound committed-message span, retaining all other turns."""
    if binding.relation == "independent":
        return history
    source_id, start, end = binding.source_message_id, binding.start, binding.end
    source_quote, alternative = binding.source_quote, binding.alternative_quote
    assert source_id is not None and start is not None and end is not None
    assert source_quote is not None and alternative is not None
    if not source_quote or not alternative or start >= end:
        raise ValueError("History witness needs nonempty source and alternative")
    if (
        unicodedata.normalize("NFKC", source_quote).casefold().strip()
        == unicodedata.normalize("NFKC", alternative).casefold().strip()
    ):
        raise ValueError("Alternative history merely repeats the source quote")
    matches = [
        index
        for index, message in enumerate(history)
        if message.message_id == source_id
        and end <= len(message.content)
        and message.content[start:end] == source_quote
    ]
    if len(matches) != 1:
        raise ValueError("History binding is not one exact committed-message span")
    index = matches[0]
    original = history[index]
    changed = original.model_copy(
        update={"content": original.content[:start] + alternative + original.content[end:]}
    )
    return history[:index] + (changed,) + history[index + 1 :]


def history_trial_id(
    case: HistoryStudyCase,
    config: PixelogueConfig,
    evaluator: Literal["generator_a", "generator_b"],
    stage: Literal["pre", "post"],
) -> str:
    """Bind each stage to the frozen case, code and active evaluator model."""
    return canonical_hash(
        {
            "case": case.model_dump(mode="json"),
            "config": config.config_hash,
            "evaluator": model_calibration_lock(getattr(config.models, evaluator)),
            "stage": stage,
        }
    )


def _complete_history_stage(
    case: HistoryStudyCase,
    config: PixelogueConfig,
    evaluator: Literal["generator_a", "generator_b"],
    stage: Literal["pre", "post"],
    artifact_root: Path,
    output_dir: Path,
    retry_failed: bool = False,
) -> HistoryTrialResult:
    """Reuse an identical saved stage or make one independent model request."""
    trial_id = history_trial_id(case, config, evaluator, stage)
    path = output_dir / "results" / f"{trial_id}.json"
    if path.exists():
        result = read_json(path, HistoryTrialResult)
        if result.trial_id != trial_id:
            raise ExternalInputError("HISTORY_TRIAL_CHANGED", "Saved history trial differs")
        if result.status == "COMPLETE" or not retry_failed:
            return result
    view = case.image.full_view
    image = ModelImage(
        view_id=view.view_id,
        path=artifact_root / view.relative_path,
        encoded_sha256=view.encoded_sha256,
        media_type=view.media_type,
    )
    alternative = resolve_alternative_history(case.public_history, case.binding)
    payload: dict[str, object] = {
        "target_language": case.target_language,
        "public_history": [message.model_dump(mode="json") for message in case.public_history],
        "alternative_history": [message.model_dump(mode="json") for message in alternative],
        "question": case.question,
        "selected_instruction": case.selected_instruction,
        "binding": case.binding.model_dump(mode="json"),
        "image_views": [
            {
                "view_id": view.view_id,
                "encoded_sha256": view.encoded_sha256,
                "width": str(view.width),
                "height": str(view.height),
            }
        ],
    }
    if stage == "post":
        payload["candidate_answer"] = case.candidate_answer
    endpoint = getattr(config.models, evaluator)
    client = VllmClient(endpoint, config.runtime, run_id=trial_id, store=None)
    started = time.monotonic()
    try:
        response = client.invoke(
            "research_history_pre" if stage == "pre" else "research_history_post",
            payload,
            (image,),
            HistoryPreRating if stage == "pre" else HistoryPostRating,
            max_tokens=256,
            temperature=0.0,
            seed=config.seed,
        )
        rating = (
            HistoryPreRating.model_validate(response.value)
            if stage == "pre"
            else HistoryPostRating.model_validate(response.value)
        )
        result = HistoryTrialResult(
            trial_id=trial_id,
            case_id=case.case_id,
            evaluator=evaluator,
            stage=stage,
            status="COMPLETE",
            rating=rating,
            error=None,
            request_hash=response.request_hash,
            response_hash=response.response_hash,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            duration_ms=round((time.monotonic() - started) * 1000),
            cost_usd=None,
        )
    except ExecutionError as exc:
        result = HistoryTrialResult(
            trial_id=trial_id,
            case_id=case.case_id,
            evaluator=evaluator,
            stage=stage,
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
    finally:
        client.client.close()
    write_json(path, result)
    return result


def run_history_study(
    cases: tuple[HistoryStudyCase, ...],
    config: PixelogueConfig,
    artifact_root: Path,
    output_dir: Path,
    *,
    retry_failed: bool = False,
) -> dict[str, object]:
    """Require two preanswer and two answer-dependent votes per strong witness."""
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("History case IDs must be unique")
    identity = canonical_hash(
        {"cases": [case.model_dump(mode="json") for case in cases], "config": config.config_hash}
    )
    frozen_path = output_dir / "cases.json"
    frozen = {"study_hash": identity, "cases": [case.model_dump(mode="json") for case in cases]}
    if (
        frozen_path.exists()
        and read_json(frozen_path, HistoryStudyManifest).model_dump(mode="json") != frozen
    ):
        raise ExternalInputError("HISTORY_CASES_CHANGED", "Frozen history study differs")
    write_json(frozen_path, frozen)
    rows: list[dict[str, Any]] = []
    for case in cases:
        if case.binding.relation == "independent":
            rows.append({"case_id": case.case_id, "state": "independent", "pre": [], "post": []})
            continue
        pre = tuple(
            _complete_history_stage(
                case, config, role, "pre", artifact_root, output_dir, retry_failed
            )
            for role in ("generator_a", "generator_b")
        )
        pre_met = all(
            isinstance(item.rating, HistoryPreRating)
            and item.rating.plausible_alternative == "MET"
            and item.rating.changed_condition == "MET"
            for item in pre
        )
        post = (
            tuple(
                _complete_history_stage(
                    case, config, role, "post", artifact_root, output_dir, retry_failed
                )
                for role in ("generator_a", "generator_b")
            )
            if pre_met
            else ()
        )
        strong = bool(post) and all(
            isinstance(item.rating, HistoryPostRating)
            and item.rating.valid_original == "MET"
            and item.rating.valid_alternative == "NOT_MET"
            for item in post
        )
        rows.append(
            {
                "case_id": case.case_id,
                "state": "strong_dependency" if strong else "unverified_dependency",
                "pre": [item.model_dump(mode="json") for item in pre],
                "post": [item.model_dump(mode="json") for item in post],
            }
        )
    report: dict[str, object] = {
        "study_hash": identity,
        "cases": len(cases),
        "strong_dependency": sum(item["state"] == "strong_dependency" for item in rows),
        "independent": sum(item["state"] == "independent" for item in rows),
        "rows": rows,
        "cost_usd": None,
    }
    write_json(output_dir / "report.json", report)
    with (output_dir / "report.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("case_id", "state", "pre_complete", "post_complete"))
        for row in rows:
            writer.writerow(
                (
                    row["case_id"],
                    row["state"],
                    sum(item["status"] == "COMPLETE" for item in row["pre"]),
                    sum(item["status"] == "COMPLETE" for item in row["post"]),
                )
            )
    (output_dir / "report.md").write_text(
        "\n".join(
            (
                "# Counterfactual history study",
                "",
                f"Cases: {len(cases)}",
                f"Strong dependency: {report['strong_dependency']}",
                f"Independent: {report['independent']}",
                "",
                "Unknown and failed votes remain in report.json.",
                "",
            )
        ),
        encoding="utf-8",
    )
    return report


class HistoryStudyManifest(StrictModel):
    """Stored frozen case input used to reject changed resumption inputs."""

    study_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases: tuple[HistoryStudyCase, ...]
