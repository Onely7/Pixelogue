"""Evaluation-only specialist execution and resumable confirmation calibration."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from pixelogue.calibration import (
    CalibrationCertificate,
    CalibrationManifest,
    CalibrationObservation,
    calibrate_observations,
    model_calibration_lock,
)
from pixelogue.catalog import task_catalog
from pixelogue.config import PixelogueConfig, StrictModel
from pixelogue.contracts import ImageArtifact, InstructionCandidate, PublicMessage, SourcePurpose
from pixelogue.errors import ExecutionError, ExternalInputError
from pixelogue.io import read_json, write_json
from pixelogue.serialization import canonical_hash, canonical_json
from pixelogue.serving import ModelImage, VllmClient
from pixelogue.store import RunStore
from pixelogue.task_evidence import PublicParameter
from pixelogue.task_registry import registration
from pixelogue.task_verification import verify_operation

ANSWER_ONLY_STAGES = frozenset(
    {
        "finite_answer",
        "quantity_answer",
        "table_answer",
        "chart_answer",
        "graph_answer",
        "scale_answer",
        "pattern_answer",
        "geometry_answer",
        "specialist_geometry_answer",
    }
)


def _schema_retry_feedback(stage: str, error: ExecutionError) -> str:
    """Describe a rejected output to the same blind evaluator for one retry."""
    guidance = {
        "specialist_ui_source": (
            "If coverage is UNKNOWN or NOT_MET, set controls=[] even when some other "
            "controls are visible. Only MET coverage may include controls. "
        ),
        "specialist_music_source": (
            "The scope_region must contain every event region in the requested bars. "
            "For a whole-view score, use left=0, top=0, right=1, bottom=1. "
            "For UNKNOWN or NOT_MET coverage, return measures=[] with no partial events. "
            "For MET, each complete measure's written durations must add to the meter: "
            "base=1 is a whole note, 2 a half note, 4 a quarter note, 8 an eighth note, "
            "16 a sixteenth note; one dot multiplies by 3/2. "
            "For a rest event, set staff_step, accidental, and tie to null; "
            "for a note event, set a resolved integer staff_step. "
        ),
        "specialist_circuit_source": (
            "closed means a complete visible component/terminal inventory. A fully "
            "read passive network without a power source has closed=true. For MET, "
            "closed and junctions_resolved must both be true after verifying every pin. "
            "If the inventory is incomplete, return UNKNOWN with closed=false, "
            "components=[] and nets=[]. "
            "Net terminals may contain only component pins, such as R1:a. "
            "Omit junction names such as n1, and include each listed component's "
            ":a and :b exactly once. "
        ),
        "specialist_geometry_answer": (
            "Always emit answer_quote and reported. For MET, quote an exact "
            "substring of the candidate answer and put its literal number in reported. "
        ),
        "specialist_geometry_source": (
            "Use the same valid variable IDs in target and premises, with no spaces. "
            "given requires one variable and one explicitly printed numeric constant; "
            "right_angle one variable/no constants; triangle_angle_sum three distinct "
            "variables/no constants; parallel_equal_angle two/no constants; "
            "similar_ratio two variables/two printed constants; pythagorean three "
            "variables/no constants in leg, leg, hypotenuse order. "
            "A MET extraction must bind target to a premise. If a required visible "
            "condition cannot be resolved, use UNKNOWN with premises=[]. "
        ),
        "specialist_chemistry_source": (
            "Return only a blind source object with exactly coverage, domain, scope_id, "
            "view_id, scope_region, notation, atoms, bonds, reason. Coverage must be MET, "
            "NOT_MET, or UNKNOWN; never output an answer or SMILES field. Use the public "
            "domain, scope ID, view ID, and notation. Each atom needs atom_id as a string "
            "such as C1 or H1, element, charge, aromatic, region. Each bond needs a and b "
            "as those exact string IDs, order as single/double/triple/aromatic, and region. "
            "Every scope_region and atom or bond region must be a JSON object with left, "
            "top, right, bottom keys, never a coordinate array. "
            "If the error identifies a specific atoms[] or bonds[] box outside scope, "
            "redraw that box inside the actual scope; use the full view scope only when "
            "the complete requested molecule is visible there. "
            "Use aromatic bond order only for an explicitly drawn closed ring; "
            "a chain bond must have its visible single, double, or triple order. "
            "Use a small positive-extent box around each visible atom and bond; widen "
            "horizontal and vertical line regions slightly so top < bottom and left < right. "
            "The image origin is top-left: x increases rightward and y downward. For a "
            "diagonal bond, top=min(endpoint y) and bottom=max(endpoint y); "
            "left=min(endpoint x) and right=max(endpoint x). Enclose the visible stroke "
            "with positive width and height inside scope. "
            "Include H atoms only if H is explicitly drawn. Unlabeled skeletal endpoints "
            "and corners are carbon atoms. Decide all bonds before emitting JSON. If any "
            "atom or bond is unresolved, use coverage UNKNOWN with empty atoms and bonds. "
            "Keep reason to one sentence and do not emit repeated whitespace. "
        ),
    }.get(stage, "")
    if error.reason == "MODEL_WHITESPACE_RUNAWAY":
        failure = (
            f"The previous {stage} output exhausted its token limit in blank lines. "
            "Finish the JSON object and stop without trailing whitespace. "
        )
    elif error.reason == "MODEL_FINISH_REASON":
        failure = f"The previous {stage} output reached its token limit before completing JSON. "
    else:
        failure = f"The previous {stage} output violated its schema: {str(error)[:900]}. "
    return (
        f"{failure}Return one complete object with the required fields. Every normalized "
        "region must satisfy 0 <= left < right <= 1 and 0 <= top < bottom <= 1. "
        f"{guidance}Do not invent unsupported image details to satisfy the schema."
    )


class SpecialistEvaluationCase(StrictModel):
    """Frozen specialist input with independent gold labels kept controller-side."""

    case_id: str = Field(min_length=1)
    image: ImageArtifact
    task_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    scope_id: str = Field(min_length=1)
    view_id: str = Field(min_length=1)
    visible_scope: str = Field(min_length=1)
    public_parameters: tuple[PublicParameter, ...]
    target_language: Literal["en", "ja", "zh-Hans"]
    public_history: tuple[PublicMessage, ...]
    question: str = Field(min_length=1)
    candidate_answer: str = Field(min_length=1)
    split: Literal["development", "confirmation"]
    gold_accept: bool | None = None
    label_provenance: str | None = None

    @model_validator(mode="after")
    def check_scope(self) -> SpecialistEvaluationCase:
        """Restrict this bypass to held-out specialists and independent labels."""
        task = next((item for item in task_catalog().tasks if item.id == self.task_id), None)
        if task is None or task.status != "validator_gated_extension":
            raise ValueError("Evaluation-only route requires a specialist task")
        if self.image.purpose is not SourcePurpose.EVALUATION:
            raise ValueError("Specialist confirmation images must be evaluation-only")
        if self.view_id != self.image.full_view.view_id:
            raise ValueError("Specialist scope references a different delivered image view")
        if self.split == "confirmation" and (self.gold_accept is None or not self.label_provenance):
            raise ValueError("Confirmation needs an independent gold label provenance")
        return self


class SpecialistEvaluationResult(StrictModel):
    """One evaluated case or explicit infrastructure failure."""

    trial_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_id: str
    status: Literal["COMPLETE", "FAILED"]
    verdict: Literal["MET", "NOT_MET", "UNKNOWN"] | None
    validator_name: str
    validator_version: str
    model_locks: tuple[str, str]
    response_artifact_hashes: tuple[str, str] | None
    error: str | None
    gold_accept: bool | None
    split: Literal["development", "confirmation"]
    image_group_id: str
    domain: str
    task_id: str


def _specialist_validator(case: SpecialistEvaluationCase) -> tuple[str, str]:
    task = next(item for item in task_catalog().tasks if item.id == case.task_id)
    names = [name for name in task.verification_contracts if name != "dual_visual_review"]
    if (
        len(names) != 1
        or (entry := registration(names[0])) is None
        or not entry.supports(case.task_id)
    ):
        raise ValueError("Specialist has no exact task-specific validator")
    return entry.name, entry.version


def _instruction(case: SpecialistEvaluationCase) -> InstructionCandidate:
    task = next(item for item in task_catalog().tasks if item.id == case.task_id)
    return InstructionCandidate(
        candidate_id=canonical_hash({"evaluation_case": case.case_id, "task": case.task_id}),
        task_id=task.id,
        family=task.family,
        profile="normal",
        visible_scope=case.visible_scope,
        instruction_summary=task.definition_en,
        required_capabilities=task.required_capabilities,
        catalog_version=task_catalog().version,
        scope_id=case.scope_id,
        view_id=case.view_id,
        evidence_refs=("evaluation-only-scoped-image",),
        public_parameters=case.public_parameters,
        verification_contracts=task.verification_contracts,
        calibrated_domain=case.domain,
    )


def specialist_trial_id(case: SpecialistEvaluationCase, config: PixelogueConfig) -> str:
    """Exclude gold labels from model-call identity but bind visible input and code."""
    return canonical_hash(
        {
            "case": case.model_dump(mode="json", exclude={"gold_accept", "label_provenance"}),
            "config": config.config_hash,
            "models": (
                model_calibration_lock(config.models.generator_a),
                model_calibration_lock(config.models.generator_b),
            ),
        }
    )


def run_specialist_evaluation(
    cases: tuple[SpecialistEvaluationCase, ...],
    config: PixelogueConfig,
    artifact_root: Path,
    output_dir: Path,
    run_id: str,
    *,
    retry_failed: bool = False,
) -> dict[str, int]:
    """Run uncalibrated validators only through a separate held-out route."""
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Specialist case IDs must be unique")
    input_hash = canonical_hash([case.model_dump(mode="json") for case in cases])
    freeze = {
        "config_hash": config.config_hash,
        "input_hash": input_hash,
        "case_ids": [case.case_id for case in cases],
    }
    freeze_path = output_dir / "input.json"
    if freeze_path.exists():
        if read_json(freeze_path, SpecialistInputManifest).model_dump(mode="json") != freeze:
            raise ExternalInputError("SPECIALIST_INPUT_CHANGED", "Evaluation input differs")
    else:
        write_json(freeze_path, freeze)
    locks = (
        model_calibration_lock(config.models.generator_a),
        model_calibration_lock(config.models.generator_b),
    )
    stats = {
        "completed": 0,
        "failed": 0,
        "reused": 0,
        "stage_reused": 0,
        "schema_retries": 0,
        "finish_retries": 0,
    }
    with RunStore(
        config.storage.run_root, run_id, require_local_wal=config.storage.require_local_wal
    ) as store:
        store.initialize_run(run_id, canonical_hash(freeze), config.profile)
        clients = (
            VllmClient(config.models.generator_a, config.runtime, run_id=run_id, store=store),
            VllmClient(config.models.generator_b, config.runtime, run_id=run_id, store=store),
        )
        try:
            for case in cases:
                trial_id = specialist_trial_id(case, config)
                result_path = output_dir / "results" / f"{trial_id}.json"
                if result_path.exists():
                    result = read_json(result_path, SpecialistEvaluationResult)
                    if result.trial_id != trial_id:
                        raise ExternalInputError(
                            "SPECIALIST_RESULT_CHANGED", "Saved result differs"
                        )
                    if result.status == "COMPLETE" or not retry_failed:
                        stats["reused"] += 1
                        continue
                name, version = _specialist_validator(case)
                view = case.image.full_view
                image = ModelImage(
                    view_id=view.view_id,
                    path=artifact_root / view.relative_path,
                    encoded_sha256=view.encoded_sha256,
                    media_type=view.media_type,
                )
                payload = {
                    "target_language": case.target_language,
                    "public_history": [
                        item.model_dump(mode="json") for item in case.public_history
                    ],
                    "question": case.question,
                    "candidate_answer": case.candidate_answer,
                    "image_views": [
                        {
                            "view_id": view.view_id,
                            "encoded_sha256": view.encoded_sha256,
                            "width": str(view.width),
                            "height": str(view.height),
                        }
                    ],
                }

                def invoke(
                    stage: str,
                    body: dict[str, object],
                    model: type[BaseModel],
                    judge: int,
                    image: ModelImage = image,
                    trial_id: str = trial_id,
                ) -> BaseModel:
                    stage_id = canonical_hash(
                        {
                            "trial": trial_id,
                            "evaluator": judge,
                            "stage": stage,
                            "schema": model.model_json_schema(),
                            "body": body,
                        }
                    )
                    stage_path = output_dir / "stages" / f"{stage_id}.json"
                    if stage_path.exists():
                        saved = read_json(stage_path, SpecialistStageResult)
                        if saved.stage_id != stage_id or saved.trial_id != trial_id:
                            raise ExternalInputError(
                                "SPECIALIST_STAGE_CHANGED", "Saved evaluator stage differs"
                            )
                        stats["stage_reused"] += 1
                        return model.model_validate_json(canonical_json(saved.value))
                    request_images = () if stage in ANSWER_ONLY_STAGES else (image,)
                    max_tokens = (
                        config.tasks.source_max_tokens
                        if stage.startswith("specialist_") and stage.endswith("_source")
                        else 2048
                    )
                    try:
                        response = clients[judge].invoke(
                            stage,
                            body,
                            request_images,
                            model,
                            max_tokens=max_tokens,
                            temperature=0.0,
                            seed=config.seed,
                            trial_id=f"blind-judge:{judge}",
                        )
                    except ExecutionError as error:
                        if error.reason == "MODEL_SCHEMA_MISMATCH":
                            stats["schema_retries"] += 1
                        elif (
                            error.reason in {"MODEL_FINISH_REASON", "MODEL_WHITESPACE_RUNAWAY"}
                            and stage.startswith("specialist_")
                            and stage.endswith("_source")
                        ):
                            stats["finish_retries"] += 1
                        else:
                            raise
                        response = clients[judge].invoke(
                            stage,
                            body,
                            request_images,
                            model,
                            max_tokens=max_tokens,
                            temperature=0.0,
                            seed=config.seed,
                            trial_id=f"blind-judge:{judge}",
                            retry_feedback=_schema_retry_feedback(stage, error),
                            json_object_fallback=(
                                error.reason in {"MODEL_FINISH_REASON", "MODEL_WHITESPACE_RUNAWAY"}
                                and stage == "specialist_chemistry_source"
                            ),
                        )
                    value = model.model_validate(response.value)
                    write_json(
                        stage_path,
                        SpecialistStageResult(
                            stage_id=stage_id,
                            trial_id=trial_id,
                            evaluator="generator_a" if judge == 0 else "generator_b",
                            stage=stage,
                            schema_name=model.__name__,
                            value=value.model_dump(mode="json"),
                            request_hash=response.request_hash,
                            response_hash=response.response_hash,
                            prompt_tokens=response.prompt_tokens,
                            completion_tokens=response.completion_tokens,
                        ),
                    )
                    return value

                try:
                    entry = registration(name)
                    assert entry is not None
                    if environment_error := entry.environment_error():
                        raise ExecutionError("VALIDATOR_ENV_MISSING", environment_error)
                    checks = verify_operation(_instruction(case), payload, invoke, image.path)
                    check = next(item for item in checks if item.name == name)
                    verdict = check.verdict.value
                    if verdict not in {"MET", "NOT_MET", "UNKNOWN"}:
                        raise ExecutionError(
                            "SPECIALIST_VERDICT_INVALID", "Invalid specialist verdict"
                        )
                    artifact_hashes = tuple(
                        store.write_json_artifact("specialist-extractions", evidence)
                        for evidence in check.evidence[:2]
                    )
                    if len(artifact_hashes) != 2:
                        raise ExecutionError(
                            "SPECIALIST_EVIDENCE_MISSING", "Need two blind readings"
                        )
                    result = SpecialistEvaluationResult(
                        trial_id=trial_id,
                        case_id=case.case_id,
                        status="COMPLETE",
                        verdict=verdict,
                        validator_name=name,
                        validator_version=version,
                        model_locks=locks,
                        response_artifact_hashes=(artifact_hashes[0], artifact_hashes[1]),
                        error=None,
                        gold_accept=case.gold_accept,
                        split=case.split,
                        image_group_id=case.image.visual_group_id,
                        domain=case.domain,
                        task_id=case.task_id,
                    )
                    stats["completed"] += 1
                except (ExecutionError, StopIteration) as exc:
                    result = SpecialistEvaluationResult(
                        trial_id=trial_id,
                        case_id=case.case_id,
                        status="FAILED",
                        verdict=None,
                        validator_name=name,
                        validator_version=version,
                        model_locks=locks,
                        response_artifact_hashes=None,
                        error=str(exc),
                        gold_accept=case.gold_accept,
                        split=case.split,
                        image_group_id=case.image.visual_group_id,
                        domain=case.domain,
                        task_id=case.task_id,
                    )
                    stats["failed"] += 1
                write_json(result_path, result)
        finally:
            for client in clients:
                client.client.close()
    return stats


class SpecialistInputManifest(StrictModel):
    """Frozen local evaluation input and exact code/config identity."""

    config_hash: str
    input_hash: str
    case_ids: tuple[str, ...]


class SpecialistStageResult(StrictModel):
    """One successful model call saved independently for sequential evaluator runs."""

    stage_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    trial_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator: Literal["generator_a", "generator_b"]
    stage: str
    schema_name: str
    value: dict[str, Any]
    request_hash: str
    response_hash: str
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None = None


def build_calibration_manifest(
    results: tuple[SpecialistEvaluationResult, ...],
    expected_case_ids: tuple[str, ...],
) -> tuple[CalibrationManifest, dict[str, Any]]:
    """Certify complete frozen input groups and report every missing expected case."""
    if not expected_case_ids or len(set(expected_case_ids)) != len(expected_case_ids):
        raise ValueError("Calibration requires unique frozen input case IDs")
    result_ids = [result.case_id for result in results]
    result_id_set = set(result_ids)
    if len(result_id_set) != len(result_ids):
        raise ValueError("Calibration result case IDs must be unique")
    unknown = result_id_set - set(expected_case_ids)
    if unknown:
        raise ValueError(f"Calibration results are outside frozen input: {sorted(unknown)}")
    missing = tuple(case_id for case_id in expected_case_ids if case_id not in result_id_set)
    groups: dict[tuple[str, str, tuple[str, str], str], list[CalibrationObservation]] = defaultdict(
        list
    )
    incomplete_groups: set[tuple[str, str, tuple[str, str], str]] = set()
    pending: list[dict[str, str]] = [
        {"case_id": case_id, "reason": "missing_result"} for case_id in missing
    ]
    for result in results:
        key = (result.task_id, result.domain, result.model_locks, result.validator_version)
        confirmation = result.split == "confirmation"
        if (
            result.status != "COMPLETE"
            or result.verdict is None
            or result.response_artifact_hashes is None
        ):
            pending.append({"case_id": result.case_id, "reason": result.error or "unprocessed"})
            if confirmation:
                incomplete_groups.add(key)
            continue
        if not confirmation:
            pending.append({"case_id": result.case_id, "reason": "development_or_unlabeled"})
            continue
        if result.gold_accept is None:
            pending.append({"case_id": result.case_id, "reason": "missing_gold_label"})
            incomplete_groups.add(key)
            continue
        observation = CalibrationObservation(
            case_id=result.case_id,
            image_group_id=result.image_group_id,
            task_id=result.task_id,
            domain=result.domain,
            split="confirmation",
            gold_accept=result.gold_accept,
            verdict=result.verdict,
            model_locks=result.model_locks,
            validator_version=result.validator_version,
            response_artifact_hashes=result.response_artifact_hashes,
        )
        groups[key].append(observation)
    certified: list[CalibrationObservation] = []
    certificates: list[CalibrationCertificate] = []
    for key, observations in sorted(groups.items()):
        if missing:
            pending.extend(
                {"case_id": item.case_id, "reason": "incomplete_input_manifest"}
                for item in observations
            )
            continue
        if key in incomplete_groups:
            pending.extend(
                {"case_id": item.case_id, "reason": "incomplete_confirmation_group"}
                for item in observations
            )
            continue
        if not any(item.gold_accept for item in observations) or not any(
            not item.gold_accept for item in observations
        ):
            pending.extend(
                {"case_id": item.case_id, "reason": "missing_positive_or_negative"}
                for item in observations
            )
            continue
        certificate = calibrate_observations(observations)
        certified.extend(observations)
        certificates.append(certificate)
    manifest = CalibrationManifest(observations=tuple(certified), certificates=tuple(certificates))
    return manifest, {
        "input_results": len(results),
        "expected_cases": len(expected_case_ids),
        "missing_results": len(missing),
        "certificates": len(certificates),
        "eligible": sum(item.eligible for item in certificates),
        "incomplete_confirmation_groups": len(incomplete_groups),
        "pending": pending,
        "pending_by_reason": dict(Counter(item["reason"] for item in pending)),
    }
