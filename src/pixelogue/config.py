"""Validated application configuration."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    ValidationError,
    model_validator,
)

from pixelogue.errors import ConfigurationError
from pixelogue.serialization import canonical_hash, canonical_json, load_yaml

PRIMARY_GENERATOR_REPOS = ("Qwen/Qwen3.8-27B-FP8", "google/gemma-4-31B-it-qat-w4a16-ct")
PILOT_GENERATOR_REPOS = ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-9B")
ALLOWED_QUANTIZATIONS = {
    "Qwen/Qwen3.8-27B-FP8": "fp8",
    "google/gemma-4-31B-it-qat-w4a16-ct": "compressed-tensors",
}
SPECIALIST_TASK_IDS = frozenset(
    {
        "geometric_constraint_solving",
        "diagram_to_code",
        "screen_to_code",
        "music_notation_reading",
        "chemical_structure_reading",
        "circuit_structure_reading",
        "ui_action_specification",
    }
)


def allocate_quotas(total: int, weights: Mapping[str, int | str]) -> dict[str, int]:
    """Allocate an exact total using rational largest remainders.

    Ties are resolved by canonical language tag order.

    Raises:
        ConfigurationError: If a weight is invalid or the input is empty.
    """
    if total < 0 or not weights:
        raise ConfigurationError(
            "INVALID_LANGUAGE_ALLOCATION", "Allocation needs a total and weights"
        )
    exact_weights: dict[str, Fraction] = {}
    for tag, raw_weight in weights.items():
        if isinstance(raw_weight, bool) or isinstance(raw_weight, float):
            raise ConfigurationError("INVALID_LANGUAGE_WEIGHT", f"Invalid weight for {tag}")
        try:
            weight = Fraction(raw_weight)
        except (ValueError, ZeroDivisionError) as error:
            raise ConfigurationError(
                "INVALID_LANGUAGE_WEIGHT", f"Invalid weight for {tag}"
            ) from error
        if weight <= 0:
            raise ConfigurationError(
                "INVALID_LANGUAGE_WEIGHT", f"Weight for {tag} must be positive"
            )
        exact_weights[tag] = weight
    weight_sum = sum(exact_weights.values())
    ideals = {tag: Fraction(total) * weight / weight_sum for tag, weight in exact_weights.items()}
    quotas = {tag: ideal.numerator // ideal.denominator for tag, ideal in ideals.items()}
    remainder = total - sum(quotas.values())
    order = sorted(ideals, key=lambda tag: (-(ideals[tag] - quotas[tag]), tag))
    for tag in order[:remainder]:
        quotas[tag] += 1
    return quotas


class StrictModel(BaseModel):
    """Base model that rejects coercion and unknown fields."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ServingRuntimeIdentity(StrictModel):
    """Controlled server manifest bound to actual launch settings and version checks."""

    vllm_version: Annotated[str, Field(min_length=1)]
    structured_output_backend: Literal[
        "auto", "xgrammar", "guidance", "outlines", "lm-format-enforcer"
    ]
    disable_any_whitespace: bool
    server_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    xgrammar_whitespace_patch_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None

    @model_validator(mode="after")
    def validate_whitespace_control(self) -> ServingRuntimeIdentity:
        """Require a backend supporting the pinned vLLM whitespace control."""
        if self.disable_any_whitespace and self.structured_output_backend not in {
            "xgrammar",
            "guidance",
        }:
            raise ValueError("whitespace suppression requires explicit xgrammar or guidance")
        return self

    @property
    def bounded_whitespace_supported(self) -> bool:
        """Match the verified grammar extension while retaining ordinary JSON whitespace."""
        return (
            self.vllm_version == "0.29.0"
            and self.structured_output_backend == "xgrammar"
            and not self.disable_any_whitespace
            and self.xgrammar_whitespace_patch_sha256
            == "97e96f276ad6ad10d536f7c88e294a8115cce11bd219763c0a708dabf913020e"
        )


class ModelEndpoint(StrictModel):
    """Pinned model and local OpenAI-compatible endpoint."""

    repo_id: str
    revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")] | None = None
    processor_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")] | None = None
    base_url: HttpUrl = HttpUrl("http://127.0.0.1:8000/v1")
    served_name: str | None = None
    tensor_parallel_size: Annotated[int, Field(ge=1)] = 1
    gpu_memory_utilization: Annotated[float, Field(gt=0, le=0.95)] = 0.9
    dtype: Literal["bfloat16"] = "bfloat16"
    quantization: Literal["fp8", "compressed-tensors"] | None = None
    max_model_len: Annotated[int, Field(ge=4096)] = 32768
    api_key_env: str = "PIXELLOGUE_API_KEY"
    serving_runtime: ServingRuntimeIdentity | None = None

    @property
    def model_name(self) -> str:
        """Return the server-visible model name."""
        return self.served_name or self.repo_id


class ModelConfig(StrictModel):
    """Assign fixed models to instruction selection, generation, and judging."""

    selector: ModelEndpoint = ModelEndpoint(repo_id="Qwen/Qwen3.5-2B")
    selector_alternative: ModelEndpoint = ModelEndpoint(repo_id="Qwen/Qwen3.6-35B-A3B")
    active_selector: Literal["default", "alternative"] = "default"
    generator_a: ModelEndpoint = ModelEndpoint(repo_id="Qwen/Qwen3.8-27B-FP8", quantization="fp8")
    generator_b: ModelEndpoint = ModelEndpoint(
        repo_id="google/gemma-4-31B-it-qat-w4a16-ct", quantization="compressed-tensors"
    )
    generation_allocation: dict[Literal["generator_a", "generator_b"], int] = {
        "generator_a": 1,
        "generator_b": 1,
    }

    @model_validator(mode="after")
    def validate_roles(self) -> ModelConfig:
        """Require the approved selector and generation repositories."""
        allowed_selectors = {"Qwen/Qwen3.5-2B", "Qwen/Qwen3.6-35B-A3B"}
        if self.selector.repo_id not in allowed_selectors:
            raise ValueError("selector must be an approved Qwen instruction selector")
        if self.selector_alternative.repo_id not in allowed_selectors:
            raise ValueError("selector_alternative must be an approved Qwen selector")
        generator_repos = (self.generator_a.repo_id, self.generator_b.repo_id)
        if generator_repos not in {PRIMARY_GENERATOR_REPOS, PILOT_GENERATOR_REPOS}:
            raise ValueError(
                "generators must use the primary Qwen3.8-FP8/Gemma-w4a16 pair or the temporary "
                "Qwen3.5-9B pilot pair"
            )
        for endpoint in (self.generator_a, self.generator_b):
            expected_quantization = ALLOWED_QUANTIZATIONS.get(endpoint.repo_id)
            if endpoint.quantization != expected_quantization:
                raise ValueError(
                    f"{endpoint.repo_id} must use its pinned quantization method "
                    f"({expected_quantization!r}), not {endpoint.quantization!r}"
                )
        if any(weight <= 0 for weight in self.generation_allocation.values()):
            raise ValueError("generation allocation weights must be positive")
        return self

    @property
    def active_selector_endpoint(self) -> ModelEndpoint:
        """Return the explicitly selected instruction model."""
        if self.active_selector == "alternative":
            return self.selector_alternative
        return self.selector


class LanguageTarget(StrictModel):
    """One conversation-level language allocation weight."""

    tag: Literal["en", "ja", "zh-Hans"]
    weight: int | str

    @model_validator(mode="after")
    def validate_weight(self) -> LanguageTarget:
        """Reject booleans, floats, zero, negative, and invalid decimals."""
        if isinstance(self.weight, bool):
            raise ValueError("language weight must not be boolean")
        try:
            exact = Fraction(self.weight)
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError("language weight must be an integer or decimal string") from error
        if exact <= 0:
            raise ValueError("language weight must be positive")
        return self


class OpenImagesConfig(StrictModel):
    """Evaluation-only Open Images V7 sample settings."""

    enabled: bool = True
    split: Literal["validation"] = "validation"
    groups: Annotated[int, Field(ge=0, le=20)] = 20
    seed: int = 20260915
    image_ids_manifest: Path | None = None
    metadata_url: HttpUrl = HttpUrl(
        "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"
    )


class DataConfig(StrictModel):
    """Dataset size, language, and pilot boundaries."""

    target_dialogues: Annotated[int, Field(ge=1)] = 30000
    min_turns: Literal[2] = 2
    max_turns: Literal[6] = 6
    languages: tuple[LanguageTarget, ...] = (LanguageTarget(tag="en", weight=100),)
    pilot: bool = False
    pilot_max_image_groups: Annotated[int, Field(ge=1, le=100)] = 100
    pilot_gpu_hours: Annotated[float, Field(gt=0, le=4)] = 4.0
    open_images: OpenImagesConfig = OpenImagesConfig()

    @model_validator(mode="after")
    def validate_languages(self) -> DataConfig:
        """Require unique language tags and non-zero integer allocations."""
        tags = [target.tag for target in self.languages]
        if len(tags) != len(set(tags)):
            raise ValueError("language tags must be unique")
        if not tags:
            raise ValueError("at least one language is required")
        quotas = allocate_quotas(
            self.target_dialogues,
            {target.tag: target.weight for target in self.languages},
        )
        if any(quota == 0 for quota in quotas.values()):
            raise ValueError("every positive language target must receive at least one dialogue")
        if self.open_images.groups > self.pilot_max_image_groups:
            raise ValueError("Open Images groups exceed the pilot image-group budget")
        return self


class StorageConfig(StrictModel):
    """Run-time local storage and optional shared backup location."""

    run_root: Path = Path("/var/tmp/pixelogue")
    backup_root: Path | None = None
    require_local_wal: bool = True


class RepetitionDetectionConfig(StrictModel):
    """Bound engine-side repetition stopping to the two extraction stages."""

    min_pattern_size: Annotated[int, Field(ge=1, le=16)] = 1
    max_pattern_size: Annotated[int, Field(ge=1, le=16)] = 4
    min_count: Annotated[int, Field(ge=2, le=1024)] = 64
    recovery: Literal["retry", "abstain"] = "retry"
    stages: Annotated[
        tuple[Literal["evidence_extraction", "candidate_binding"], ...],
        Field(min_length=1, max_length=2),
    ] = ("evidence_extraction", "candidate_binding")

    @model_validator(mode="after")
    def validate_patterns(self) -> RepetitionDetectionConfig:
        """Reject inverted pattern ranges and repeated stage names."""
        if self.min_pattern_size > self.max_pattern_size:
            raise ValueError("minimum pattern size exceeds maximum")
        if len(self.stages) != len(set(self.stages)):
            raise ValueError("repetition detection stages must be unique")
        return self


class RuntimeConfig(StrictModel):
    """Inference request and retry boundaries."""

    request_timeout_seconds: Annotated[int, Field(ge=1, le=600)] = 180
    transport_max_attempts: Literal[1, 2, 3] = 3
    structured_output_max_attempts: Literal[1, 2, 3] = 2
    max_concurrent_images: Annotated[int, Field(ge=1, le=64)] = 1
    refill_completed_images: bool = False
    repetition_detection: RepetitionDetectionConfig | None = None
    json_whitespace_max_chars: Annotated[int, Field(ge=1, le=256)] | None = None
    max_total_requests: Annotated[int, Field(ge=1)] = 10_000_000
    max_total_output_tokens: Annotated[int, Field(ge=1)] = 1_000_000_000
    allow_external_inference: Literal[False] = False


class DecisionRoutingConfig(StrictModel):
    """Explicit experimental classifier identity, separate from generator roles."""

    evidence_enabled: bool = False
    binding_enabled: bool = False
    model_repository: Literal["autotrust/JEV-27B-VL", "akhilaaa3/Jev-Omni"] | None = None
    model_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")] | None = None
    proposal_max_tokens: Annotated[int, Field(ge=256, le=4096)] = 1536
    base_url: HttpUrl | None = None
    runtime_python: Path | None = None
    snapshot_path: Path | None = None
    startup_timeout_seconds: Annotated[int, Field(ge=1, le=1800)] = 600

    @model_validator(mode="after")
    def validate_explicit_identity(self) -> DecisionRoutingConfig:
        """Require a pinned classifier before enabling either experimental path."""
        if self.evidence_enabled or self.binding_enabled:
            if self.model_repository is None or self.model_revision is None:
                raise ValueError("Decision routing requires an explicit repository and revision")
        if (self.model_repository is None) != (self.model_revision is None):
            raise ValueError("Decision repository and revision must be specified together")
        return self


# Whole-structure reconstructions are kept reachable but drafted less often: their
# verifiers need complete grids or series and abstain far more than targeted lookups.
DEFAULT_TASK_WEIGHTS = {
    "table_structure_reconstruction": 0.25,
    "chart_data_reconstruction": 0.25,
    "document_structure_reconstruction": 0.25,
}


class TaskRuntimeConfig(StrictModel):
    """Bound candidate and evidence work independently of taxonomy size."""

    catalog_version: Literal["7.0"] = "7.0"
    candidate_limit: Annotated[int, Field(ge=1, le=8)] = 8
    max_scopes: Annotated[int, Field(ge=1, le=8)] = 4
    max_observations_per_scope: Annotated[int, Field(ge=1, le=50)] = 20
    evidence_max_tokens: Annotated[int, Field(ge=512, le=16384)] = 4096
    evidence_format: Literal["keyed", "array", "compact"] = "keyed"
    attribute_recheck_enabled: bool = False
    max_candidate_attempts: Literal[1, 2] = 1
    question_operation_guidance: Literal["baseline", "object_identification_v1"] = "baseline"
    fact_novelty_enabled: bool = False
    initial_binding_batch_size: Literal[2, 8] = 8
    binding_max_tokens: Annotated[int, Field(ge=512, le=16384)] = 4096
    answer_max_tokens: Annotated[int, Field(ge=256, le=8192)] = 1024
    profiles: tuple[Literal["normal", "limitation", "false_premise"], ...] = ("normal",)
    enabled_extensions: tuple[str, ...] = ()
    calibration_manifest: Path | None = None
    decision_routing: DecisionRoutingConfig = DecisionRoutingConfig()
    planner: Literal["scoped", "direct"] = "scoped"
    draft_count: Annotated[int, Field(ge=1, le=4)] = 3
    draft_max_tokens: Annotated[int, Field(ge=256, le=4096)] = 1024
    extra_draft_calls_per_turn: Literal[0, 1] = 1
    profile_max_tokens: Annotated[int, Field(ge=128, le=1024)] = 256
    family_targets: Literal["uniform"] | dict[str, Annotated[float, Field(gt=0)]] = "uniform"
    task_weights: dict[str, Annotated[float, Field(gt=0, le=1)]] = Field(
        default_factory=lambda: dict(DEFAULT_TASK_WEIGHTS)
    )

    @model_validator(mode="after")
    def validate_admission_settings(self) -> TaskRuntimeConfig:
        """Fail closed on unsupported extensions and repeated profile settings."""
        from pixelogue.catalog import task_catalog

        catalog = task_catalog()
        if isinstance(self.family_targets, dict) and not set(self.family_targets) <= set(
            catalog.families
        ):
            raise ValueError("Unknown task family in family_targets")
        if not set(self.task_weights) <= {task.id for task in catalog.tasks}:
            raise ValueError("Unknown task in task_weights")
        if self.planner == "direct":
            scoped_only = {
                "evidence_format": self.evidence_format != "keyed",
                "attribute_recheck_enabled": self.attribute_recheck_enabled,
                "max_candidate_attempts": self.max_candidate_attempts != 1,
                "question_operation_guidance": self.question_operation_guidance != "baseline",
                "fact_novelty_enabled": self.fact_novelty_enabled,
                "initial_binding_batch_size": self.initial_binding_batch_size != 8,
                "decision_routing": self.decision_routing.evidence_enabled
                or self.decision_routing.binding_enabled,
            }
            if changed := sorted(name for name, used in scoped_only.items() if used):
                raise ValueError(
                    f"Scoped-planner settings do not apply to direct drafting: {changed}"
                )
            if self.profiles != ("normal",):
                raise ValueError("Direct drafting supports only the normal profile")
        if self.decision_routing.evidence_enabled and self.attribute_recheck_enabled:
            raise ValueError(
                "Decision evidence routing cannot mix with generator-only attribute rechecks"
            )
        if not self.profiles or len(self.profiles) != len(set(self.profiles)):
            raise ValueError("At least one unique task profile is required")
        if len(self.enabled_extensions) != len(set(self.enabled_extensions)):
            raise ValueError("Repeated specialized extension")
        if not set(self.enabled_extensions) <= SPECIALIST_TASK_IDS:
            raise ValueError("Unknown specialized extension")
        return self


class StudentViewConfig(StrictModel):
    """Training-side image processor lock kept independent from teacher models."""

    processor_repo_id: Literal["Qwen/Qwen3-VL-8B-Instruct"] = "Qwen/Qwen3-VL-8B-Instruct"
    processor_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")] = (
        "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    )
    min_pixels: Annotated[int, Field(ge=1)] = 128 * 128
    max_pixels: Annotated[int, Field(ge=1)] = 2048 * 2048

    @model_validator(mode="after")
    def validate_pixel_window(self) -> StudentViewConfig:
        """Require a non-empty supported pixel window."""
        if self.max_pixels < self.min_pixels:
            raise ValueError("student max_pixels must be at least min_pixels")
        return self


class EvaluationConfig(StrictModel):
    """Choose joint review or the legacy decomposed rubric."""

    mode: Literal["holistic", "detailed"] = "holistic"
    retain_accepted_prefix: bool = True
    question_gate_label_policy: Literal["strict", "same_contract"] = "strict"
    judge_views: Literal["crop", "full_and_crop"] = "crop"
    holistic_tiebreak: Literal["none", "full_view"] = "none"
    repair_once: bool = False

    @model_validator(mode="after")
    def validate_holistic_options(self) -> EvaluationConfig:
        """Keep v8 judge rules out of the legacy decomposed evaluator."""
        changed = (
            self.question_gate_label_policy != "strict"
            or self.judge_views != "crop"
            or self.holistic_tiebreak != "none"
            or self.repair_once
        )
        if self.mode == "detailed" and changed:
            raise ValueError("v8 judge options require evaluation.mode=holistic")
        return self


class PixelogueConfig(StrictModel):
    """Complete, immutable configuration for one Pixelogue run."""

    version: Literal["1.0"] = "1.0"
    profile: Literal["standard", "pilot", "ablation"] = "standard"
    seed: int = 20260915
    data: DataConfig = DataConfig()
    models: ModelConfig = ModelConfig()
    storage: StorageConfig = StorageConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    student_view: StudentViewConfig = StudentViewConfig()
    tasks: TaskRuntimeConfig = TaskRuntimeConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    @model_validator(mode="after")
    def validate_profile(self) -> PixelogueConfig:
        """Keep pilot outputs outside the standard release path."""
        if self.profile == "pilot" and not self.data.pilot:
            raise ValueError("pilot profile requires data.pilot=true")
        if self.profile == "standard" and self.data.pilot:
            raise ValueError("standard profile cannot enable pilot mode")
        generator_repos = (self.models.generator_a.repo_id, self.models.generator_b.repo_id)
        if self.tasks.planner == "direct" and self.evaluation.mode != "holistic":
            raise ValueError("Direct drafting requires evaluation.mode=holistic")
        if self.profile == "standard" and generator_repos != PRIMARY_GENERATOR_REPOS:
            raise ValueError("standard profile requires the primary Qwen3.8/Gemma model pair")
        if self.runtime.json_whitespace_max_chars is not None:
            for endpoint in (self.models.generator_a, self.models.generator_b):
                if (
                    endpoint.serving_runtime is None
                    or not endpoint.serving_runtime.bounded_whitespace_supported
                ):
                    raise ValueError(
                        "Bounded JSON whitespace requires a verified patched xgrammar runtime "
                        f"with ordinary whitespace enabled for {endpoint.repo_id}"
                    )
        return self

    @property
    def config_hash(self) -> str:
        """Bind settings, code, schemas, prompts, locks and pinned input identity."""
        from pixelogue.calibration import CalibrationManifest
        from pixelogue.catalog import TASK_CONTRACT_VERSION, load_task_catalog
        from pixelogue.io import read_json

        calibration = (
            read_json(self.tasks.calibration_manifest, CalibrationManifest).model_dump(mode="json")
            if self.tasks.calibration_manifest is not None
            else None
        )
        package_root = Path(__file__).resolve().parent
        project_root = package_root.parents[1]
        identity_files = [
            *package_root.rglob("*.py"),
            *package_root.joinpath("resources").rglob("*.yaml"),
            *package_root.joinpath("resources").rglob("*.json"),
        ]
        identity_files.extend(
            path
            for path in (
                project_root / "uv.lock",
                project_root / "runtime/vllm/pyproject.toml",
                project_root / "runtime/vllm/uv.lock",
                project_root / "runtime/validators/pyproject.toml",
                project_root / "runtime/validators/uv.lock",
                project_root / "runtime/validators/worker.py",
                project_root / "runtime/validators/render_worker.py",
            )
            if path.is_file()
        )
        code_identity = {
            str(path.relative_to(project_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(identity_files)
        }
        image_manifest = self.data.open_images.image_ids_manifest
        pinned_input_hash = (
            hashlib.sha256(image_manifest.read_bytes()).hexdigest()
            if image_manifest is not None and image_manifest.is_file()
            else None
        )

        return canonical_hash(
            {
                "config": self.model_dump(mode="json"),
                "task_catalog": load_task_catalog(),
                "task_contract_version": TASK_CONTRACT_VERSION,
                "calibration": calibration,
                "source_and_lock_files": code_identity,
                "config_schema": self.model_json_schema(),
                "pinned_input_manifest_hash": pinned_input_hash,
            }
        )

    @property
    def language_quotas(self) -> dict[str, int]:
        """Return exact conversation quotas using largest remainder allocation."""
        return allocate_quotas(
            self.data.target_dialogues,
            {target.tag: target.weight for target in self.data.languages},
        )


def load_config(path: Path) -> PixelogueConfig:
    """Load and validate a configuration file.

    Relative paths are resolved against the configuration file's directory.

    Raises:
        ConfigurationError: If configuration is invalid or contradictory.
    """
    raw = load_yaml(path)
    try:
        config = PixelogueConfig.model_validate_json(canonical_json(raw))
    except ValidationError as error:
        raise ConfigurationError("INVALID_CONFIG", str(error)) from error
    base = path.resolve().parent
    storage_update: dict[str, object] = {}
    if not config.storage.run_root.is_absolute():
        storage_update["run_root"] = (base / config.storage.run_root).resolve()
    if config.storage.backup_root is not None and not config.storage.backup_root.is_absolute():
        storage_update["backup_root"] = (base / config.storage.backup_root).resolve()
    updated = (
        config.model_copy(update={"storage": config.storage.model_copy(update=storage_update)})
        if storage_update
        else config
    )
    calibration_manifest = updated.tasks.calibration_manifest
    if calibration_manifest is not None and not calibration_manifest.is_absolute():
        updated = updated.model_copy(
            update={
                "tasks": updated.tasks.model_copy(
                    update={"calibration_manifest": (base / calibration_manifest).resolve()}
                )
            }
        )
    manifest = updated.data.open_images.image_ids_manifest
    if manifest is None or manifest.is_absolute():
        return updated
    open_images = updated.data.open_images.model_copy(
        update={"image_ids_manifest": (base / manifest).resolve()}
    )
    return updated.model_copy(
        update={"data": updated.data.model_copy(update={"open_images": open_images})}
    )
