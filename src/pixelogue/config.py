"""Validated application configuration."""

from __future__ import annotations

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

PRIMARY_GENERATOR_REPOS = ("Qwen/Qwen3.8-27B", "google/gemma-4-31B-it")
PILOT_GENERATOR_REPOS = ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-9B")


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
    quantization: None = None
    max_model_len: Annotated[int, Field(ge=4096)] = 32768
    api_key_env: str = "PIXELLOGUE_API_KEY"

    @property
    def model_name(self) -> str:
        """Return the server-visible model name."""
        return self.served_name or self.repo_id


class ModelConfig(StrictModel):
    """Assign fixed models to instruction selection, generation, and judging."""

    selector: ModelEndpoint = ModelEndpoint(repo_id="Qwen/Qwen3.5-2B")
    selector_alternative: ModelEndpoint = ModelEndpoint(repo_id="Qwen/Qwen3.6-35B-A3B")
    active_selector: Literal["default", "alternative"] = "default"
    generator_a: ModelEndpoint = ModelEndpoint(repo_id="Qwen/Qwen3.8-27B")
    generator_b: ModelEndpoint = ModelEndpoint(repo_id="google/gemma-4-31B-it")
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
                "generators must use the primary Qwen3.8/Gemma pair or the temporary "
                "Qwen3.5-9B pilot pair"
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


class RuntimeConfig(StrictModel):
    """Inference request and retry boundaries."""

    request_timeout_seconds: Annotated[int, Field(ge=1, le=180)] = 180
    transport_max_attempts: Literal[1, 2, 3] = 3
    structured_output_max_attempts: Literal[1, 2, 3] = 2
    max_concurrent_images: Annotated[int, Field(ge=1, le=32)] = 1
    max_total_requests: Annotated[int, Field(ge=1)] = 10_000_000
    max_total_output_tokens: Annotated[int, Field(ge=1)] = 1_000_000_000
    allow_external_inference: Literal[False] = False


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

    @model_validator(mode="after")
    def validate_profile(self) -> PixelogueConfig:
        """Keep pilot outputs outside the standard release path."""
        if self.profile == "pilot" and not self.data.pilot:
            raise ValueError("pilot profile requires data.pilot=true")
        if self.profile == "standard" and self.data.pilot:
            raise ValueError("standard profile cannot enable pilot mode")
        generator_repos = (self.models.generator_a.repo_id, self.models.generator_b.repo_id)
        if self.profile == "standard" and generator_repos != PRIMARY_GENERATOR_REPOS:
            raise ValueError("standard profile requires the primary Qwen3.8/Gemma model pair")
        return self

    @property
    def config_hash(self) -> str:
        """Return a stable identity for all effective settings."""
        return canonical_hash(self.model_dump(mode="json"))

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
    if not storage_update:
        updated = config
    else:
        updated = config.model_copy(
            update={"storage": config.storage.model_copy(update=storage_update)}
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
