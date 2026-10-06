from pathlib import Path

import pytest
from pydantic import ValidationError

from pixelogue.config import PixelogueConfig, allocate_quotas, load_config
from pixelogue.errors import ConfigurationError, ExternalInputError
from pixelogue.planner import exact_schedule
from pixelogue.serialization import strict_json_object


def test_example_profiles_are_valid_and_separate() -> None:
    pilot = load_config(Path("configs/pilot.yaml"))
    standard = load_config(Path("configs/standard.yaml"))

    assert pilot.profile == "pilot" and pilot.data.pilot
    assert standard.profile == "standard" and not standard.data.pilot
    assert standard.data.target_dialogues == 30_000
    assert pilot.models.router.repo_id == "Qwen/Qwen3.5-2B"
    assert (standard.models.generator_a.repo_id, standard.models.generator_b.repo_id) == (
        "Qwen/Qwen3.8-27B-FP8",
        "google/gemma-4-31B-it-qat-w4a16-ct",
    )
    assert standard.models.generator_a.quantization == "fp8"
    assert standard.models.generator_b.quantization == "compressed-tensors"
    assert (pilot.models.generator_a.repo_id, pilot.models.generator_b.repo_id) == (
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-9B",
    )
    assert all(
        endpoint.revision and endpoint.processor_revision
        for endpoint in (
            pilot.models.router,
            pilot.models.generator_a,
            pilot.models.generator_b,
        )
    )


def test_language_allocation_is_exact_and_deterministic() -> None:
    assert allocate_quotas(7, {"zh-Hans": "1", "en": "1", "ja": "1"}) == {
        "zh-Hans": 2,
        "en": 3,
        "ja": 2,
    }


@pytest.mark.parametrize("weight", [True, 0, -1, 0.5, "nan"])
def test_invalid_language_weights_are_rejected(weight: object) -> None:
    with pytest.raises((ConfigurationError, ValidationError, ValueError)):
        PixelogueConfig.model_validate({"data": {"languages": [{"tag": "en", "weight": weight}]}})


def test_duplicate_yaml_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text('version: "1.0"\nversion: "1.0"\n', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Duplicate YAML key"):
        load_config(path)


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ('{"a": 1, "a": 2}', "DUPLICATE_JSON_KEY"),
        ('{"a": NaN}', "NON_FINITE_JSON_NUMBER"),
        ("[]", "JSON_ROOT_NOT_OBJECT"),
        ('{"a": 1} trailing', "INVALID_JSON"),
    ],
)
def test_strict_json_rejects_ambiguous_inputs(payload: str, reason: str) -> None:
    with pytest.raises(ExternalInputError) as caught:
        strict_json_object(payload)
    assert caught.value.reason == reason


def test_unknown_model_and_quantization_are_rejected() -> None:
    base = load_config(Path("configs/pilot.yaml")).model_dump(mode="json")
    base["models"]["generator_a"]["repo_id"] = "unknown/model"
    with pytest.raises(ValidationError):
        PixelogueConfig.model_validate(base)

    base = load_config(Path("configs/pilot.yaml")).model_dump(mode="json")
    base["models"]["router"]["quantization"] = "int4"
    with pytest.raises(ValidationError):
        PixelogueConfig.model_validate(base)


def test_generator_pairs_cannot_mix_standard_and_pilot_models() -> None:
    base = load_config(Path("configs/pilot.yaml")).model_dump(mode="json")
    base["models"]["generator_a"]["repo_id"] = "Qwen/Qwen3.8-27B-FP8"
    with pytest.raises(ValidationError, match="quantized or bf16 Qwen3.8/Gemma 4 pair"):
        PixelogueConfig.model_validate(base)


def _bf16_standard() -> dict:
    base = load_config(Path("configs/standard.yaml")).model_dump()
    for role, repo in (
        ("generator_a", "Qwen/Qwen3.8-27B"),
        ("generator_b", "google/gemma-4-31B-it"),
    ):
        base["models"][role].update(repo_id=repo, quantization=None)
    return base


def test_standard_profile_accepts_the_unquantized_pair_and_a_qwen_router() -> None:
    base = _bf16_standard()
    config = PixelogueConfig.model_validate(base)
    assert config.models.generator_b.quantization is None
    base["models"]["router"] = {**base["models"]["generator_a"]}
    assert PixelogueConfig.model_validate(base).models.router.repo_id == "Qwen/Qwen3.8-27B"


@pytest.mark.parametrize(
    ("role", "change"),
    [
        ("generator_a", {"quantization": "fp8"}),
        ("generator_b", {"repo_id": "google/gemma-4-31B-it-qat-w4a16-ct"}),
        ("router", {"repo_id": "Qwen/Qwen3.8-27B-FP8"}),
        ("router", {"repo_id": "Qwen/Qwen3.8-27B", "quantization": "fp8"}),
    ],
)
def test_unquantized_pair_rejects_quantization_and_mixed_checkpoints(role, change) -> None:
    base = _bf16_standard()
    base["models"][role].update(change)
    with pytest.raises(ValidationError):
        PixelogueConfig.model_validate(base)


def test_generator_quantization_must_match_pinned_method() -> None:
    base = load_config(Path("configs/standard.yaml")).model_dump(mode="json")
    base["models"]["generator_a"]["quantization"] = "compressed-tensors"
    with pytest.raises(ValidationError, match="pinned quantization method"):
        PixelogueConfig.model_validate(base)


def test_standard_profile_rejects_temporary_pilot_pair() -> None:
    base = load_config(Path("configs/pilot.yaml")).model_dump()
    base["profile"] = "standard"
    base["data"]["pilot"] = False
    with pytest.raises(ValidationError, match="standard profile requires"):
        PixelogueConfig.model_validate(base)


def test_exact_schedule_preserves_every_quota() -> None:
    schedule = exact_schedule(10, {"generator_a": 1, "generator_b": 1}, seed=3, namespace="test")
    assert schedule.count("generator_a") == 5
    assert schedule.count("generator_b") == 5
    assert schedule == exact_schedule(
        10,
        {"generator_a": 1, "generator_b": 1},
        seed=3,
        namespace="test",
    )
