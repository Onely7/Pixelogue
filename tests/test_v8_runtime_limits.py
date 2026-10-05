from __future__ import annotations

import pytest

from pixelogue.config import ModelEndpoint, RepetitionDetectionConfig, RuntimeConfig
from pixelogue.serving import VllmClient


def client(runtime: RuntimeConfig) -> VllmClient:
    return VllmClient(ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"), runtime, run_id="limits")


def test_request_timeout_scales_with_output_budget() -> None:
    fixed = client(RuntimeConfig(request_timeout_seconds=180))
    assert fixed._timeout(8192).read == 180
    scaled = client(
        RuntimeConfig(request_timeout_seconds=180, timeout_seconds_per_1k_output_tokens=45)
    )
    assert scaled._timeout(1024).read == 180
    assert scaled._timeout(8192).read == pytest.approx(60 + 8192 * 45 / 1000)
    assert scaled._timeout(32768).read == 900


def test_repetition_guard_patterns_cover_only_structured_extraction() -> None:
    guard = RepetitionDetectionConfig(stages=("*_source", "question_draft", "image_profile"))
    assert guard.applies_to("table_source") and guard.applies_to("question_draft")
    assert not guard.applies_to("answer_generation")
    assert not guard.applies_to("holistic_review")
    for pattern in ("answer_generation", "*", "question_*", "holistic_review"):
        with pytest.raises(ValueError):
            RepetitionDetectionConfig(stages=(pattern,))
