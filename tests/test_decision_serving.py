from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image

from pixelogue.decision import DecisionRequest
from pixelogue.decision_serving import (
    JevHttpDecisionClient,
    LocalOmniDecisionClient,
    ThreadedAsyncDecisionClient,
)
from pixelogue.errors import ExecutionError
from pixelogue.serialization import strict_json_object
from pixelogue.task_evidence import ImageRegion

REVISION = "a" * 40


def request_for(tmp_path: Path) -> DecisionRequest:
    image = tmp_path / "view.png"
    Image.new("RGB", (8, 8), "red").save(image)
    return DecisionRequest(
        request_id="visual-check",
        trial_id="trial",
        image_id="image",
        image_path=str(image),
        image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        view_id="original-view",
        region=ImageRegion(left=0.0, top=0.0, right=1.0, bottom=1.0),
        question="Is the target visible?",
        public_context=("The target is the central surface.",),
    )


def native_response(**changes: object) -> dict[str, object]:
    return {
        "kind": "choice",
        "options": ["MET", "NOT_MET", "UNKNOWN"],
        "probabilities": [0.7, 0.2, 0.1],
        "choice_index": 0,
        "choice": "MET",
        "protocol": "jev27-bare-v1",
        "model": "autotrust/JEV-27B-VL",
        "num_model_requests": 1,
        **changes,
    }


def http_adapter(handler: Any) -> JevHttpDecisionClient:
    return JevHttpDecisionClient(
        "http://localhost:18104/v1",
        "autotrust/JEV-27B-VL",
        REVISION,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_native_http_request_has_three_choices_and_no_generative_fallback(tmp_path: Path) -> None:
    observed = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(strict_json_object(request.content))
        assert str(request.url) == "http://localhost:18104/v1/decide"
        return httpx.Response(200, json=native_response())

    request = request_for(tmp_path)
    result = http_adapter(handler).decide(request)
    result.validate_request(request)
    assert result.verdict == "MET"
    assert result.prompt_tokens is None
    assert result.completion_tokens is None
    assert result.num_model_requests == 1
    assert observed[0]["thinking"] == "off"
    assert observed[0]["options"] == ["MET", "NOT_MET", "UNKNOWN"]
    assert observed[0]["return_reasoning"] is False
    assert observed[0]["state"][1]["image"].startswith("data:image/png;base64,")
    assert json.loads(observed[0]["state"][0])["scope_region"] == request.region.model_dump()


@pytest.mark.parametrize(
    "changes",
    [
        {"options": ["MET", "UNKNOWN", "NOT_MET"]},
        {"choice": "NOT_MET"},
        {"choice_index": True},
        {"probabilities": [7.0, 2.0, 1.0]},
        {"probabilities": [0.7, 0.2]},
        {"model": "unconfigured/model"},
        {"thinking": {"used": True}},
        {"protocol": "unverified-head-v1"},
    ],
    ids=[
        "option-order",
        "choice-index",
        "bool-index",
        "logits",
        "missing-score",
        "model",
        "thinking",
        "protocol",
    ],
)
def test_native_http_output_is_strictly_rejected(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    adapter = http_adapter(lambda _: httpx.Response(200, json=native_response(**changes)))
    with pytest.raises(ExecutionError) as caught:
        adapter.decide(request_for(tmp_path))
    assert caught.value.reason == "DECISION_OUTPUT_INVALID"


def test_duplicate_native_json_key_is_rejected(tmp_path: Path) -> None:
    payload = json.dumps(native_response()).replace(
        '"choice": "MET"', '"choice": "MET", "choice": "NOT_MET"'
    )
    adapter = http_adapter(lambda _: httpx.Response(200, text=payload))
    with pytest.raises(ExecutionError) as caught:
        adapter.decide(request_for(tmp_path))
    assert caught.value.reason == "DECISION_OUTPUT_INVALID"


def test_changed_image_is_rejected_before_spending_a_native_call(tmp_path: Path) -> None:
    calls = []
    adapter = http_adapter(
        lambda request: calls.append(request) or httpx.Response(200, json=native_response())
    )
    request = request_for(tmp_path)
    Path(request.image_path).write_bytes(b"changed")
    with pytest.raises(ExecutionError) as caught:
        adapter.decide(request)
    assert caught.value.reason == "DECISION_IMAGE_CHANGED"
    assert calls == []


def test_http_failure_does_not_call_another_endpoint(tmp_path: Path) -> None:
    calls = []
    adapter = http_adapter(lambda request: calls.append(request) or httpx.Response(503))
    with pytest.raises(ExecutionError) as caught:
        adapter.decide(request_for(tmp_path))
    assert caught.value.reason == "DECISION_TRANSPORT_FAILED"
    assert len(calls) == 1


def test_native_http_external_host_requires_explicit_configuration() -> None:
    with pytest.raises(ExecutionError) as caught:
        JevHttpDecisionClient("http://172.24.21.2:18104/v1", "autotrust/JEV-27B-VL", REVISION)
    assert caught.value.reason == "EXTERNAL_INFERENCE_FORBIDDEN"
    adapter = JevHttpDecisionClient(
        "http://172.24.21.2:18104/v1",
        "autotrust/JEV-27B-VL",
        REVISION,
        allow_external_inference=True,
    )
    adapter.close()


class CapturingPredictor:
    def __init__(self) -> None:
        self.active = 0
        self.maximum_active = 0
        self.calls: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def predict(self, **kwargs: Any) -> dict[str, Any]:
        with self.lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            self.calls.append({**kwargs, "image_bytes": Path(kwargs["media"]).read_bytes()})
        time.sleep(0.01)
        with self.lock:
            self.active -= 1
        return {
            "prediction": "UNKNOWN",
            "prediction_index": 2,
            "confidence": 0.8,
            "probabilities": {"MET": 0.1, "NOT_MET": 0.1, "UNKNOWN": 0.8},
        }


def omni_adapter(tmp_path: Path, predictor: CapturingPredictor) -> LocalOmniDecisionClient:
    return LocalOmniDecisionClient(tmp_path / REVISION, REVISION, predictor=predictor)


def test_omni_serializes_shared_capture_and_passes_immutable_image_bytes(tmp_path: Path) -> None:
    predictor = CapturingPredictor()
    adapter = omni_adapter(tmp_path, predictor)
    request = request_for(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(adapter.decide, [request] * 4))
    assert predictor.maximum_active == 1
    assert len(predictor.calls) == 4
    assert all(result.verdict == "UNKNOWN" for result in results)
    assert all(
        hashlib.sha256(call["image_bytes"]).hexdigest() == request.image_sha256
        and not Path(call["media"]).exists()
        for call in predictor.calls
    )
    assert all(call["modality"] == "image" for call in predictor.calls)
    assert all(result.prompt_tokens is None for result in results)


def test_omni_changed_image_prevents_predictor_call(tmp_path: Path) -> None:
    predictor = CapturingPredictor()
    request = request_for(tmp_path)
    Path(request.image_path).write_bytes(b"changed")
    with pytest.raises(ExecutionError):
        omni_adapter(tmp_path, predictor).decide(request)
    assert predictor.calls == []


@pytest.mark.asyncio
async def test_async_adapter_preserves_native_result(tmp_path: Path) -> None:
    predictor = CapturingPredictor()
    request = request_for(tmp_path)
    result = await ThreadedAsyncDecisionClient(omni_adapter(tmp_path, predictor)).decide(request)
    result.validate_request(request)
    assert result.verdict == "UNKNOWN"
    assert result.probability_map["UNKNOWN"] == 0.8
