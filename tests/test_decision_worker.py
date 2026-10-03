from __future__ import annotations

import hashlib
import io
import json
import selectors
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from pixelogue import decision_worker
from pixelogue.decision import DecisionProbability, DecisionRequest, DecisionResult
from pixelogue.decision_serving import SubprocessOmniDecisionClient
from pixelogue.errors import ExecutionError
from pixelogue.task_evidence import ImageRegion

REVISION = "a" * 40
REPOSITORY = "akhilaaa3/Jev-Omni"


def request_for(tmp_path: Path) -> DecisionRequest:
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "red").save(image)
    return DecisionRequest(
        request_id="request-1",
        trial_id="trial-1",
        image_id="image-1",
        image_path=str(image),
        image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        view_id="full",
        region=ImageRegion(left=0.0, top=0.0, right=1.0, bottom=1.0),
        question="Is the target visible?",
        public_context=(),
    )


def result_for(request: DecisionRequest) -> DecisionResult:
    return DecisionResult(
        request_id=request.request_id,
        request_hash=request.identity(REPOSITORY, REVISION),
        model_repository=REPOSITORY,
        model_revision=REVISION,
        verdict="UNKNOWN",
        probabilities=(
            DecisionProbability(label="MET", probability=0.1),
            DecisionProbability(label="NOT_MET", probability=0.1),
            DecisionProbability(label="UNKNOWN", probability=0.8),
        ),
        elapsed_seconds=0.01,
    )


class WorkerClient:
    def __init__(self, failure: Exception | None = None) -> None:
        self.requests = []
        self.failure = failure

    def decide(self, request: DecisionRequest) -> DecisionResult:
        self.requests.append(request)
        print("native progress")
        if self.failure is not None:
            raise self.failure
        return result_for(request)


def test_worker_rejects_invalid_json_then_handles_the_next_request(
    tmp_path: Path, capsys: Any
) -> None:
    request = request_for(tmp_path)
    source = io.StringIO(
        '{"request_id":"duplicate","request_id":"other"}\n' + request.model_dump_json() + "\n"
    )
    output = io.StringIO()
    client = WorkerClient()
    assert decision_worker.serve_decisions(client, source, output) == 0
    frames = [json.loads(line) for line in output.getvalue().splitlines()]
    assert frames[0]["status"] == "error"
    assert frames[0]["reason"] == "DECISION_REQUEST_INVALID"
    result = DecisionResult.model_validate_json(json.dumps(frames[1]["result"]))
    result.validate_request(request)
    assert result.verdict == "UNKNOWN"
    assert len(client.requests) == 1
    assert "native progress" in capsys.readouterr().err


def test_worker_keeps_typed_failure_separate_from_a_verdict(tmp_path: Path) -> None:
    output = io.StringIO()
    source = io.StringIO(request_for(tmp_path).model_dump_json() + "\n")
    code = decision_worker.serve_decisions(
        WorkerClient(ExecutionError("DECISION_OUTPUT_INVALID", "bad scores")), source, output
    )
    assert code == 0
    assert json.loads(output.getvalue()) == {
        "status": "error",
        "request_id": "request-1",
        "reason": "DECISION_OUTPUT_INVALID",
        "message": "bad scores",
    }


def test_worker_stops_after_an_unexpected_native_failure(tmp_path: Path) -> None:
    request = request_for(tmp_path)
    output = io.StringIO()
    client = WorkerClient(RuntimeError("device failure"))
    assert (
        decision_worker.serve_decisions(
            client, io.StringIO((request.model_dump_json() + "\n") * 2), output
        )
        == 7
    )
    assert len(client.requests) == 1
    assert json.loads(output.getvalue())["reason"] == "DECISION_WORKER_INFERENCE_FAILED"


def test_worker_requires_explicit_gpu_before_loading(monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert (
        decision_worker.main(["--snapshot-path", "/unused/" + REVISION, "--revision", REVISION])
        == 2
    )
    frame = json.loads(capsys.readouterr().out)
    assert frame["reason"] == "DECISION_GPU_NOT_SELECTED"


def start_fake_worker(
    tmp_path: Path, monkeypatch: Any, request: DecisionRequest, *, mode: str = "normal"
) -> SubprocessOmniDecisionClient:
    ready = {
        "status": "ready",
        "model_repository": REPOSITORY,
        "model_revision": REVISION,
        "model_load_seconds": 0.05,
    }
    result = {"status": "result", "result": result_for(request).model_dump(mode="json")}
    script = tmp_path / "fake-worker.py"
    script.write_text(
        "import json,sys,time\n"
        f"ready={ready!r}\nresult={result!r}\nmode={mode!r}\n"
        "print(json.dumps(ready),flush=True)\n"
        "for line in sys.stdin:\n"
        " request=json.loads(line)\n"
        " if mode=='exit': sys.exit(4)\n"
        " if mode=='timeout': time.sleep(10)\n"
        " if mode=='mismatch': result['result']['request_id']='wrong-request'\n"
        " if mode=='error':\n"
        "  print(json.dumps({'status':'error','request_id':request['request_id'],'reason':'DECISION_OUTPUT_INVALID','message':'bad native output'}),flush=True)\n"
        " else: print(json.dumps(result),flush=True)\n"
    )
    actual_popen = subprocess.Popen

    def fake_popen(command: list[str], **kwargs: Any) -> Any:
        assert command[1:4] == ["-u", "-m", "pixelogue.decision_worker"]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-test-explicit"
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        return actual_popen([sys.executable, "-u", str(script)], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return SubprocessOmniDecisionClient(
        Path(sys.executable).resolve(),
        (tmp_path / REVISION).resolve(),
        REVISION,
        env={"CUDA_VISIBLE_DEVICES": "GPU-test-explicit"},
        stderr_path=tmp_path / "worker.log",
        startup_timeout_seconds=5.0,
        request_timeout_seconds=0.2,
    )


def test_subprocess_worker_is_reused_and_close_is_idempotent(
    tmp_path: Path, monkeypatch: Any
) -> None:
    request = request_for(tmp_path)
    adapter = start_fake_worker(tmp_path, monkeypatch, request)
    try:
        for _ in range(2):
            adapter.decide(request).validate_request(request)
        assert adapter.model_load_seconds == 0.05
    finally:
        adapter.close()
        adapter.close()
    with pytest.raises(ExecutionError) as caught:
        adapter.decide(request)
    assert caught.value.reason == "DECISION_WORKER_EXITED"


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("exit", "DECISION_WORKER_EXITED"),
        ("timeout", "DECISION_WORKER_TIMEOUT"),
        ("mismatch", "DECISION_WORKER_PROTOCOL"),
    ],
)
def test_failed_process_or_mismatched_response_is_not_adopted(
    tmp_path: Path, monkeypatch: Any, mode: str, reason: str
) -> None:
    request = request_for(tmp_path)
    adapter = start_fake_worker(tmp_path, monkeypatch, request, mode=mode)
    try:
        with pytest.raises(ExecutionError) as caught:
            adapter.decide(request)
        assert caught.value.reason == reason
    finally:
        adapter.close()


def test_native_error_does_not_force_automatic_worker_restart(
    tmp_path: Path, monkeypatch: Any
) -> None:
    request = request_for(tmp_path)
    adapter = start_fake_worker(tmp_path, monkeypatch, request, mode="error")
    try:
        for _ in range(2):
            with pytest.raises(ExecutionError) as caught:
                adapter.decide(request)
            assert caught.value.reason == "DECISION_OUTPUT_INVALID"
    finally:
        adapter.close()


def capture_started_child(monkeypatch: Any) -> tuple[list[Any], list[Any]]:
    processes = []
    logs = []
    actual_popen = subprocess.Popen

    def fake_popen(command: list[str], **kwargs: Any) -> Any:
        assert command[1:4] == ["-u", "-m", "pixelogue.decision_worker"]
        logs.append(kwargs["stderr"])
        process = actual_popen(
            [sys.executable, "-u", "-c", "import time; time.sleep(60)"], **kwargs
        )
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return processes, logs


def assert_child_resources_were_closed(processes: list[Any], logs: list[Any]) -> None:
    assert len(processes) == len(logs) == 1
    process = processes[0]
    stopped = process.poll() is not None
    pipes_closed = process.stdin.closed and process.stdout.closed
    log_closed = logs[0].closed
    # Clean up even when this regression is exercised against the broken code.
    if not stopped:
        process.terminate()
        process.wait(timeout=5)
    for stream in (process.stdin, process.stdout, logs[0]):
        stream.close()
    assert stopped, "Startup failure left the owned worker alive"
    assert pipes_closed, "Startup failure left the owned worker pipes open"
    assert log_closed, "Startup failure left the worker log open"


def construct_worker(tmp_path: Path) -> SubprocessOmniDecisionClient:
    return SubprocessOmniDecisionClient(
        Path(sys.executable).resolve(),
        (tmp_path / REVISION).resolve(),
        REVISION,
        env={"CUDA_VISIBLE_DEVICES": "GPU-test-explicit"},
        stderr_path=tmp_path / "startup-worker.log",
    )


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), SystemExit(23)])
def test_startup_interrupt_closes_owned_worker_before_propagating(
    tmp_path: Path, monkeypatch: Any, failure: BaseException
) -> None:
    processes, logs = capture_started_child(monkeypatch)

    def interrupt_read(self: Any, timeout_seconds: float) -> Any:
        raise failure

    monkeypatch.setattr(SubprocessOmniDecisionClient, "_read_frame", interrupt_read)
    with pytest.raises(type(failure)) as caught:
        construct_worker(tmp_path)
    assert caught.value is failure
    assert_child_resources_were_closed(processes, logs)


def test_selector_registration_failure_closes_owned_worker(
    tmp_path: Path, monkeypatch: Any
) -> None:
    processes, logs = capture_started_child(monkeypatch)
    failure = RuntimeError("selector registration failed")

    def fail_register(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise failure

    monkeypatch.setattr(selectors.DefaultSelector, "register", fail_register)
    with pytest.raises(RuntimeError) as caught:
        construct_worker(tmp_path)
    assert caught.value is failure
    assert_child_resources_were_closed(processes, logs)


def test_selector_construction_failure_closes_owned_worker(
    tmp_path: Path, monkeypatch: Any
) -> None:
    processes, logs = capture_started_child(monkeypatch)
    failure = RuntimeError("selector construction failed")

    def fail_selector() -> Any:
        raise failure

    monkeypatch.setattr(selectors, "DefaultSelector", fail_selector)
    with pytest.raises(RuntimeError) as caught:
        construct_worker(tmp_path)
    assert caught.value is failure
    assert_child_resources_were_closed(processes, logs)
