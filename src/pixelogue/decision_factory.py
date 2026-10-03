"""Explicit deployment boundary for opt-in native decision clients."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pixelogue.config import PixelogueConfig
from pixelogue.decision import DecisionClient
from pixelogue.errors import ConfigurationError


def create_decision_client(config: PixelogueConfig, run_directory: Path) -> DecisionClient | None:
    """Create only the selected classifier, using the operator's endpoint or GPU UUID.

    No client is created when both experimental routes are disabled. Native Omni
    uses a long-lived worker in an explicitly supplied separate Python runtime;
    the application environment does not import its GPU libraries. This function
    neither chooses devices nor changes model, dtype or quantization.

    Raises:
        ConfigurationError: If an enabled deployment lacks its explicit inputs.
    """
    routing = config.tasks.decision_routing
    if not (routing.evidence_enabled or routing.binding_enabled):
        return None
    if routing.model_repository is None or routing.model_revision is None:
        raise ConfigurationError(
            "DECISION_MODEL_UNCONFIGURED", "A pinned classifier identity is required"
        )

    from pixelogue.decision_serving import JevHttpDecisionClient, SubprocessOmniDecisionClient

    if routing.model_repository == "autotrust/JEV-27B-VL":
        if routing.base_url is None:
            raise ConfigurationError(
                "DECISION_ENDPOINT_REQUIRED", "JEV routing needs its explicit native endpoint"
            )
        if routing.runtime_python is not None or routing.snapshot_path is not None:
            raise ConfigurationError(
                "DECISION_DEPLOYMENT_MISMATCH", "JEV HTTP routing does not use Omni worker paths"
            )
        return JevHttpDecisionClient(
            str(routing.base_url),
            routing.model_repository,
            routing.model_revision,
            timeout_seconds=float(config.runtime.request_timeout_seconds),
            allow_external_inference=config.runtime.allow_external_inference,
        )
    if routing.base_url is not None:
        raise ConfigurationError(
            "DECISION_DEPLOYMENT_MISMATCH", "Local Omni routing cannot use a JEV HTTP endpoint"
        )
    if routing.runtime_python is None or routing.snapshot_path is None:
        raise ConfigurationError(
            "DECISION_WORKER_INPUTS_REQUIRED",
            "Omni routing needs separate runtime_python and pinned snapshot_path",
        )
    runtime_python, snapshot = routing.runtime_python, routing.snapshot_path
    if (
        not runtime_python.is_absolute()
        or not runtime_python.is_file()
        or not os.access(runtime_python, os.X_OK)
    ):
        raise ConfigurationError(
            "DECISION_RUNTIME_INVALID", "Omni runtime_python must name an absolute executable"
        )
    if (
        not snapshot.is_absolute()
        or not snapshot.is_dir()
        or snapshot.name != routing.model_revision
    ):
        raise ConfigurationError(
            "DECISION_SNAPSHOT_INVALID",
            "Omni snapshot_path must name its absolute pinned revision directory",
        )
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu.startswith("GPU-") or "," in gpu:
        raise ConfigurationError(
            "DECISION_GPU_REQUIRED",
            "Select one inspected physical GPU UUID in CUDA_VISIBLE_DEVICES before starting Omni",
        )
    return SubprocessOmniDecisionClient(
        runtime_python,
        snapshot,
        routing.model_revision,
        env={"CUDA_VISIBLE_DEVICES": gpu},
        stderr_path=run_directory / "decision-worker.log",
        startup_timeout_seconds=float(routing.startup_timeout_seconds),
        request_timeout_seconds=float(config.runtime.request_timeout_seconds),
    )


@contextmanager
def decision_client_context(
    config: PixelogueConfig, run_directory: Path
) -> Iterator[DecisionClient | None]:
    """Close the owned client after success, failure or an interrupted synthesis job."""
    client = create_decision_client(config, run_directory)
    try:
        yield client
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            close()
