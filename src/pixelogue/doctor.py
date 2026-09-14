"""Read-only environment and model-serving diagnostics."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from itertools import combinations
from typing import Literal

from pydantic import Field

from pixelogue.config import ModelEndpoint, PixelogueConfig, StrictModel
from pixelogue.serving import VllmClient

PARAMETER_COUNTS = {
    "Qwen/Qwen3.5-2B": 2_000_000_000,
    "Qwen/Qwen3.6-35B-A3B": 35_951_822_704,
    "Qwen/Qwen3.5-9B": 9_000_000_000,
    "Qwen/Qwen3.8-27B": 27_000_000_000,
    "google/gemma-4-31B-it": 31_000_000_000,
}


class GpuDevice(StrictModel):
    """One GPU's scheduling-relevant state at inspection time."""

    index: int = Field(ge=0)
    name: str
    total_mib: int = Field(ge=0)
    used_mib: int = Field(ge=0)
    free_mib: int = Field(ge=0)
    utilization_percent: int = Field(ge=0, le=100)
    idle: bool


class ModelCheck(StrictModel):
    """Static lock, memory, and optional server result for one configured role."""

    role: str
    repo_id: str
    required_for_current_run: bool
    revision_pinned: bool
    tensor_parallel_size: int
    gpu_memory_utilization: float
    estimated_weight_mib: int
    assigned_gpu_indices: tuple[int, ...]
    enough_idle_memory: bool
    server_status: Literal["NOT_CHECKED", "READY", "UNAVAILABLE", "WRONG_MODEL"]
    detail: str


class DoctorReport(StrictModel):
    """Complete read-only diagnosis without claiming that inference ran."""

    gpus: tuple[GpuDevice, ...]
    models: tuple[ModelCheck, ...]
    local_wal_root: str
    ready: bool


def inspect_gpus() -> tuple[GpuDevice, ...]:
    """Return all NVIDIA devices, or an empty tuple when `nvidia-smi` is absent."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return ()
    command = [
        executable,
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return ()
    devices: list[GpuDevice] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            continue
        index, name, total, used, free, utilization = fields
        devices.append(
            GpuDevice(
                index=int(index),
                name=name,
                total_mib=int(total),
                used_mib=int(used),
                free_mib=int(free),
                utilization_percent=int(utilization),
                idle=int(used) < 1024 and int(utilization) == 0,
            )
        )
    return tuple(devices)


def diagnose(config: PixelogueConfig, *, check_servers: bool = False) -> DoctorReport:
    """Inspect revision locks, idle GPU capacity, and optional local endpoints."""
    gpus = inspect_gpus()
    endpoints: Sequence[tuple[str, ModelEndpoint, bool]] = (
        (
            "selector",
            config.models.active_selector_endpoint,
            True,
        ),
        (
            "selector_alternative",
            config.models.selector_alternative
            if config.models.active_selector == "default"
            else config.models.selector,
            False,
        ),
        ("generator_a", config.models.generator_a, True),
        ("generator_b", config.models.generator_b, True),
    )
    allocations = _allocate_required_gpus(endpoints, gpus)
    idle_gpus = tuple(gpu for gpu in gpus if gpu.idle)
    checks: list[ModelCheck] = []
    for role, endpoint, required in endpoints:
        parameters = PARAMETER_COUNTS[endpoint.repo_id]
        estimate_mib = int(parameters * 2 * 1.1 / (1024 * 1024))
        assigned = allocations.get(role, ())
        if required:
            enough = bool(assigned)
        else:
            enough = any(
                sum(gpu.free_mib for gpu in group) >= estimate_mib
                for group in combinations(idle_gpus, endpoint.tensor_parallel_size)
            )
        server_status: Literal["NOT_CHECKED", "READY", "UNAVAILABLE", "WRONG_MODEL"] = "NOT_CHECKED"
        detail = "Static checks only"
        if check_servers:
            try:
                listing = VllmClient(
                    endpoint,
                    config.runtime,
                    run_id="doctor",
                ).health()
                model_ids = {
                    item.get("id") for item in listing.get("data", []) if isinstance(item, dict)
                }
                if endpoint.model_name in model_ids:
                    server_status = "READY"
                    detail = "Configured model is served"
                else:
                    server_status = "WRONG_MODEL"
                    detail = f"Server model IDs: {sorted(str(value) for value in model_ids)}"
            except Exception as error:
                server_status = "UNAVAILABLE"
                detail = str(error)
        checks.append(
            ModelCheck(
                role=role,
                repo_id=endpoint.repo_id,
                required_for_current_run=required,
                revision_pinned=(
                    endpoint.revision is not None and endpoint.processor_revision is not None
                ),
                tensor_parallel_size=endpoint.tensor_parallel_size,
                gpu_memory_utilization=endpoint.gpu_memory_utilization,
                estimated_weight_mib=estimate_mib,
                assigned_gpu_indices=assigned,
                enough_idle_memory=enough,
                server_status=server_status,
                detail=detail,
            )
        )
    ready = all(
        (not check.required_for_current_run)
        or check.revision_pinned
        and (check.server_status == "READY" if check_servers else check.enough_idle_memory)
        for check in checks
    )
    return DoctorReport(
        gpus=gpus,
        models=tuple(checks),
        local_wal_root=str(config.storage.run_root),
        ready=ready,
    )


def _allocate_required_gpus(
    endpoints: Sequence[tuple[str, ModelEndpoint, bool]],
    gpus: Sequence[GpuDevice],
) -> dict[str, tuple[int, ...]]:
    """Allocate idle GPU fractions to every unique concurrently required server."""
    idle = {gpu.index: gpu for gpu in gpus if gpu.idle}
    remaining_fraction = {index: 1.0 for index in idle}
    allocations: dict[str, tuple[int, ...]] = {}
    grouped: dict[tuple[str, str, str | None], tuple[ModelEndpoint, list[str]]] = {}
    for role, endpoint, is_required in endpoints:
        if not is_required:
            continue
        key = (str(endpoint.base_url), endpoint.model_name, endpoint.revision)
        if key not in grouped:
            grouped[key] = (endpoint, [])
        grouped[key][1].append(role)
    required = sorted(
        grouped.values(),
        key=lambda item: PARAMETER_COUNTS[item[0].repo_id],
        reverse=True,
    )
    for endpoint, roles in required:
        estimate_mib = int(PARAMETER_COUNTS[endpoint.repo_id] * 2 * 1.1 / (1024 * 1024))
        per_shard_mib = (estimate_mib + endpoint.tensor_parallel_size - 1) // (
            endpoint.tensor_parallel_size
        )
        eligible = [
            group
            for group in combinations(idle.values(), endpoint.tensor_parallel_size)
            if all(
                remaining_fraction[gpu.index] >= endpoint.gpu_memory_utilization
                and gpu.free_mib >= int(gpu.total_mib * endpoint.gpu_memory_utilization)
                and int(gpu.total_mib * endpoint.gpu_memory_utilization) >= per_shard_mib
                for gpu in group
            )
        ]
        if not eligible:
            for role in roles:
                allocations[role] = ()
            continue
        selected = min(
            eligible,
            key=lambda group: (
                sum(remaining_fraction[gpu.index] for gpu in group),
                tuple(gpu.index for gpu in group),
            ),
        )
        assigned = tuple(sorted(gpu.index for gpu in selected))
        for role in roles:
            allocations[role] = assigned
        for gpu in selected:
            remaining_fraction[gpu.index] -= endpoint.gpu_memory_utilization
    return allocations
