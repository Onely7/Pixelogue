import subprocess

import pytest
from pydantic import HttpUrl

from pixelogue import doctor
from pixelogue.config import ModelEndpoint
from pixelogue.doctor import GpuDevice, _allocate_required_gpus


def _endpoint(repo_id: str, tensor_parallel_size: int) -> ModelEndpoint:
    return ModelEndpoint(
        repo_id=repo_id,
        base_url=HttpUrl("http://127.0.0.1:8000/v1"),
        tensor_parallel_size=tensor_parallel_size,
    )


def _gpus(count: int) -> tuple[GpuDevice, ...]:
    return tuple(
        GpuDevice(
            index=index,
            name="test",
            total_mib=48_000,
            used_mib=0,
            free_mib=48_000,
            utilization_percent=0,
            idle=True,
        )
        for index in range(count)
    )


def test_shared_router_is_one_server_and_bf16_models_need_two_48gb_gpus() -> None:
    qwen = _endpoint("Qwen/Qwen3.8-27B", 2)
    endpoints = (
        ("router", qwen, True),
        ("generator_a", qwen, True),
        ("generator_b", _endpoint("google/gemma-4-31B-it", 2), True),
    )
    enough = _allocate_required_gpus(endpoints, _gpus(4))
    assert all(enough.values())
    assert enough["router"] == enough["generator_a"]
    assert len({gpu for group in enough.values() for gpu in group}) == 4

    # The larger Gemma server takes two GPUs; the one shared Qwen server, and so both of its
    # roles, are then short.
    short = _allocate_required_gpus(endpoints, _gpus(3))
    assert short["generator_b"] and not short["generator_a"] and not short["router"]


@pytest.mark.parametrize("missing", ["N/A", "[N/A]"])
def test_unknown_utilization_is_reported_without_admitting_the_device(
    missing: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: "nvidia-smi")
    response = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=f"0, faulty, 48000, 15, 47985, {missing}\n1, healthy, 48000, 15, 47985, 0\n",
    )
    monkeypatch.setattr(doctor.subprocess, "run", lambda *_args, **_kwargs: response)
    gpus = doctor.inspect_gpus()
    assert gpus[0].utilization_percent is None
    assert not gpus[0].idle
    assert gpus[1].idle
    endpoints = (("router", _endpoint("Qwen/Qwen3.5-2B", 1), True),)
    assert _allocate_required_gpus(endpoints, gpus) == {"router": (1,)}


def test_split_layout_places_qwen_on_the_96gb_gpu() -> None:
    large = GpuDevice(
        index=2,
        name="96 GiB test device",
        total_mib=97_887,
        used_mib=0,
        free_mib=97_887,
        utilization_percent=0,
        idle=True,
    )
    qwen = _endpoint("Qwen/Qwen3.8-27B", 1).model_copy(update={"gpu_memory_utilization": 0.88})
    endpoints = (
        ("router", qwen, True),
        ("generator_a", qwen, True),
        ("generator_b", _endpoint("google/gemma-4-31B-it", 2), True),
    )
    assert _allocate_required_gpus(endpoints, (*_gpus(2), large)) == {
        "generator_b": (0, 1),
        "router": (2,),
        "generator_a": (2,),
    }
    assert not _allocate_required_gpus(endpoints, _gpus(3))["generator_a"]
