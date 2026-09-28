from pydantic import HttpUrl

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


def test_required_servers_receive_distinct_gpus_or_report_shortfall() -> None:
    endpoints = (
        ("selector", _endpoint("Qwen/Qwen3.5-2B", 1), True),
        ("generator_a", _endpoint("Qwen/Qwen3.8-27B-FP8", 2), True),
        ("generator_b", _endpoint("google/gemma-4-31B-it-qat-w4a16-ct", 2), True),
    )
    enough = _allocate_required_gpus(endpoints, _gpus(5))
    assigned = [gpu for group in enough.values() for gpu in group]
    assert all(enough.values())
    assert len(assigned) == len(set(assigned)) == 5

    short = _allocate_required_gpus(endpoints, _gpus(4))
    assert sum(not group for group in short.values()) == 1


def test_quantized_standard_pair_can_share_one_large_idle_gpu() -> None:
    gpu = GpuDevice(
        index=4,
        name="96 GiB test device",
        total_mib=97_887,
        used_mib=0,
        free_mib=97_887,
        utilization_percent=0,
        idle=True,
    )
    endpoints = (
        (
            "selector",
            _endpoint("Qwen/Qwen3.5-2B", 1).model_copy(update={"gpu_memory_utilization": 0.10}),
            True,
        ),
        (
            "generator_a",
            _endpoint("Qwen/Qwen3.8-27B-FP8", 1).model_copy(
                update={"gpu_memory_utilization": 0.44, "quantization": "fp8"}
            ),
            True,
        ),
        (
            "generator_b",
            _endpoint("google/gemma-4-31B-it-qat-w4a16-ct", 1).model_copy(
                update={"gpu_memory_utilization": 0.35, "quantization": "compressed-tensors"}
            ),
            True,
        ),
    )
    assert _allocate_required_gpus(endpoints, (gpu,)) == {
        "generator_a": (4,),
        "generator_b": (4,),
        "selector": (4,),
    }
