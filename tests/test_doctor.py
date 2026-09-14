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
        ("generator_a", _endpoint("Qwen/Qwen3.8-27B", 2), True),
        ("generator_b", _endpoint("google/gemma-4-31B-it", 2), True),
    )
    enough = _allocate_required_gpus(endpoints, _gpus(5))
    assigned = [gpu for group in enough.values() for gpu in group]
    assert all(enough.values())
    assert len(assigned) == len(set(assigned)) == 5

    short = _allocate_required_gpus(endpoints, _gpus(4))
    assert sum(not group for group in short.values()) == 1
