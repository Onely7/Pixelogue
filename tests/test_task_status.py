"""Task inventory separates implementation, CPU fixtures and actual admission."""

from __future__ import annotations

from pathlib import Path

from pixelogue.config import load_config
from pixelogue.task_status import task_status_report


def test_status_covers_all_tasks_without_claiming_unrun_cpu_checks(tmp_path: Path) -> None:
    config = load_config(Path("configs/specialist-pilot.yaml"))
    untested = task_status_report(config)
    assert untested["summary"]["tasks"] == 72
    assert untested["summary"]["implemented"] == 72
    assert untested["summary"]["normal_selectable"] == 65
    assert all(item["cpu_contract_status"] == "not_recorded" for item in untested["tasks"])
    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuite><testcase classname="tests.test_specialist_render" name="test_parser"/></testsuite>'
    )
    tested = task_status_report(config, junit)
    renderer = next(item for item in tested["tasks"] if item["task_id"] == "diagram_to_code")
    assert renderer["implemented"]
    assert renderer["cpu_contract_status"] == "shared_contract_tests_passed"
    assert renderer["calibration"] == "pending"
