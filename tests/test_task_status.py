"""Task inventory separates implementation, CPU fixtures and actual admission."""

from __future__ import annotations

from pathlib import Path

from pixelogue.config import load_config
from pixelogue.contracts import (
    ConversationArtifact,
    InstructionCandidate,
    PublicMessage,
    TurnArtifact,
    TurnRating,
)
from pixelogue.io import write_jsonl
from pixelogue.serialization import canonical_hash
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


def test_status_records_actual_gpu_turn_without_claiming_all_boundaries(
    tmp_path: Path, image_artifact
) -> None:
    image, _ = image_artifact
    turn = TurnArtifact(
        turn_index=1,
        instruction=InstructionCandidate(
            candidate_id="candidate",
            task_id="entity_count",
            family="set_logic",
            visible_scope="whole image",
            instruction_summary="Count visible objects",
            required_capabilities=("visible_entity",),
        ),
        question=PublicMessage(message_id="q1", turn_index=1, role="user", content="How many?"),
        answer=PublicMessage(message_id="a1", turn_index=1, role="assistant", content="Two."),
        history_hash=canonical_hash([]),
        generation_model="pilot-model",
        selector_model="selector-model",
        rating=TurnRating(items=(), aggregate="PASS"),
        status="COMMITTED",
    )
    conversation = ConversationArtifact(
        conversation_id="conversation-1",
        image=image,
        target_language="en",
        generation_model="pilot-model",
        turns=(turn,),
        status="REJECTED",
    )
    path = tmp_path / "conversations.jsonl"
    write_jsonl(path, [conversation])
    report = task_status_report(load_config(Path("configs/specialist-pilot.yaml")), None, path)
    task = next(item for item in report["tasks"] if item["task_id"] == "entity_count")
    assert task["gpu_turn_case_count"] == 1
    assert task["gpu_turn_cases"][0]["generator_model"] == "pilot-model"
    assert task["cpu_task_specific_boundaries"] == "not_recorded"
