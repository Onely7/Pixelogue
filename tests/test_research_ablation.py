"""Comparison plans isolate all agreed factors and count every starting case."""

from __future__ import annotations

from pathlib import Path

from pixelogue.catalog import task_catalog
from pixelogue.config import load_config
from pixelogue.contracts import ImageArtifact, ImageView, InstructionCandidate, SourcePurpose
from pixelogue.io import write_json
from pixelogue.research_ablation import (
    AblationCase,
    AblationResult,
    AblationTurn,
    AblationTurnResult,
    ablation_report,
    build_ablation_plan,
    write_ablation_reports,
)
from pixelogue.research_exposure import FiveQuestionRating


def _case() -> AblationCase:
    sha = "a" * 64
    image = ImageArtifact(
        image_id="image",
        source_id="source",
        purpose=SourcePurpose.EVALUATION,
        raw_sha256=sha,
        canonical_pixel_sha256=sha,
        source_group_ids=("group",),
        visual_group_id="group",
        full_view=ImageView(
            view_id="view",
            relative_path="view.png",
            encoded_sha256=sha,
            pixel_sha256=sha,
            width=100,
            height=100,
            media_type="image/png",
        ),
    )
    task = next(item for item in task_catalog().tasks if item.id == "object_identification")
    instruction = InstructionCandidate(
        candidate_id="candidate",
        task_id=task.id,
        family=task.family,
        visible_scope="whole view",
        instruction_summary=task.definition_en,
        required_capabilities=task.required_capabilities,
        catalog_version="7.0",
        scope_id="scope",
        view_id="view",
        evidence_refs=("evidence",),
        verification_contracts=task.verification_contracts,
    )
    return AblationCase(
        case_id="case",
        image=image,
        target_language="en",
        generator_role="generator_a",
        turns=(AblationTurn(question="What object is shown?", instruction=instruction),),
    )


def test_full_factorial_plan_and_unbiased_depth_denominators(tmp_path: Path) -> None:
    config = load_config(Path("configs/pilot.yaml"))
    plan = build_ablation_plan((_case(),), config, 11)
    assert len(plan.trials) == len({trial.trial_id for trial in plan.trials}) == 18
    assert {trial.cell.question_gate for trial in plan.trials} == {
        "pre_answer",
        "post_answer",
        "omitted",
    }
    assert {trial.cell.history for trial in plan.trials} == {"verified_only", "generated"}
    report = ablation_report(plan, tmp_path)
    assert len(report["cells"]) == 18
    assert all(cell["depth"][1]["started"] == 1 for cell in report["cells"])
    assert all(cell["depth"][1]["reached"] == 0 for cell in report["cells"])
    assert report["training_export_allowed"] is False


def test_report_counts_human_error_overlap_and_writes_depth_csv(tmp_path: Path) -> None:
    plan = build_ablation_plan((_case(),), load_config(Path("configs/pilot.yaml")), 11)
    trial = next(item for item in plan.trials if item.cell.question_evaluator == "both")
    vote = FiveQuestionRating(
        grounding="MET",
        operation_match="MET",
        answerability="MET",
        nonredundancy="MET",
        naturalness="MET",
        reason="Fixture acceptance",
    )
    write_json(
        tmp_path / "results" / f"{trial.trial_id}.json",
        AblationResult(
            trial_id=trial.trial_id,
            case_id=trial.case_id,
            cell=trial.cell,
            status="COMPLETE",
            error=None,
            turns=(
                AblationTurnResult(
                    depth=1,
                    gate_verdict="MET",
                    gate_votes=(vote, vote),
                    answer="Wrong",
                    review="FAIL",
                    status="REJECTED",
                    history_includes_unverified=False,
                    human_question_admissible=False,
                ),
            ),
        ),
    )
    report = write_ablation_reports(plan, tmp_path)
    row = next(
        item for item in report["cells"] if item["cell"] == trial.cell.model_dump(mode="json")
    )
    assert row["depth"][1]["false_accept"] == 1
    assert row["depth"][1]["both_evaluator_errors"] == 1
    assert "false_accept" in (tmp_path / "depth.csv").read_text(encoding="utf-8")
