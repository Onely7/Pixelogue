from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from pixelogue.catalog import task_catalog
from pixelogue.config import ModelEndpoint, RuntimeConfig, load_config
from pixelogue.contracts import (
    GateVerdict,
    InstructionCandidate,
    PublicMessage,
    RubricVerdict,
    TextPayload,
)
from pixelogue.errors import ExecutionError
from pixelogue.pipeline import SynthesisCoordinator
from pixelogue.rules import CountGroup, SetInventory
from pixelogue.serving import ModelImage, VllmClient
from pixelogue.store import RunStore
from pixelogue.task_evidence import ImageRegion, TranscriptSource
from pixelogue.task_verification import verify_operation


def operation(task_id: str) -> InstructionCandidate:
    task = next(t for t in task_catalog().tasks if t.id == task_id)
    return InstructionCandidate(
        candidate_id="bound-operation",
        task_id=task.id,
        family=task.family,
        visible_scope="the whole canvas",
        instruction_summary=task.definition,
        required_capabilities=task.required_capabilities,
        catalog_version="8.0",
        scope_id="canvas",
        view_id="view",
        evidence_refs=("scope-evidence",),
        verification_contracts=task.verification_contracts,
    )


@pytest.mark.parametrize(("reported", "expected_verdict"), [(2, "MET"), (3, "NOT_MET")])
def test_closed_set_verification_uses_the_operation_contract_after_family_rename(
    reported, expected_verdict
):
    candidate = operation("entity_count")
    assert candidate.family == "counting_and_sets"
    votes = []

    def invoke(stage, payload, model, judge):
        assert stage == "set_inventory"
        assert payload["expected_operation"]["task_id"] == "entity_count"
        assert "other_judge" not in payload
        votes.append(judge)
        return SetInventory(
            coverage="MET",
            mode="count",
            counts=(CountGroup(scope="cups", expected=2, reported=reported),),
            expected_members=(),
            reported_members=(),
            empty_scope_is_explicit=False,
            reason="Visible",
        )

    result = verify_operation(
        candidate, {"question": "How many cups?", "candidate_answer": str(reported)}, invoke
    )
    assert votes == [0, 1]
    assert result[0].verdict.value == expected_verdict


def test_closed_scope_uncertainty_cannot_be_overridden_by_matching_answers():
    def invoke(stage, payload, model, judge):
        return SetInventory(
            coverage="UNKNOWN",
            mode="count",
            counts=(),
            expected_members=(),
            reported_members=(),
            empty_scope_is_explicit=False,
            reason="Part of the set is occluded",
        )

    result = verify_operation(operation("entity_count"), {"candidate_answer": "2"}, invoke)
    assert result[0].verdict is GateVerdict.UNKNOWN


@pytest.mark.parametrize(
    "answer,verdict",
    [
        ("Six volumes.", "MET"),
        ("six volumes", "MET"),
        ("Six volumes and two maps.", "NOT_MET"),
        ("Six", "NOT_MET"),
        ("Seven volumes.", "NOT_MET"),
        ("six VOLUMES", "NOT_MET"),
    ],
)
def test_extractive_qa_reads_blind_and_preserves_the_entire_answer_span(answer, verdict):
    def invoke(stage, payload, model, judge):
        assert stage == "extractive_source"
        assert "candidate_answer" not in payload
        return TranscriptSource(
            coverage="MET",
            expected_lines=("six volumes",),
            source_region=ImageRegion(left=0, top=0, right=1, bottom=1),
            requested_unit_complete=True,
            reason="All answer-bearing text is readable",
        )

    result = verify_operation(
        operation("document_extractive_qa"), {"candidate_answer": answer}, invoke
    )
    assert result[0].verdict.value == verdict


def test_extractive_qa_disagreement_and_truncated_qualifiers_still_abstain():
    def invoke(stage, payload, model, judge):
        return TranscriptSource(
            coverage="MET",
            expected_lines=(("at least six volumes" if judge else "six volumes"),),
            source_region=ImageRegion(left=0, top=0, right=1, bottom=1),
            requested_unit_complete=True,
            reason="Independent source reading",
        )

    result = verify_operation(
        operation("document_extractive_qa"), {"candidate_answer": "Six volumes."}, invoke
    )
    assert result[0].verdict is GateVerdict.UNKNOWN


@pytest.mark.parametrize(
    ("expected_text", "answer_text", "verdict"),
    [
        ("STOP", "STOP", "MET"),
        ("ST0P", "STOP", "NOT_MET"),
        ("a\n  b", "a\nb", "NOT_MET"),
        ("STOP", "STOP\nextra", "NOT_MET"),
    ],
)
def test_transcription_preserves_source_errors_and_meaningful_whitespace(
    expected_text, answer_text, verdict
):
    def invoke(stage, payload, model, judge):
        assert stage == "transcript_source"
        assert "candidate_answer" not in payload
        return TranscriptSource(
            coverage="MET",
            expected_lines=tuple(expected_text.split("\n")),
            source_region=ImageRegion(left=0, top=0, right=1, bottom=1),
            requested_unit_complete=True,
            reason="Visible source",
        )

    result = verify_operation(
        operation("text_transcription"), {"candidate_answer": answer_text}, invoke
    )
    assert result[0].verdict.value == verdict


def test_a_holistic_pass_cannot_bypass_required_count_verification(
    tmp_path, image_artifact, monkeypatch
):
    image, root = image_artifact
    config = load_config(Path("configs/pilot.yaml"))
    with RunStore(tmp_path, "operation-check", require_local_wal=False) as store:
        # The transport is never called: the scripted method supplies blind, typed votes.
        client = VllmClient(config.models.generator_a, config.runtime, run_id="operation-check")
        co = SynthesisCoordinator(config, "operation-check", store, client, client, client)
        stages = []

        def invoke(client, stage, payload, images, model, **kwargs):
            stages.append(stage)
            if stage == "holistic_review":
                assert payload["expected_operation"]["task_id"] == "entity_count"
                return RubricVerdict(verdict="MET", reason="Scripted false positive")
            assert stage == "set_inventory"
            return SetInventory(
                coverage="MET",
                mode="count",
                counts=(CountGroup(scope="cups", expected=2, reported=3),),
                expected_members=(),
                reported_members=(),
                empty_scope_is_explicit=False,
                reason="Counted",
            )

        monkeypatch.setattr(co, "_invoke", invoke)
        result = co._rate_turn(
            "conversation",
            "0" * 64,
            (),
            PublicMessage(message_id="q", turn_index=1, role="user", content="How many cups?"),
            PublicMessage(message_id="a", turn_index=1, role="assistant", content="3"),
            operation("entity_count").model_copy(update={"view_id": image.full_view.view_id}),
            "en",
            [],
            ModelImage(
                view_id=image.full_view.view_id,
                path=root / image.full_view.relative_path,
                encoded_sha256=image.full_view.encoded_sha256,
                media_type=image.full_view.media_type,
            ),
            1,
        )
        client.client.close()
        assert result.aggregate == "FAIL"
        assert stages == ["holistic_review", "holistic_review", "set_inventory", "set_inventory"]


def test_length_terminated_public_text_is_not_certified_as_complete():
    def handler(request):
        return httpx.Response(
            200,
            request=request,
            json={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": json.dumps({"text": "Only the first row"})},
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 20},
            },
        )

    with httpx.Client(
        base_url="http://127.0.0.1:8000/v1/", transport=httpx.MockTransport(handler)
    ) as http:
        client = VllmClient(
            ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"), RuntimeConfig(), run_id="length", client=http
        )
        with pytest.raises(ExecutionError, match="Public text must finish"):
            client.invoke(
                "answer_generation",
                {
                    "target_language": "en",
                    "question": "Transcribe every row.",
                    "public_history": [],
                    "active_requirements": [],
                    "image_views": [],
                },
                (),
                TextPayload,
                max_tokens=256,
                temperature=0.0,
                seed=1,
            )


@pytest.mark.parametrize(
    ("extracted_operator", "reported", "expected"),
    [
        ("add", "5", "MET"),
        ("add", "6", "NOT_MET"),
        ("subtract", "1", "UNKNOWN"),
    ],
)
def test_arithmetic_recomputes_only_the_bound_public_operator(
    extracted_operator, reported, expected
):
    from pixelogue.quantitative_verifiers import (
        QuantityAnswer,
        QuantityOperand,
        QuantityQuery,
        QuantitySource,
    )
    from pixelogue.rules import NumericValue
    from pixelogue.task_evidence import ImageRegion, PublicParameter

    candidate = operation("grounded_arithmetic").model_copy(
        update={
            "public_parameters": (
                PublicParameter(name="operators", value="add", origin="instruction"),
            ),
        }
    )

    def invoke(stage, payload, model, judge):
        if stage == "quantity_answer":
            assert "image_views" not in payload
            return QuantityAnswer(
                coverage="MET",
                answer_quote=f"{reported} kg",
                value=NumericValue(value=reported, unit="kg"),
                reason="Literal answer",
            )
        assert stage == "quantity_source"
        assert "candidate_answer" not in payload
        region = ImageRegion(left=0, top=0, right=1, bottom=1)
        return QuantitySource(
            task_id="grounded_arithmetic",
            coverage="MET",
            closed=True,
            scope_id="canvas",
            view_id="view",
            scope_region=region,
            operands=(
                QuantityOperand(
                    operand_id="a",
                    value=NumericValue(value="3", unit="kg"),
                    printed_text="3 kg",
                    region=region,
                ),
                QuantityOperand(
                    operand_id="b",
                    value=NumericValue(value="2", unit="kg"),
                    printed_text="2 kg",
                    region=region,
                ),
            ),
            query=QuantityQuery(operation=extracted_operator, operand_ids=("a", "b")),
            reason="Visible quantities",
        )

    result = verify_operation(
        candidate, {"question": "What is the sum?", "candidate_answer": f"{reported} kg"}, invoke
    )
    assert result[0].verdict.value == expected
