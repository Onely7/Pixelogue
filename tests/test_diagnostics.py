"""Run diagnostics preserve private stop reasons and attribute model calls."""

from __future__ import annotations

import json
from pathlib import Path

from pixelogue.contracts import ConversationArtifact
from pixelogue.diagnostics import build_diagnostic_report, write_diagnostic_reports
from pixelogue.io import write_jsonl
from pixelogue.store import RunStore


def test_diagnostics_join_image_calls_and_private_stop_without_loading_image(
    tmp_path: Path, image_artifact
) -> None:
    image, _ = image_artifact
    conversation = ConversationArtifact(
        conversation_id="conversation-1",
        image=image,
        target_language="en",
        generation_model="pilot-model",
        turns=(),
        status="ABSTAINED",
    )
    conversations = tmp_path / "conversations.jsonl"
    write_jsonl(conversations, [conversation])
    request = {
        "archive_format": "image-refs-v1",
        "request": {
            "messages": [
                {"role": "system", "content": "fixture"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "input": {
                                        "image_views": [{"view_id": image.full_view.view_id}]
                                    },
                                    "retry_feedback": "Use exact checks",
                                }
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "pixelogue-image:missing-fixture"},
                        },
                    ],
                },
            ]
        },
    }
    with RunStore(tmp_path / "runs", "diagnostic", require_local_wal=False) as store:
        store.initialize_run("diagnostic", "config-hash", "pilot")
        request_hash = store.write_json_artifact("requests", request)
        store.write_json_artifact(
            "binding-abstentions",
            {
                "conversation_id": conversation.conversation_id,
                "turn_index": 1,
                "reason": "CANDIDATE_CHECKS_MISMATCH",
            },
        )
        store.write_json_artifact(
            "structured-output-failures",
            {
                "view_ids": [image.full_view.view_id],
                "stage": "candidate_binding",
                "turn_index": 1,
                "attempt": 2,
                "reason": "CANDIDATE_CHECKS_MISMATCH",
                "response_hash": "0" * 64,
                "parsed_output": {"bindings": []},
                "next_retry_feedback": "Include exact checks",
            },
        )
        store.write_json_artifact(
            "candidate-binding-rejections",
            {
                "conversation_id": conversation.conversation_id,
                "turn_index": 1,
                "rejections": [
                    {
                        "candidate_id": "invalid-candidate",
                        "reason": "CANDIDATE_PARAMETER_SOURCE",
                        "message": "The cited target is missing",
                    }
                ],
            },
        )
        with store.transaction() as connection:
            connection.execute(
                """INSERT INTO model_call(
                    call_id, stage, model_repo, model_lock_hash, request_hash,
                    request_artifact_hash, input_tokens, output_tokens, duration_ms, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "call-1",
                    "candidate_binding",
                    "pilot-model",
                    "lock",
                    "request-hash",
                    request_hash,
                    120,
                    30,
                    2500,
                    "INVALID",
                ),
            )
        report = build_diagnostic_report(store, conversations)
    row = report["rows"][0]
    assert report["status_counts"] == {"ABSTAINED": 1}
    assert report["attempted_turns"] == 1
    assert report["committed_turns"] == 0
    assert report["completed_conversations"] == 0
    assert row["attempted_turns"] == 1
    assert report["retry_calls"] == 1
    assert row["stage_calls"] == {"candidate_binding": 1}
    assert row["turn_stage_calls"] == {"1": {"candidate_binding": 1}}
    assert row["stop_category"] == "MALFORMED_MODEL_OUTPUT"
    assert row["stop_stage"] == "candidate_binding"
    assert row["stop_stage_evidence"] == "binding_stop_record"
    assert row["stop_reason"] == "CANDIDATE_CHECKS_MISMATCH"
    assert row["recorded_stops"][0]["reason"] == "CANDIDATE_CHECKS_MISMATCH"
    assert row["input_tokens"] == 0
    assert row["token_usage_missing_calls"] == 1
    assert row["contract_failure_attempts"] == 1
    assert report["binding_rejections"] == 1
    assert row["binding_rejections"][0]["candidate_id"] == "invalid-candidate"
    assert row["attempt_failures"][0]["next_retry_feedback"] == "Include exact checks"
    assert row["cost_usd"] is None

    stem = tmp_path / "diagnostics"
    write_diagnostic_reports(report, stem)
    assert json.loads(stem.with_suffix(".json").read_text())["model_calls"] == 1
    assert "CANDIDATE_CHECKS_MISMATCH" in stem.with_suffix(".csv").read_text()
    assert "CANDIDATE_PARAMETER_SOURCE" in stem.with_suffix(".csv").read_text()
    assert "ABSTAINED" in stem.with_suffix(".md").read_text()
    assert "| source-1 | MALFORMED_MODEL_OUTPUT |" in stem.with_suffix(".md").read_text()
