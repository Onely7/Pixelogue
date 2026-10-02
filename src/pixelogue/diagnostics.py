"""Private, reproducible diagnostics for completed synthesis runs."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, TypedDict

from pixelogue.contracts import ConversationArtifact
from pixelogue.errors import ExternalInputError
from pixelogue.io import read_jsonl, write_json
from pixelogue.serialization import strict_json_object
from pixelogue.store import RunStore

STOP_ARTIFACTS = (
    "conversation-stop-reasons",
    "errors",
    "model-output-abstentions",
    "binding-abstentions",
    "public-text-rejections",
    "question-intent-decisions",
    "rating-decisions",
)


class DiagnosticRow(TypedDict):
    """One image outcome with private execution evidence."""

    conversation_id: str
    image_id: str
    source_id: str
    status: str
    stop_category: str
    stop_stage: str | None
    stop_stage_evidence: str
    stop_reason: str | None
    attempted_turns: int
    committed_turns: int
    stage_calls: dict[str, int]
    turn_stage_calls: dict[str, dict[str, int]]
    model_calls: int
    invalid_calls: int
    contract_failure_attempts: int
    attempt_failures: list[dict[str, Any]]
    binding_rejections: list[dict[str, Any]]
    retry_calls: int
    input_tokens: int | None
    output_tokens: int | None
    known_input_tokens: int
    known_output_tokens: int
    status_counts: dict[str, int]
    duration_ms: int
    token_usage_missing_calls: int
    recorded_stops: list[dict[str, Any]]
    cost_usd: None


def _request_input(store: RunStore, artifact_hash: str) -> tuple[dict[str, Any], bool]:
    """Read request text without materializing stored image bytes."""
    archive = strict_json_object(store.read_artifact(artifact_hash))
    request = archive.get("request")
    if not isinstance(request, dict):
        raise ExternalInputError("DIAGNOSTIC_REQUEST", "Saved model request is malformed")
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise ExternalInputError("DIAGNOSTIC_REQUEST", "Saved model messages are malformed")
    user = next((message for message in messages if message.get("role") == "user"), None)
    if not isinstance(user, dict):
        raise ExternalInputError("DIAGNOSTIC_REQUEST", "Saved model request has no user message")
    content = user.get("content")
    text_part = (
        next((part.get("text") for part in content if part.get("type") == "text"), None)
        if isinstance(content, list)
        else content
    )
    if not isinstance(text_part, str):
        raise ExternalInputError("DIAGNOSTIC_REQUEST", "Saved model request has no text")
    envelope = strict_json_object(text_part)
    payload = envelope.get("input")
    if not isinstance(payload, dict):
        raise ExternalInputError("DIAGNOSTIC_REQUEST", "Saved request input is malformed")
    return payload, "retry_feedback" in envelope


def _stop_records(store: RunStore) -> dict[str, list[dict[str, Any]]]:
    """Collect recorded stop evidence without interpreting private model responses."""
    by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    placeholders = ", ".join("?" for _ in STOP_ARTIFACTS)
    rows = store.connection.execute(
        f"SELECT artifact_hash, kind FROM artifact WHERE kind IN ({placeholders})",
        STOP_ARTIFACTS,
    )
    for row in rows:
        record = strict_json_object(store.read_artifact(row["artifact_hash"]))
        conversation_id = record.get("conversation_id")
        if conversation_id is None and isinstance(record.get("question_message_id"), str):
            conversation_id = record["question_message_id"].split(":q:", 1)[0]
        if not isinstance(conversation_id, str):
            continue
        verdict = record.get("verdict")
        if row["kind"] in {"question-intent-decisions", "rating-decisions"} and verdict == "MET":
            continue
        by_conversation[conversation_id].append(
            {
                "kind": row["kind"],
                "reason": record.get("reason") or record.get("controller_reason") or verdict,
                "stage": record.get("stage"),
                "turn_index": record.get("turn_index"),
                "artifact_hash": row["artifact_hash"],
            }
        )
    return by_conversation


def _stop_category(status: str, stops: list[dict[str, Any]]) -> str:
    """Classify only outcomes supported by an explicit status or private stop record."""
    if status == "ERROR" or any(stop["kind"] == "errors" for stop in stops):
        return "EXECUTION_ERROR"
    if status == "QUALITY_CANDIDATE":
        return "QUALITY_CANDIDATE"
    if status == "REJECTED":
        return "QUALITY_REJECTION"
    if any(stop["kind"] in {"binding-abstentions", "model-output-abstentions"} for stop in stops):
        return "MALFORMED_MODEL_OUTPUT"
    if any(
        stop["kind"] == "conversation-stop-reasons"
        and stop["reason"] in {"QUESTION_GATE_UNKNOWN", "REQUIREMENT_UNKNOWN", "RATING_ABSTAIN"}
        for stop in stops
    ):
        return "EVIDENCE_OR_JUDGE_UNCERTAINTY"
    return "UNRESOLVED_ABSTENTION"


def _attempt_records(
    store: RunStore, view_to_conversation: dict[str, str]
) -> dict[str, list[dict[str, Any]]]:
    """Link each private malformed attempt to the image that caused it."""
    by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows = store.connection.execute(
        "SELECT artifact_hash FROM artifact WHERE kind = 'structured-output-failures'"
    )
    for row in rows:
        record = strict_json_object(store.read_artifact(row["artifact_hash"]))
        views = record.get("view_ids")
        view_id = views[0] if isinstance(views, list) and views else None
        conversation_id = view_to_conversation.get(view_id)
        if conversation_id is None and isinstance(record.get("question_message_id"), str):
            conversation_id = record["question_message_id"].split(":q:", 1)[0]
        if conversation_id is None:
            continue
        by_conversation[conversation_id].append(
            {
                "artifact_hash": row["artifact_hash"],
                "stage": record.get("stage"),
                "turn_index": record.get("turn_index"),
                "attempt": record.get("attempt"),
                "reason": record.get("reason"),
                "response_hash": record.get("response_hash"),
                "has_parsed_output": record.get("parsed_output") is not None,
                "next_retry_feedback": record.get("next_retry_feedback"),
            }
        )
    return by_conversation


def _binding_rejections(store: RunStore) -> dict[str, list[dict[str, Any]]]:
    """Collect candidate-local failures that did not stop the whole response."""
    by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows = store.connection.execute(
        "SELECT artifact_hash FROM artifact WHERE kind = 'candidate-binding-rejections'"
    )
    for row in rows:
        record = strict_json_object(store.read_artifact(row["artifact_hash"]))
        conversation_id = record.get("conversation_id")
        rejections = record.get("rejections")
        if not isinstance(conversation_id, str) or not isinstance(rejections, list):
            raise ExternalInputError("DIAGNOSTIC_BINDING", "Saved binding rejections are malformed")
        for rejection in rejections:
            if not isinstance(rejection, dict):
                raise ExternalInputError(
                    "DIAGNOSTIC_BINDING", "Saved binding rejection is malformed"
                )
            by_conversation[conversation_id].append(
                {
                    "artifact_hash": row["artifact_hash"],
                    "turn_index": record.get("turn_index"),
                    "candidate_id": rejection.get("candidate_id"),
                    "reason": rejection.get("reason"),
                    "message": rejection.get("message"),
                }
            )
    return by_conversation


def _markdown_cell(value: str | None) -> str:
    """Keep private model text inside one report table cell."""
    return (value or "unrecorded")[:120].replace("|", "\\|").replace("\n", " ")


def build_diagnostic_report(store: RunStore, conversations_path: Path) -> dict[str, Any]:
    """Join public outcomes with private stage and stop evidence for one run."""
    if (
        store.connection.execute(
            "SELECT 1 FROM run WHERE run_id = ?", (store.run_dir.name,)
        ).fetchone()
        is None
    ):
        raise ExternalInputError("DIAGNOSTIC_RUN_MISSING", "The requested run is not initialized")
    conversations = read_jsonl(conversations_path, ConversationArtifact)
    view_to_conversation = {
        conversation.image.full_view.view_id: conversation.conversation_id
        for conversation in conversations
    }
    if len(view_to_conversation) != len(conversations):
        raise ExternalInputError("DIAGNOSTIC_INPUT", "Image views must be unique")
    question_turns: dict[tuple[str, str], set[int]] = defaultdict(set)
    for conversation in conversations:
        for turn in conversation.turns:
            question_turns[conversation.conversation_id, turn.question.content].add(turn.turn_index)
    metrics: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "stage_calls": Counter(),
            "stage_sequence": [],
            "turn_stage_calls": defaultdict(Counter),
            "model_calls": 0,
            "invalid_calls": 0,
            "retry_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "duration_ms": 0,
            "token_usage_missing_calls": 0,
            "missing_input_calls": 0,
            "missing_output_calls": 0,
            "status_counts": Counter(),
        }
    )
    unattributed = 0
    all_calls: Counter[str] = Counter()
    unattributed_usage: Counter[str] = Counter()
    use_attempts = (
        store.connection.execute("SELECT 1 FROM model_call_attempt LIMIT 1").fetchone() is not None
    )
    query = (
        "SELECT stage, status, request_artifact_hash, input_tokens, output_tokens, duration_ms, transport_attempt FROM model_call_attempt ORDER BY rowid"
        if use_attempts
        else """SELECT stage, status, request_artifact_hash,
                  CASE WHEN status='COMPLETE' THEN input_tokens END AS input_tokens,
                  CASE WHEN status='COMPLETE' THEN output_tokens END AS output_tokens,
                  duration_ms, 1 AS transport_attempt FROM model_call ORDER BY rowid"""
    )
    for call in store.connection.execute(query):
        payload, retry = _request_input(store, call["request_artifact_hash"])
        usage = {
            "model_calls": 1,
            "duration_ms": call["duration_ms"],
            "known_input_tokens": call["input_tokens"] or 0,
            "known_output_tokens": call["output_tokens"] or 0,
            "token_usage_missing_calls": int(
                call["input_tokens"] is None or call["output_tokens"] is None
            ),
            "retry_calls": int(retry or call["transport_attempt"] > 1),
            "invalid_calls": int(call["status"] == "INVALID"),
        }
        all_calls.update(usage)
        views = payload.get("image_views")
        view_id = views[0].get("view_id") if isinstance(views, list) and views else None
        operation = payload.get("expected_operation")
        if view_id is None and isinstance(operation, dict):
            view_id = operation.get("view_id")
        conversation_id = view_to_conversation.get(view_id)
        if conversation_id is None:
            unattributed += 1
            unattributed_usage.update(usage)
            continue
        item = metrics[conversation_id]
        item["stage_calls"][call["stage"]] += 1
        item["stage_sequence"].append(call["stage"])
        history = payload.get("public_history")
        turn_index = (
            0
            if call["stage"] == "evidence_extraction"
            else len(history) // 2 + 1
            if isinstance(history, list)
            else payload.get("turn_index", 1)
        )
        if not views and isinstance(payload.get("question"), str):
            matching_turns = question_turns.get((conversation_id, payload["question"]), set())
            turn_index = next(iter(matching_turns)) if len(matching_turns) == 1 else 0
        item["turn_stage_calls"][str(turn_index)][call["stage"]] += 1
        item["model_calls"] += 1
        item["invalid_calls"] += call["status"] == "INVALID"
        item["status_counts"][call["status"]] += 1
        item["retry_calls"] += retry or call["transport_attempt"] > 1
        if call["input_tokens"] is not None:
            item["input_tokens"] += call["input_tokens"]
        else:
            item["missing_input_calls"] += 1
        if call["output_tokens"] is not None:
            item["output_tokens"] += call["output_tokens"]
        else:
            item["missing_output_calls"] += 1
        item["token_usage_missing_calls"] += (
            call["input_tokens"] is None or call["output_tokens"] is None
        )
        item["duration_ms"] += call["duration_ms"]
    stops = _stop_records(store)
    attempts = _attempt_records(store, view_to_conversation)
    binding_rejections = _binding_rejections(store)
    rows: list[DiagnosticRow] = []
    for conversation in conversations:
        item = metrics[conversation.conversation_id]
        recorded_stops = sorted(
            stops.get(conversation.conversation_id, []),
            key=lambda stop: (stop["turn_index"] or 0, stop["kind"], stop["artifact_hash"]),
        )
        explicit_stop = next(
            (
                stop
                for stop in reversed(recorded_stops)
                if stop["kind"] == "conversation-stop-reasons" and stop["stage"]
            ),
            None,
        )
        if conversation.status == "QUALITY_CANDIDATE":
            stop_stage, stop_stage_evidence = None, "completed"
        elif explicit_stop:
            stop_stage, stop_stage_evidence = explicit_stop["stage"], "explicit_stop_record"
        elif any(stop["kind"] == "binding-abstentions" for stop in recorded_stops):
            stop_stage, stop_stage_evidence = "candidate_binding", "binding_stop_record"
        elif any(stop["kind"] == "model-output-abstentions" for stop in recorded_stops):
            stop_stage, stop_stage_evidence = (
                item["stage_sequence"][-1] if item["stage_sequence"] else None,
                "last_model_call",
            )
        else:
            stop_stage, stop_stage_evidence = (
                item["stage_sequence"][-1] if item["stage_sequence"] else None,
                "last_model_call" if item["stage_sequence"] else "unrecorded",
            )
        attempted_indices = {turn.turn_index for turn in conversation.turns}
        attempted_indices.update(int(index) for index in item["turn_stage_calls"] if int(index) > 0)
        attempted_indices.update(
            stop["turn_index"]
            for stop in recorded_stops
            if isinstance(stop["turn_index"], int) and stop["turn_index"] > 0
        )
        rows.append(
            {
                "conversation_id": conversation.conversation_id,
                "image_id": conversation.image.image_id,
                "source_id": conversation.image.source_id,
                "status": conversation.status,
                "stop_category": _stop_category(conversation.status, recorded_stops),
                "stop_stage": stop_stage,
                "stop_stage_evidence": stop_stage_evidence,
                "stop_reason": (
                    None
                    if conversation.status == "QUALITY_CANDIDATE"
                    else explicit_stop["reason"]
                    if explicit_stop
                    else recorded_stops[-1]["reason"]
                    if recorded_stops
                    else None
                ),
                "attempted_turns": len(attempted_indices),
                "committed_turns": sum(turn.status == "COMMITTED" for turn in conversation.turns),
                "stage_calls": dict(sorted(item["stage_calls"].items())),
                "turn_stage_calls": {
                    turn: dict(sorted(stages.items()))
                    for turn, stages in sorted(item["turn_stage_calls"].items())
                },
                "model_calls": item["model_calls"],
                "invalid_calls": item["invalid_calls"],
                "contract_failure_attempts": len(attempts.get(conversation.conversation_id, [])),
                "attempt_failures": attempts.get(conversation.conversation_id, []),
                "binding_rejections": binding_rejections.get(conversation.conversation_id, []),
                "retry_calls": item["retry_calls"],
                "input_tokens": None if item["missing_input_calls"] else item["input_tokens"],
                "output_tokens": None if item["missing_output_calls"] else item["output_tokens"],
                "known_input_tokens": item["input_tokens"],
                "known_output_tokens": item["output_tokens"],
                "status_counts": dict(sorted(item["status_counts"].items())),
                "duration_ms": item["duration_ms"],
                "token_usage_missing_calls": item["token_usage_missing_calls"],
                "recorded_stops": recorded_stops,
                "cost_usd": None,
            }
        )
    return {
        "run_id": store.run_dir.name,
        "conversations": len(rows),
        "attempted_turns": sum(row["attempted_turns"] for row in rows),
        "committed_turns": sum(row["committed_turns"] for row in rows),
        "completed_conversations": sum(row["status"] == "QUALITY_CANDIDATE" for row in rows),
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "stop_category_counts": dict(sorted(Counter(row["stop_category"] for row in rows).items())),
        "committed_turn_counts": dict(
            sorted(Counter(row["committed_turns"] for row in rows).items())
        ),
        "model_calls": all_calls["model_calls"],
        "attributed_model_calls": sum(row["model_calls"] for row in rows),
        "unattributed_model_calls": unattributed,
        "unattributed_usage": dict(unattributed_usage),
        "call_measurement": "actual_http_attempts" if use_attempts else "legacy_canonical_calls",
        "sum_http_duration_ms": all_calls["duration_ms"],
        "gpu_allocation_seconds": None,
        "known_input_tokens": all_calls["known_input_tokens"],
        "known_output_tokens": all_calls["known_output_tokens"],
        "cache_accesses": store.connection.execute(
            "SELECT COUNT(*) FROM artifact WHERE kind='cache-accesses'"
        ).fetchone()[0],
        "retry_calls": all_calls["retry_calls"],
        "invalid_calls": all_calls["invalid_calls"],
        "contract_failure_attempts": sum(row["contract_failure_attempts"] for row in rows),
        "binding_rejections": sum(len(row["binding_rejections"]) for row in rows),
        "stage_reach_images": dict(
            sorted(Counter(stage for row in rows for stage in row["stage_calls"]).items())
        ),
        "stage_reach_turns": dict(
            sorted(
                Counter(
                    stage
                    for row in rows
                    for stages in row["turn_stage_calls"].values()
                    for stage in stages
                ).items()
            )
        ),
        "token_usage_missing_calls": all_calls["token_usage_missing_calls"],
        "cost_usd": None,
        "rows": rows,
    }


def write_diagnostic_reports(report: dict[str, Any], output_stem: Path) -> None:
    """Write private JSON, CSV, and Markdown views of the same run evidence."""
    write_json(output_stem.with_suffix(".json"), report)
    csv_path = output_stem.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "conversation_id",
                "source_id",
                "status",
                "stop_category",
                "stop_stage",
                "stop_stage_evidence",
                "stop_reason",
                "attempted_turns",
                "committed_turns",
                "model_calls",
                "invalid_calls",
                "contract_failure_attempts",
                "attempt_failures",
                "binding_rejections",
                "retry_calls",
                "input_tokens",
                "output_tokens",
                "duration_ms",
                "token_usage_missing_calls",
                "stage_calls",
                "turn_stage_calls",
                "recorded_stops",
            ),
        )
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow(
                {
                    key: json.dumps(row[key], ensure_ascii=False, sort_keys=True)
                    if key
                    in {
                        "stage_calls",
                        "turn_stage_calls",
                        "recorded_stops",
                        "attempt_failures",
                        "binding_rejections",
                    }
                    else row[key]
                    for key in writer.fieldnames
                }
            )
    lines = [
        f"# Run diagnostics: {report['run_id']}",
        "",
        f"Conversations: {report['conversations']}; model calls: {report['model_calls']}; "
        f"retries: {report['retry_calls']}; invalid calls: {report['invalid_calls']}.",
        f"Attempted turns: {report['attempted_turns']}; committed turns: "
        f"{report['committed_turns']}; completed conversations: "
        f"{report['completed_conversations']}.",
        f"Recorded structured-output contract failures: {report['contract_failure_attempts']}.",
        f"Rejected individual candidate bindings: {report['binding_rejections']}.",
        f"Token usage is missing for {report['token_usage_missing_calls']} calls.",
        "",
        "| Status | Count |",
        "|---|---:|",
        *(f"| {status} | {count} |" for status, count in report["status_counts"].items()),
        "",
        "| Source | Status | Turns | Stop stage | Reason | Stage evidence | Calls | Retries | Invalid | Contract failures | Duration ms |",
        "|---|---|---:|---|---|---|---:|---:|---:|---:|---:|",
        *(
            f"| {row['source_id']} | {row['stop_category']} | {row['committed_turns']} | "
            f"{row['stop_stage'] or 'unknown'} | {_markdown_cell(row['stop_reason'])} | "
            f"{row['stop_stage_evidence']} | "
            f"{row['model_calls']} | {row['retry_calls']} | {row['invalid_calls']} | "
            f"{row['contract_failure_attempts']} | "
            f"{row['duration_ms']} |"
            for row in report["rows"]
        ),
        "",
        "Cost is unknown without a recorded price schedule. Stop records are private run evidence; "
        "their absence does not imply that the conversation completed successfully.",
        "Older runs may lack per-attempt contract failure artifacts; a zero count in those runs "
        "does not establish that every model response met its contract.",
        "",
    ]
    output_stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
