"""Training, rating, and provenance export boundaries."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

from pixelogue.contracts import (
    ConversationArtifact,
    SelectionManifest,
    SourcePurpose,
    TrainingContent,
    TrainingMessage,
    TrainingRecord,
)
from pixelogue.errors import AuditError


def training_record(conversation: ConversationArtifact) -> TrainingRecord:
    """Convert one fully accepted training conversation to public-only messages.

    Raises:
        AuditError: If the source or conversation is ineligible for training.
    """
    if conversation.status != "QUALITY_CANDIDATE":
        raise AuditError("CONVERSATION_NOT_ACCEPTED", conversation.conversation_id)
    if conversation.image.purpose is not SourcePurpose.TRAINING:
        raise AuditError("EVALUATION_IMAGE_EXPORT", conversation.conversation_id)
    if not 2 <= len(conversation.turns) <= 6:
        raise AuditError("TURN_COUNT_INVALID", conversation.conversation_id)
    messages: list[TrainingMessage] = []
    for index, turn in enumerate(conversation.turns):
        if turn.status != "COMMITTED":
            raise AuditError("UNCOMMITTED_TURN_EXPORT", conversation.conversation_id)
        user_content: list[TrainingContent] = []
        if index == 0:
            user_content.append(
                TrainingContent(type="image", image=conversation.image.full_view.relative_path)
            )
        user_content.append(TrainingContent(type="text", text=turn.question.content))
        messages.append(TrainingMessage(role="user", content=tuple(user_content)))
        messages.append(
            TrainingMessage(
                role="assistant",
                content=(TrainingContent(type="text", text=turn.answer.content),),
            )
        )
    return TrainingRecord(record_id=conversation.conversation_id, messages=tuple(messages))


def export_bundle(
    conversations: Sequence[ConversationArtifact],
    selection: SelectionManifest,
    destination: Path,
    *,
    profile: str,
    student_processor_lock: dict[str, object] | None = None,
) -> dict[str, Path]:
    """Write immutable training, rating, and provenance JSONL outputs."""
    if profile != "standard":
        raise AuditError("PILOT_EXPORT_FORBIDDEN", "Only standard runs may export training data")
    if not selection.has_valid_audit_binding():
        raise AuditError("AUDIT_REQUIRED", "Selection manifest is not bound to an audit")
    by_id = {conversation.conversation_id: conversation for conversation in conversations}
    selected = [
        by_id[selected_id] for selected_id in selection.selected_ids if selected_id in by_id
    ]
    if len(selected) != len(selection.selected_ids):
        raise AuditError(
            "SELECTED_CONVERSATION_MISSING", "Selection references missing conversations"
        )
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "training": destination / "training.jsonl",
        "ratings": destination / "ratings.jsonl",
        "provenance": destination / "provenance.jsonl",
        "selection": destination / "selection.json",
    }
    _write_jsonl(
        outputs["training"], [training_record(item).model_dump(mode="json") for item in selected]
    )
    _write_jsonl(
        outputs["ratings"],
        [
            {
                "conversation_id": item.conversation_id,
                "turns": [turn.rating.model_dump(mode="json") for turn in item.turns],
            }
            for item in selected
        ],
    )
    _write_jsonl(
        outputs["provenance"],
        [
            {
                "conversation_id": item.conversation_id,
                "source_id": item.image.source_id,
                "image_id": item.image.image_id,
                "visual_group_id": item.image.visual_group_id,
                "generation_model": item.generation_model,
                "student_processor_lock": student_processor_lock,
            }
            for item in selected
        ],
    )
    _atomic_write(
        outputs["selection"],
        json.dumps(selection.model_dump(mode="json"), sort_keys=True, indent=2).encode(),
    )
    return outputs


def _write_jsonl(path: Path, rows: Sequence[object]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    _atomic_write(path, payload.encode())


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
