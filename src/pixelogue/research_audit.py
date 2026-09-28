"""Blind local audit packs and independent human rating resolution."""

from __future__ import annotations

import csv
import html
import math
import os
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from pixelogue.config import StrictModel
from pixelogue.contracts import ConversationArtifact
from pixelogue.errors import ExternalInputError
from pixelogue.io import read_json, write_json, write_jsonl
from pixelogue.serialization import canonical_hash

Verdict = Literal["MET", "NOT_MET", "UNKNOWN"]
AuditStatus = Literal["accepted", "rejected", "abstained"]


class AuditCase(StrictModel):
    """Public material and sampling stratum, without model or automatic verdicts."""

    case_id: str = Field(min_length=1)
    source_status: AuditStatus
    image_path: Path
    public_history: tuple[str, ...]
    question: str | None = None
    answer: str | None = None

    @model_validator(mode="after")
    def validate_available_text(self) -> AuditCase:
        """Keep unanswered and pre-question stops explicit in the sampling frame."""
        if self.question == "" or self.answer == "":
            raise ValueError("Audit text must be non-empty when present")
        if self.answer is not None and self.question is None:
            raise ValueError("An audit answer requires a public question")
        if self.source_status == "accepted" and self.answer is None:
            raise ValueError("An accepted turn requires a public answer")
        return self


class AuditItem(StrictModel):
    """One selected case with an opaque identity for both separate ballots."""

    audit_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    case: AuditCase


class AuditPack(StrictModel):
    """Frozen sampling frame and selected blind cases."""

    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    requested_rate: Annotated[float, Field(gt=0, le=1)]
    frame_counts: dict[str, int]
    selected_counts: dict[str, int]
    actual_rates: dict[str, float | None]
    items: tuple[AuditItem, ...]
    pack_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def verify_hash(self) -> AuditPack:
        """Reject changed sampling details when a pack is resumed."""
        if self.pack_hash != canonical_hash(self.model_dump(mode="json", exclude={"pack_hash"})):
            raise ValueError("Audit pack identity changed")
        return self


class QuestionBallot(StrictModel):
    """Question-only human ratings; candidate answers never enter this ballot."""

    audit_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    rater_id: str = Field(min_length=1)
    grounding: Verdict
    operation_match: Verdict
    answerability: Verdict
    nonredundancy: Verdict
    naturalness: Verdict
    note: str


class AnswerBallot(StrictModel):
    """Answer-quality human rating on a separate form."""

    audit_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    rater_id: str = Field(min_length=1)
    correctness: Verdict
    evidence: Verdict
    note: str


class Adjudication(StrictModel):
    """Explicit human decision after disagreement, without replacing original votes."""

    audit_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    ballot: Literal["question", "answer"]
    adjudicator_id: str = Field(min_length=1)
    label: Verdict
    reason: str = Field(min_length=1)


def audit_cases_from_conversations(
    conversations: tuple[ConversationArtifact, ...], artifact_root: Path
) -> tuple[tuple[AuditCase, ...], int]:
    """Build a turn-level audit frame, retaining stops without a public question."""
    cases: list[AuditCase] = []
    skipped_errors = 0
    root = artifact_root.resolve()
    turn_status: dict[str, AuditStatus] = {
        "COMMITTED": "accepted",
        "REJECTED": "rejected",
        "ABSTAINED": "abstained",
    }
    conversation_status: dict[str, AuditStatus] = {
        "REJECTED": "rejected",
        "ABSTAINED": "abstained",
    }
    for conversation in conversations:
        if conversation.status == "ERROR":
            skipped_errors += 1
            continue
        image_path = (root / conversation.image.full_view.relative_path).resolve()
        if not image_path.is_relative_to(root) or not image_path.is_file():
            raise ExternalInputError("AUDIT_IMAGE_MISSING", "Audit image view is unavailable")
        history: list[str] = []
        for turn in conversation.turns:
            if turn.status == "ERROR":
                skipped_errors += 1
                break
            cases.append(
                AuditCase(
                    case_id=f"{conversation.conversation_id}:turn:{turn.turn_index}",
                    source_status=turn_status[turn.status],
                    image_path=image_path,
                    public_history=tuple(history),
                    question=turn.question.content,
                    answer=turn.answer.content,
                )
            )
            if turn.status == "COMMITTED":
                history.extend((turn.question.content, turn.answer.content))
        else:
            if conversation.status in conversation_status and (
                not conversation.turns or conversation.turns[-1].status == "COMMITTED"
            ):
                cases.append(
                    AuditCase(
                        case_id=f"{conversation.conversation_id}:stop:{len(conversation.turns) + 1}",
                        source_status=conversation_status[conversation.status],
                        image_path=image_path,
                        public_history=tuple(history),
                    )
                )
    return tuple(cases), skipped_errors


def build_audit_pack(cases: tuple[AuditCase, ...], *, rate: float, seed: int) -> AuditPack:
    """Sample each acceptance stratum deterministically and record its full denominator."""
    if not 0 < rate <= 1 or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Audit needs a valid rate and distinct case IDs")
    input_hash = canonical_hash([case.model_dump(mode="json") for case in cases])
    frame = Counter(case.source_status for case in cases)
    chosen: list[AuditCase] = []
    for status in ("accepted", "rejected", "abstained"):
        candidates = sorted(
            (case for case in cases if case.source_status == status),
            key=lambda case: canonical_hash({"seed": seed, "case": case.case_id}),
        )
        chosen.extend(candidates[: math.ceil(rate * len(candidates))])
    items = tuple(
        AuditItem(audit_id=canonical_hash({"input": input_hash, "case": case.case_id}), case=case)
        for case in sorted(chosen, key=lambda case: case.case_id)
    )
    selected = Counter(item.case.source_status for item in items)
    body = {
        "input_hash": input_hash,
        "seed": seed,
        "requested_rate": rate,
        "frame_counts": {status: frame[status] for status in ("accepted", "rejected", "abstained")},
        "selected_counts": {
            status: selected[status] for status in ("accepted", "rejected", "abstained")
        },
        "actual_rates": {
            status: selected[status] / frame[status] if frame[status] else None
            for status in ("accepted", "rejected", "abstained")
        },
        "items": items,
    }
    return AuditPack(
        **body,
        pack_hash=canonical_hash(
            {
                **body,
                "items": [item.model_dump(mode="json") for item in items],
            }
        ),
    )


def _card(item: AuditItem, output_dir: Path, *, show_answer: bool) -> str:
    image = os.path.relpath(item.case.image_path.resolve(), output_dir.resolve())
    history = "".join(f"<li>{html.escape(message)}</li>" for message in item.case.public_history)
    question = (
        html.escape(item.case.question)
        if item.case.question is not None
        else "No public question was recorded."
    )
    answer = (
        f"<h3>Answer</h3><p>{html.escape(item.case.answer)}</p>"
        if show_answer and item.case.answer is not None
        else "<h3>Answer</h3><p>No public answer was recorded.</p>"
        if show_answer
        else ""
    )
    return (
        f"<article><h2>{html.escape(item.audit_id)}</h2>"
        f'<img src="{html.escape(image, quote=True)}" alt="Audit image" />'
        f"<h3>Public history</h3><ol>{history}</ol>"
        f"<h3>Question</h3><p>{question}</p>{answer}</article>"
    )


def write_audit_pack(pack: AuditPack, output_dir: Path) -> None:
    """Write separate, static question and answer views plus empty JSONL templates."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "pack.json").exists() and read_json(
        output_dir / "pack.json", AuditPack
    ) != pack:
        raise ExternalInputError("AUDIT_PACK_CHANGED", "Existing audit sampling frame differs")
    write_json(output_dir / "pack.json", pack)
    for name, show_answer in (("questions", False), ("answers", True)):
        cards = "\n".join(_card(item, output_dir, show_answer=show_answer) for item in pack.items)
        page = (
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            "<style>body{font:16px sans-serif;max-width:60rem;margin:auto}"
            "article{border:1px solid #bbb;padding:1rem;margin:2rem 0}"
            "img{max-width:100%;height:auto}</style>"
            f"<title>{name.title()} audit</title><h1>{name.title()} audit</h1>{cards}</html>"
        )
        (output_dir / f"{name}.html").write_text(page, encoding="utf-8")
    write_jsonl(
        output_dir / "question-ballot.template.jsonl",
        (
            {
                "audit_id": item.audit_id,
                "rater_id": None,
                "grounding": None,
                "operation_match": None,
                "answerability": None,
                "nonredundancy": None,
                "naturalness": None,
                "note": "",
            }
            for item in pack.items
            if item.case.question is not None
        ),
    )
    write_jsonl(
        output_dir / "answer-ballot.template.jsonl",
        (
            {
                "audit_id": item.audit_id,
                "rater_id": None,
                "correctness": None,
                "evidence": None,
                "note": "",
            }
            for item in pack.items
            if item.case.answer is not None
        ),
    )


def resolve_audit(
    pack: AuditPack,
    question_votes: tuple[QuestionBallot, ...],
    answer_votes: tuple[AnswerBallot, ...],
    adjudications: tuple[Adjudication, ...] = (),
    *,
    required_raters: int = 3,
) -> dict[str, Any]:
    """Keep all original human votes and unresolved states in the report."""
    if required_raters < 2:
        raise ValueError("Independent audit needs at least two raters")
    ids = {item.audit_id for item in pack.items}
    result: dict[str, Any] = {
        "pack_hash": pack.pack_hash,
        "required_raters": required_raters,
        "frame_counts": pack.frame_counts,
        "selected_counts": pack.selected_counts,
        "actual_rates": pack.actual_rates,
        "items": [],
    }
    for votes, name in ((question_votes, "question"), (answer_votes, "answer")):
        keys = [(vote.audit_id, vote.rater_id) for vote in votes]
        applicable = {
            item.audit_id
            for item in pack.items
            if (item.case.question if name == "question" else item.case.answer) is not None
        }
        if len(keys) != len(set(keys)) or any(vote.audit_id not in applicable for vote in votes):
            raise ValueError(f"Duplicate or unknown {name} vote")
    if len({(item.audit_id, item.ballot) for item in adjudications}) != len(adjudications):
        raise ValueError("Duplicate adjudications")
    if any(item.audit_id not in ids for item in adjudications):
        raise ValueError("Adjudication references unknown item")
    if any(
        (item.case.question if decision.ballot == "question" else item.case.answer) is None
        for decision in adjudications
        for item in pack.items
        if item.audit_id == decision.audit_id
    ):
        raise ValueError("Adjudication references an unavailable ballot")
    rows: list[dict[str, Any]] = []
    for item in pack.items:
        output: dict[str, object] = {
            "audit_id": item.audit_id,
            "source_status": item.case.source_status,
        }
        for kind, ballots in (("question", question_votes), ("answer", answer_votes)):
            if (item.case.question if kind == "question" else item.case.answer) is None:
                output[kind] = {
                    "state": "not_applicable",
                    "label": None,
                    "votes": [],
                    "adjudication": None,
                }
                continue
            current = [vote for vote in ballots if vote.audit_id == item.audit_id]
            # Question consensus requires all five dimensions; answer consensus requires both.
            labels = [
                tuple(
                    getattr(vote, key)
                    for key in (
                        (
                            "grounding",
                            "operation_match",
                            "answerability",
                            "nonredundancy",
                            "naturalness",
                        )
                        if kind == "question"
                        else ("correctness", "evidence")
                    )
                )
                for vote in current
            ]
            decision = next(
                (
                    vote
                    for vote in adjudications
                    if vote.audit_id == item.audit_id and vote.ballot == kind
                ),
                None,
            )
            if len(current) < required_raters:
                state = "unreviewed"
                label = None
            elif any("UNKNOWN" in label_set for label_set in labels):
                state = "unknown"
                label = None
            elif len(set(labels)) != 1:
                state = "disagreement"
                label = None
            else:
                state = "agreed"
                label = "MET" if all(value == "MET" for value in labels[0]) else "NOT_MET"
            if decision is not None and len(current) >= required_raters:
                state = "adjudicated"
                label = decision.label
            output[kind] = {
                "state": state,
                "label": label,
                "votes": [vote.model_dump(mode="json") for vote in current],
                "adjudication": decision.model_dump(mode="json") if decision is not None else None,
            }
        rows.append(output)
    result["items"] = rows
    result["summary"] = {}
    for ballot in ("question", "answer"):
        states = Counter(row[ballot]["state"] for row in rows)
        rated = len(rows) - states["unreviewed"] - states["not_applicable"]
        result["summary"][ballot] = {
            "states": dict(states),
            "original_agreement_rate": states["agreed"] / rated if rated else None,
            "rated": rated,
        }
    return result


def write_audit_resolution(report: dict[str, Any], output: Path) -> None:
    """Save human-only resolution as JSON, CSV and Markdown without collapsing unknowns."""
    write_json(output, report)
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "audit_id",
                "source_status",
                "question_state",
                "question_label",
                "answer_state",
                "answer_label",
            )
        )
        for item in report["items"]:
            writer.writerow(
                (
                    item["audit_id"],
                    item["source_status"],
                    item["question"]["state"],
                    item["question"]["label"],
                    item["answer"]["state"],
                    item["answer"]["label"],
                )
            )
    lines = [
        "# Human audit resolution",
        "",
        f"Pack: `{report['pack_hash']}`",
        "",
        "| Ballot | Rated | Original agreement rate |",
        "|---|---:|---:|",
    ]
    for ballot in ("question", "answer"):
        item = report["summary"][ballot]
        lines.append(f"| {ballot} | {item['rated']} | {item['original_agreement_rate']} |")
    lines += ["", "Unreviewed, unknown and disagreements remain separate in the JSON and CSV.", ""]
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
