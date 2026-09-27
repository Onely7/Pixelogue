"""Human audit separates ballots, preserves disagreements and records sampling rates."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pixelogue.research_audit import (
    Adjudication,
    AnswerBallot,
    AuditCase,
    QuestionBallot,
    build_audit_pack,
    resolve_audit,
    write_audit_pack,
)


def _cases(image: Path) -> tuple[AuditCase, ...]:
    return tuple(
        AuditCase(
            case_id=f"{status}-{index}",
            source_status=status,
            image_path=image,
            public_history=("Earlier public turn",),
            question=f"Question {index}?",
            answer=f"SECRET-ANSWER-{status}-{index}",
        )
        for status in ("accepted", "rejected", "abstained")
        for index in range(2)
    )


def test_question_pack_hides_answers_and_sampling_frame_is_explicit(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"placeholder")
    pack = build_audit_pack(_cases(image), rate=0.5, seed=13)
    write_audit_pack(pack, tmp_path / "audit")
    question_html = (tmp_path / "audit/questions.html").read_text()
    answer_html = (tmp_path / "audit/answers.html").read_text()
    assert "SECRET-ANSWER" not in question_html
    assert "SECRET-ANSWER" in answer_html
    assert pack.frame_counts == {"accepted": 2, "rejected": 2, "abstained": 2}
    assert pack.selected_counts == {"accepted": 1, "rejected": 1, "abstained": 1}
    assert pack.actual_rates["rejected"] == 0.5


def test_three_independent_votes_and_adjudication_keep_originals(tmp_path: Path) -> None:
    pack = build_audit_pack(_cases(tmp_path / "image.png"), rate=0.5, seed=13)
    audit_id = pack.items[0].audit_id
    verdicts: tuple[Literal["MET", "NOT_MET"], ...] = ("MET", "NOT_MET", "MET")
    question_votes = tuple(
        QuestionBallot(
            audit_id=audit_id,
            rater_id=f"r{index}",
            grounding=verdict,
            operation_match="MET",
            answerability="MET",
            nonredundancy="MET",
            naturalness="MET",
            note="",
        )
        for index, verdict in enumerate(verdicts)
    )
    answer_votes = tuple(
        AnswerBallot(
            audit_id=audit_id,
            rater_id=f"r{index}",
            correctness="MET",
            evidence="MET",
            note="",
        )
        for index in range(3)
    )
    report = resolve_audit(pack, question_votes, answer_votes)
    row = next(item for item in report["items"] if item["audit_id"] == audit_id)
    assert row["question"]["state"] == "disagreement"
    assert row["question"]["label"] is None
    assert row["answer"]["state"] == "agreed"
    assert any(item["question"]["state"] == "unreviewed" for item in report["items"])
    decision = Adjudication(
        audit_id=audit_id,
        ballot="question",
        adjudicator_id="lead",
        label="NOT_MET",
        reason="Image evidence was insufficient",
    )
    adjudicated = resolve_audit(pack, question_votes, answer_votes, (decision,))
    row = next(item for item in adjudicated["items"] if item["audit_id"] == audit_id)
    assert row["question"]["state"] == "adjudicated"
    assert len(row["question"]["votes"]) == 3
