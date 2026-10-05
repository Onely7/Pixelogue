"""Human audit separates ballots, preserves disagreements and records sampling rates."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from pixelogue.contracts import (
    ConversationArtifact,
    InstructionCandidate,
    PublicMessage,
    TurnArtifact,
    TurnRating,
)
from pixelogue.research_audit import (
    Adjudication,
    AnswerBallot,
    AuditCase,
    AuditPack,
    QuestionBallot,
    audit_cases_from_conversations,
    build_audit_pack,
    resolve_audit,
    write_audit_pack,
)
from pixelogue.serialization import canonical_hash


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


def test_audit_samples_statuses_separately_and_reads_legacy_pack(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"placeholder")
    cases = _cases(image)
    pack = build_audit_pack(
        cases,
        rate=0.5,
        seed=13,
        rates={"accepted": 1.0, "rejected": 0.5, "abstained": 0.5},
    )
    assert pack.selected_counts == {"accepted": 2, "rejected": 1, "abstained": 1}
    assert pack.actual_rates == {"accepted": 1.0, "rejected": 0.5, "abstained": 0.5}
    assert AuditPack.model_validate_json(pack.model_dump_json()) == pack

    legacy = build_audit_pack(cases, rate=0.5, seed=13)
    assert (
        AuditPack.model_validate_json(legacy.model_dump_json(exclude={"requested_rates"})) == legacy
    )


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


def test_single_rater_resolution_keeps_provisional_labels_and_unknowns(tmp_path: Path) -> None:
    pack = build_audit_pack(_cases(tmp_path / "image.png"), rate=1.0, seed=13)
    met_id, not_met_id, unknown_id = (item.audit_id for item in pack.items[:3])
    question_votes = (
        QuestionBallot(
            audit_id=met_id,
            rater_id="human-1",
            grounding="MET",
            operation_match="MET",
            answerability="MET",
            nonredundancy="MET",
            naturalness="MET",
            note="",
        ),
        QuestionBallot(
            audit_id=not_met_id,
            rater_id="human-1",
            grounding="MET",
            operation_match="MET",
            answerability="MET",
            nonredundancy="NOT_MET",
            naturalness="MET",
            note="Question repeats a public fact",
        ),
        QuestionBallot(
            audit_id=unknown_id,
            rater_id="human-1",
            grounding="MET",
            operation_match="UNKNOWN",
            answerability="MET",
            nonredundancy="MET",
            naturalness="MET",
            note="Operation not supplied",
        ),
    )
    answer_votes = (
        AnswerBallot(
            audit_id=met_id,
            rater_id="human-1",
            correctness="MET",
            evidence="MET",
            note="",
        ),
        AnswerBallot(
            audit_id=not_met_id,
            rater_id="human-1",
            correctness="NOT_MET",
            evidence="MET",
            note="Incorrect answer",
        ),
        AnswerBallot(
            audit_id=unknown_id,
            rater_id="human-1",
            correctness="MET",
            evidence="UNKNOWN",
            note="Image detail is unclear",
        ),
    )

    report = resolve_audit(pack, question_votes, answer_votes, required_raters=1)

    assert report["required_raters"] == 1
    for ballot in ("question", "answer"):
        rows = {item["audit_id"]: item[ballot] for item in report["items"]}
        assert (rows[met_id]["state"], rows[met_id]["label"]) == ("single_rater", "MET")
        assert (rows[not_met_id]["state"], rows[not_met_id]["label"]) == (
            "single_rater",
            "NOT_MET",
        )
        assert (rows[unknown_id]["state"], rows[unknown_id]["label"]) == ("unknown", None)
        assert all(rows[item.audit_id]["state"] == "unreviewed" for item in pack.items[3:])
        assert report["summary"][ballot]["original_agreement_rate"] is None
        assert report["summary"][ballot]["rated"] == 3


def test_one_vote_does_not_complete_default_three_rater_audit(tmp_path: Path) -> None:
    pack = build_audit_pack(_cases(tmp_path / "image.png"), rate=0.5, seed=13)
    audit_id = pack.items[0].audit_id
    question_vote = QuestionBallot(
        audit_id=audit_id,
        rater_id="human-1",
        grounding="MET",
        operation_match="MET",
        answerability="MET",
        nonredundancy="MET",
        naturalness="MET",
        note="",
    )
    answer_vote = AnswerBallot(
        audit_id=audit_id,
        rater_id="human-1",
        correctness="MET",
        evidence="MET",
        note="",
    )

    report = resolve_audit(pack, (question_vote,), (answer_vote,))

    assert report["required_raters"] == 3
    row = next(item for item in report["items"] if item["audit_id"] == audit_id)
    assert row["question"]["state"] == "unreviewed"
    assert row["answer"]["state"] == "unreviewed"
    assert row["question"]["votes"] == [question_vote.model_dump(mode="json")]
    assert row["answer"]["votes"] == [answer_vote.model_dump(mode="json")]


def test_audit_resolution_rejects_zero_required_raters(tmp_path: Path) -> None:
    pack = build_audit_pack(_cases(tmp_path / "image.png"), rate=0.5, seed=13)

    with pytest.raises(ValueError):
        resolve_audit(pack, (), (), required_raters=0)


def test_audit_frame_keeps_accepted_prefix_and_unanswered_stops(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    question = PublicMessage(message_id="q1", turn_index=1, role="user", content="What is shown?")
    answer = PublicMessage(message_id="a1", turn_index=1, role="assistant", content="A shape.")
    turn = TurnArtifact(
        turn_index=1,
        instruction=InstructionCandidate(
            candidate_id="candidate",
            task_id="object_identification",
            family="object",
            visible_scope="whole image",
            instruction_summary="Identify an object",
            required_capabilities=("visible_entity",),
        ),
        question=question,
        answer=answer,
        history_hash=canonical_hash([]),
        generation_model="pilot-model",
        selector_model="selector-model",
        rating=TurnRating(items=(), aggregate="PASS"),
        status="COMMITTED",
    )
    conversations = (
        ConversationArtifact(
            conversation_id="partial",
            image=image,
            target_language="en",
            generation_model="pilot-model",
            turns=(turn,),
            status="REJECTED",
        ),
        ConversationArtifact(
            conversation_id="no-question",
            image=image,
            target_language="en",
            generation_model="pilot-model",
            turns=(),
            status="ABSTAINED",
        ),
    )
    cases, skipped = audit_cases_from_conversations(conversations, root)
    assert skipped == 0
    assert [case.source_status for case in cases] == ["accepted", "rejected", "abstained"]
    assert cases[1].question is None
    assert cases[1].public_history == (question.content, answer.content)
    pack = build_audit_pack(cases, rate=1.0, seed=13)
    write_audit_pack(pack, tmp_path / "audit")
    assert len((tmp_path / "audit/question-ballot.template.jsonl").read_text().splitlines()) == 1
    assert len((tmp_path / "audit/answer-ballot.template.jsonl").read_text().splitlines()) == 1
    report = resolve_audit(pack, (), ())
    assert report["summary"]["question"]["states"]["not_applicable"] == 2
    assert report["summary"]["answer"]["states"]["not_applicable"] == 2
