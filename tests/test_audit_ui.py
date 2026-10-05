"""Local human annotation preserves blindness, ballots and resumable state."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import pytest
from typer.testing import CliRunner

from pixelogue.audit_ui import (
    AnnotationCriterion,
    AuditSession,
    SupplementalItem,
    SupplementalPack,
    make_audit_server,
)
from pixelogue.research_audit import AuditCase, AuditPack, build_audit_pack


def _pack(image: Path) -> AuditPack:
    """Use case IDs in an order that differs from public-history depth."""
    image.write_bytes(b"known-image")
    return build_audit_pack(
        (
            AuditCase(
                case_id="z-first",
                source_status="accepted",
                image_path=image,
                public_history=(),
                question="FIRST-QUESTION",
                answer="FIRST-ANSWER",
            ),
            AuditCase(
                case_id="m-second",
                source_status="accepted",
                image_path=image,
                public_history=("FIRST-QUESTION", "FIRST-ANSWER"),
                question="SECOND-QUESTION",
                answer="SECOND-ANSWER",
            ),
            AuditCase(
                case_id="a-third",
                source_status="accepted",
                image_path=image,
                public_history=(
                    "FIRST-QUESTION",
                    "FIRST-ANSWER",
                    "SECOND-QUESTION",
                    "SECOND-ANSWER",
                ),
                question="THIRD-QUESTION",
                answer="THIRD-ANSWER",
            ),
        ),
        rate=1.0,
        seed=13,
    )


def _id(pack: AuditPack, question: str) -> str:
    return next(item.audit_id for item in pack.items if item.case.question == question)


def _question_vote(item_id: str, **changes: Any) -> dict[str, Any]:
    return {
        "audit_id": item_id,
        "rater_id": "human-1",
        "grounding": "MET",
        "operation_match": "UNKNOWN",
        "answerability": "MET",
        "nonredundancy": "MET",
        "naturalness": "MET",
        "note": "Operation instructions were unavailable.",
        **changes,
    }


def _answer_vote(item_id: str, **changes: Any) -> dict[str, Any]:
    return {
        "audit_id": item_id,
        "rater_id": "human-1",
        "correctness": "MET",
        "evidence": "UNKNOWN",
        "note": "The detail is unclear.",
        **changes,
    }


def _submit_questions(session: AuditSession, pack: AuditPack) -> None:
    for question in ("FIRST-QUESTION", "SECOND-QUESTION", "THIRD-QUESTION"):
        session.submit("question", _question_vote(_id(pack, question)))


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _supplement(image: Path) -> SupplementalPack:
    criterion = AnnotationCriterion(
        key="visible", label="Visible evidence", help="Rate only visible evidence."
    )
    return SupplementalPack(
        items=(
            SupplementalItem(
                audit_id="1" * 64,
                image_path=image,
                kind="opportunity",
                title="Image opportunities",
                description=("Rate opportunities from the image alone.",),
                criteria=(criterion,),
            ),
            SupplementalItem(
                audit_id="2" * 64,
                image_path=image,
                kind="extraction",
                title="Extracted evidence",
                description=("SECRET-EXTRACTION-TEXT",),
                criteria=(criterion,),
            ),
        )
    )


def _supplement_vote(audit_id: str, **changes: Any) -> dict[str, Any]:
    return {
        "audit_id": audit_id,
        "rater_id": "human-1",
        "ratings": {"visible": "UNKNOWN"},
        "note": "Cannot determine from the image.",
        **changes,
    }


@contextmanager
def _server(session: AuditSession) -> Iterator[HTTPServer]:
    server = make_audit_server(session)
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        assert not thread.is_alive()


def _request(
    server: HTTPServer,
    method: str,
    path: str,
    *,
    body: str | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def test_new_session_has_no_automatic_votes(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "image.png")
    output = tmp_path / "annotations"

    AuditSession(pack, output, "human-1")

    assert json.loads((output / "state.json").read_text())
    assert _rows(output / "question-votes.jsonl") == []
    assert _rows(output / "answer-votes.jsonl") == []
    report = json.loads((output / "resolution.json").read_text())
    assert report["required_raters"] == 1
    assert report["evidence_level"] == "single_rater_provisional"
    assert report["summary"]["question"]["rated"] == 0
    assert report["summary"]["answer"]["rated"] == 0


def test_question_snapshot_reveals_history_only_after_prior_question_vote(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "image.png")
    session = AuditSession(pack, tmp_path / "annotations", "human-1")

    before = json.dumps(session.snapshot())

    assert "FIRST-QUESTION" in before
    assert "SECOND-QUESTION" not in before
    assert "THIRD-QUESTION" not in before
    assert "FIRST-ANSWER" not in before
    assert "SECOND-ANSWER" not in before
    assert "THIRD-ANSWER" not in before
    for private_field in ("case_id", "source_status", "generation_model", "selector_model"):
        assert private_field not in before

    session.submit("question", _question_vote(_id(pack, "FIRST-QUESTION")))
    after_first = json.dumps(session.snapshot())
    assert "SECOND-QUESTION" in after_first
    assert "FIRST-ANSWER" in after_first
    assert "THIRD-QUESTION" not in after_first
    assert "SECOND-ANSWER" not in after_first
    assert "THIRD-ANSWER" not in after_first


def test_future_question_cannot_be_submitted_before_history_is_unlocked(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "image.png")
    output = tmp_path / "annotations"
    session = AuditSession(pack, output, "human-1")

    with pytest.raises(ValueError):
        session.submit("question", _question_vote(_id(pack, "THIRD-QUESTION")))

    assert _rows(output / "question-votes.jsonl") == []


def test_question_items_follow_public_history_depth(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "image.png")
    session = AuditSession(pack, tmp_path / "annotations", "human-1")
    _submit_questions(session, pack)

    assert [row["question"] for row in session.snapshot()["items"]] == [
        "FIRST-QUESTION",
        "SECOND-QUESTION",
        "THIRD-QUESTION",
    ]


def test_answer_access_and_ballots_require_all_question_ballots(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "image.png")
    output = tmp_path / "annotations"
    session = AuditSession(pack, output, "human-1")
    first_id = _id(pack, "FIRST-QUESTION")
    session.submit("question", _question_vote(first_id))

    with pytest.raises(ValueError):
        session.answer(first_id)
    with pytest.raises(ValueError):
        session.submit("answer", _answer_vote(first_id))
    assert _rows(output / "answer-votes.jsonl") == []

    for question in ("SECOND-QUESTION", "THIRD-QUESTION"):
        session.submit("question", _question_vote(_id(pack, question)))
    assert session.answer(first_id) == "FIRST-ANSWER"
    session.submit("answer", _answer_vote(first_id))
    assert _rows(output / "answer-votes.jsonl") == [_answer_vote(first_id)]


def test_saved_question_vote_is_immutable_after_answer_becomes_available(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "image.png")
    output = tmp_path / "annotations"
    session = AuditSession(pack, output, "human-1")
    _submit_questions(session, pack)
    first_id = _id(pack, "FIRST-QUESTION")
    session.answer(first_id)
    saved = (output / "question-votes.jsonl").read_bytes()

    with pytest.raises(ValueError):
        session.submit("question", _question_vote(first_id, grounding="NOT_MET"))

    assert (output / "question-votes.jsonl").read_bytes() == saved


def test_saved_ballots_resume_and_unknown_is_preserved(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "image.png")
    output = tmp_path / "annotations"
    session = AuditSession(pack, output, "human-1")
    _submit_questions(session, pack)
    first_id = _id(pack, "FIRST-QUESTION")
    session.submit("answer", _answer_vote(first_id))

    resumed = AuditSession(pack, output, "human-1")

    assert resumed.snapshot() == session.snapshot()
    assert resumed.answer(first_id) == "FIRST-ANSWER"
    assert _rows(output / "answer-votes.jsonl") == [_answer_vote(first_id)]
    report = json.loads((output / "resolution.json").read_text())
    first = next(item for item in report["items"] if item["audit_id"] == first_id)
    assert first["question"]["state"] == "unknown"
    assert first["question"]["label"] is None
    assert first["answer"]["state"] == "unknown"
    assert first["answer"]["label"] is None
    assert report["summary"]["question"]["original_agreement_rate"] is None


def test_resume_rejects_another_rater(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "image.png")
    output = tmp_path / "annotations"
    AuditSession(pack, output, "human-1")

    with pytest.raises(ValueError):
        AuditSession(pack, output, "human-2")


def test_resume_rejects_changed_sampling_pack(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "image.png")
    output = tmp_path / "annotations"
    AuditSession(pack, output, "human-1")
    changed = build_audit_pack(tuple(item.case for item in pack.items), rate=1.0, seed=14)

    with pytest.raises(ValueError):
        AuditSession(changed, output, "human-1")


def test_resume_rejects_changed_image_bytes(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    pack = _pack(image)
    output = tmp_path / "annotations"
    AuditSession(pack, output, "human-1")
    image.write_bytes(b"changed-image")

    with pytest.raises(ValueError):
        AuditSession(pack, output, "human-1")


@pytest.mark.parametrize(
    "changes",
    (
        {"grounding": "PASS"},
        {"audit_id": "0" * 64},
        {"rater_id": "human-2"},
        {"note": None},
        {"extra": "not allowed"},
    ),
    ids=("invalid-verdict", "unknown-item", "wrong-rater", "null-note", "extra-field"),
)
def test_invalid_question_ballot_does_not_change_saved_votes(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    pack = _pack(tmp_path / "image.png")
    output = tmp_path / "annotations"
    session = AuditSession(pack, output, "human-1")
    before = session.snapshot()

    with pytest.raises(ValueError):
        session.submit("question", _question_vote(_id(pack, "FIRST-QUESTION"), **changes))

    assert session.snapshot() == before
    assert _rows(output / "question-votes.jsonl") == []


def test_ballot_with_missing_dimension_is_rejected(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "image.png")
    output = tmp_path / "annotations"
    session = AuditSession(pack, output, "human-1")
    incomplete = _question_vote(_id(pack, "FIRST-QUESTION"))
    del incomplete["naturalness"]

    with pytest.raises(ValueError):
        session.submit("question", incomplete)

    assert _rows(output / "question-votes.jsonl") == []


def test_failed_atomic_state_write_does_not_accept_vote_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = _pack(tmp_path / "image.png")
    output = tmp_path / "annotations"
    session = AuditSession(pack, output, "human-1")
    before_snapshot = session.snapshot()
    before_state = (output / "state.json").read_bytes()
    replace = os.replace

    def fail_state_replace(source: Any, destination: Any, *args: Any, **kwargs: Any) -> None:
        if Path(destination) == output / "state.json":
            raise OSError("Simulated state storage failure")
        replace(source, destination, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail_state_replace)
        with pytest.raises(OSError):
            session.submit("question", _question_vote(_id(pack, "FIRST-QUESTION")))

    assert session.snapshot() == before_snapshot
    assert (output / "state.json").read_bytes() == before_state
    resumed = AuditSession(pack, output, "human-1")
    assert resumed.snapshot() == before_snapshot
    assert _rows(output / "question-votes.jsonl") == []


def test_opportunities_are_rated_before_quality_is_visible(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    pack = _pack(image)
    session = AuditSession(pack, tmp_path / "annotations", "human-1", supplement=_supplement(image))

    before = json.dumps(session.snapshot())

    assert "Image opportunities" in before
    assert "SECRET-EXTRACTION-TEXT" not in before
    assert "FIRST-QUESTION" not in before
    with pytest.raises(ValueError):
        session.submit("question", _question_vote(_id(pack, "FIRST-QUESTION")))
    with pytest.raises(ValueError):
        session.submit("supplemental", _supplement_vote("2" * 64))

    session.submit("supplemental", _supplement_vote("1" * 64))
    after = json.dumps(session.snapshot())
    assert "SECRET-EXTRACTION-TEXT" not in after
    assert "FIRST-QUESTION" in after
    assert "SECOND-QUESTION" not in after
    assert "THIRD-QUESTION" not in after


def test_extraction_requires_all_question_and_answer_quality_ballots(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    pack = _pack(image)
    output = tmp_path / "annotations"
    supplement = _supplement(image)
    session = AuditSession(pack, output, "human-1", supplement=supplement)
    opportunity_vote = _supplement_vote("1" * 64)
    extraction_vote = _supplement_vote("2" * 64)
    session.submit("supplemental", opportunity_vote)

    assert session.snapshot()["quality_complete"] is False
    assert "SECRET-EXTRACTION-TEXT" not in json.dumps(session.snapshot())
    with pytest.raises(ValueError):
        session.submit("supplemental", extraction_vote)

    _submit_questions(session, pack)
    assert session.snapshot()["quality_complete"] is False
    assert "SECRET-EXTRACTION-TEXT" not in json.dumps(session.snapshot())
    with pytest.raises(ValueError):
        session.submit("supplemental", extraction_vote)

    for question in ("FIRST-QUESTION", "SECOND-QUESTION"):
        session.submit("answer", _answer_vote(_id(pack, question)))
    assert session.snapshot()["quality_complete"] is False
    assert "SECRET-EXTRACTION-TEXT" not in json.dumps(session.snapshot())
    with pytest.raises(ValueError):
        session.submit("supplemental", extraction_vote)
    assert _rows(output / "supplemental-votes.jsonl") == [opportunity_vote]

    session.submit("answer", _answer_vote(_id(pack, "THIRD-QUESTION")))
    assert session.snapshot()["quality_complete"] is True
    assert "SECRET-EXTRACTION-TEXT" in json.dumps(session.snapshot())
    session.submit("supplemental", extraction_vote)
    assert _rows(output / "supplemental-votes.jsonl") == [opportunity_vote, extraction_vote]
    assert (
        AuditSession(pack, output, "human-1", supplement=supplement).snapshot()
        == session.snapshot()
    )


@pytest.mark.parametrize(
    "changes",
    (
        {"ratings": {}},
        {"ratings": {"visible": "MET", "extra": "MET"}},
        {"ratings": {"visible": "PASS"}},
        {"rater_id": "human-2"},
        {"extra": "not allowed"},
    ),
    ids=("missing-criterion", "extra-criterion", "invalid-verdict", "wrong-rater", "extra-field"),
)
def test_supplemental_ballot_requires_exact_criteria(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    image = tmp_path / "image.png"
    pack = _pack(image)
    output = tmp_path / "annotations"
    session = AuditSession(pack, output, "human-1", supplement=_supplement(image))
    before = session.snapshot()

    with pytest.raises(ValueError):
        session.submit("supplemental", _supplement_vote("1" * 64, **changes))

    assert session.snapshot() == before


def test_server_binds_loopback_and_serves_only_registered_images(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "image.png")
    session = AuditSession(pack, tmp_path / "annotations", "human-1")
    unrelated = tmp_path / "private.png"
    unrelated.write_bytes(b"private-image")

    with _server(session) as server:
        assert server.server_address[0] == "127.0.0.1"
        status, payload = _request(server, "GET", f"/images/{_id(pack, 'FIRST-QUESTION')}")
        assert (status, payload) == (200, b"known-image")
        for path in (
            "/images/" + "0" * 64,
            "/images/../private.png",
            "/images/%2e%2e/private.png",
            "/private.png",
        ):
            status, payload = _request(server, "GET", path)
            assert status in (400, 404)
            assert b"private-image" not in payload


@pytest.mark.parametrize(
    "headers",
    (
        {"Host": "attacker.example"},
        {"Origin": "https://attacker.example"},
        {"Origin": "null"},
    ),
    ids=("foreign-host", "foreign-origin", "null-origin"),
)
def test_server_rejects_foreign_host_or_origin(tmp_path: Path, headers: dict[str, str]) -> None:
    pack = _pack(tmp_path / "image.png")
    output = tmp_path / "annotations"
    session = AuditSession(pack, output, "human-1")
    body = json.dumps({"kind": "question", "ballot": _question_vote(_id(pack, "FIRST-QUESTION"))})

    with _server(session) as server:
        assert _request(server, "GET", "/api/session", headers=headers)[0] == 403
        assert (
            _request(
                server,
                "POST",
                "/api/ballot",
                body=body,
                headers={"Content-Type": "application/json", **headers},
            )[0]
            == 403
        )

    assert _rows(output / "question-votes.jsonl") == []


@pytest.mark.parametrize(
    "host",
    (
        "localhost",
        "127.0.0.1",
        "[::1]",
        "localhost:50838",
        "127.0.0.1:50838",
        "[::1]:50838",
        "localhost:1",
        "127.0.0.1:65535",
    ),
)
def test_forwarded_loopback_authorities_can_load_annotation_material(
    tmp_path: Path, host: str
) -> None:
    pack = _pack(tmp_path / "image.png")
    session = AuditSession(pack, tmp_path / "annotations", "human-1")
    headers = {"Host": host, "Origin": f"http://{host}"}

    with _server(session) as server:
        status, html = _request(server, "GET", "/", headers=headers)
        assert status == 200
        assert b"<!doctype html>" in html.lower()
        status, payload = _request(server, "GET", "/api/session", headers=headers)
        assert status == 200
        assert b"FIRST-QUESTION" in payload
        assert b"FIRST-ANSWER" not in payload
        assert b"SECOND-QUESTION" not in payload
        assert _request(
            server, "GET", f"/images/{_id(pack, 'FIRST-QUESTION')}", headers=headers
        ) == (200, b"known-image")


@pytest.mark.parametrize(
    "host",
    (
        "",
        "localhost.attacker.example:50838",
        "localhost.:50838",
        "127.0.0.2:50838",
        "0.0.0.0:50838",
        "[::]:50838",
        "::1:50838",
        "http://localhost:50838",
        "user@localhost:50838",
        "localhost:50838@attacker.example",
        "localhost:50838/",
        "localhost:50838?query",
        "localhost:50838#fragment",
        "localhost:",
        "localhost:not-a-port",
        "localhost:0",
        "localhost:-1",
        "localhost:65536",
        "localhost:50838:50839",
        "[::1]:0",
        "[::1]:65536",
        "[::1]:50838/path",
    ),
)
def test_malformed_or_nonloopback_forwarded_host_cannot_read_or_submit(
    tmp_path: Path, host: str
) -> None:
    pack = _pack(tmp_path / "image.png")
    output = tmp_path / "annotations"
    session = AuditSession(pack, output, "human-1")
    before = session.snapshot()
    body = json.dumps({"kind": "question", "ballot": _question_vote(_id(pack, "FIRST-QUESTION"))})

    with _server(session) as server:
        assert _request(server, "GET", "/api/session", headers={"Host": host})[0] == 403
        assert (
            _request(
                server,
                "POST",
                "/api/ballot",
                body=body,
                headers={"Host": host, "Content-Type": "application/json"},
            )[0]
            == 403
        )

    assert session.snapshot() == before
    assert _rows(output / "question-votes.jsonl") == []


@pytest.mark.parametrize(
    "origin",
    (
        "http://localhost:50839",
        "http://127.0.0.1:50838",
        "http://[::1]:50838",
        "https://localhost:50838",
        "http://localhost:50838/",
    ),
)
def test_forwarded_requests_require_origin_to_match_browser_authority(
    tmp_path: Path, origin: str
) -> None:
    pack = _pack(tmp_path / "image.png")
    output = tmp_path / "annotations"
    session = AuditSession(pack, output, "human-1")
    body = json.dumps({"kind": "question", "ballot": _question_vote(_id(pack, "FIRST-QUESTION"))})
    headers = {"Host": "localhost:50838", "Origin": origin}

    with _server(session) as server:
        assert _request(server, "GET", "/api/session", headers=headers)[0] == 403
        assert (
            _request(
                server,
                "POST",
                "/api/ballot",
                body=body,
                headers={"Content-Type": "application/json", **headers},
            )[0]
            == 403
        )

    assert _rows(output / "question-votes.jsonl") == []


def test_different_forwarded_port_preserves_blindness_and_saved_ballots(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "image.png")
    output = tmp_path / "annotations"
    session = AuditSession(pack, output, "human-1")
    first_id = _id(pack, "FIRST-QUESTION")
    ballots = []

    with _server(session) as server:
        forwarded_port = 50838 if server.server_port != 50838 else 50839
        host = f"localhost:{forwarded_port}"
        headers = {"Host": host, "Origin": f"http://{host}"}
        assert _request(server, "GET", "/", headers=headers)[0] == 200
        status, payload = _request(server, "GET", "/api/session", headers=headers)
        assert status == 200
        assert b"FIRST-ANSWER" not in payload
        assert b"SECOND-QUESTION" not in payload
        assert _request(server, "GET", f"/api/answer/{first_id}", headers=headers)[0] == 400
        for question in ("FIRST-QUESTION", "SECOND-QUESTION", "THIRD-QUESTION"):
            ballot = _question_vote(_id(pack, question))
            ballots.append(ballot)
            status, payload = _request(
                server,
                "POST",
                "/api/ballot",
                body=json.dumps({"kind": "question", "ballot": ballot}),
                headers={"Content-Type": "application/json", **headers},
            )
            assert status == 200
            assert json.loads(payload) == {"saved": True}
        status, payload = _request(server, "GET", f"/api/answer/{first_id}", headers=headers)
        assert status == 200
        assert json.loads(payload) == {"answer": "FIRST-ANSWER"}
        status, payload = _request(
            server,
            "POST",
            "/api/ballot",
            body=json.dumps({"kind": "answer", "ballot": _answer_vote(first_id)}),
            headers={"Content-Type": "application/json", **headers},
        )
        assert status == 200
        assert json.loads(payload) == {"saved": True}

    assert _rows(output / "question-votes.jsonl") == ballots
    assert _rows(output / "answer-votes.jsonl") == [_answer_vote(first_id)]
    assert AuditSession(pack, output, "human-1").snapshot() == session.snapshot()


@pytest.mark.parametrize(
    "body",
    (
        "",
        "{",
        "[]",
        '{"kind":"question","ballot":{}}',
        '{"kind":"question","ballot":{},"extra":true}',
        '{"kind":"unrecognized","ballot":{}}',
    ),
    ids=("empty", "broken-json", "array", "empty-ballot", "extra-field", "unknown-kind"),
)
def test_bad_http_submission_keeps_votes_empty(tmp_path: Path, body: str) -> None:
    pack = _pack(tmp_path / "image.png")
    output = tmp_path / "annotations"
    session = AuditSession(pack, output, "human-1")
    before = session.snapshot()

    with _server(session) as server:
        status, _ = _request(
            server,
            "POST",
            "/api/ballot",
            body=body,
            headers={"Content-Type": "application/json"},
        )

    assert status == 400
    assert session.snapshot() == before
    assert _rows(output / "question-votes.jsonl") == []
    assert _rows(output / "answer-votes.jsonl") == []


def test_http_question_flow_saves_votes_before_unlocking_answers(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "image.png")
    output = tmp_path / "annotations"
    session = AuditSession(pack, output, "human-1")
    first_id = _id(pack, "FIRST-QUESTION")

    with _server(session) as server:
        status, payload = _request(server, "GET", "/api/session")
        assert status == 200
        assert b"FIRST-ANSWER" not in payload
        assert _request(server, "GET", f"/api/answer/{first_id}")[0] == 400
        for question in ("FIRST-QUESTION", "SECOND-QUESTION", "THIRD-QUESTION"):
            ballot = _question_vote(_id(pack, question))
            status, payload = _request(
                server,
                "POST",
                "/api/ballot",
                body=json.dumps({"kind": "question", "ballot": ballot}),
                headers={
                    "Content-Type": "application/json",
                    "Origin": f"http://127.0.0.1:{server.server_port}",
                },
            )
            assert status == 200
            assert json.loads(payload) == {"saved": True}
        status, payload = _request(server, "GET", f"/api/answer/{first_id}")
        assert status == 200
        assert json.loads(payload) == {"answer": "FIRST-ANSWER"}

    assert len(_rows(output / "question-votes.jsonl")) == 3
    assert AuditSession(pack, output, "human-1").snapshot()["question_locked"] is True


@pytest.mark.parametrize("required_raters", ("0", "1"), ids=("reject-zero", "accept-one"))
def test_audit_resolve_cli_accepts_one_rater_and_rejects_zero(
    tmp_path: Path, required_raters: str
) -> None:
    from pixelogue.cli import app

    pack = _pack(tmp_path / "image.png")
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(pack.model_dump_json())
    question_path = tmp_path / "question-votes.jsonl"
    question_path.write_text(json.dumps(_question_vote(_id(pack, "FIRST-QUESTION"))) + "\n")
    answer_path = tmp_path / "answer-votes.jsonl"
    answer_path.write_text("")
    output = tmp_path / "resolution.json"

    result = CliRunner().invoke(
        app,
        [
            "audit-resolve",
            "--pack",
            str(pack_path),
            "--question-votes",
            str(question_path),
            "--answer-votes",
            str(answer_path),
            "--output",
            str(output),
            "--required-raters",
            required_raters,
        ],
    )

    if required_raters == "0":
        assert result.exit_code == 2
        assert not output.exists()
    else:
        assert result.exit_code == 0, result.output
        report = json.loads(output.read_text())
        assert report["required_raters"] == 1
        assert report["evidence_level"] == "single_rater_provisional"
