"""Local human annotation with durable ballots and staged blind access."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from pixelogue.config import StrictModel
from pixelogue.errors import ExternalInputError
from pixelogue.io import read_json, write_json, write_jsonl
from pixelogue.research_audit import (
    AnswerBallot,
    AuditPack,
    QuestionBallot,
    Verdict,
    resolve_audit,
    write_audit_resolution,
)
from pixelogue.serialization import canonical_hash, strict_json_object


class AnnotationCriterion(StrictModel):
    """A public question to answer about an image or extracted claim."""

    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    help: str


class SupplementalItem(StrictModel):
    """Image-only opportunities or extracted claims, separate from dialogue ratings."""

    audit_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_path: Path
    kind: Literal["opportunity", "extraction"]
    title: str = Field(min_length=1)
    description: tuple[str, ...]
    criteria: tuple[AnnotationCriterion, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_criteria(self) -> SupplementalItem:
        """Require an unambiguous response for each criterion."""
        if len({criterion.key for criterion in self.criteria}) != len(self.criteria):
            raise ValueError("Repeated annotation criterion")
        return self


class SupplementalPack(StrictModel):
    """Frozen additional tasks, excluding private model identities and verdicts."""

    items: tuple[SupplementalItem, ...]


class SupplementalBallot(StrictModel):
    """One human's image opportunity or extraction ratings."""

    audit_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    rater_id: str = Field(min_length=1)
    ratings: dict[str, Verdict]
    note: str


class AnnotationState(StrictModel):
    """Authoritative snapshot binding votes and image exposure to the exact inputs."""

    identity: str
    rater_id: str = Field(min_length=1)
    question_votes: tuple[QuestionBallot, ...] = ()
    answer_votes: tuple[AnswerBallot, ...] = ()
    supplemental_votes: tuple[SupplementalBallot, ...] = ()
    answer_exposed: bool = False


class AuditSession:
    """Serialize a single person's submissions, without supplying judgments.

    Question ballots are final once submitted. Later public history is withheld
    until earlier questions are rated; candidate answers require all questions.
    Image-only opportunities precede every model-derived claim in this session.
    """

    def __init__(
        self,
        pack: AuditPack,
        output_dir: Path,
        rater_id: str,
        context: dict[str, Any] | None = None,
        supplement: SupplementalPack | None = None,
    ) -> None:
        """Bind resumable ballots to a pack, rater, contexts and image bytes."""
        if not rater_id.strip():
            raise ValueError("An annotation rater ID is required")
        self.pack = pack
        self.output_dir = output_dir
        self.context = context or {}
        self.supplement = supplement or SupplementalPack(items=())
        self.items = {item.audit_id: item for item in pack.items}
        self.additional = {item.audit_id: item for item in self.supplement.items}
        if len(self.items) != len(pack.items) or len(self.additional) != len(self.supplement.items):
            raise ValueError("Repeated annotation identity")
        if set(self.items) & set(self.additional) or set(self.context) - set(self.items):
            raise ValueError("Annotation context references an unknown or ambiguous identity")
        self.images = {
            **{key: item.case.image_path.resolve() for key, item in self.items.items()},
            **{key: item.image_path.resolve() for key, item in self.additional.items()},
        }
        image_hashes = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in set(self.images.values())
        }
        identity = canonical_hash(
            {
                "pack": pack.pack_hash,
                "context": self.context,
                "supplement": self.supplement.model_dump(mode="json"),
                "images": image_hashes,
            }
        )
        self.image_hashes = image_hashes
        state_path = output_dir / "state.json"
        state = (
            read_json(state_path, AnnotationState)
            if state_path.exists()
            else AnnotationState(identity=identity, rater_id=rater_id)
        )
        if state.identity != identity or state.rater_id != rater_id:
            raise ValueError("Saved annotations belong to different inputs or a different rater")
        resolve_audit(pack, state.question_votes, state.answer_votes, required_raters=1)
        for votes in (state.question_votes, state.answer_votes, state.supplemental_votes):
            if any(vote.rater_id != rater_id for vote in votes):
                raise ValueError("Saved annotations contain another rater")
        self.state = state
        if len({vote.audit_id for vote in state.supplemental_votes}) != len(
            state.supplemental_votes
        ):
            raise ValueError("Repeated supplemental ballot")
        for vote in state.supplemental_votes:
            self._validate_supplemental(vote)
        if not self.opportunities_complete and (
            state.question_votes
            or state.answer_votes
            or any(
                self.additional[vote.audit_id].kind == "extraction"
                for vote in state.supplemental_votes
            )
        ):
            raise ValueError("Saved model-derived ratings precede image-only opportunities")
        if (state.answer_votes or state.answer_exposed) and not self.answers_unlocked:
            raise ValueError("Saved answer ratings precede required question ratings")
        if not self.quality_complete and any(
            self.additional[vote.audit_id].kind == "extraction" for vote in state.supplemental_votes
        ):
            raise ValueError("Saved extraction ratings precede dialogue quality ratings")
        if not state_path.exists():
            write_json(state_path, state)
        self._export()

    @property
    def opportunities_complete(self) -> bool:
        """Return whether image-only ratings are saved before showing model outputs."""
        saved = {vote.audit_id for vote in self.state.supplemental_votes}
        return all(
            item.audit_id in saved for item in self.supplement.items if item.kind == "opportunity"
        )

    @property
    def answers_unlocked(self) -> bool:
        """Return whether every applicable question has an explicit human ballot."""
        saved = {vote.audit_id for vote in self.state.question_votes}
        return self.opportunities_complete and all(
            item.audit_id in saved for item in self.pack.items if item.case.question is not None
        )

    @property
    def quality_complete(self) -> bool:
        """Return whether quality judgments are final before revealing extracted claims."""
        saved = {vote.audit_id for vote in self.state.answer_votes}
        return self.answers_unlocked and all(
            item.audit_id in saved for item in self.pack.items if item.case.answer is not None
        )

    def _pending_depth(self) -> int:
        saved = {vote.audit_id for vote in self.state.question_votes}
        return min(
            (
                len(item.case.public_history)
                for item in self.pack.items
                if item.case.question is not None and item.audit_id not in saved
            ),
            default=10**9,
        )

    def snapshot(self) -> dict[str, Any]:
        """Return public material while masking future history and candidate answers."""
        depth = self._pending_depth()
        rows = []
        for item in sorted(self.pack.items, key=lambda row: len(row.case.public_history)):
            available = self.opportunities_complete and len(item.case.public_history) <= depth
            context = self.context.get(item.audit_id)
            operation = context.get("operation") if isinstance(context, dict) else context
            if isinstance(operation, dict):
                operation = json.dumps(operation, ensure_ascii=False, indent=2)
            rows.append(
                {
                    "audit_id": item.audit_id,
                    "available": available,
                    "has_question": item.case.question is not None,
                    "question": item.case.question if available else None,
                    "public_history": list(item.case.public_history) if available else [],
                    "operation": operation if available else None,
                    "has_answer": item.case.answer is not None,
                }
            )
        additional = [
            item.model_dump(mode="json", exclude={"image_path"})
            for item in self.supplement.items
            if item.kind == "opportunity" or self.quality_complete
        ]
        return {
            "pack_hash": self.state.identity,
            "rater_id": self.state.rater_id,
            "required_raters": 1,
            "evidence_level": "single_rater_provisional",
            "question_locked": self.state.answer_exposed,
            "answers_unlocked": self.answers_unlocked,
            "opportunities_complete": self.opportunities_complete,
            "quality_complete": self.quality_complete,
            "items": rows,
            "supplemental_items": additional,
            "supplemental_counts": dict(Counter(item.kind for item in self.supplement.items)),
            "question_votes": [vote.model_dump(mode="json") for vote in self.state.question_votes],
            "answer_votes": [vote.model_dump(mode="json") for vote in self.state.answer_votes],
            "supplemental_votes": [
                vote.model_dump(mode="json") for vote in self.state.supplemental_votes
            ],
        }

    def _validate_supplemental(self, vote: SupplementalBallot) -> SupplementalItem:
        item = self.additional.get(vote.audit_id)
        if item is None or set(vote.ratings) != {criterion.key for criterion in item.criteria}:
            raise ValueError("Unknown task or incomplete annotation criteria")
        return item

    def submit(self, kind: str, payload: dict[str, Any]) -> None:
        """Validate and atomically save explicit ratings; submitted ballots are final.

        Raises:
            ValueError: If a ballot is out of order, incomplete or changes a saved vote.
        """
        models = {
            "question": QuestionBallot,
            "answer": AnswerBallot,
            "supplemental": SupplementalBallot,
        }
        if not isinstance(kind, str) or kind not in models:
            raise ValueError("Unknown annotation form")
        vote = models[kind].model_validate(payload)
        if vote.rater_id != self.state.rater_id:
            raise ValueError("This form belongs to a different rater")
        field = f"{kind}_votes"
        votes = getattr(self.state, field)
        previous = next((row for row in votes if row.audit_id == vote.audit_id), None)
        if previous is not None:
            if vote != previous:
                raise ValueError(
                    "Submitted ballots are final; start a separate review for revisions"
                )
            self._export()
            return
        if isinstance(vote, SupplementalBallot):
            item = self._validate_supplemental(vote)
            if item.kind == "extraction" and not self.quality_complete:
                raise ValueError(
                    "Complete dialogue quality ballots before reading extracted claims"
                )
        else:
            item = self.items.get(vote.audit_id)
            if item is None:
                raise ValueError("Unknown annotation identity")
            if kind == "question":
                if not self.opportunities_complete or item.case.question is None:
                    raise ValueError("Question form is unavailable")
                if len(item.case.public_history) > self._pending_depth():
                    raise ValueError(
                        "Complete earlier questions before reading later public history"
                    )
            elif not self.answers_unlocked or item.case.answer is None:
                raise ValueError("Complete every question ballot before rating candidate answers")
        state = self.state.model_copy(update={field: (*votes, vote)})
        self._save(state)

    def _save(self, state: AnnotationState) -> None:
        write_json(self.output_dir / "state.json", state)
        self.state = state
        self._export()

    def _export(self) -> None:
        write_jsonl(self.output_dir / "question-votes.jsonl", self.state.question_votes)
        write_jsonl(self.output_dir / "answer-votes.jsonl", self.state.answer_votes)
        write_jsonl(self.output_dir / "supplemental-votes.jsonl", self.state.supplemental_votes)
        report = resolve_audit(
            self.pack, self.state.question_votes, self.state.answer_votes, required_raters=1
        )
        write_audit_resolution(report, self.output_dir / "resolution.json")
        supplemental_summary = {}
        for kind in ("opportunity", "extraction"):
            items = [item for item in self.supplement.items if item.kind == kind]
            votes = [
                vote
                for vote in self.state.supplemental_votes
                if self.additional[vote.audit_id].kind == kind
            ]
            supplemental_summary[kind] = {
                "total": len(items),
                "submitted": len(votes),
                "unreviewed": len(items) - len(votes),
                "criterion_counts": dict(
                    Counter(label for vote in votes for label in vote.ratings.values())
                ),
            }
        write_json(
            self.output_dir / "supplemental-summary.json",
            {
                "identity": self.state.identity,
                "required_raters": 1,
                "evidence_level": "single_rater_provisional",
                "items": supplemental_summary,
            },
        )

    def answer(self, audit_id: str) -> str:
        """Persist exposure before returning an existing answer after all questions."""
        if not self.answers_unlocked:
            raise ValueError("Complete every question ballot before viewing candidate answers")
        item = self.items.get(audit_id)
        if item is None or item.case.answer is None:
            raise ValueError("No candidate answer exists for this item")
        if not self.state.answer_exposed:
            self._save(self.state.model_copy(update={"answer_exposed": True}))
        return item.case.answer

    def image(self, audit_id: str) -> bytes:
        """Read only a registered image and verify that its pixels have not changed."""
        path = self.images.get(audit_id)
        if path is None:
            raise ValueError("Unknown annotation image")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != self.image_hashes[str(path)]:
            raise ValueError("Annotation image changed")
        return payload


def _is_loopback_authority(authority: str) -> bool:
    """Accept local browser addresses with ports assigned by SSH or app forwarding."""
    match = re.fullmatch(r"(?:localhost|127\.0\.0\.1|\[::1\])(?::([0-9]{1,5}))?", authority)
    return match is not None and (match[1] is None or 1 <= int(match[1]) <= 65535)


def make_audit_server(session: AuditSession, port: int = 0) -> HTTPServer:
    """Create a loopback-only server with serialized submissions and no file browsing."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _respond(self, status: int, payload: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, status: int, payload: Any) -> None:
            self._respond(
                status,
                json.dumps(payload, ensure_ascii=False).encode(),
                "application/json; charset=utf-8",
            )

        def _allowed(self) -> bool:
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            if not _is_loopback_authority(host) or (
                origin is not None and origin != f"http://{host}"
            ):
                self._json(403, {"error": "この画面からアクセスしてください。"})
                return False
            return True

        def do_GET(self) -> None:
            """Serve staged public material and known images only."""
            if not self._allowed():
                return
            path = urlsplit(self.path).path
            try:
                if path == "/":
                    self._respond(
                        200,
                        Path(__file__).with_name("audit_ui.html").read_bytes(),
                        "text/html; charset=utf-8",
                    )
                elif path == "/api/session":
                    self._json(200, session.snapshot())
                elif path == "/api/export":
                    self._json(200, session.state.model_dump(mode="json"))
                elif path.startswith("/api/answer/"):
                    self._json(200, {"answer": session.answer(path.removeprefix("/api/answer/"))})
                elif path.startswith("/images/"):
                    audit_id = path.removeprefix("/images/")
                    payload = session.image(audit_id)
                    suffix = session.images[audit_id].suffix.lower()
                    media_type = {
                        ".png": "image/png",
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".webp": "image/webp",
                    }.get(suffix)
                    if media_type is None:
                        raise ValueError("Unsupported annotation image format")
                    self._respond(200, payload, media_type)
                else:
                    self._json(404, {"error": "画面が見つかりません。"})
            except (ValueError, ExternalInputError) as error:
                self._json(400, {"error": str(error)})
            except OSError:
                self._json(
                    500, {"error": "ファイルの読み書きに失敗しました。保存先を確認してください。"}
                )

        def do_POST(self) -> None:
            """Accept complete explicit ballots from this local interface."""
            if not self._allowed():
                return
            if urlsplit(self.path).path != "/api/ballot":
                self._json(404, {"error": "送信先が見つかりません。"})
                return
            try:
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("Ballots require JSON")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024 * 1024:
                    raise ValueError("Invalid ballot length")
                payload = strict_json_object(self.rfile.read(length))
                if set(payload) != {"kind", "ballot"} or not isinstance(payload["ballot"], dict):
                    raise ValueError("Malformed ballot submission")
                session.submit(payload["kind"], payload["ballot"])
                self._json(200, {"saved": True})
            except (ValueError, ExternalInputError) as error:
                self._json(400, {"error": str(error)})
            except OSError:
                self._json(500, {"error": "保存できませんでした。再度送信してください。"})

    return HTTPServer(("127.0.0.1", port), Handler)
