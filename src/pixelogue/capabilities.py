"""Answer-keyed capability evaluation over procedurally generated fixtures."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from pixelogue.config import ModelEndpoint, StrictModel
from pixelogue.contracts import GateVerdict, RubricVerdict
from pixelogue.errors import ExternalInputError
from pixelogue.evaluation import consensus
from pixelogue.fixtures import FixtureRecord
from pixelogue.serving import ModelImage, ModelResponse

OutputModel = TypeVar("OutputModel", bound=BaseModel)


class CapabilityClient(Protocol):
    """Minimum inference interface needed by capability evaluation."""

    endpoint: ModelEndpoint

    def invoke(
        self,
        stage: str,
        payload: dict[str, Any],
        images: tuple[ModelImage, ...],
        response_model: type[OutputModel],
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
        bypass_cache: bool = False,
    ) -> ModelResponse:
        """Return one schema-validated response."""


class CapabilityOutcome(StrictModel):
    """Dual-judge result for one positive or negative fixture."""

    fixture_id: str
    split: str
    stratum: str
    expected_accept: bool
    verdict: GateVerdict
    correct: bool


class CapabilityReport(StrictModel):
    """Per-stratum counts kept separate for development and confirmation."""

    outcomes: tuple[CapabilityOutcome, ...]
    counts: dict[str, dict[str, int]]


def evaluate_capabilities(
    records: Sequence[FixtureRecord],
    fixture_root: Path,
    judges: Sequence[CapabilityClient],
    *,
    seed: int,
) -> CapabilityReport:
    """Evaluate fixture answers without exposing private expected labels to models."""
    if len(judges) != 2:
        raise ValueError("capability evaluation requires exactly two judges")
    outcomes: list[CapabilityOutcome] = []
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    root = fixture_root.resolve(strict=True)
    for index, record in enumerate(records):
        image_path = (root / record.image_path).resolve(strict=True)
        try:
            image_path.relative_to(root)
        except ValueError as error:
            raise ExternalInputError("FIXTURE_PATH_ESCAPE", record.image_path) from error
        image_bytes = image_path.read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != record.image_sha256:
            raise ExternalInputError("FIXTURE_HASH_MISMATCH", record.fixture_id)
        model_image = ModelImage(
            view_id=f"fixture:{record.fixture_id}",
            path=image_path,
            encoded_sha256=record.image_sha256,
            media_type="image/png",
        )
        payload = {
            "target_language": record.target_language,
            "public_history": [],
            "question": record.question,
            "candidate_answer": record.candidate_answer,
            "image_views": [
                {"view_id": model_image.view_id, "encoded_sha256": model_image.encoded_sha256}
            ],
            "criterion": {
                "template_id": "CAPABILITY_CORRECTNESS",
                "question": "Is the candidate answer fully correct and responsive to the image?",
                "met_anchor": "All answer content is supported and follows the requested format.",
                "not_met_anchor": "Any content is wrong, unsupported, or violates the format.",
            },
        }
        votes = [
            judge.invoke(
                "rubric_item",
                payload,
                (model_image,),
                RubricVerdict,
                max_tokens=512,
                temperature=0.0,
                seed=seed + index,
                bypass_cache=True,
            ).value
            for judge in judges
        ]
        verdict = consensus(
            [GateVerdict(vote.verdict) for vote in votes if isinstance(vote, RubricVerdict)]
        )
        predicted_accept = verdict is GateVerdict.MET
        correct = predicted_accept == record.expected_accept and verdict in {
            GateVerdict.MET,
            GateVerdict.NOT_MET,
        }
        outcome = CapabilityOutcome(
            fixture_id=record.fixture_id,
            split=record.split,
            stratum=record.stratum,
            expected_accept=record.expected_accept,
            verdict=verdict,
            correct=correct,
        )
        outcomes.append(outcome)
        key = f"{record.split}/{record.stratum}"
        counts[key]["total"] += 1
        counts[key]["correct"] += int(correct)
        counts[key]["unknown"] += int(verdict is GateVerdict.UNKNOWN)
    return CapabilityReport(
        outcomes=tuple(outcomes),
        counts={key: dict(value) for key, value in sorted(counts.items())},
    )
