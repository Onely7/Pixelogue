"""Answer-independent contracts for finite visual decisions."""

from __future__ import annotations

import math
from typing import Annotated, Literal, Protocol

from pydantic import Field, model_validator

from pixelogue.config import StrictModel
from pixelogue.serialization import canonical_hash
from pixelogue.task_evidence import ImageRegion

DecisionLabel = Literal["MET", "NOT_MET", "UNKNOWN"]
DECISION_LABELS: tuple[DecisionLabel, ...] = ("MET", "NOT_MET", "UNKNOWN")
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Revision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Nonempty = Annotated[str, Field(min_length=1, max_length=512)]
FiniteSeconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class DecisionModelIdentity(StrictModel):
    """A repository and immutable commit, independent of generator roles."""

    repository: Nonempty
    revision: Revision


class DecisionRequest(StrictModel):
    """One visual condition with only the public context permitted for it.

    The region uses normalized coordinates in the exact delivered view. A path
    is an operational locator; the image hash, view, region, public prefix and
    trial identify the decision. Reference labels and candidate answers have no
    fields in this contract. Free text still needs the caller's information-
    boundary validation.
    """

    request_id: Nonempty
    trial_id: Nonempty
    image_id: Nonempty
    image_path: Annotated[str, Field(min_length=1)]
    image_sha256: Sha256
    view_id: Nonempty
    region: ImageRegion
    question: Annotated[str, Field(min_length=1, max_length=4096)]
    public_context: Annotated[tuple[str, ...], Field(max_length=64)] = ()
    contract_version: Nonempty = "finite-decision-v1"

    @model_validator(mode="after")
    def validate_context(self) -> DecisionRequest:
        """Bound the public prefix without introducing hidden metadata."""
        if any(not item or len(item) > 16384 for item in self.public_context):
            raise ValueError("Public context entries must contain 1 to 16384 characters")
        return self

    def identity(self, model_repository: str, model_revision: str) -> str:
        """Bind a replay to its trial, public input and exact decision model.

        Moving identical bytes to another local path does not change identity.
        Different trials deliberately do, even when all visible inputs match.
        """
        return canonical_hash(
            {
                "request": self.model_dump(mode="json", exclude={"image_path"}),
                "model_repository": model_repository,
                "model_revision": model_revision,
            }
        )


class DecisionProbability(StrictModel):
    """One explicit label's finite probability."""

    label: DecisionLabel
    probability: Probability


class DecisionResult(StrictModel):
    """A private native decision and its observed transport duration.

    Probabilities are model scores, not a task calibration certificate. Missing
    token usage stays missing. Timing includes image verification and processing
    in the adapter, but model loading is recorded by the job supervisor.
    """

    request_id: Nonempty
    request_hash: Sha256
    model_repository: Nonempty
    model_revision: Revision
    verdict: DecisionLabel
    probabilities: Annotated[tuple[DecisionProbability, ...], Field(min_length=3, max_length=3)]
    elapsed_seconds: FiniteSeconds
    prompt_tokens: Annotated[int, Field(ge=0)] | None = None
    completion_tokens: Annotated[int, Field(ge=0)] | None = None
    num_model_requests: Annotated[int, Field(ge=1)] = 1
    native_protocol: Nonempty | None = None

    @model_validator(mode="after")
    def validate_distribution(self) -> DecisionResult:
        """Reject missing labels, unnormalized scores and inconsistent choices."""
        labels = tuple(item.label for item in self.probabilities)
        if len(set(labels)) != 3 or set(labels) != set(DECISION_LABELS):
            raise ValueError("Decision distribution must contain each label exactly once")
        total = math.fsum(item.probability for item in self.probabilities)
        if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-5):
            raise ValueError("Decision probabilities must sum to one")
        scores = self.probability_map
        if scores[self.verdict] != max(scores.values()):
            raise ValueError("Decision verdict must be a maximum-probability label")
        return self

    @property
    def probability_map(self) -> dict[DecisionLabel, float]:
        """Return a detached mapping, preserving the immutable stored scores."""
        return {item.label: item.probability for item in self.probabilities}

    @property
    def confidence(self) -> float:
        """Return the selected label's score without implying calibration."""
        return self.probability_map[self.verdict]

    def validate_request(self, request: DecisionRequest) -> None:
        """Reject a result replayed for different visible inputs or a new trial."""
        if self.request_id != request.request_id or self.request_hash != request.identity(
            self.model_repository, self.model_revision
        ):
            raise ValueError("Decision result does not belong to this request")


class DecisionClient(Protocol):
    """Synchronous decision service usable by the threaded synthesis pipeline."""

    def decide(self, request: DecisionRequest) -> DecisionResult:
        """Evaluate one visual condition, raising on malformed or failed output."""
        ...


class AsyncDecisionClient(Protocol):
    """Optional asynchronous decision service for bounded experiment workers."""

    async def decide(self, request: DecisionRequest) -> DecisionResult:
        """Evaluate one visual condition without blocking an event loop."""
        ...
