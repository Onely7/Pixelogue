"""Exact, model-bound acceptance calibration for specialist validators."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import Field, model_validator

from pixelogue.config import StrictModel
from pixelogue.serialization import canonical_hash

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class CalibrationObservation(StrictModel):
    """One held-out, independently labelled image and validator result."""

    case_id: str = Field(min_length=1)
    image_group_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    split: Literal["development", "confirmation"]
    gold_accept: bool
    verdict: Literal["MET", "NOT_MET", "UNKNOWN"]
    model_locks: tuple[Sha256, Sha256]
    validator_version: str = Field(min_length=1)
    response_artifact_hashes: tuple[Sha256, Sha256]


class CalibrationCertificate(StrictModel):
    """A computed decision bound to exact inputs, model roles, and validator code."""

    task_id: str
    domain: str
    model_locks: tuple[Sha256, Sha256]
    validator_version: str
    observations_hash: Sha256
    positive_images: int
    negative_images: int
    accepted_positive: int
    accepted_negative: int
    false_accept_upper_95: float
    positive_accept_lower_95: float
    eligible: bool


class CalibrationManifest(StrictModel):
    """Immutable confirmation evidence and its recomputed conclusion."""

    observations: tuple[CalibrationObservation, ...]
    certificates: tuple[CalibrationCertificate, ...]

    @model_validator(mode="after")
    def verify_certificates(self) -> CalibrationManifest:
        """Reject hand-edited conclusions or certificates without evidence."""
        groups: dict[tuple[str, str, tuple[str, str], str], list[CalibrationObservation]] = {}
        for item in self.observations:
            key = (item.task_id, item.domain, item.model_locks, item.validator_version)
            groups.setdefault(key, []).append(item)
        computed = tuple(
            calibrate_observations(items)
            for _, items in sorted(groups.items(), key=lambda item: item[0])
        )
        if computed != self.certificates:
            raise ValueError("Calibration certificates differ from their observations")
        return self


def model_calibration_lock(endpoint: object) -> str:
    """Bind calibration to a served model and image processor, not a local port."""
    from pixelogue.config import ModelEndpoint

    if not isinstance(endpoint, ModelEndpoint):
        raise TypeError("Calibration requires a configured model endpoint")
    return canonical_hash(
        {
            "repo_id": endpoint.repo_id,
            "revision": endpoint.revision,
            "processor_revision": endpoint.processor_revision,
            "dtype": endpoint.dtype,
            "quantization": endpoint.quantization,
            "max_model_len": endpoint.max_model_len,
        }
    )


def eligible_domains(
    manifest_path: object,
    task_id: str,
    model_locks: tuple[str, str],
    validator_version: str,
) -> tuple[str, ...]:
    """Return only domains certified for the active model pair and code."""
    from pathlib import Path

    from pixelogue.io import read_json

    if not isinstance(manifest_path, Path):
        return ()
    manifest = read_json(manifest_path, CalibrationManifest)
    return tuple(
        certificate.domain
        for certificate in manifest.certificates
        if certificate.task_id == task_id
        and certificate.model_locks == model_locks
        and certificate.validator_version == validator_version
        and certificate.eligible
    )


def _binomial_cdf(successes: int, trials: int, probability: float) -> float:
    """Evaluate the exact binomial CDF with stable log probabilities."""
    if probability == 0:
        return 1.0
    if probability == 1:
        return 0.0 if successes < trials else 1.0
    terms = (
        math.exp(
            math.lgamma(trials + 1)
            - math.lgamma(index + 1)
            - math.lgamma(trials - index + 1)
            + index * math.log(probability)
            + (trials - index) * math.log1p(-probability)
        )
        for index in range(successes + 1)
    )
    return min(1.0, math.fsum(terms))


def upper_false_accept_bound(successes: int, trials: int) -> float:
    """Return the one-sided 95% Clopper-Pearson upper binomial bound."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("A nonempty valid negative sample is required")
    if successes == trials:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(60):
        midpoint = (low + high) / 2
        if _binomial_cdf(successes, trials, midpoint) > 0.05:
            low = midpoint
        else:
            high = midpoint
    return high


def lower_positive_accept_bound(successes: int, trials: int) -> float:
    """Return the one-sided 95% Clopper-Pearson lower binomial bound."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("A nonempty valid positive sample is required")
    if successes == 0:
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(60):
        midpoint = (low + high) / 2
        if 1 - _binomial_cdf(successes - 1, trials, midpoint) < 0.05:
            low = midpoint
        else:
            high = midpoint
    return low


def calibrate_observations(
    observations: Sequence[CalibrationObservation],
) -> CalibrationCertificate:
    """Certify only complete held-out records from independent image groups."""
    if not observations:
        raise ValueError("Calibration needs observations")
    first = observations[0]
    identity = (first.task_id, first.domain, first.model_locks, first.validator_version)
    if any(
        (item.task_id, item.domain, item.model_locks, item.validator_version) != identity
        or item.split != "confirmation"
        for item in observations
    ):
        raise ValueError("Confirmation records must share the exact calibrated identity")
    if len({item.case_id for item in observations}) != len(observations):
        raise ValueError("Calibration case IDs must be unique")
    if len({item.image_group_id for item in observations}) != len(observations):
        raise ValueError("Repeated image groups are not independent samples")
    positives = [item for item in observations if item.gold_accept]
    negatives = [item for item in observations if not item.gold_accept]
    if not positives or not negatives:
        raise ValueError("Both positive and negative confirmation images are required")
    accepted_positive = sum(item.verdict == "MET" for item in positives)
    accepted_negative = sum(item.verdict == "MET" for item in negatives)
    false_upper = upper_false_accept_bound(accepted_negative, len(negatives))
    positive_lower = lower_positive_accept_bound(accepted_positive, len(positives))
    return CalibrationCertificate(
        task_id=first.task_id,
        domain=first.domain,
        model_locks=first.model_locks,
        validator_version=first.validator_version,
        observations_hash=canonical_hash(
            [item.model_dump(mode="json") for item in sorted(observations, key=lambda x: x.case_id)]
        ),
        positive_images=len(positives),
        negative_images=len(negatives),
        accepted_positive=accepted_positive,
        accepted_negative=accepted_negative,
        false_accept_upper_95=false_upper,
        positive_accept_lower_95=positive_lower,
        eligible=false_upper <= 0.05 and positive_lower >= 0.80,
    )
