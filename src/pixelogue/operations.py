"""Application operations shared by the CLI and programmatic callers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from PIL import Image
from pydantic import Field

from pixelogue.catalog import load_rubric_catalog, load_task_catalog
from pixelogue.config import PixelogueConfig, StrictModel
from pixelogue.contracts import (
    ClaimInventory,
    ConversationArtifact,
    EvidenceInventory,
    HistorySnapshot,
    ImageArtifact,
    InstructionCandidate,
    InstructionSelection,
    PublicMessage,
    QuestionFit,
    RightsRecord,
    RubricItem,
    SelectionCandidate,
    SelectionManifest,
    SourcePurpose,
    SourceRecord,
    TextPayload,
    TrainingRecord,
    TurnArtifact,
    TurnRating,
)
from pixelogue.errors import ExternalInputError
from pixelogue.images import assign_split, canonicalize_image, group_visual_sources
from pixelogue.ledger import Requirement, RequirementInventory
from pixelogue.rules import ComputationInventory, NumericValue, SetCheck, SetInventory
from pixelogue.serialization import canonical_hash


class IngestFailure(StrictModel):
    """One source rejected at a named validation boundary."""

    source_id: str
    reason: str
    message: str


class PreparedDataset(StrictModel):
    """Accepted image artifacts, rejected inputs, and stable split assignments."""

    images: tuple[ImageArtifact, ...]
    failures: tuple[IngestFailure, ...]
    splits: dict[str, str]
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class FrozenPool(StrictModel):
    """Immutable quality-candidate input to constrained selection."""

    candidates: tuple[SelectionCandidate, ...]
    pool_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class ImageEmbedder(Protocol):
    """Interface for an optional pinned visual-copy embedding model."""

    def embed(self, image: Image.Image) -> tuple[float, ...]:
        """Return one normalized visual embedding."""


def compile_configuration(config: PixelogueConfig) -> dict[str, Any]:
    """Resolve configuration, catalogs, quotas, and public JSON Schemas."""
    task_catalog = load_task_catalog()
    rubric_catalog = load_rubric_catalog()
    schemas = {
        model.__name__: model.model_json_schema()
        for model in (
            PixelogueConfig,
            SourceRecord,
            RightsRecord,
            ImageArtifact,
            PublicMessage,
            HistorySnapshot,
            InstructionCandidate,
            EvidenceInventory,
            InstructionSelection,
            TextPayload,
            QuestionFit,
            Requirement,
            RequirementInventory,
            ClaimInventory,
            NumericValue,
            ComputationInventory,
            SetCheck,
            SetInventory,
            RubricItem,
            TurnRating,
            TurnArtifact,
            ConversationArtifact,
            SelectionCandidate,
            SelectionManifest,
            TrainingRecord,
            PreparedDataset,
            FrozenPool,
        )
    }
    body = {
        "config": config.model_dump(mode="json"),
        "config_hash": config.config_hash,
        "language_quotas": config.language_quotas,
        "task_catalog": task_catalog,
        "rubric_catalog": rubric_catalog,
        "schemas": schemas,
    }
    return {**body, "compiled_hash": canonical_hash(body)}


def prepare_sources(
    sources: Sequence[SourceRecord],
    rights_records: Sequence[RightsRecord],
    image_root: Path,
    artifact_root: Path,
    *,
    seed: int,
    embeddings: Mapping[str, Sequence[float]] | None = None,
    embedder: ImageEmbedder | None = None,
    similarity_threshold: float = 0.95,
    at: datetime | None = None,
) -> PreparedDataset:
    """Validate, canonicalize, group, and split a source manifest.

    Invalid individual sources are preserved as failures so deterministic replacement can be
    performed by the caller. Duplicate identities and dangling rights references reject the whole
    manifest because continuing would make resumption ambiguous.
    """
    if len({source.source_id for source in sources}) != len(sources):
        raise ExternalInputError("DUPLICATE_SOURCE_ID", "Source IDs must be unique")
    if len({record.rights_record_id for record in rights_records}) != len(rights_records):
        raise ExternalInputError("DUPLICATE_RIGHTS_ID", "Rights record IDs must be unique")
    rights_by_id = {record.rights_record_id: record for record in rights_records}
    accepted: list[ImageArtifact] = []
    failures: list[IngestFailure] = []
    evaluated_at = at or datetime.now(UTC)
    for source in sorted(sources, key=lambda item: item.source_id):
        rights = rights_by_id.get(source.rights_record_id)
        if rights is None:
            raise ExternalInputError(
                "RIGHTS_REFERENCE_MISSING",
                f"{source.source_id} references {source.rights_record_id}",
            )
        try:
            accepted.append(
                canonicalize_image(
                    source,
                    rights,
                    image_root,
                    artifact_root,
                    at=evaluated_at,
                )
            )
        except ExternalInputError as error:
            failures.append(
                IngestFailure(
                    source_id=source.source_id,
                    reason=error.reason,
                    message=str(error),
                )
            )
    if embeddings is not None and embedder is not None:
        raise ValueError("supply existing embeddings or an embedder, not both")
    effective_embeddings: Mapping[str, Sequence[float]] = embeddings or {}
    if embedder is not None:
        computed: dict[str, Sequence[float]] = {}
        for image in accepted:
            with Image.open(artifact_root / image.full_view.relative_path) as raster:
                computed[image.image_id] = embedder.embed(raster)
        effective_embeddings = computed
    group_ids = group_visual_sources(
        accepted,
        effective_embeddings,
        threshold=similarity_threshold,
    )
    grouped = tuple(
        image.model_copy(
            update={
                "visual_group_id": group_ids[image.image_id],
                "purpose": (
                    SourcePurpose.EVALUATION
                    if any(
                        peer.purpose is SourcePurpose.EVALUATION
                        and group_ids[peer.image_id] == group_ids[image.image_id]
                        for peer in accepted
                    )
                    else image.purpose
                ),
            }
        )
        for image in accepted
    )
    splits = {
        image.image_id: (
            "evaluation"
            if image.purpose is SourcePurpose.EVALUATION
            else assign_split(image.visual_group_id, seed)
        )
        for image in grouped
    }
    body = {
        "images": [image.model_dump(mode="json") for image in grouped],
        "failures": [failure.model_dump(mode="json") for failure in failures],
        "splits": splits,
    }
    return PreparedDataset(
        images=grouped,
        failures=tuple(failures),
        splits=splits,
        manifest_hash=canonical_hash(body),
    )


def candidate_from_conversation(conversation: ConversationArtifact) -> SelectionCandidate:
    """Reduce one fully accepted conversation to solver-visible metadata."""
    if conversation.status != "QUALITY_CANDIDATE" or not conversation.turns:
        raise ExternalInputError(
            "CONVERSATION_NOT_QUALITY_CANDIDATE",
            conversation.conversation_id,
        )
    families = tuple(turn.instruction.family for turn in conversation.turns)
    tasks = tuple(turn.instruction.task_id for turn in conversation.turns)
    profiles = tuple(turn.instruction.profile for turn in conversation.turns)
    return SelectionCandidate(
        conversation_id=conversation.conversation_id,
        language=conversation.target_language,
        image_id=conversation.image.image_id,
        visual_group_id=conversation.image.visual_group_id,
        task_family=families[0],
        semantic_family=canonical_hash(tasks),
        pattern="/".join(profiles),
        purpose=conversation.image.purpose,
    )


def make_frozen_pool(conversations: Sequence[ConversationArtifact]) -> FrozenPool:
    """Freeze eligible conversations in stable identity order."""
    candidates = tuple(
        sorted(
            (
                candidate_from_conversation(conversation)
                for conversation in conversations
                if conversation.status == "QUALITY_CANDIDATE"
            ),
            key=lambda item: item.conversation_id,
        )
    )
    if len({candidate.conversation_id for candidate in candidates}) != len(candidates):
        raise ExternalInputError("DUPLICATE_CONVERSATION_ID", "Conversation IDs must be unique")
    pool_hash = canonical_hash([candidate.model_dump(mode="json") for candidate in candidates])
    return FrozenPool(candidates=candidates, pool_hash=pool_hash)


def summarize_conversations(
    conversations: Sequence[ConversationArtifact],
) -> dict[str, dict[str, int]]:
    """Count results by source so Open Images is never hidden in a mixed average."""
    summary: dict[str, dict[str, int]] = {}
    for conversation in conversations:
        source = conversation.image.dataset or "unclassified"
        counts = summary.setdefault(source, {})
        counts[conversation.status] = counts.get(conversation.status, 0) + 1
        counts["conversations"] = counts.get("conversations", 0) + 1
        counts["committed_turns"] = counts.get("committed_turns", 0) + sum(
            turn.status == "COMMITTED" for turn in conversation.turns
        )
    return {source: dict(sorted(counts.items())) for source, counts in sorted(summary.items())}
