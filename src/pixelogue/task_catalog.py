"""Versioned operation definitions, independent of model and transport code."""

from __future__ import annotations

from collections import Counter
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Text = Annotated[str, Field(min_length=1)]


class CatalogModel(BaseModel):
    """Reject misspelled fields and implicit coercion in catalog definitions."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class CatalogCounts(CatalogModel):
    """Declared counts independently reconciled with the actual definitions."""

    tasks: Literal[72]
    core_candidates: Literal[65]
    validator_gated_extensions: Literal[7]
    families: Literal[14]


class TaskFamily(CatalogModel):
    """A sampling family of explicitly listed semantic operations."""

    label_en: Text
    task_ids: Annotated[tuple[Text, ...], Field(min_length=1)]


class VerificationContract(CatalogModel):
    """Required verification semantics, without claiming implementation availability."""

    contract: Text
    applies_when: Text


class TaskDefinition(CatalogModel):
    """One operation with scope guards and required verification contracts."""

    number: Annotated[int, Field(ge=1, le=72)]
    id: Text
    family: Text
    label_en: Text
    status: Literal["core_candidate", "validator_gated_extension"]
    definition_en: Text
    required_capabilities: Annotated[tuple[Text, ...], Field(min_length=1)]
    eligibility_checks: Annotated[tuple[Text, ...], Field(min_length=1)]
    example_question_if_eligible: Text
    do_not_infer: Text
    verification_contracts: Annotated[tuple[Text, ...], Field(min_length=1)]
    parameters: dict[str, str | tuple[str, ...] | bool]
    inspiration_subset_numbers: tuple[int, ...]
    inspiration_subsets: tuple[str, ...]
    source_urls: tuple[str, ...]

    @model_validator(mode="after")
    def validate_contract(self) -> TaskDefinition:
        """Require explicit scope/visual checks and disabled specialist defaults."""
        for names in (
            self.required_capabilities,
            self.eligibility_checks,
            self.verification_contracts,
        ):
            if len(names) != len(set(names)):
                raise ValueError("Repeated task contract reference")
        if "scope_resolved" not in self.eligibility_checks:
            raise ValueError("Every operation must bind its scope")
        if "dual_visual_review" not in self.verification_contracts:
            raise ValueError("Every operation needs dual visual review")
        if self.status == "validator_gated_extension":
            if self.parameters.get("enabled_by_default") is not False:
                raise ValueError("Specialized extensions must be disabled by default")
            if len(self.verification_contracts) < 2:
                raise ValueError("Specialized extensions need a specialized validator")
        return self


class ProfileContract(CatalogModel):
    """Alternative answerability guards, distinct from normal capability requirements."""

    eligible_task_ids: tuple[str, ...] = ()
    eligibility: Text
    answer: Text


class Facet(CatalogModel):
    """An orthogonal modifier, never an additional operation ID."""

    values: tuple[Text, ...]
    rule: Text


class InputContract(CatalogModel):
    """Single-canvas facts and permitted public context."""

    source_images: Literal[1]
    raw_content_inputs: tuple[Literal["image_pixels"], ...]
    permitted_context: tuple[Text, ...]
    forbidden_context: tuple[Text, ...]
    image_only_interpretation: Text
    multi_panel_policy: Text
    source_rights: Text


class SourceTable(CatalogModel):
    """Research provenance excluded from all inference payloads."""

    filename: Text
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    subset_count: Literal[185]
    operation_phrase_count: Literal[527]
    limitations: Text


class CatalogProvenance(CatalogModel):
    """Identity of the supplied design and its research map."""

    proposal_version: Text
    proposal_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    source_table: SourceTable


class TaskCatalog(CatalogModel):
    """Validated v7 taxonomy; runtime admission is a separate decision."""

    version: Literal["7.0"]
    counts: CatalogCounts
    families: dict[str, TaskFamily]
    capabilities: dict[str, Text]
    eligibility_checks: dict[str, Text]
    verification_contracts: dict[str, VerificationContract]
    profile_contracts: dict[str, ProfileContract]
    facets: dict[str, Facet]
    input_contract: InputContract
    global_admission_rules: tuple[Text, ...]
    classification_rules: tuple[Text, ...]
    routing_precedence: tuple[Text, ...]
    runtime_feasibility: tuple[Text, ...]
    provenance: CatalogProvenance
    tasks: tuple[TaskDefinition, ...]

    @model_validator(mode="after")
    def validate_references(self) -> TaskCatalog:
        """Reconcile counts, family membership, and every executable reference."""
        by_id = {task.id: task for task in self.tasks}
        counts = Counter(task.status for task in self.tasks)
        if len(by_id) != self.counts.tasks or len(self.tasks) != self.counts.tasks:
            raise ValueError("Task count or unique IDs do not match declared counts")
        if {task.number for task in self.tasks} != set(range(1, self.counts.tasks + 1)):
            raise ValueError("Task numbers must cover the declared range exactly")
        if counts != {"core_candidate": 65, "validator_gated_extension": 7}:
            raise ValueError("Catalog must contain 65 core candidates and 7 extensions")
        if len(self.families) != self.counts.families:
            raise ValueError("Family count does not match declared count")
        members = [name for family in self.families.values() for name in family.task_ids]
        if Counter(members) != Counter(by_id.keys()):
            raise ValueError("Each task must belong to exactly one family")
        for task in self.tasks:
            if (
                task.family not in self.families
                or task.id not in self.families[task.family].task_ids
            ):
                raise ValueError(f"Invalid family for {task.id}")
            for names, vocabulary in (
                (task.required_capabilities, self.capabilities),
                (task.eligibility_checks, self.eligibility_checks),
                (task.verification_contracts, self.verification_contracts),
            ):
                if not set(names) <= vocabulary.keys():
                    raise ValueError(f"Unknown contract reference in {task.id}")
        if set(self.profile_contracts) != {"normal", "limitation", "false_premise"}:
            raise ValueError("Unknown or missing answerability profile")
        for profile in self.profile_contracts.values():
            if not set(profile.eligible_task_ids) <= by_id.keys():
                raise ValueError("Unknown task in answerability profile")
        return self
