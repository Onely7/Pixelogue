"""Versioned operation definitions, independent of model and transport code."""

from __future__ import annotations

from collections import Counter
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Text = Annotated[str, Field(min_length=1)]
Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)]


class CatalogModel(BaseModel):
    """Reject misspelled fields and implicit coercion in catalog definitions."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class TaskFamily(CatalogModel):
    """A routing family of explicitly listed operations."""

    label: Text
    task_ids: Annotated[tuple[Identifier, ...], Field(min_length=1)]


class VerificationContract(CatalogModel):
    """Required verification semantics, without claiming implementation availability."""

    contract: Text
    applies_when: Text


class TaskParameter(CatalogModel):
    """One public choice that a question binds for its operation.

    ``text`` is free text, ``choice`` one of ``values``, ``list`` a list of distinct strings
    within the item bounds, and ``integer`` a whole number.
    """

    kind: Literal["text", "choice", "list", "integer"]
    description: Text
    required: bool
    values: Annotated[tuple[Text, ...], Field(min_length=1)] | None = None
    min_items: Annotated[int, Field(ge=1)] | None = None
    max_items: Annotated[int, Field(ge=1)] | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> TaskParameter:
        """Keep each field to the kind that uses it."""
        if (self.kind == "choice") != (self.values is not None):
            raise ValueError("Only choice parameters list values, and every choice lists them")
        if self.values is not None and len(set(self.values)) != len(self.values):
            raise ValueError("Choice values must be distinct")
        if self.kind != "list" and (self.min_items is not None or self.max_items is not None):
            raise ValueError("Item bounds apply only to list parameters")
        if (
            self.min_items is not None
            and self.max_items is not None
            and self.min_items > self.max_items
        ):
            raise ValueError("min_items cannot exceed max_items")
        return self


class TaskDefinition(CatalogModel):
    """One operation with its scope guards, verification contracts and public choices."""

    id: Identifier
    family: Identifier
    label: Text
    status: Literal["core", "extension"]
    definition: Text
    required_capabilities: Annotated[tuple[Identifier, ...], Field(min_length=1)]
    eligibility_checks: Annotated[tuple[Identifier, ...], Field(min_length=1)]
    verification_contracts: Annotated[tuple[Identifier, ...], Field(min_length=1)]
    parameters: dict[Identifier, TaskParameter]
    example_question: Text
    do_not_infer: Text
    answer_format: Text | None = None
    related_finevision_subsets: tuple[Text, ...] = ()

    @model_validator(mode="after")
    def validate_contract(self) -> TaskDefinition:
        """Require the scope and dual-review guards and a validator for every extension."""
        for names in (
            self.required_capabilities,
            self.eligibility_checks,
            self.verification_contracts,
            self.related_finevision_subsets,
        ):
            if len(names) != len(set(names)):
                raise ValueError("Repeated task reference")
        if "scope_resolved" not in self.eligibility_checks:
            raise ValueError("Every operation must bind its scope")
        if "dual_visual_review" not in self.verification_contracts:
            raise ValueError("Every operation needs dual visual review")
        if self.status == "extension" and len(self.verification_contracts) < 2:
            raise ValueError("Extensions need a specialized validator")
        if "target" in self.parameters:
            raise ValueError("The target locator is implicit in every operation")
        return self


class InputContract(CatalogModel):
    """Single-image facts and permitted public context."""

    source_images: Literal[1]
    raw_content_inputs: tuple[Literal["image_pixels"], ...]
    permitted_context: tuple[Text, ...]
    forbidden_context: tuple[Text, ...]
    image_only_interpretation: Text
    multi_panel_policy: Text
    source_rights: Text


class TaskCatalog(CatalogModel):
    """Validated v8 taxonomy; runtime admission is a separate decision."""

    version: Literal["8.0"]
    families: dict[Identifier, TaskFamily]
    capabilities: dict[Identifier, Text]
    eligibility_checks: dict[Identifier, Text]
    verification_contracts: dict[Identifier, VerificationContract]
    input_contract: InputContract
    global_admission_rules: tuple[Text, ...]
    tasks: tuple[TaskDefinition, ...]

    @model_validator(mode="after")
    def validate_references(self) -> TaskCatalog:
        """Reconcile family membership and require every vocabulary entry to be used."""
        by_id = {task.id: task for task in self.tasks}
        if len(by_id) != len(self.tasks):
            raise ValueError("Task IDs must be unique")
        members = [name for family in self.families.values() for name in family.task_ids]
        if Counter(members) != Counter(by_id.keys()):
            raise ValueError("Each task must belong to exactly one family")
        for task in self.tasks:
            if (
                task.family not in self.families
                or task.id not in self.families[task.family].task_ids
            ):
                raise ValueError(f"Invalid family for {task.id}")
        for field, vocabulary in (
            ("required_capabilities", self.capabilities),
            ("eligibility_checks", self.eligibility_checks),
            ("verification_contracts", self.verification_contracts),
        ):
            used = {name for task in self.tasks for name in getattr(task, field)}
            if unknown := used - vocabulary.keys():
                raise ValueError(f"Unknown {field}: {sorted(unknown)}")
            if unused := vocabulary.keys() - used:
                raise ValueError(f"Unused {field}: {sorted(unused)}")
        return self
