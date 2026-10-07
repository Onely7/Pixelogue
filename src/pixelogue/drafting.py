"""Direct question drafts and their conversion into public operation contracts.

A draft is written by the conversation's generator from the image and committed public
history alone. It names one catalog operation, its public choices and the visual scope it
used. The controller validates those choices against the catalog before any judge sees the
question; no evidence verdict, private label or answer is created here.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection
from typing import Annotated, Any

from pydantic import Field, model_validator

from pixelogue.catalog import task_catalog
from pixelogue.config import StrictModel
from pixelogue.contracts import InstructionCandidate
from pixelogue.errors import ExecutionError
from pixelogue.task_catalog import TaskDefinition
from pixelogue.task_evidence import ImageRegion, PublicParameter
from pixelogue.task_runtime import (
    bindable_parameter_names,
    fingerprint,
    parameter_contract,
    required_parameter_names,
)

DraftValue = Annotated[str, Field(min_length=1, max_length=512)]
# The whole requested text unit must fit the bound region, so the drafted scope is used.
TEXT_UNIT_TASKS = frozenset({"text_transcription"})
_ARTICLES = frozenset({"a", "an", "the"})


class DraftParameter(StrictModel):
    """One public operation choice written in the draft."""

    name: Annotated[str, Field(min_length=1, max_length=64)]
    value: DraftValue | int | bool | tuple[DraftValue, ...]


class FactKey(StrictModel):
    """Private identity of the requested fact: subject and dimension, never its value."""

    subject: Annotated[str, Field(min_length=1, max_length=80)]
    dimension: Annotated[str, Field(min_length=1, max_length=40)]


class QuestionDraft(StrictModel):
    """One candidate question and the public operation it realizes.

    ``target`` is the public locator of the subject or region the question is about; it is
    also the operation's public scope description and never contains the answer.
    """

    task_id: Annotated[str, Field(min_length=1, max_length=64)]
    question: Annotated[str, Field(min_length=1, max_length=600)]
    target: Annotated[str, Field(min_length=1, max_length=160)]
    public_parameters: Annotated[tuple[DraftParameter, ...], Field(max_length=12)]
    scope_region: ImageRegion
    target_region: ImageRegion | None
    fact_key: FactKey

    @model_validator(mode="after")
    def validate_draft(self) -> QuestionDraft:
        """Keep the target inside its scope and parameter names unique."""
        if self.target_region is not None:
            parent, target = self.scope_region, self.target_region
            if not (
                parent.left <= target.left < target.right <= parent.right
                and parent.top <= target.top < target.bottom <= parent.bottom
            ):
                raise ValueError("Draft target region must stay inside its scope region")
        names = [parameter.name for parameter in self.public_parameters]
        if len(names) != len(set(names)) or "target" in names:
            raise ValueError("Draft parameter names must be unique and exclude target")
        return self


class QuestionDraftBatch(StrictModel):
    """Ordered drafts for one turn; an empty batch is a valid abstention."""

    drafts: Annotated[tuple[QuestionDraft, ...], Field(max_length=4)]
    reason: Annotated[str, Field(min_length=1, max_length=240)] | None = None


def _words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("_", " ")
    words = [word for word in re.findall(r"\w+", normalized) if word not in _ARTICLES]
    return ["color" if word == "colour" else word for word in words]


def request_key(fact: FactKey) -> str:
    """Normalize a private fact identity so spelling and articles do not create new facts."""
    return " ".join(_words(fact.subject)) + "|" + " ".join(_words(fact.dimension))


def draft_task_contract(task: TaskDefinition) -> dict[str, Any]:
    """Expose the fixed public contract a drafter must satisfy for one operation."""
    catalog = task_catalog()
    return {
        "task_id": task.id,
        "family": task.family,
        "definition": task.definition,
        "do_not_infer": task.do_not_infer,
        "eligibility_checks": {
            name: catalog.eligibility_checks[name] for name in task.eligibility_checks
        },
        "parameter_contract": parameter_contract(task),
        "required_parameter_names": list(required_parameter_names(task)),
        "bindable_parameter_names": list(bindable_parameter_names(task)),
        "answer_format": task.answer_format,
    }


def _reject(reason: str, message: str) -> ExecutionError:
    return ExecutionError(reason, message)


def validate_draft_parameters(task: TaskDefinition, draft: QuestionDraft) -> None:
    """Apply the catalog's public parameter rules to a draft.

    Raises:
        ExecutionError: If a name, a required choice, a value's kind or a choice is invalid.
    """
    parameters = {parameter.name: parameter.value for parameter in draft.public_parameters}
    parameters["target"] = draft.target
    allowed = set(bindable_parameter_names(task))
    if unknown := sorted(parameters.keys() - allowed):
        raise _reject("DRAFT_PARAMETER_UNKNOWN", f"Unknown parameters {unknown}")
    if missing := sorted(set(required_parameter_names(task)) - parameters.keys()):
        raise _reject("DRAFT_PARAMETER_MISSING", f"Missing parameters {missing}")
    for name, value in parameters.items():
        item = task.parameters.get(name)
        if item is None:
            continue
        if item.kind == "text" and not isinstance(value, str):
            raise _reject("DRAFT_PARAMETER_VALUE", f"{name} must be text")
        if item.kind == "choice" and value not in (item.values or ()):
            raise _reject("DRAFT_PARAMETER_VALUE", f"Unsupported choice for {name}")
        if item.kind == "integer" and type(value) is not int:
            raise _reject("DRAFT_PARAMETER_VALUE", f"{name} must be a whole number")
        if item.kind == "list":
            if not isinstance(value, tuple):
                raise _reject("DRAFT_PARAMETER_VALUE", f"{name} must be a list")
            if len({entry.casefold() for entry in value}) != len(value):
                raise _reject("DRAFT_PARAMETER_VALUE", f"{name} repeats an item")
            if len(value) < (item.min_items or 1) or (
                item.max_items is not None and len(value) > item.max_items
            ):
                raise _reject("DRAFT_PARAMETER_VALUE", f"{name} has an unsupported item count")


def draft_to_candidate(
    draft: QuestionDraft,
    *,
    image_id: str,
    view_id: str,
    turn_index: int,
    draft_index: int,
    allowed_task_ids: Collection[str],
    task_id: str | None = None,
) -> InstructionCandidate:
    """Convert a validated draft into a direct-origin operation contract.

    ``task_id`` relabels the draft only when a caller has independent agreement for an
    operation with the same verification contracts.

    Raises:
        ExecutionError: If the operation is not offered or its public choices are invalid.
    """
    chosen = task_id or draft.task_id
    if chosen not in allowed_task_ids:
        raise _reject("DRAFT_TASK_NOT_ALLOWED", f"Task {chosen} was not offered for this turn")
    catalog = task_catalog()
    task = next(item for item in catalog.tasks if item.id == chosen)
    validate_draft_parameters(task, draft)
    target_region = draft.scope_region if task.id in TEXT_UNIT_TASKS else draft.target_region
    candidate = InstructionCandidate(
        candidate_id="unbound",
        task_id=task.id,
        family=task.family,
        visible_scope=draft.target,
        instruction_summary=task.definition,
        required_capabilities=task.required_capabilities,
        catalog_version=catalog.version,
        scope_id=f"draft:{turn_index}:{draft_index}",
        view_id=view_id,
        scope_region=draft.scope_region,
        target_region=target_region,
        public_parameters=(
            PublicParameter(name="target", value=draft.target, origin="instruction"),
            *(
                PublicParameter(name=parameter.name, value=parameter.value, origin="instruction")
                for parameter in draft.public_parameters
            ),
        ),
        evidence_refs=(),
        verification_contracts=task.verification_contracts,
        origin="direct",
        request_key=request_key(draft.fact_key),
    )
    return candidate.model_copy(update={"candidate_id": fingerprint(candidate, image_id)})
