"""Render the task catalog as the human-readable reference in ``docs/tasks/TASKS.md``.

The reference is generated, not written by hand, so the definitions that people read are
the definitions that the question drafter, the judges and the validators receive.
"""

from __future__ import annotations

from collections.abc import Iterable

from pixelogue.routing import LIGHT_CONTRACTS, lightly_verified
from pixelogue.task_catalog import TaskCatalog, TaskDefinition, TaskParameter

CATALOG_SOURCE = "src/pixelogue/resources/task_catalog.yaml"
REGENERATE_COMMAND = "uv run --locked pixelogue compile --tasks-markdown docs/tasks/TASKS.md"


def render_tasks_markdown(catalog: TaskCatalog) -> str:
    """Return the complete task reference as deterministic Markdown.

    Args:
        catalog: The validated catalog whose families, tasks and vocabularies are listed.

    Returns:
        Markdown text ending in one newline, identical for identical catalogs.
    """
    tasks = {task.id: task for task in catalog.tasks}
    # The loader sorts mapping keys; list families in the order their tasks are written.
    family_ids = list(dict.fromkeys(task.family for task in catalog.tasks))
    core = sum(task.status == "core" for task in catalog.tasks)
    light = sum(task.status == "core" and lightly_verified(task) for task in catalog.tasks)
    extensions = len(catalog.tasks) - core
    lines = [
        f"# Pixelogue task catalog {catalog.version}",
        "",
        f"<!-- Generated from {CATALOG_SOURCE}. Do not edit by hand; run: {REGENERATE_COMMAND} -->",
        "",
        "[Runtime admission guide](README.md)",
        "",
        f"The catalog defines {len(catalog.tasks)} tasks in {len(catalog.families)} families: "
        f"{core} core tasks that the question drafter may propose and {extensions} extensions "
        "that stay off until their specialized validator is configured and calibrated. A task "
        "is offered only when the image supports it and every verification contract it needs "
        "has a working implementation.",
        "",
        f"{light} core tasks are light: they are checked only by "
        f"{_names(sorted(LIGHT_CONTRACTS), last='or')}. The opening turns of a conversation "
        "can be limited to light tasks with `tasks.anchor_turns`.",
        "",
        "Each task lists the question it answers, the parameters that the question states, "
        "the image capabilities and eligibility checks it needs, and its verification "
        "contracts. Related FineVision subsets show where a similar task appears in public "
        "data; they are never shown to a model.",
        "",
        "## Input contract",
        "",
        *_input_contract(catalog),
        "",
        "## Admission rules",
        "",
        *(f"- {rule}" for rule in catalog.global_admission_rules),
        "",
        "## Families",
        "",
        "| Family | Label | Core | Extensions | Tasks |",
        "|---|---|---:|---:|---|",
    ]
    for family_id in family_ids:
        family = catalog.families[family_id]
        members = [tasks[task_id] for task_id in family.task_ids]
        family_core = sum(task.status == "core" for task in members)
        lines.append(
            f"| `{family_id}` | {_cell(family.label)} | {family_core} | "
            f"{len(members) - family_core} | {_names(family.task_ids)} |"
        )
    lines.extend(["", "## Tasks", ""])
    for family_id in family_ids:
        family = catalog.families[family_id]
        lines.extend([f"### {family.label} (`{family_id}`)", ""])
        for task_id in family.task_ids:
            lines.extend(_task_section(tasks[task_id]))
    lines.extend(
        [
            "## Capabilities",
            "",
            "A capability is something the image must show for a task to apply.",
            "",
            "| Capability | Meaning |",
            "|---|---|",
            *(f"| `{key}` | {_cell(text)} |" for key, text in catalog.capabilities.items()),
            "",
            "## Eligibility checks",
            "",
            "An eligibility check is a condition on the question and the image that the "
            "drafter must satisfy and the question judges confirm.",
            "",
            "| Check | Condition |",
            "|---|---|",
            *(f"| `{key}` | {_cell(text)} |" for key, text in catalog.eligibility_checks.items()),
            "",
            "## Verification contracts",
            "",
            "| Contract | What it checks | Applies when |",
            "|---|---|---|",
            *(
                f"| `{key}` | {_cell(item.contract)} | {_cell(item.applies_when)} |"
                for key, item in catalog.verification_contracts.items()
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _input_contract(catalog: TaskCatalog) -> list[str]:
    """List the single-image input rules."""
    contract = catalog.input_contract
    return [
        f"- **Source images.** {contract.source_images}; the only raw content is "
        f"{_names(contract.raw_content_inputs)}.",
        f"- **Image-only interpretation.** {contract.image_only_interpretation}",
        f"- **Multi-panel images.** {contract.multi_panel_policy}",
        f"- **Source rights.** {contract.source_rights}",
        "- **Permitted context:**",
        *(f"  - {item}" for item in contract.permitted_context),
        "- **Forbidden context:**",
        *(f"  - {item}" for item in contract.forbidden_context),
    ]


def _task_section(task: TaskDefinition) -> list[str]:
    """Describe one task for a reader who has not seen the code."""
    weight = "light" if lightly_verified(task) else "structured"
    lines = [
        f"#### `{task.id}`: {task.label}",
        "",
        task.definition,
        "",
        f"- **Status.** {task.status}; {weight} verification.",
        f"- **Example question.** {task.example_question}",
    ]
    if task.answer_format is not None:
        lines.append(f"- **Answer format.** {task.answer_format}")
    lines.extend(
        [
            f"- **Do not infer.** {task.do_not_infer}",
            f"- **Required capabilities.** {_names(task.required_capabilities)}.",
            f"- **Eligibility checks.** {_names(task.eligibility_checks)}.",
            f"- **Verification contracts.** {_names(task.verification_contracts)}.",
        ]
    )
    if task.related_finevision_subsets:
        lines.append(
            f"- **Related FineVision subsets.** {_names(task.related_finevision_subsets)}."
        )
    if task.parameters:
        lines.extend(
            [
                "",
                "| Parameter | Required | Form | Meaning |",
                "|---|---|---|---|",
                *(
                    f"| `{name}` | {'yes' if parameter.required else 'no'} | "
                    f"{_form(parameter)} | {_cell(parameter.description)} |"
                    for name, parameter in task.parameters.items()
                ),
            ]
        )
    else:
        lines.append("- **Parameters.** None.")
    lines.append("")
    return lines


def _form(parameter: TaskParameter) -> str:
    """Describe the accepted value of one parameter."""
    if parameter.kind == "choice":
        return "one of " + _names(parameter.values or ())
    if parameter.kind != "list":
        return parameter.kind
    low, high = parameter.min_items, parameter.max_items
    if low is not None and low == high:
        return f"list of exactly {low}"
    if low is not None and high is not None:
        return f"list of {low} to {high}"
    if high is not None:
        return f"list of at most {high}"
    if low is not None:
        return f"list of at least {low}"
    return "list"


def _names(names: Iterable[str], last: str | None = None) -> str:
    """Format identifiers as inline code separated by commas.

    Args:
        names: Identifiers in display order.
        last: A conjunction such as ``or`` before the final identifier; commas only if absent.
    """
    items = [f"`{name}`" for name in names]
    if last is None or len(items) < 2:
        return ", ".join(items)
    return f"{', '.join(items[:-1])} {last} {items[-1]}"


def _cell(text: str) -> str:
    """Escape a table cell so a vertical bar cannot split it."""
    return text.replace("|", "\\|")
