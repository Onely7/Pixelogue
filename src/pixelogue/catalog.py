"""Load and validate bundled task and rating catalogs."""

from __future__ import annotations

from importlib.resources import as_file, files
from typing import Any

from pixelogue.errors import ConfigurationError
from pixelogue.serialization import load_yaml

EXPECTED_TASK_COUNT = 24
EXPECTED_RUBRIC_COUNT = 28
EXPECTED_AXES = {
    "format",
    "relevance",
    "visual_dependency",
    "image_correspondence",
    "factual_correctness",
    "history",
    "uncertainty",
    "safety",
}


def load_task_catalog() -> dict[str, Any]:
    """Return the bundled 24-task taxonomy after structural checks."""
    path = files("pixelogue.resources").joinpath("task_catalog.yaml")
    with as_file(path) as resource:
        catalog = load_yaml(resource)
    tasks = catalog.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != EXPECTED_TASK_COUNT:
        raise ConfigurationError("TASK_CATALOG_MISMATCH", "Task catalog must define 24 tasks")
    ids = [task.get("id") for task in tasks if isinstance(task, dict)]
    if len(ids) != len(set(ids)) or len(ids) != EXPECTED_TASK_COUNT:
        raise ConfigurationError("TASK_CATALOG_MISMATCH", "Task IDs must be unique")
    return catalog


def load_rubric_catalog() -> dict[str, Any]:
    """Return the bundled 28-item rubric after structural checks."""
    path = files("pixelogue.resources").joinpath("rubric_catalog.yaml")
    with as_file(path) as resource:
        catalog = load_yaml(resource)
    items = catalog.get("items")
    if not isinstance(items, list) or len(items) != EXPECTED_RUBRIC_COUNT:
        raise ConfigurationError("RUBRIC_CATALOG_MISMATCH", "Rubric catalog must define 28 items")
    axes = set(catalog.get("axes", []))
    if axes != EXPECTED_AXES:
        raise ConfigurationError("RUBRIC_CATALOG_MISMATCH", "Rubric axes do not match contract")
    return catalog
