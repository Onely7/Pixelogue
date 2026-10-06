"""Load and validate the bundled task catalog and legacy task migration."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import as_file, files
from typing import Any

from pydantic import ValidationError

from pixelogue.errors import ConfigurationError
from pixelogue.serialization import canonical_json, load_yaml, strict_json_object
from pixelogue.task_catalog import TaskCatalog

TASK_CONTRACT_VERSION = "scope-operations-v2"


@lru_cache(maxsize=1)
def task_catalog() -> TaskCatalog:
    """Load v7 with strict schema and cross-reference validation."""
    path = files("pixelogue.resources").joinpath("task_catalog.yaml")
    with as_file(path) as resource:
        catalog = load_yaml(resource)
    try:
        return TaskCatalog.model_validate_json(canonical_json(catalog))
    except ValidationError as error:
        raise ConfigurationError("TASK_CATALOG_MISMATCH", str(error)) from error


def load_task_catalog() -> dict[str, Any]:
    """Return all 65 standard candidates and 7 disabled specialized extensions."""
    return task_catalog().model_dump(mode="json")


def load_legacy_migration() -> list[dict[str, Any]]:
    """Read advisory mappings without relabeling any saved conversation."""
    root = files("pixelogue.resources")
    rows = strict_json_object(
        '{"rows":' + root.joinpath("legacy_24_migration.json").read_text() + "}"
    )["rows"]
    with as_file(root.joinpath("task_catalog_legacy.yaml")) as resource:
        old_ids = {task["id"] for task in load_yaml(resource)["tasks"]}
    new_ids = {task.id for task in task_catalog().tasks}
    if not isinstance(rows, list) or len(rows) != len(old_ids):
        raise ConfigurationError("TASK_MIGRATION_MISMATCH", "Every legacy ID needs a mapping")
    seen: set[str] = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"old_id", "new_ids", "parameter_changes", "historical_artifacts"}
            or row["old_id"] not in old_ids
            or row["old_id"] in seen
            or not row["new_ids"]
            or not set(row["new_ids"]) <= new_ids
        ):
            raise ConfigurationError("TASK_MIGRATION_MISMATCH", "Invalid legacy mapping")
        seen.add(row["old_id"])
    return rows
