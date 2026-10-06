"""Load and validate the bundled task catalog."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import as_file, files
from typing import Any

from pydantic import ValidationError

from pixelogue.errors import ConfigurationError
from pixelogue.serialization import canonical_json, load_yaml
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
