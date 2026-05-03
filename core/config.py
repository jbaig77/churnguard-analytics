"""Loads and merges YAML config files; provides a single access point."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).parent.parent


def _load(path: Path) -> dict:
    """Read a single YAML file from disk and return its contents as a dict.

    Args:
        path: Absolute or relative path to the YAML file to read.

    Returns:
        A dict representing the parsed YAML contents. Returns an empty dict
        if the file is empty or contains only null/whitespace.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_config(
    schema_path: str | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    """Load, merge, and return the full project configuration.

    Reads the schema config (column definitions, data sources, data directory)
    and the model config (hyperparameters, thresholds, cache settings) from
    their default YAML locations inside ``config/``, or from caller-supplied
    paths. Resolves the data directory against the project root and honours a
    ``DATA_DIR`` environment variable override.

    Args:
        schema_path: Optional path to a custom schema YAML file. When omitted,
            defaults to ``<project_root>/config/schema.yaml``.
        model_path: Optional path to a custom model config YAML file. When
            omitted, defaults to ``<project_root>/config/model_config.yaml``.

    Returns:
        A dict with three keys:

        - ``"schema"``  — parsed schema config with ``data_dir`` resolved to
          an absolute path string.
        - ``"model"``   — parsed model config as-is.
        - ``"project_root"`` — absolute path string of the repository root.
    """
    root = _PROJECT_ROOT
    schema_cfg = _load(Path(schema_path) if schema_path else root / "config" / "schema.yaml")
    model_cfg = _load(Path(model_path) if model_path else root / "config" / "model_config.yaml")

    # Allow DATA_DIR env var override
    data_dir = os.environ.get("DATA_DIR", schema_cfg.get("data_dir", "."))
    schema_cfg["data_dir"] = str(root / data_dir)

    return {"schema": schema_cfg, "model": model_cfg, "project_root": str(root)}


def get_project_root() -> Path:
    """Return the absolute path to the repository root directory.

    The root is determined at import time as the parent of the ``core/``
    package directory, so it is stable regardless of the working directory
    from which Python is invoked.

    Returns:
        A ``pathlib.Path`` object pointing to the project root.
    """
    return _PROJECT_ROOT
