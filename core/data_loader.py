"""Schema-driven data loader.

Column names come from schema.yaml — no hardcoded field names here.
To support a new data source, add it to schema.yaml and it will be
automatically loaded and made available to the feature engineer.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads tabular data files described in a schema config into DataFrames.

    The loader is entirely driven by the ``data_sources`` section of
    ``schema.yaml``.  Each source entry specifies a file path (relative to
    ``data_dir``), a file type (csv or parquet), and an optional column
    mapping.  No source-specific logic lives in this class.
    """

    def __init__(self, schema_cfg: dict[str, Any]):
        """Initialise the loader with the parsed schema configuration.

        Args:
            schema_cfg: The ``"schema"`` sub-dict returned by
                ``load_config()``.  Must contain at minimum the keys
                ``"data_dir"`` (absolute path string) and
                ``"data_sources"`` (mapping of source name → source
                definition dict).
        """
        self.schema = schema_cfg
        self.data_dir = Path(schema_cfg["data_dir"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load every source defined in schema.yaml and return as a dict.

        Iterates over every entry in ``schema["data_sources"]``.  Sources
        whose files are missing are skipped with a warning; other errors are
        logged and also skipped so that a single bad source does not abort
        the entire load.

        Returns:
            A dict mapping each successfully loaded source name (str) to its
            corresponding ``pd.DataFrame``.  Sources that failed to load are
            absent from the dict.
        """
        frames: dict[str, pd.DataFrame] = {}
        for source_name, source_def in self.schema["data_sources"].items():
            try:
                frames[source_name] = self._load_source(source_name, source_def)
                logger.info(
                    "Loaded '%s': %d rows, %d cols",
                    source_name,
                    len(frames[source_name]),
                    len(frames[source_name].columns),
                )
            except FileNotFoundError:
                logger.warning("Source '%s' file not found, skipping.", source_name)
            except Exception as exc:
                logger.error("Failed to load source '%s': %s", source_name, exc)
        return frames

    def load_source(self, source_name: str) -> pd.DataFrame:
        """Load and return a single named data source.

        Looks up the source definition in ``schema["data_sources"]`` by
        name, then delegates to :meth:`_load_source` for the actual file
        reading, column renaming, and type coercion.

        Args:
            source_name: The key that identifies the source in
                ``schema["data_sources"]`` (e.g. ``"accounts"``).

        Returns:
            A cleaned ``pd.DataFrame`` for the requested source with logical
            column names and coerced types.

        Raises:
            KeyError: If ``source_name`` is not present in the schema.
            FileNotFoundError: If the corresponding data file does not exist.
            ValueError: If the file type specified in the schema is not
                supported.
        """
        source_def = self.schema["data_sources"][source_name]
        return self._load_source(source_name, source_def)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_source(self, name: str, defn: dict) -> pd.DataFrame:
        """Read a single source file, rename columns, and coerce types.

        Resolves the file path relative to ``self.data_dir``, reads it as
        CSV or Parquet according to the ``"type"`` key in ``defn``, applies
        the column mapping defined in the schema, and finally runs
        best-effort type coercion on well-known timestamp and boolean
        columns.

        Args:
            name: The logical source name (used only for error messages).
            defn: The source definition dict from schema.yaml.  Expected keys:

                - ``"path"`` (str): file path relative to ``data_dir``.
                - ``"type"`` (str, optional): ``"csv"`` (default) or
                  ``"parquet"``.
                - ``"columns"`` (dict, optional): mapping of logical column
                  name → raw column name in the file.

        Returns:
            A ``pd.DataFrame`` with logical column names and coerced types.

        Raises:
            FileNotFoundError: If the resolved file path does not exist.
            ValueError: If the file type is neither ``"csv"`` nor
                ``"parquet"``.
        """
        path = self.data_dir / defn["path"]
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        file_type = defn.get("type", "csv").lower()
        if file_type == "csv":
            df = pd.read_csv(path, low_memory=False)
        elif file_type in ("parquet",):
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file type '{file_type}' for source '{name}'")

        df = self._apply_column_mapping(df, defn)
        df = self._coerce_types(df, name)
        return df

    def _apply_column_mapping(self, df: pd.DataFrame, defn: dict) -> pd.DataFrame:
        """Rename columns according to the schema's column mapping.

        The mapping is logical_name → raw_name (so we rename raw → logical).
        Only columns present in the file are renamed; extras are kept as-is
        so that Phase 2 additions don't break anything.

        Args:
            df: The raw ``pd.DataFrame`` as read from disk.
            defn: The source definition dict.  The optional ``"columns"``
                key should be a dict mapping logical column names to the
                raw column names that appear in the file.

        Returns:
            A ``pd.DataFrame`` with raw column names replaced by their
            logical equivalents wherever a mapping is defined.  Unmapped
            columns are preserved unchanged.
        """
        col_map = defn.get("columns", {})
        # col_map: {logical: raw}  →  we want raw→logical rename dict
        rename = {raw: logical for logical, raw in col_map.items() if raw in df.columns}
        if rename:
            df = df.rename(columns=rename)
        return df

    def _coerce_types(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Apply best-effort type coercion for well-known fields.

        Converts any column whose name appears in the hard-coded list of
        timestamp columns to ``datetime64`` (invalid values become ``NaT``).
        Converts any column whose name appears in the boolean column list to
        Python ``bool`` or ``None``, recognising the string variants
        ``"true"``/``"1"``/``"yes"`` and ``"false"``/``"0"``/``"no"``.
        Columns not in either list are left untouched.

        Args:
            df: The ``pd.DataFrame`` after column renaming.
            source_name: The logical source name (reserved for future
                per-source coercion rules; currently unused in the logic
                but kept for forward compatibility).

        Returns:
            The same ``pd.DataFrame`` with timestamp and boolean columns
            coerced to their proper Python/pandas types in-place.
        """
        timestamp_cols = [
            "created_timestamp", "last_activity_timestamp", "status_change_date",
            "contract_end_date", "user_created_date", "last_login_date",
            "created_date", "resolved_date",
        ]
        bool_cols = [
            "auto_renew_enabled", "api_calls_enabled", "sso_enabled",
            "white_label_enabled", "risk_flag", "api_key_active", "mobile_app_user",
            "onboarding_completed", "certification_earned", "beta_features_enabled",
            "escalated", "cancellation_requested", "retention_offer_made",
            "retention_offer_accepted", "account_pause_requested",
            "downgrade_requested", "sla_breach",
        ]
        for col in timestamp_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].map(
                    lambda v: True if str(v).lower() in ("true", "1", "yes") else
                              (False if str(v).lower() in ("false", "0", "no") else None)
                )
        return df
