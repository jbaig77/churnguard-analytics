"""Curveball scenario definitions.

Each Scenario has:
  apply(orchestrator)    — modify state/config/data in-place; return {"ok": True, "notes": "..."}
  rollback(orchestrator) — undo the modification

Adding a new Phase 2 scenario: subclass Scenario, implement apply/rollback,
and add to the SCENARIOS dict.
"""
from __future__ import annotations

import copy
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from agents.orchestrator import Orchestrator


class Scenario(ABC):
    """Base class for all curveball test scenarios; each subclass implements apply/rollback to inject and undo a single adversarial condition."""

    name: str = ""
    description: str = ""

    @abstractmethod
    def apply(self, orch: "Orchestrator") -> dict[str, Any]:
        """Apply the scenario to the orchestrator state/config in-place.

        Args:
            orch: The Orchestrator instance whose state and config will be mutated.

        Returns:
            dict with "ok" (bool) indicating success and "notes" (str) describing what changed.
        """
        ...

    @abstractmethod
    def rollback(self, orch: "Orchestrator") -> None:
        """Undo all changes made by apply(), restoring the original state.

        Args:
            orch: The Orchestrator instance to restore.

        Returns:
            None
        """
        ...


# =====================================================================
# Scenario 1: Column rename (schema drift)
# =====================================================================
class ColumnRenameScenario(Scenario):
    """Simulates a CSV column rename by remapping account_health_score to health_index in schema and live frame."""

    name = "column_rename"
    description = "Renames 'account_health_score' to 'health_index' in accounts source"

    def __init__(self):
        self._original_col_map = None

    def apply(self, orch: "Orchestrator") -> dict[str, Any]:
        """Apply the scenario: rename account_health_score to health_index in schema and live DataFrame."""
        source_def = orch.config["schema"]["data_sources"]["accounts"]
        self._original_col_map = copy.deepcopy(source_def.get("columns", {}))
        # Remap logical name in schema
        cols = source_def.setdefault("columns", {})
        if "account_health_score" in cols:
            cols["account_health_score"] = "health_index"
        # Rename column in the live DataFrame to simulate the file changing
        if "accounts" in orch.state.frames:
            df = orch.state.frames["accounts"]
            if "account_health_score" in df.columns:
                orch.state.frames["accounts"] = df.rename(columns={"account_health_score": "health_index"})
        return {"ok": True, "notes": "Renamed 'account_health_score' → 'health_index' in accounts frame."}

    def rollback(self, orch: "Orchestrator") -> None:
        """Restore original state by reverting the column mapping and renaming health_index back to account_health_score."""
        if self._original_col_map is not None:
            orch.config["schema"]["data_sources"]["accounts"]["columns"] = self._original_col_map
        # Rename back in live frame
        if "accounts" in orch.state.frames:
            df = orch.state.frames["accounts"]
            if "health_index" in df.columns:
                orch.state.frames["accounts"] = df.rename(columns={"health_index": "account_health_score"})


# =====================================================================
# Scenario 2: New data source (extensibility test)
# =====================================================================
class NewSourceScenario(Scenario):
    """Injects a synthetic nps_scores DataFrame to verify the generic aggregation path for unknown data sources."""

    name = "new_source"
    description = "Adds a synthetic 'nps_scores' data source with account-level NPS"

    def __init__(self):
        self._added_source = False

    def apply(self, orch: "Orchestrator") -> dict[str, Any]:
        """Apply the scenario: inject a synthetic nps_scores frame and register it in the schema config."""
        # Build a synthetic NPS DataFrame from existing account IDs
        if orch.state.frames.get("accounts") is None:
            return {"ok": False, "reason": "accounts not loaded"}

        accounts_df = orch.state.frames["accounts"]
        account_ids = accounts_df["account_id"].dropna().unique()
        rng = np.random.default_rng(seed=99)
        nps_df = pd.DataFrame({
            "account_id": account_ids,
            "nps_score": rng.integers(-100, 101, size=len(account_ids)).astype(float),
            "survey_response_rate": rng.uniform(0, 1, size=len(account_ids)),
        })
        # Drop 20% to simulate missing data
        drop_idx = rng.choice(len(nps_df), size=int(0.2 * len(nps_df)), replace=False)
        nps_df.loc[drop_idx, "nps_score"] = np.nan

        orch.state.frames["nps_scores"] = nps_df
        orch.config["schema"]["data_sources"]["nps_scores"] = {
            "path": "_synthetic_nps_.csv",
            "type": "csv",
            "primary_key": "account_id",
            "join_key": "account_id",
            "columns": {"account_id": "account_id", "nps_score": "nps_score",
                        "survey_response_rate": "survey_response_rate"},
        }
        self._added_source = True
        return {
            "ok": True,
            "notes": (
                f"Injected synthetic 'nps_scores' source: {len(nps_df)} rows, "
                f"~20% missing nps_score.\n"
                "Verifies: new source columns appear in features without code changes."
            ),
        }

    def rollback(self, orch: "Orchestrator") -> None:
        """Restore original state by removing the nps_scores frame and its schema entry."""
        orch.state.frames.pop("nps_scores", None)
        orch.config["schema"]["data_sources"].pop("nps_scores", None)


# =====================================================================
# Scenario 3: Change churn definition
# =====================================================================
class ChurnRedefinitionScenario(Scenario):
    """Expands the churn target to include 'suspended' accounts via extra_positive_values."""

    name = "churn_redefine"
    description = "Expands churn to include 'suspended' accounts in addition to 'churned'"

    def __init__(self):
        self._original_extra = None

    def apply(self, orch: "Orchestrator") -> dict[str, Any]:
        """Apply the scenario: add 'suspended' to the target's extra_positive_values to expand the churn definition."""
        target = orch.config["schema"]["target"]
        self._original_extra = list(target.get("extra_positive_values", []))
        target["extra_positive_values"] = ["suspended"]

        if "accounts" in orch.state.frames:
            suspended = (orch.state.frames["accounts"]["account_status"] == "suspended").sum()
        else:
            suspended = "unknown"

        return {
            "ok": True,
            "notes": (
                f"Target redefined: churned OR suspended.\n"
                f"Approx {suspended} additional 'positive' accounts added.\n"
                "Verifies: target_cfg.extra_positive_values is respected."
            ),
        }

    def rollback(self, orch: "Orchestrator") -> None:
        """Restore original state by resetting extra_positive_values to its pre-scenario list."""
        if self._original_extra is not None:
            orch.config["schema"]["target"]["extra_positive_values"] = self._original_extra


# =====================================================================
# Scenario 4: Inject nulls into critical features
# =====================================================================
class NullInjectionScenario(Scenario):
    """Nulls 30% of mrr and account_health_score values to exercise the median imputation path."""

    name = "null_injection"
    description = "Nulls out 30% of 'mrr' and 'account_health_score' values"

    def __init__(self):
        self._backup: dict[str, pd.Series] = {}

    def apply(self, orch: "Orchestrator") -> dict[str, Any]:
        """Apply the scenario: randomly null 30% of mrr and account_health_score values in the accounts frame."""
        df = orch.state.frames.get("accounts")
        if df is None:
            return {"ok": False, "reason": "accounts frame not loaded"}
        rng = np.random.default_rng(42)
        targets = ["mrr", "account_health_score"]
        for col in targets:
            if col in df.columns:
                self._backup[col] = df[col].copy()
                null_idx = rng.choice(len(df), size=int(0.3 * len(df)), replace=False)
                df.loc[df.index[null_idx], col] = np.nan
        orch.state.frames["accounts"] = df
        return {
            "ok": True,
            "notes": (
                "Nulled 30% of 'mrr' and 'account_health_score'.\n"
                "Verifies: median imputation in feature engineer handles missing data."
            ),
        }

    def rollback(self, orch: "Orchestrator") -> None:
        """Restore original state by reinserting the backed-up mrr and account_health_score series."""
        df = orch.state.frames.get("accounts")
        if df is not None:
            for col, series in self._backup.items():
                if col in df.columns:
                    df[col] = series
            orch.state.frames["accounts"] = df
        self._backup.clear()


# =====================================================================
# Scenario 5: Unseen subscription tier
# =====================================================================
class NewTierScenario(Scenario):
    """Assigns 5% of accounts to an unseen 'platinum' tier to test the model's unknown-category fallback."""

    name = "new_tier"
    description = "Introduces 'platinum' subscription tier not seen during training"

    def __init__(self):
        self._modified_indices: list = []

    def apply(self, orch: "Orchestrator") -> dict[str, Any]:
        """Apply the scenario: set subscription_tier to 'platinum' for 5% of accounts."""
        df = orch.state.frames.get("accounts")
        if df is None or "subscription_tier" not in df.columns:
            return {"ok": False, "reason": "subscription_tier not available"}
        rng = np.random.default_rng(7)
        idx = rng.choice(len(df), size=max(1, int(0.05 * len(df))), replace=False)
        self._modified_indices = list(df.index[idx])
        df.loc[self._modified_indices, "subscription_tier"] = "platinum"
        orch.state.frames["accounts"] = df
        return {
            "ok": True,
            "notes": (
                f"Set {len(idx)} accounts to tier='platinum' (unseen during training).\n"
                "Verifies: LabelEncoder __unknown__ fallback in model._sanitize()."
            ),
        }

    def rollback(self, orch: "Orchestrator") -> None:
        """Restore original state by resetting the modified accounts' subscription_tier back to 'enterprise'."""
        df = orch.state.frames.get("accounts")
        if df is not None and self._modified_indices:
            # Restore to 'enterprise' (highest known tier)
            df.loc[self._modified_indices, "subscription_tier"] = "enterprise"
            orch.state.frames["accounts"] = df


# =====================================================================
# Scenario 6: Drop a non-critical column
# =====================================================================
class DropColumnScenario(Scenario):
    """Drops integration_count from accounts to verify graceful column-skipping in feature engineering."""

    name = "drop_column"
    description = "Removes 'integration_count' from accounts (simulates source schema trim)"

    def __init__(self):
        self._backup: pd.Series | None = None

    def apply(self, orch: "Orchestrator") -> dict[str, Any]:
        """Apply the scenario: drop the integration_count column from the accounts frame."""
        df = orch.state.frames.get("accounts")
        if df is None or "integration_count" not in df.columns:
            return {"ok": False, "reason": "column not present"}
        self._backup = df["integration_count"].copy()
        orch.state.frames["accounts"] = df.drop(columns=["integration_count"])
        return {
            "ok": True,
            "notes": (
                "Dropped 'integration_count' column.\n"
                "Verifies: feature engineer skips missing columns gracefully;\n"
                "model.reindex(fill_value=0) handles missing features at prediction time."
            ),
        }

    def rollback(self, orch: "Orchestrator") -> None:
        """Restore original state by reinserting the backed-up integration_count column."""
        df = orch.state.frames.get("accounts")
        if df is not None and self._backup is not None:
            df["integration_count"] = self._backup
            orch.state.frames["accounts"] = df


# =====================================================================
# Scenario 7: Scale test — duplicate accounts 2×
# =====================================================================
class ScaleTestScenario(Scenario):
    """Doubles the account dataset to ~10k rows to test throughput under load."""

    name = "scale_test"
    description = "Doubles account dataset to ~10,000 rows to test throughput"

    def __init__(self):
        self._original_frames: dict | None = None

    def apply(self, orch: "Orchestrator") -> dict[str, Any]:
        """Apply the scenario: concatenate a duplicate accounts frame (with suffixed IDs) to double the dataset size."""
        self._original_frames = {k: v.copy() for k, v in orch.state.frames.items()}
        accounts = orch.state.frames.get("accounts")
        if accounts is None:
            return {"ok": False, "reason": "accounts not loaded"}

        duped = accounts.copy()
        duped["account_id"] = duped["account_id"].apply(lambda x: x + "_DUP")
        # Status: force duplicates to a mix of churned/active
        rng = np.random.default_rng(0)
        duped["account_status"] = rng.choice(["active", "churned"], size=len(duped), p=[0.7, 0.3])
        orch.state.frames["accounts"] = pd.concat([accounts, duped], ignore_index=True)

        return {
            "ok": True,
            "notes": (
                f"Dataset doubled: {len(accounts)} → {len(orch.state.frames['accounts'])} accounts.\n"
                "Verifies: feature engineering and model training handle 10k accounts."
            ),
        }

    def rollback(self, orch: "Orchestrator") -> None:
        """Restore original state by replacing all frames with the snapshot taken before duplication."""
        if self._original_frames is not None:
            orch.state.frames = self._original_frames


# =====================================================================
# Registry
# =====================================================================
SCENARIOS: dict[str, Scenario] = {
    s.name: s() for s in [
        ColumnRenameScenario,
        NewSourceScenario,
        ChurnRedefinitionScenario,
        NullInjectionScenario,
        NewTierScenario,
        DropColumnScenario,
        ScaleTestScenario,
    ]
}
