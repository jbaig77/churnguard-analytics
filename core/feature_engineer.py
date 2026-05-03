"""Feature engineering pipeline.

Reads column names from schema config so renamed/remapped columns work
automatically. To add features for a new Phase 2 data source, implement
a _agg_<source_name> method and register it in build().
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TIER_ORDER = {"free": 0, "starter": 1, "professional": 2, "enterprise": 3}
REGION_ORDER = {"NA": 0, "EU": 1, "APAC": 2, "LATAM": 3}
BILLING_ORDER = {"monthly": 0, "quarterly": 1, "annual": 2}
PROVISIONING_ORDER = {"self_serve": 0, "partner": 1, "sales_led": 2}


class FeatureEngineer:
    """Transforms raw source DataFrames into a flat feature matrix ready for model training or inference.

    The engineer reads column names from the schema config, making it
    resilient to column renames. A fixed reference date (or today's date)
    is used for all relative time calculations so that features are
    reproducible across runs.

    Attributes:
        schema: The full schema config dict passed at construction time.
        ref_date: The timezone-naive Timestamp used as "today" for all
            age/recency calculations.
    """

    def __init__(self, schema_cfg: dict[str, Any]):
        """Initialise the engineer and resolve the reference date.

        Args:
            schema_cfg: Schema configuration dict. May contain a
                ``reference_date`` key whose value is either ``"auto"``
                (use today's date) or an ISO-8601 date string that is
                parsed into a fixed ``pd.Timestamp``.
        """
        self.schema = schema_cfg
        ref = schema_cfg.get("reference_date", "auto")
        if ref == "auto":
            self.ref_date = pd.Timestamp.now(tz=None).normalize()
        else:
            self.ref_date = pd.Timestamp(ref)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        frames: dict[str, pd.DataFrame],
        target_cfg: dict[str, Any],
        exclude: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Build the complete feature matrix and binary target series from raw source frames.

        Starts with account-level features, then iterates over every
        additional source frame, dispatching to a dedicated ``_agg_<name>``
        method when one exists or falling back to ``_agg_generic``.
        Missing numeric values are imputed with per-column medians before
        returning.

        Args:
            frames: Mapping of source name to DataFrame. Must include the
                key ``"accounts"``; all other entries are treated as
                secondary sources that are aggregated and left-joined onto
                the account index.
            target_cfg: Dictionary that must contain:
                - ``"column"`` (str): Column in the accounts frame that
                  holds the churn label.
                - ``"positive_value"`` (str): The label value that maps to
                  ``1`` (churned).
                - ``"extra_positive_values"`` (list[str], optional):
                  Additional label values also treated as positive.

        Returns:
            A two-element tuple ``(features, target)`` where ``features``
            is a ``pd.DataFrame`` indexed by ``account_id`` containing all
            engineered numeric columns, and ``target`` is a ``pd.Series``
            of ``0``/``1`` integers indexed by ``account_id``.

        Raises:
            ValueError: If the ``"accounts"`` key is absent from ``frames``.
        """
        accounts = frames.get("accounts")
        if accounts is None:
            raise ValueError("'accounts' source is required but not loaded.")

        # Start with account-level features
        features = self._build_account_features(accounts)

        # Aggregate and join each non-primary source
        for source_name, df in frames.items():
            if source_name == "accounts":
                continue
            source_def = self.schema["data_sources"].get(source_name, {})
            join_key = source_def.get("join_key", "account_id")
            if join_key not in df.columns:
                logger.warning(
                    "Source '%s' has no join key '%s'; skipping.", source_name, join_key
                )
                continue
            agg_fn = getattr(self, f"_agg_{source_name}", self._agg_generic)
            agg_df = agg_fn(df, source_name)
            if agg_df is not None and not agg_df.empty:
                features = features.join(agg_df, how="left")

        # Build target
        pos_vals = {target_cfg["positive_value"]} | set(target_cfg.get("extra_positive_values", []))
        target = accounts.set_index("account_id")[target_cfg["column"]].map(
            lambda v: 1 if v in pos_vals else 0
        )

        features = features.fillna(features.median(numeric_only=True))

        if exclude:
            drop = [c for c in exclude if c in features.columns]
            if drop:
                features = features.drop(columns=drop)
                logger.info("Excluded features: %s", drop)

        logger.info(
            "Feature matrix: %d accounts × %d features | churn rate=%.1f%%",
            len(features),
            len(features.columns),
            100 * target.mean(),
        )
        return features, target

    # ------------------------------------------------------------------
    # Account features
    # ------------------------------------------------------------------

    def _build_account_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive all account-level features from the primary accounts DataFrame.

        Produces time-based recency/age columns, seat utilisation, numeric
        pass-through columns, boolean flag encodings, ordinal category
        encodings, and a post-health-algorithm era indicator. Columns
        absent from the input are silently skipped so the method works
        against partial schemas.

        Args:
            df: The raw accounts DataFrame. Must contain an ``account_id``
                column which becomes the index of the returned frame.

        Returns:
            A ``pd.DataFrame`` indexed by ``account_id`` containing all
            account-level engineered features as numeric columns. Rows
            correspond 1-to-1 with the input accounts.
        """
        acc = df.set_index("account_id").copy()
        ref = self.ref_date
        feats = pd.DataFrame(index=acc.index)

        # -- Time-based --
        if "created_timestamp" in acc.columns:
            feats["account_age_days"] = (ref - acc["created_timestamp"].dt.tz_localize(None)).dt.days.clip(lower=0)
        if "last_activity_timestamp" in acc.columns:
            feats["days_since_last_activity"] = (ref - acc["last_activity_timestamp"].dt.tz_localize(None)).dt.days.clip(lower=0)
        if "contract_end_date" in acc.columns:
            feats["days_until_contract_end"] = (acc["contract_end_date"].dt.tz_localize(None) - ref).dt.days
        # status_change_date intentionally excluded: it records the churn event itself
        # and would cause direct data leakage (days_since_status_change ≈ 0 for churned).

        # -- Seat utilization --
        if "seats_purchased" in acc.columns and "seats_active" in acc.columns:
            feats["seat_utilization_rate"] = (
                acc["seats_active"] / acc["seats_purchased"].replace(0, np.nan)
            ).clip(0, 1)

        # -- Numeric pass-through --
        for col in ["mrr", "integration_count", "account_health_score", "internal_notes_count",
                    "seats_purchased", "seats_active"]:
            if col in acc.columns:
                feats[col] = pd.to_numeric(acc[col], errors="coerce")

        # -- Boolean flags --
        for col in ["auto_renew_enabled", "api_calls_enabled", "sso_enabled",
                    "white_label_enabled", "risk_flag"]:
            if col in acc.columns:
                feats[col] = acc[col].map(lambda v: 1 if v is True else (0 if v is False else np.nan))

        feats["has_account_manager"] = acc.get("account_manager_id", pd.Series(dtype=str)).notna().astype(int)
        feats["risk_flag_known"] = feats.get("risk_flag", pd.Series(np.nan, index=acc.index)).notna().astype(int)

        # -- Ordinal encodings --
        if "subscription_tier" in acc.columns:
            feats["tier_rank"] = acc["subscription_tier"].map(TIER_ORDER).fillna(0)
        if "region" in acc.columns:
            feats["region_code"] = acc["region"].map(REGION_ORDER).fillna(0)
        if "billing_cycle" in acc.columns:
            feats["billing_rank"] = acc["billing_cycle"].map(BILLING_ORDER).fillna(0)
        if "provisioning_method" in acc.columns:
            feats["provisioning_rank"] = acc["provisioning_method"].map(PROVISIONING_ORDER).fillna(0)

        # -- Health score era flag (algorithm changed Q3 2023) --
        if "created_timestamp" in acc.columns:
            cutoff = pd.Timestamp("2023-07-01")
            feats["post_health_algo"] = (acc["created_timestamp"].dt.tz_localize(None) >= cutoff).astype(int)

        return feats

    # ------------------------------------------------------------------
    # User engagement aggregation
    # ------------------------------------------------------------------

    def _agg_users(self, df: pd.DataFrame, _name: str) -> pd.DataFrame:
        """Aggregate per-user engagement rows into one row of features per account.

        Computes user counts, active-user percentages, admin counts,
        aggregated login and session metrics, feature-usage scores, and
        adoption rates for boolean flags such as mobile-app usage and
        certification completion.

        Args:
            df: Users DataFrame. Must contain an ``account_id`` column used
                as the grouping key. All other recognised columns are
                optional; absent columns are silently skipped.
            _name: Source name string (unused; present for interface
                consistency with other ``_agg_*`` methods).

        Returns:
            A ``pd.DataFrame`` indexed by ``account_id`` containing
            aggregated user-engagement features, one row per account.
        """
        g = df.groupby("account_id")
        agg = pd.DataFrame(index=g.groups.keys())

        agg["user_count"] = g.size()
        if "user_status" in df.columns:
            agg["active_user_pct"] = df.groupby("account_id")["user_status"].apply(
                lambda x: (x == "active").mean()
            )
        if "user_role" in df.columns:
            agg["admin_count"] = df.groupby("account_id")["user_role"].apply(
                lambda x: (x == "admin").sum()
            )

        for col, agg_type in [
            ("login_count_30d", "mean"), ("login_count_30d", "max"),
            ("login_count_90d", "mean"), ("feature_usage_score", "mean"),
            ("avg_session_duration_minutes", "mean"), ("sessions_count_30d", "sum"),
            ("reports_generated_30d", "sum"), ("exports_count_30d", "sum"),
            ("dashboards_created", "sum"), ("dashboards_shared", "sum"),
            ("profile_completeness_pct", "mean"),
        ]:
            if col not in df.columns:
                continue
            col_out = f"{col}_{agg_type}" if agg_type != "mean" else col
            if agg_type == "mean":
                agg[col_out] = g[col].mean()
            elif agg_type == "max":
                agg[f"{col}_max"] = g[col].max()
            elif agg_type == "sum":
                agg[f"{col}_total"] = g[col].sum()

        for flag in ["api_key_active", "mobile_app_user", "onboarding_completed",
                     "certification_earned", "beta_features_enabled"]:
            if flag in df.columns:
                agg[f"{flag}_rate"] = g[flag].apply(
                    lambda x: x.map(lambda v: 1 if v is True else (0 if v is False else np.nan)).mean()
                )

        agg.index.name = "account_id"
        return agg

    # ------------------------------------------------------------------
    # Support aggregation
    # ------------------------------------------------------------------

    def _agg_support(self, df: pd.DataFrame, _name: str) -> pd.DataFrame:
        """Aggregate per-ticket support rows into one row of features per account.

        Computes total and recency-windowed ticket counts, mean resolution
        time and satisfaction scores, boolean-flag event counts (escalations,
        SLA breaches, cancellation requests, etc.), sentiment rates,
        per-category ticket counts, competitor mention counts, and the
        retention-offer acceptance rate.

        Args:
            df: Support-tickets DataFrame. Must contain an ``account_id``
                column. A ``created_date`` datetime column is used for
                recency windows if present; all other recognised columns
                are optional.
            _name: Source name string (unused; present for interface
                consistency with other ``_agg_*`` methods).

        Returns:
            A ``pd.DataFrame`` indexed by ``account_id`` containing
            aggregated support features, one row per account.
        """
        df = df.copy()  # avoid mutating caller's frame
        if "created_date" in df.columns:
            df["_age_days"] = (self.ref_date - df["created_date"].dt.tz_localize(None)).dt.days

        g = df.groupby("account_id")
        agg = pd.DataFrame(index=g.groups.keys())

        agg["ticket_count"] = g.size()

        if "_age_days" in df.columns:
            agg["tickets_last_90d"] = g["_age_days"].apply(lambda x: (x <= 90).sum())
            agg["tickets_last_30d"] = g["_age_days"].apply(lambda x: (x <= 30).sum())

        for col in ["resolution_time_hours", "satisfaction_rating", "interaction_count",
                    "reopened_count", "kb_articles_referenced"]:
            if col in df.columns:
                agg[f"{col}_mean"] = g[col].mean()

        for flag in ["escalated", "sla_breach", "cancellation_requested",
                     "retention_offer_made", "downgrade_requested", "account_pause_requested"]:
            if flag in df.columns:
                agg[f"{flag}_count"] = g[flag].apply(
                    lambda x: x.map(lambda v: 1 if v is True else 0).sum()
                )

        if "ticket_sentiment" in df.columns:
            agg["negative_sentiment_rate"] = g["ticket_sentiment"].apply(
                lambda x: x.isin(["negative", "frustrated"]).mean()
            )

        if "ticket_category" in df.columns:
            for cat in ["billing", "technical", "cancellation", "bug", "feature_request"]:
                agg[f"ticket_cat_{cat}"] = g["ticket_category"].apply(lambda x: (x == cat).sum())

        if "competitor_mentioned" in df.columns:
            agg["competitor_mention_count"] = g["competitor_mentioned"].apply(
                lambda x: x.notna().sum()
            )

        if "retention_offer_made" in df.columns and "retention_offer_accepted" in df.columns:
            made = df["retention_offer_made"].map(lambda v: v is True)
            accepted = df["retention_offer_accepted"].map(lambda v: 1 if v is True else 0)
            agg["retention_offer_acceptance_rate"] = (
                df[made].assign(_acc=accepted[made]).groupby("account_id")["_acc"].mean()
            )

        agg.index.name = "account_id"
        return agg

    # ------------------------------------------------------------------
    # Generic aggregation fallback for unknown Phase 2 sources
    # ------------------------------------------------------------------

    def _agg_generic(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Aggregate an unknown source by taking the mean of numeric columns and the true-rate of boolean-string columns.

        This is the fallback used when no dedicated ``_agg_<source_name>``
        method exists. Output column names are prefixed with
        ``<source_name>__`` to avoid collisions across sources.

        Args:
            df: Source DataFrame. Must contain an ``account_id`` column.
                Numeric columns (excluding ``account_id``) are aggregated
                by mean. Object columns whose non-null values are all
                ``"True"``/``"False"`` (case-insensitive) are treated as
                boolean and aggregated by true-rate.
            source_name: Name of the data source, used as a column prefix
                in the output (e.g. ``"nps"`` produces columns like
                ``nps__score_mean``).

        Returns:
            A ``pd.DataFrame`` indexed by ``account_id`` with aggregated
            feature columns, or an empty ``pd.DataFrame`` if the input
            contains no ``account_id`` column.
        """
        join_key = "account_id"
        if join_key not in df.columns:
            return pd.DataFrame()
        g = df.groupby(join_key)
        agg = pd.DataFrame(index=g.groups.keys())
        agg.index.name = "account_id"

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col == join_key:
                continue
            agg[f"{source_name}__{col}_mean"] = g[col].mean()

        bool_cols = [c for c in df.columns if df[c].dtype == object and
                     df[c].dropna().isin(["True", "False", "true", "false"]).all()]
        for col in bool_cols:
            agg[f"{source_name}__{col}_rate"] = g[col].apply(
                lambda x: x.map(lambda v: 1 if str(v).lower() == "true" else 0).mean()
            )
        return agg
