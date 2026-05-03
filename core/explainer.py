"""Per-account and global feature contribution explanations.

Uses XGBoost's built-in pred_contribs (TreeSHAP) instead of the shap
library directly — avoids shap/xgboost version incompatibilities while
producing identical values.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)

FEATURE_LABELS: dict[str, str] = {
    "days_since_last_activity": "Days since last activity",
    "account_age_days": "Account age (days)",
    "seat_utilization_rate": "Seat utilization rate",
    "mrr": "Monthly recurring revenue ($)",
    "login_count_30d": "Avg logins per user (30d)",
    "login_count_30d_max": "Max logins by any user (30d)",
    "login_count_90d": "Avg logins per user (90d)",
    "feature_usage_score": "Feature usage score",
    "avg_session_duration_minutes": "Avg session duration (min)",
    "sessions_count_30d_total": "Total sessions (30d)",
    "reports_generated_30d_total": "Total reports generated (30d)",
    "exports_count_30d_total": "Total exports (30d)",
    "ticket_count": "Total support tickets",
    "tickets_last_90d": "Tickets in last 90 days",
    "tickets_last_30d": "Tickets in last 30 days",
    "cancellation_requested_count": "Cancellation requests",
    "negative_sentiment_rate": "Negative ticket sentiment rate",
    "satisfaction_rating_mean": "Avg CSAT score",
    "escalated_count": "Escalated tickets",
    "sla_breach_count": "SLA breaches",
    "competitor_mention_count": "Competitor mentions",
    "integration_count": "Active integrations",
    "account_health_score": "Account health score",
    "risk_flag": "CS risk flag",
    "auto_renew_enabled": "Auto-renew enabled",
    "tier_rank": "Subscription tier",
    "billing_rank": "Billing cycle",
    "days_until_contract_end": "Days until contract end",
    "user_count": "Number of users",
    "active_user_pct": "Active user percentage",
    "onboarding_completed_rate": "Onboarding completion rate",
    "certification_earned_rate": "Certification earned rate",
    "downgrade_requested_count": "Downgrade requests",
    "retention_offer_acceptance_rate": "Retention offer acceptance rate",
    "retention_offer_made_count": "Retention offers made",
    "account_pause_requested_count": "Account pause requests",
    "admin_count": "Admin user count",
    "has_account_manager": "Has account manager",
    "api_calls_enabled": "API access enabled",
    "sso_enabled": "SSO enabled",
    "region_code": "Region",
    "provisioning_rank": "Provisioning method",
    "post_health_algo": "Post-2023 health algorithm",
    "resolution_time_hours_mean": "Avg ticket resolution time (hrs)",
    "interaction_count_mean": "Avg ticket interactions",
    "ticket_cat_cancellation": "Cancellation-category tickets",
    "ticket_cat_billing": "Billing-category tickets",
    "ticket_cat_bug": "Bug-category tickets",
}


class Explainer:
    """Computes and surfaces per-account and global TreeSHAP feature contributions.

    After calling :meth:`fit`, the instance holds a (n_accounts, n_features)
    matrix of exact TreeSHAP values computed via XGBoost's native
    ``pred_contribs`` path.  These values can then be queried globally
    (mean absolute contribution across all accounts) or per-account (top
    drivers for a specific customer).

    Attributes:
        cfg: The ``"model"`` sub-dict from the model config, used to locate
            the SHAP value cache path.
        root: Absolute ``pathlib.Path`` to the project root.
        _contribs: NumPy array of shape ``(n_accounts, n_features)`` holding
            the TreeSHAP contribution values once :meth:`fit` has been called,
            or ``None`` beforehand.
        _features: Copy of the feature ``DataFrame`` passed to :meth:`fit`,
            retained so that account-level feature values can be surfaced
            alongside contributions.
        _model: Reference to the fitted model object passed to :meth:`fit`.
    """

    def __init__(self, model_cfg: dict[str, Any], project_root: str):
        """Initialise the explainer with configuration and path information.

        Args:
            model_cfg: The full model configuration dict as returned by
                ``load_config()["model"]``.  Must contain a ``"model"`` key
                whose value includes an optional ``"cache"`` sub-dict with a
                ``"shap_path"`` entry (default ``"models/shap_values.npy"``).
            project_root: Absolute path string to the repository root, used
                to resolve the SHAP cache file path.
        """
        self.cfg = model_cfg["model"]
        self.root = Path(project_root)
        self._contribs: np.ndarray | None = None   # shape (n_accounts, n_features)
        self._features: pd.DataFrame | None = None
        self._model = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def fit(self, model, features: pd.DataFrame) -> None:
        """Compute per-account feature contributions via XGBoost TreeSHAP.

        Attempts to load a previously computed contributions array from the
        NumPy cache file configured in the model config.  If the cache does
        not exist, has a shape mismatch, or cannot be read, the contributions
        are recomputed via :meth:`_compute` and saved to the cache path for
        future runs.

        Args:
            model: A fitted ``XGBClassifier`` instance (or any object that
                exposes a ``get_booster()`` method compatible with
                ``xgb.Booster.predict``).
            features: A ``pd.DataFrame`` of feature values indexed by account
                ID.  The number of rows and columns must match the model's
                training data for the cache shape check to pass.
        """
        self._features = features.copy()
        self._model = model

        cache = self.cfg.get("cache", {})
        shap_path = self.root / cache.get("shap_path", "models/shap_values.npy")

        # Try valid cache
        if shap_path.exists():
            try:
                cached = np.load(str(shap_path))
                if cached.shape == (len(features), len(features.columns)):
                    self._contribs = cached
                    logger.info("Loaded feature contributions from cache.")
                    return
                else:
                    logger.warning(
                        "Cache shape %s doesn't match features %s — recomputing.",
                        cached.shape, (len(features), len(features.columns)),
                    )
            except Exception as exc:
                logger.warning("Could not load contributions cache: %s", exc)
            shap_path.unlink(missing_ok=True)

        logger.info("Computing feature contributions for %d accounts…", len(features))
        self._contribs = self._compute(model, features)
        shap_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(shap_path), self._contribs)
        logger.info("Feature contributions cached to %s", shap_path)

    def _compute(self, model, features: pd.DataFrame) -> np.ndarray:
        """Run XGBoost pred_contribs and return a pure feature-contribution array.

        Converts ``features`` to an ``xgb.DMatrix``, calls
        ``predict(pred_contribs=True)`` on the booster (which uses exact
        TreeSHAP), and strips the trailing bias column that XGBoost appends
        so that the output has the same number of columns as the input
        feature matrix.

        Args:
            model: A fitted model object with a ``get_booster()`` method
                that returns an ``xgb.Booster``.
            features: A ``pd.DataFrame`` of feature values.  Preprocessing
                (NaN filling, categorical encoding) is applied via
                :meth:`_prepare` before the DMatrix is constructed.

        Returns:
            A NumPy array of shape ``(n_samples, n_features)`` where each
            entry is the TreeSHAP contribution of that feature for that
            sample.  Positive values push the prediction toward churn;
            negative values push it away from churn.
        """
        X = self._prepare(features)
        dm = xgb.DMatrix(X, feature_names=list(features.columns))
        # pred_contribs=True uses TreeSHAP; approx_contribs=False is exact
        raw = model.get_booster().predict(dm, pred_contribs=True)
        return raw[:, :-1]   # drop bias column

    def _prepare(self, features: pd.DataFrame) -> pd.DataFrame:
        """Preprocess a feature matrix so it can be wrapped in an xgb.DMatrix.

        Applies lightweight cleaning: replaces infinities with NaN, fills
        NaNs with column medians (falling back to 0 for all-NaN columns),
        and integer-encodes categorical columns via pandas category codes.
        This mirrors the sanitisation done inside the model without requiring
        access to the model's ``LabelEncoder`` instances.

        Args:
            features: Raw feature ``pd.DataFrame`` to preprocess.

        Returns:
            A cleaned ``pd.DataFrame`` with the same shape as ``features``
            but with all values finite and all dtypes numeric.
        """
        X = features.copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True)).fillna(0)
        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = X[col].astype("category").cat.codes
        return X

    # ------------------------------------------------------------------
    # Global importance
    # ------------------------------------------------------------------

    def global_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Return the top globally important features ranked by mean absolute SHAP value.

        Aggregates the per-account contribution matrix by computing the mean
        of the absolute values across all accounts, giving a single
        importance score per feature that reflects its average influence on
        churn predictions across the whole customer base.

        Args:
            top_n: Maximum number of features to return, sorted descending by
                mean absolute contribution.  Defaults to 10.

        Returns:
            A ``pd.DataFrame`` with columns:

            - ``"feature"`` (str) — internal feature column name.
            - ``"importance"`` (float) — mean absolute TreeSHAP value.
            - ``"label"`` (str) — human-readable label from ``FEATURE_LABELS``,
              or a title-cased version of the feature name if no label is
              defined.

            Returns an empty ``pd.DataFrame`` if :meth:`fit` has not been
            called yet.
        """
        if self._contribs is None or self._features is None:
            return pd.DataFrame()
        mean_abs = np.abs(self._contribs).mean(axis=0)
        df = pd.DataFrame({
            "feature": self._features.columns,
            "importance": mean_abs,
        }).sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
        df["label"] = df["feature"].map(lambda f: FEATURE_LABELS.get(f, f.replace("_", " ").title()))
        return df

    # ------------------------------------------------------------------
    # Per-account explanation
    # ------------------------------------------------------------------

    def explain_account(self, account_id: str, top_n: int = 5) -> dict[str, Any]:
        """Return the top contributing features for a single account as a dict.

        Looks up the account's row in the stored contribution matrix and
        feature values, identifies the ``top_n`` features with the largest
        absolute SHAP contribution, and packages each into a structured dict
        describing its label, contribution magnitude, raw value, and direction
        of influence.

        Args:
            account_id: The account identifier as it appears in the index of
                the feature ``DataFrame`` passed to :meth:`fit`.
            top_n: How many top-contributing features to include in the
                result.  Defaults to 5.

        Returns:
            A dict with two keys on success:

            - ``"account_id"`` (str) — the requested account ID.
            - ``"factors"`` (list[dict]) — list of up to ``top_n`` factor
              dicts, each containing:

              - ``"feature"`` (str) — internal feature name.
              - ``"label"`` (str) — human-readable feature label.
              - ``"contribution"`` (float) — signed SHAP value rounded to
                4 decimal places (positive = increases churn probability).
              - ``"feature_value"`` (float | None) — raw feature value
                rounded to 3 decimal places, or ``None`` if missing.
              - ``"direction"`` (str) — ``"increases"`` or ``"decreases"``.

            On failure returns a dict with a single ``"error"`` key:

            - If the explainer has not been initialised:
              ``{"error": "Explainer not initialized."}``.
            - If the account is not found:
              ``{"error": "Account '<id>' not found."}``.
        """
        if self._contribs is None or self._features is None:
            return {"error": "Explainer not initialized."}
        if account_id not in self._features.index:
            return {"error": f"Account '{account_id}' not found."}

        idx = self._features.index.get_loc(account_id)
        contrib_row = self._contribs[idx]
        feature_vals = self._features.iloc[idx]

        order = np.argsort(np.abs(contrib_row))[::-1][:top_n]
        factors = []
        for i in order:
            fname = self._features.columns[i]
            cv = float(contrib_row[i])
            fv = feature_vals.iloc[i]
            fv_f = float(fv) if pd.notna(fv) else None
            label = FEATURE_LABELS.get(fname, fname.replace("_", " ").title())
            factors.append({
                "feature": fname,
                "label": label,
                "contribution": round(cv, 4),
                "feature_value": round(fv_f, 3) if fv_f is not None else None,
                "direction": "increases" if cv > 0 else "decreases",
            })
        return {"account_id": account_id, "factors": factors}

    def explain_account_text(self, account_id: str, churn_prob: float, top_n: int = 5) -> str:
        """Format a single-account explanation as a human-readable text block.

        Calls :meth:`explain_account` internally and renders the result as a
        multi-line string suitable for printing to a terminal or embedding in
        a report.  Each contributing factor is shown with an arrow indicator
        (up = increases churn risk, down = decreases it), its human-readable
        label, and the raw feature value when available.

        Args:
            account_id: The account identifier to explain (must match the
                index of the feature ``DataFrame`` passed to :meth:`fit`).
            churn_prob: The model's predicted churn probability for this
                account, expressed as a float in [0, 1].  Used to display
                the probability and derive the risk tier label
                (HIGH >= 0.70, MEDIUM >= 0.40, LOW otherwise).
            top_n: Number of top contributing factors to include.  Defaults
                to 5.

        Returns:
            A formatted multi-line string containing the account ID, churn
            probability, risk tier, and the numbered list of contributing
            factors with directional arrows and feature values.  Returns the
            error message string if :meth:`explain_account` reports an error.
        """
        result = self.explain_account(account_id, top_n)
        if "error" in result:
            return result["error"]

        risk = "HIGH" if churn_prob >= 0.70 else ("MEDIUM" if churn_prob >= 0.40 else "LOW")
        lines = [
            f"Account {account_id}",
            f"Churn probability: {churn_prob:.1%}  [{risk} RISK]",
            "",
            f"Top {top_n} contributing factors:",
        ]
        for i, f in enumerate(result["factors"], 1):
            val_str = f" (value: {f['feature_value']})" if f["feature_value"] is not None else ""
            arrow = "▲" if f["direction"] == "increases" else "▼"
            lines.append(
                f"  {i}. {arrow} {f['label']}{val_str}"
            )
        return "\n".join(lines)
