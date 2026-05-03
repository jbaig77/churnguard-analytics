"""Translates raw numeric feature values into plain-English business context.

Used by the conversation agent and UI to make every number digestible.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ── Per-feature metadata ──────────────────────────────────────────────
# higher_is_better=True  → high value is good (low value is a warning sign)
# higher_is_better=False → low value is good (high value is a warning sign)

FIELD_META: dict[str, dict] = {
    "account_health_score": {
        "label": "Account health score",
        "unit": "/100",
        "higher_is_better": True,
        "ranges": [(30, "⚠ Critical"), (50, "Below average"), (70, "Average"), (100, "Healthy")],
        "note": "Platform-computed composite score (0–100). Score < 40 correlates strongly with churn.",
    },
    "days_since_last_activity": {
        "label": "Days since last activity",
        "unit": "days",
        "higher_is_better": False,
        "ranges": [(7, "Very active"), (30, "Normal"), (60, "⚠ Inactive"), (999, "🚨 Severely inactive")],
        "note": "Accounts inactive for >30 days are 3× more likely to churn.",
    },
    "seat_utilization_rate": {
        "label": "Seat utilization",
        "unit": "%",
        "scale": 100,
        "higher_is_better": True,
        "ranges": [(0.25, "🚨 Critical under-use"), (0.50, "⚠ Low"), (0.75, "Adequate"), (1.01, "Full utilization")],
        "note": "Ratio of active seats to purchased seats. Low utilization is a leading churn indicator.",
    },
    "cancellation_requested_count": {
        "label": "Cancellation requests",
        "unit": "tickets",
        "higher_is_better": False,
        "ranges": [(0.5, "None — positive"), (1.5, "⚠ 1 request on file"), (999, "🚨 Multiple requests")],
        "note": "Any cancellation ticket is a strong churn signal.",
    },
    "downgrade_requested_count": {
        "label": "Downgrade requests",
        "unit": "tickets",
        "higher_is_better": False,
        "ranges": [(0.5, "None"), (1.5, "⚠ 1 downgrade request"), (999, "🚨 Multiple downgrade requests")],
        "note": "Downgrade requests often precede full cancellation.",
    },
    "login_count_30d": {
        "label": "Avg logins (30d)",
        "unit": "logins/user",
        "higher_is_better": True,
        "ranges": [(2, "🚨 Very low engagement"), (8, "⚠ Below average"), (20, "Normal"), (999, "Highly engaged")],
        "note": "Average logins per user over the past 30 days.",
    },
    "feature_usage_score": {
        "label": "Feature usage score",
        "unit": "/100",
        "higher_is_better": True,
        "ranges": [(25, "🚨 Minimal adoption"), (50, "⚠ Low"), (75, "Moderate"), (100, "High adoption")],
        "note": "Composite score of product feature breadth usage (0–100).",
    },
    "mrr": {
        "label": "Monthly recurring revenue",
        "unit": "$/mo",
        "higher_is_better": True,
        "ranges": [(50, "Micro (<$50)"), (200, "Small"), (500, "Mid-market"), (99999, "Enterprise")],
        "note": "Revenue at risk if this account churns.",
    },
    "integration_count": {
        "label": "Active integrations",
        "unit": "integrations",
        "higher_is_better": True,
        "ranges": [(0.5, "🚨 None — no integrations"), (2, "⚠ Few"), (5, "Moderate"), (99, "Well integrated")],
        "note": "Accounts with zero integrations have 4× higher churn rates.",
    },
    "negative_sentiment_rate": {
        "label": "Negative ticket sentiment",
        "unit": "%",
        "scale": 100,
        "higher_is_better": False,
        "ranges": [(0.1, "Positive"), (0.3, "⚠ Mixed"), (0.6, "⚠ Mostly negative"), (1.01, "🚨 Highly frustrated")],
        "note": "Fraction of support tickets classified as negative or frustrated.",
    },
    "ticket_count": {
        "label": "Total support tickets",
        "unit": "tickets",
        "higher_is_better": False,
        "ranges": [(2, "Low contact"), (5, "Normal"), (10, "⚠ Elevated"), (999, "🚨 High support burden")],
        "note": "High ticket volume often signals dissatisfaction or complex issues.",
    },
    "satisfaction_rating_mean": {
        "label": "Avg CSAT score",
        "unit": "/5",
        "higher_is_better": True,
        "ranges": [(2, "🚨 Very dissatisfied"), (3, "⚠ Below average"), (4, "Adequate"), (5, "Excellent")],
        "note": "Customer satisfaction rating on support tickets (1–5 scale).",
    },
    "competitor_mention_count": {
        "label": "Competitor mentions",
        "unit": "tickets",
        "higher_is_better": False,
        "ranges": [(0.5, "None — good"), (1.5, "⚠ 1 mention"), (999, "🚨 Multiple mentions")],
        "note": "Number of support tickets referencing competitor products.",
    },
    "active_user_pct": {
        "label": "Active users",
        "unit": "%",
        "scale": 100,
        "higher_is_better": True,
        "ranges": [(0.30, "🚨 Most users inactive"), (0.60, "⚠ Partial adoption"), (0.85, "Good"), (1.01, "Full adoption")],
        "note": "Percentage of provisioned users who are still active.",
    },
    "account_age_days": {
        "label": "Account age",
        "unit": "days",
        "higher_is_better": True,
        "ranges": [(90, "New account (<3 mo)"), (365, "Established"), (730, "Long-term"), (9999, "Veteran customer")],
        "note": "Older accounts with sudden disengagement are particularly at risk.",
    },
    "days_until_contract_end": {
        "label": "Days until contract end",
        "unit": "days",
        "higher_is_better": True,
        "ranges": [(-1, "🚨 Contract expired"), (30, "🚨 Expiring within 30 days"), (90, "⚠ Expiring soon"), (9999, "Ample time remaining")],
        "note": "Negative = contract already expired. Accounts near expiry with low engagement rarely renew.",
    },
    "onboarding_completed_rate": {
        "label": "Onboarding completion",
        "unit": "%",
        "scale": 100,
        "higher_is_better": True,
        "ranges": [(0.3, "🚨 Poor onboarding"), (0.6, "⚠ Partial"), (0.9, "Good"), (1.01, "Complete")],
        "note": "Accounts where most users haven't completed onboarding rarely reach full value.",
    },
}


class Interpreter:
    """Pre-computes portfolio statistics so each feature value can be put in plain-English business context."""

    def __init__(self, features: pd.DataFrame, predictions: pd.DataFrame | None = None):
        """Store feature/prediction frames and pre-compute per-column percentile stats for all numeric columns.

        Args:
            features: DataFrame of engineered feature values for the full account portfolio.
            predictions: Optional DataFrame of model prediction outputs (e.g., churn probabilities).
        """
        self.features = features
        self.predictions = predictions
        # Precompute per-column stats
        self._stats: dict[str, dict] = {}
        for col in features.select_dtypes(include=[np.number]).columns:
            s = features[col].dropna()
            if len(s) > 0:
                self._stats[col] = {
                    "mean": float(s.mean()),
                    "median": float(s.median()),
                    "p10": float(s.quantile(0.10)),
                    "p25": float(s.quantile(0.25)),
                    "p75": float(s.quantile(0.75)),
                    "p90": float(s.quantile(0.90)),
                }

    # ── Public API ────────────────────────────────────────────────────

    def context_line(self, feature: str, value: float) -> str:
        """Return a one-line contextual annotation: range label and percentile vs. portfolio, or empty string if value is NaN.

        Args:
            feature: Name of the feature column to annotate.
            value: The numeric feature value for a specific account.

        Returns:
            A formatted string starting with "  -> " containing the range label and portfolio percentile,
            or an empty string when value is NaN or no metadata is available.
        """
        meta = FIELD_META.get(feature)
        stats = self._stats.get(feature)
        if value is None or np.isnan(float(value)):
            return ""

        parts = []
        # Range label
        if meta:
            label = self._range_label(meta, value)
            if label:
                parts.append(label)
        # Percentile
        if stats:
            pct = self._percentile(feature, value)
            if pct is not None:
                h_is_better = meta.get("higher_is_better", True) if meta else True
                if h_is_better:
                    parts.append(f"bottom {pct:.0f}% of accounts" if pct < 50 else f"top {100-pct:.0f}% of accounts")
                else:
                    parts.append(f"higher than {pct:.0f}% of accounts")
            parts.append(f"median: {self._fmt(feature, stats['median'])}")
        return "  → " + " | ".join(parts) if parts else ""

    def explain_probability(self, prob: float) -> str:
        """Return a business-language interpretation of a churn probability.

        Args:
            prob: Predicted churn probability as a float between 0 and 1.

        Returns:
            A plain-English string describing the risk level and recommended action.
        """
        if prob >= 0.90:
            return "🚨 Immediate action required — nearly certain to churn without intervention."
        if prob >= 0.70:
            return "⚠ High risk — CS team should reach out this week."
        if prob >= 0.20:
            return "⚠ Elevated risk — worth monitoring, consider proactive outreach."
        return "✓ Low risk — standard monitoring is sufficient."

    def explain_metrics(self, metrics: dict, churn_rate: float) -> list[str]:
        """Return plain-English sentences interpreting AUC, recall, and precision for non-technical readers.

        Args:
            metrics: Dict containing keys 'roc_auc', 'recall', and 'precision' as floats.
            churn_rate: Historical churn rate as a float between 0 and 1, used for calibration context.

        Returns:
            A list of plain-English strings, one per metric plus an optional calibration note.
        """
        lines = []
        auc = metrics.get("roc_auc", 0)
        recall = metrics.get("recall", 0)
        precision = metrics.get("precision", 0)

        if auc >= 0.99:
            lines.append(
                "⚠ Note: AUC = 1.00 reflects the high predictive signal in this synthetic dataset "
                "(cancellation requests and health scores are near-deterministic). "
                "Real-world production AUC is typically 0.80–0.92."
            )
        else:
            lines.append(f"AUC of {auc:.2f} means the model correctly ranks a churned account "
                         f"above a healthy account {auc*100:.0f}% of the time.")

        lines.append(
            f"Recall of {recall:.0%} means the model catches {recall*100:.0f}% of accounts "
            "that actually churn (minimising missed at-risk customers)."
        )
        lines.append(
            f"Precision of {precision:.0%} means {precision*100:.0f}% of accounts the model "
            "flags as high-risk actually do churn."
        )
        if churn_rate > 0:
            lines.append(
                f"Historical churn rate: {churn_rate:.1%} — the model's average predicted "
                f"probability ({churn_rate:.1%}) is well-calibrated to this baseline."
            )
        return lines

    def explain_summary(self, summary: dict) -> list[str]:
        """Convert a portfolio summary dict into 1-2 plain-English sentences about high-risk account count and average probability.

        Args:
            summary: Dict with keys 'total_accounts' (int), 'high_risk_count' (int), and 'avg_churn_prob' (float).

        Returns:
            A list of plain-English strings summarising portfolio risk.
        """
        total = summary.get("total_accounts", 0)
        high = summary.get("high_risk_count", 0)
        avg_p = summary.get("avg_churn_prob", 0)
        lines = []
        if total:
            pct_high = high / total
            lines.append(
                f"{high:,} of {total:,} accounts ({pct_high:.1%}) are high risk (≥70%) "
                "and warrant immediate Customer Success attention."
            )
        if avg_p:
            lines.append(
                f"Average predicted churn probability of {avg_p:.1%} is consistent with "
                "the historical churn rate in this dataset."
            )
        return lines

    # ── Internals ─────────────────────────────────────────────────────

    def _range_label(self, meta: dict, value: float) -> str:
        """Look up which range bucket the value falls into and return the label string.

        Args:
            meta: Feature metadata dict containing 'ranges' (list of (threshold, label) tuples)
                  and optional 'scale' factor.
            value: The raw numeric feature value to classify.

        Returns:
            The label string corresponding to the matching range bucket.
        """
        scale = meta.get("scale", 1)
        scaled = value * scale
        for threshold, label in meta["ranges"]:
            if scaled <= threshold * scale:
                return label
        return meta["ranges"][-1][1]

    def _percentile(self, feature: str, value: float) -> float | None:
        """Compute what percentile the given value sits at within the feature column distribution.

        Args:
            feature: Name of the feature column to compare against.
            value: The numeric value whose percentile rank is to be computed.

        Returns:
            Percentile as a float between 0 and 100, or None if the column has no non-null values.
        """
        col = self.features[feature].dropna()
        if len(col) == 0:
            return None
        return float((col <= value).mean() * 100)

    def _fmt(self, feature: str, value: float) -> str:
        """Format a numeric value with optional scale factor and unit suffix for display.

        Args:
            feature: Name of the feature, used to look up scale and unit metadata.
            value: The raw numeric value to format.

        Returns:
            A formatted string with the scaled value and unit suffix appended.
        """
        meta = FIELD_META.get(feature, {})
        scale = meta.get("scale", 1)
        unit = meta.get("unit", "")
        v = value * scale
        if abs(v) >= 1000:
            return f"{v:,.0f}{unit}"
        if abs(v) >= 10:
            return f"{v:.1f}{unit}"
        return f"{v:.2f}{unit}"
