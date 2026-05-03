"""AnalyticsAgent — pre-computed summaries and on-demand account lookups."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import pandas as pd

from agents.base_agent import BaseAgent

if TYPE_CHECKING:
    from agents.orchestrator import Orchestrator


class AnalyticsAgent(BaseAgent):
    """Agent responsible for all read-only analytics operations.

    Handles portfolio-level summaries, top-risk rankings, per-account detail
    lookups, fuzzy account searches, and churn-probability distribution
    bucketing.  All data is read from the shared
    :class:`~agents.orchestrator.SharedState`; this agent never writes to it.
    """

    def __init__(self, orchestrator: "Orchestrator"):
        """Initialise the analytics agent and bind it to the orchestrator.

        Args:
            orchestrator: The central :class:`~agents.orchestrator.Orchestrator`
                          instance whose ``state`` and ``config`` will be
                          consulted for every analytics request.
        """
        super().__init__("analytics")
        self.orch = orchestrator

    def handle(self, msg_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Dispatch an inbound analytics message to the appropriate handler.

        Args:
            msg_type: One of ``"summary"``, ``"top_risk"``,
                      ``"account_detail"``, ``"search_account"``, or
                      ``"churn_distribution"``.
            payload:  Parameter dict whose expected keys depend on
                      ``msg_type``:

                      * ``"top_risk"`` — ``n`` (int, default 20): number of
                        accounts to return.
                      * ``"account_detail"`` — ``account_id`` (str): the
                        account to look up.
                      * ``"search_account"`` — ``query`` (str): substring to
                        match against account IDs.
                      * All other types ignore ``payload``.

        Returns:
            A response dict with ``"status": "ok"`` and type-specific keys on
            success, or ``{"status": "error", "message": <str>}`` for unknown
            message types or missing data.
        """
        if msg_type == "summary":
            return self._summary()
        if msg_type == "top_risk":
            return self._top_risk(payload.get("n", 20))
        if msg_type == "account_detail":
            return self._account_detail(payload.get("account_id", ""))
        if msg_type == "search_account":
            return self._search_account(payload.get("query", ""))
        if msg_type == "churn_distribution":
            return self._churn_distribution()
        return self._err(f"Unknown message type: {msg_type}")

    # ------------------------------------------------------------------

    def _summary(self) -> dict[str, Any]:
        """Compute a portfolio-level churn risk summary from current predictions.

        Reads risk-threshold configuration values from the orchestrator config
        and counts how many accounts fall into each risk band.

        Returns:
            Ok response with keys:

            * ``total_accounts`` (int): total number of scored accounts.
            * ``high_risk_count`` (int): accounts at or above the high-risk
              threshold.
            * ``medium_risk_count`` (int): accounts between the medium and
              high thresholds.
            * ``low_risk_count`` (int): accounts below the medium threshold.
            * ``avg_churn_prob`` (float, 4 d.p.): mean churn probability.
            * ``max_churn_prob`` (float, 4 d.p.): highest churn probability.
            * ``metrics`` (dict): training/evaluation metrics from shared state.

            Returns an error response if no predictions are available.
        """
        preds = self.orch.state.predictions
        metrics = self.orch.state.metrics
        if preds is None:
            return self._err("No predictions available.")

        high_t = self.orch.config["model"]["model"]["high_risk_threshold"]
        med_t = self.orch.config["model"]["model"]["medium_risk_threshold"]
        probs = preds["churn_probability"]

        return self._ok(
            total_accounts=len(preds),
            high_risk_count=int((probs >= high_t).sum()),
            medium_risk_count=int(((probs >= med_t) & (probs < high_t)).sum()),
            low_risk_count=int((probs < med_t).sum()),
            avg_churn_prob=round(float(probs.mean()), 4),
            max_churn_prob=round(float(probs.max()), 4),
            metrics=metrics,
        )

    def _top_risk(self, n: int) -> dict[str, Any]:
        """Return the ``n`` accounts with the highest predicted churn probability.

        Predictions are sorted descending by ``churn_probability``.  When an
        ``accounts`` source frame is available the result is enriched with
        ``subscription_tier``, ``mrr``, and ``region`` where present.

        Args:
            n: Maximum number of accounts to return.  Defaults to 20 when
               called via :meth:`handle`.

        Returns:
            Ok response with key ``"accounts"`` containing a list of record
            dicts, each including at minimum ``account_id``,
            ``churn_probability`` (rounded to 4 d.p.), and ``risk_level``,
            plus any available metadata columns.

            Returns an error response if no predictions are available.
        """
        preds = self.orch.state.predictions
        accounts_df = self.orch.state.frames.get("accounts", pd.DataFrame())
        if preds is None:
            return self._err("No predictions available.")

        top = preds.sort_values("churn_probability", ascending=False).head(n).copy()
        top = top.reset_index()

        # Enrich with account metadata
        if not accounts_df.empty and "account_id" in accounts_df.columns:
            meta_cols = [c for c in ["subscription_tier", "mrr", "region"] if c in accounts_df.columns]
            meta = accounts_df.set_index("account_id")[meta_cols]
            top = top.join(meta, on="account_id", how="left")

        records = top.to_dict("records")
        for r in records:
            r["churn_probability"] = round(float(r["churn_probability"]), 4)
        return self._ok(accounts=records)

    def _account_detail(self, account_id: str) -> dict[str, Any]:
        """Retrieve the full prediction and raw account data for a single account.

        Args:
            account_id: The unique account identifier to look up, matching an
                        index value in the predictions DataFrame.

        Returns:
            Ok response with keys:

            * ``account_id`` (str): echoed back for convenience.
            * ``churn_probability`` (float, 4 d.p.): model-predicted probability.
            * ``risk_level`` (str): categorical risk band (e.g. ``"HIGH"``).
            * ``account_data`` (dict): raw fields from the accounts source frame,
              with NaN values dropped; empty dict if the source frame is absent.

            Returns an error response if predictions are unavailable or the
            account ID is not found.
        """
        preds = self.orch.state.predictions
        if preds is None:
            return self._err("No predictions available.")
        if account_id not in preds.index:
            return self._err(f"Account '{account_id}' not found.")

        prob = float(preds.loc[account_id, "churn_probability"])
        risk = str(preds.loc[account_id, "risk_level"])

        # Raw account fields
        accounts_df = self.orch.state.frames.get("accounts", pd.DataFrame())
        acct_row = {}
        if not accounts_df.empty and "account_id" in accounts_df.columns:
            row = accounts_df[accounts_df["account_id"] == account_id]
            if not row.empty:
                acct_row = row.iloc[0].dropna().to_dict()

        return self._ok(
            account_id=account_id,
            churn_probability=round(prob, 4),
            risk_level=risk,
            account_data=acct_row,
        )

    def _search_account(self, query: str) -> dict[str, Any]:
        """Find accounts whose ID contains the query string.

        The search is case-insensitive and returns at most 10 matching
        accounts from the current predictions index.

        Args:
            query: Substring to look for within account IDs.  Leading and
                   trailing whitespace is stripped before matching.

        Returns:
            Ok response with key ``"accounts"`` containing a list of dicts,
            each with ``account_id``, ``churn_probability`` (rounded to 4
            d.p.), and ``risk_level``.

            Returns an error response if predictions are unavailable or no
            accounts match the query.
        """
        preds = self.orch.state.predictions
        if preds is None:
            return self._err("No predictions available.")
        q = query.upper().strip()
        matches = [aid for aid in preds.index if q in str(aid).upper()][:10]
        if not matches:
            return self._err(f"No accounts matching '{query}'.")
        results = []
        for aid in matches:
            prob = float(preds.loc[aid, "churn_probability"])
            risk = str(preds.loc[aid, "risk_level"])
            results.append({"account_id": aid, "churn_probability": round(prob, 4), "risk_level": risk})
        return self._ok(accounts=results)

    def _churn_distribution(self) -> dict[str, Any]:
        """Bucket all accounts by churn probability into 10-percentage-point bands.

        Produces ten non-overlapping intervals from 0 % to 100 % (labels such
        as ``"0-10%"``, ``"10-20%"``, …, ``"90-100%"``) and counts how many
        accounts fall in each band.

        Returns:
            Ok response with key ``"distribution"`` containing a dict mapping
            each band label (str) to the number of accounts (int) in that band,
            sorted by band order.

            Returns an error response if no predictions are available.
        """
        preds = self.orch.state.predictions
        if preds is None:
            return self._err("No predictions available.")
        buckets = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        labels = [f"{int(b*100)}-{int(buckets[i+1]*100)}%" for i, b in enumerate(buckets[:-1])]
        counts = pd.cut(preds["churn_probability"], bins=buckets, labels=labels, right=False).value_counts().sort_index()
        return self._ok(distribution=counts.to_dict())
