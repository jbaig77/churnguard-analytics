"""ConversationAgent — pattern-based natural language router.

Handles the five required query types from the brief plus common
variations. No external NLP dependencies needed.
"""
from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

from agents.base_agent import BaseAgent

if TYPE_CHECKING:
    from agents.orchestrator import Orchestrator

# ── Intent patterns ───────────────────────────────────────────────────

_ACCOUNT_RE = re.compile(
    r"\b(ACC\d[\w-]*|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
    re.IGNORECASE,
)
_TOP_N_RE = re.compile(
    r"\btop\s*(\d+)\b"
    r"|(?:give\s+me|show|list|get|find|what|which|return)\s+(?:me\s+)?(?:the\s+)?(\d+)\b"
    r"|\b(\d+)\s+(?:account|compan|company|customer|record)\b",
    re.IGNORECASE,
)

_WHY_RE = re.compile(
    r"\b(why|explain|reason|factor|driver|cause|contribut|"
    r"what.*risk|what.*churn|break.?down|detail)\b", re.IGNORECASE,
)
_PROB_RE = re.compile(
    r"\b(probability|prob|chance|likelihood|percent|how\s+likely|"
    r"churn\s+risk|risk\s+score|predicted)\b", re.IGNORECASE,
)
_TOP_RISK_RE = re.compile(
    r"\b(top.*account|most\s+likely|highest\s+risk|riskiest|"
    r"at.risk|likely\s+to\s+churn|show.*churn|list.*churn)\b", re.IGNORECASE,
)
_IMPORTANCE_RE = re.compile(
    r"\b(feature\s+import|most\s+import|top\s+feature|top\s+predictor|"
    r"what\s+drive|what\s+predict|which\s+feature|feature\w*\s+predict|overall\s+factor)", re.IGNORECASE,
)
_PERF_RE = re.compile(
    r"\b(model\s+perform|how\s+(good|well|accurat)|auc|roc|"
    r"recall|precision|f1|metric|evaluate|accuracy)", re.IGNORECASE,
)
_SUMMARY_RE = re.compile(
    r"\b(summary|overview|portfolio|how\s+many|total|stat|dashboard)\b", re.IGNORECASE,
)
_LOWEST_RE = re.compile(
    r"\b(lowest|safest|least\s+likely|minimum\s+churn|healthiest)\b", re.IGNORECASE,
)
_MRR_RISK_RE = re.compile(
    r"\b(mrr|revenue|arr).{0,30}\b(risk|churn|los)|"
    r"\b(risk|churn|los).{0,30}\b(mrr|revenue|arr)", re.IGNORECASE,
)
_TIER_RE = re.compile(r"\b(free|starter|professional|enterprise)\b", re.IGNORECASE)
_REGION_RE = re.compile(r"\b(NA|EU|APAC|LATAM)\b", re.IGNORECASE)
_BREAKDOWN_RE = re.compile(r"\bby\s+(tier|plan|region|geography)\b|\bbreakdown\b", re.IGNORECASE)


class ConversationAgent(BaseAgent):
    def __init__(self, orchestrator: "Orchestrator"):
        """Store a reference to the orchestrator for inter-agent communication."""
        super().__init__("conversation")
        self.orch = orchestrator

    def handle(self, msg_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Dispatch an incoming message by type, delegating 'query' messages to _route.

        Args:
            msg_type: The message type string (e.g. 'query').
            payload: A dict containing message data; 'query' messages must include a 'text' key.

        Returns:
            A response dict produced by the appropriate handler or an error dict for unknown types.
        """
        if msg_type == "query":
            return self._route(payload.get("text", "").strip())
        return self._err(f"Unknown message type: {msg_type}")

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------

    def _route(self, text: str) -> dict[str, Any]:
        """Route a natural-language query to the correct handler by matching intent patterns in priority order.

        Args:
            text: The raw user query string.

        Returns:
            A response dict from the matched handler, or a guidance message if no pattern matches.
        """
        if not text:
            return self._ok(response="Please enter a query. Type /help for examples.")
        if not self.orch.state.is_ready:
            return self._ok(response="System is still initializing — please wait a moment.")

        tl = text.lower()
        account_id = _extract_account_id(text)
        top_n = _extract_top_n(text)

        # 1. Slash commands
        if tl.startswith("/"):
            return self._slash(text, tl)

        # 2. Account-specific queries (required query types 1, 2, 4)
        if account_id:
            if _WHY_RE.search(tl) or _TOP_N_RE.search(tl):
                return self._explain(account_id, top_n or 5)
            if _PROB_RE.search(tl):
                return self._probability(account_id)
            return self._explain(account_id, top_n or 5)

        # 3. Lowest-risk accounts
        if _LOWEST_RE.search(tl):
            n = top_n or 1
            return self._lowest_risk(n)

        # 4. MRR at risk
        if _MRR_RISK_RE.search(tl):
            return self._mrr_at_risk()

        # 5. Filter by tier or region
        tier_m = _TIER_RE.search(tl)
        region_m = _REGION_RE.search(tl)
        if tier_m and not _BREAKDOWN_RE.search(tl):
            return self._accounts_by_filter(tier=tier_m.group(1).lower(), n=top_n or 10)
        if region_m and not _BREAKDOWN_RE.search(tl):
            return self._accounts_by_filter(region=region_m.group(1).upper(), n=top_n or 10)

        # 6. Breakdowns by tier/region
        if _BREAKDOWN_RE.search(tl):
            field = "region" if re.search(r"\b(region|geography)\b", tl) else "subscription_tier"
            return self._breakdown(field)

        # 7. Required query type 3: feature importance
        if _IMPORTANCE_RE.search(tl):
            return self._feature_importance()

        # 8. Required query type 5: top-risk accounts
        if _TOP_RISK_RE.search(tl) or top_n:
            return self._top_risk(top_n or 5)

        # 9. Model performance / portfolio summary
        if _PERF_RE.search(tl):
            return self._model_performance()
        if _SUMMARY_RE.search(tl):
            return self._summary()

        return self._ok(response=(
            "I can help with:\n"
            "  • \"Why is ACC000123 predicted to churn?\"\n"
            "  • \"Top 3 factors for ACC000456\"\n"
            "  • \"Churn probability for ACC000789\"\n"
            "  • \"Show top 5 accounts most likely to churn\"\n"
            "  • \"Which features are most important?\"\n"
            "  • \"Portfolio summary\" / \"Model performance\"\n"
            "  • /help for all commands"
        ))

    # ------------------------------------------------------------------
    # Response builders
    # ------------------------------------------------------------------

    def _explain(self, account_id: str, top_n: int) -> dict[str, Any]:
        """Request a SHAP-based churn explanation for a single account from the model agent.

        Args:
            account_id: The identifier of the account to explain.
            top_n: The number of top contributing features to include in the explanation.

        Returns:
            A dict containing the explanation text, contributing factors list, and churn probability.
        """
        resp = self.orch.send("model", "explain_account",
                              {"account_id": account_id, "top_n": top_n})
        if resp.get("status") == "error":
            return self._ok(response=f"Could not explain {account_id}: {resp['message']}")
        return self._ok(response=resp["text"], account_id=account_id,
                        factors=resp.get("factors", []),
                        probability=resp.get("probability", 0.0))

    def _probability(self, account_id: str) -> dict[str, Any]:
        """Look up the predicted churn probability for an account from the current state predictions.

        Args:
            account_id: The identifier of the account to look up.

        Returns:
            A dict containing a formatted probability string and the raw probability value.
        """
        preds = self.orch.state.predictions
        if preds is None or account_id not in preds.index:
            return self._ok(response=f"Account {account_id} not found.")
        prob = float(preds.loc[account_id, "churn_probability"])
        risk = str(preds.loc[account_id, "risk_level"]).upper()
        return self._ok(
            response=f"Account {account_id} — Churn probability: {prob:.1%} [{risk} RISK]",
            account_id=account_id, probability=round(prob, 4),
        )

    def _top_risk(self, n: int) -> dict[str, Any]:
        """Query the analytics agent for the N accounts with the highest churn probability.

        Args:
            n: The number of highest-risk accounts to retrieve.

        Returns:
            A dict containing a formatted ranked list of the top-N at-risk accounts.
        """
        resp = self.orch.send("analytics", "top_risk", {"n": n})
        if resp.get("status") == "error":
            return self._ok(response=resp["message"])
        accounts = resp["accounts"]
        lines = [f"Top {len(accounts)} highest churn-risk accounts:\n"]
        for i, a in enumerate(accounts, 1):
            tier = a.get("subscription_tier") or "—"
            mrr = a.get("mrr")
            mrr_s = f"  ${mrr:,.0f}/mo" if isinstance(mrr, (int, float)) else ""
            lines.append(
                f"  {i:>2}. {a['account_id']}  {a['churn_probability']:.1%}"
                f"  [{a['risk_level'].upper()}]  {tier}{mrr_s}"
            )
        return self._ok(response="\n".join(lines))

    def _lowest_risk(self, n: int) -> dict[str, Any]:
        """Sort predictions by churn probability ascending and return the N accounts least likely to churn.

        Args:
            n: The number of lowest-risk accounts to return.

        Returns:
            A dict containing a formatted list of the N accounts with the lowest churn probability.
        """
        preds = self.orch.state.predictions
        if preds is None:
            return self._ok(response="No predictions available.")
        df = preds.sort_values("churn_probability", ascending=True).head(n)
        if n == 1:
            row = df.iloc[0]
            return self._ok(response=(
                f"Account with the lowest churn probability:\n"
                f"  {df.index[0]}  —  {row['churn_probability']:.1%} [{row['risk_level'].upper()}]"
            ))
        lines = [f"{n} accounts with lowest churn probability:\n"]
        for i, (acc_id, row) in enumerate(df.iterrows(), 1):
            lines.append(f"  {i:>2}. {acc_id}  {row['churn_probability']:.1%}  [{row['risk_level'].upper()}]")
        return self._ok(response="\n".join(lines))

    def _accounts_by_filter(self, tier: str | None = None,
                            region: str | None = None, n: int = 10) -> dict[str, Any]:
        """Filter predictions by subscription tier or region and return the top-N by churn probability.

        Args:
            tier: Subscription tier to filter by (e.g. 'enterprise'), or None to skip tier filtering.
            region: Region code to filter by (e.g. 'APAC'), or None to skip region filtering.
            n: Maximum number of results to return.

        Returns:
            A dict containing a formatted ranked list of matching accounts sorted by churn probability.
        """
        preds = self.orch.state.predictions
        accounts = self.orch.state.frames.get("accounts")
        if preds is None or accounts is None:
            return self._ok(response="Data not available.")
        df = preds.join(accounts.set_index("account_id")[
            [c for c in ["subscription_tier", "region", "mrr"] if c in accounts.columns]
        ], how="left")
        label = tier or region
        if tier and "subscription_tier" in df.columns:
            df = df[df["subscription_tier"].str.lower() == tier]
        if region and "region" in df.columns:
            df = df[df["region"].str.upper() == region]
        df = df.sort_values("churn_probability", ascending=False).head(n)
        if df.empty:
            return self._ok(response=f"No accounts found for {label}.")
        lines = [f"Top {len(df)} {label} accounts by churn risk:\n"]
        for i, (acc_id, row) in enumerate(df.iterrows(), 1):
            mrr = row.get("mrr")
            mrr_s = f"  ${mrr:,.0f}" if isinstance(mrr, (int, float)) else ""
            lines.append(f"  {i:>2}. {acc_id}  {row['churn_probability']:.1%}"
                         f"  [{row['risk_level'].upper()}]{mrr_s}")
        return self._ok(response="\n".join(lines))

    def _breakdown(self, field: str) -> dict[str, Any]:
        """Aggregate churn risk statistics grouped by a categorical field such as tier or region.

        Args:
            field: The column name to group by (e.g. 'subscription_tier' or 'region').

        Returns:
            A dict containing a formatted table of per-group account counts, average churn probability, and high-risk counts.
        """
        preds = self.orch.state.predictions
        accounts = self.orch.state.frames.get("accounts")
        if preds is None or accounts is None or field not in accounts.columns:
            return self._ok(response=f"Breakdown by {field} not available.")
        df = preds.join(accounts.set_index("account_id")[[field]], how="left")
        rows = []
        for val, gdf in df.groupby(field):
            rows.append({
                "group": str(val),
                "count": len(gdf),
                "avg_prob": gdf["churn_probability"].mean(),
                "high_risk": (gdf["churn_probability"] >= 0.70).sum(),
            })
        rows.sort(key=lambda x: x["avg_prob"], reverse=True)
        lines = [f"Churn risk by {field.replace('_', ' ')}:\n",
                 f"  {'Group':<18} {'Accounts':>8}  {'Avg Churn':>10}  {'High Risk':>10}"]
        for r in rows:
            lines.append(f"  {r['group']:<18} {r['count']:>8}  {r['avg_prob']:>9.1%}  {r['high_risk']:>10}")
        return self._ok(response="\n".join(lines))

    def _mrr_at_risk(self) -> dict[str, Any]:
        """Calculate total MRR broken down by high, medium, and low churn-risk tiers.

        Returns:
            A dict containing a formatted summary of portfolio MRR segmented by risk level.
        """
        preds = self.orch.state.predictions
        accounts = self.orch.state.frames.get("accounts")
        if preds is None or accounts is None or "mrr" not in accounts.columns:
            return self._ok(response="MRR data not available.")
        df = preds.join(accounts.set_index("account_id")[["mrr"]], how="left")
        total = df["mrr"].dropna().sum()
        high = df[df["churn_probability"] >= 0.70]["mrr"].dropna().sum()
        at_risk = df[df["churn_probability"] >= 0.20]["mrr"].dropna().sum()
        return self._ok(response=(
            f"MRR at risk:\n"
            f"  Total portfolio MRR:    ${total:,.0f}/mo\n"
            f"  High risk MRR (≥70%):   ${high:,.0f}/mo  ({high/total*100:.1f}% of total)\n"
            f"  At-risk MRR (≥20%):     ${at_risk:,.0f}/mo  ({at_risk/total*100:.1f}% of total)\n"
            f"  Low risk MRR (<20%):    ${total-at_risk:,.0f}/mo"
        ))

    def _feature_importance(self) -> dict[str, Any]:
        """Fetch the top-N global SHAP feature importances from the model agent and format them as a numbered list.

        Returns:
            A dict containing a formatted numbered list of features and their mean absolute SHAP values.
        """
        top_n = self.orch.config["model"]["dashboard"].get("top_n_features", 10)
        resp = self.orch.send("model", "feature_importance", {"top_n": top_n})
        if resp.get("status") == "error":
            return self._ok(response=resp["message"])
        items = resp["importance"]
        lines = ["Top predictive features (by mean |SHAP contribution|):\n"]
        for i, item in enumerate(items, 1):
            label = item.get("label", item["feature"])
            lines.append(f"  {i:>2}. {label:<45} {item['importance']:.4f}")
        return self._ok(response="\n".join(lines), importance=items)

    def _model_performance(self) -> dict[str, Any]:
        """Read stored evaluation metrics from state and format AUC, recall, precision, F1, and CV scores.

        Returns:
            A dict containing a formatted string of model performance metrics from the held-out test set.
        """
        m = self.orch.state.metrics
        if not m:
            return self._ok(response="Model metrics not available yet.")
        lines = [
            "Model performance (held-out test set):\n",
            f"  Accuracy:  {m.get('accuracy', 0):.4f}",
            f"  ROC-AUC:   {m.get('roc_auc', 0):.4f}",
            f"  PR-AUC:    {m.get('pr_auc', 0):.4f}",
            f"  Recall:    {m.get('recall', 0):.4f}",
            f"  Precision: {m.get('precision', 0):.4f}",
            f"  F1:        {m.get('f1', 0):.4f}",
            f"  CV AUC:    {m.get('cv_auc_mean', 0):.4f} ± {m.get('cv_auc_std', 0):.4f}",
        ]
        return self._ok(response="\n".join(lines))

    def _summary(self) -> dict[str, Any]:
        """Query the analytics agent for portfolio-level totals and return a formatted overview.

        Returns:
            A dict containing a formatted overview of account counts by risk tier, average churn probability, and model AUC.
        """
        resp = self.orch.send("analytics", "summary")
        if resp.get("status") == "error":
            return self._ok(response=resp["message"])
        r = resp
        m = r.get("metrics", {})
        lines = [
            "Portfolio overview:\n",
            f"  Total accounts:    {r['total_accounts']:,}",
            f"  High risk (≥70%):  {r['high_risk_count']:,}",
            f"  Low risk (<20%):   {r['low_risk_count']:,}",
            f"  Avg churn prob:    {r['avg_churn_prob']:.1%}",
            f"  Max churn prob:    {r['max_churn_prob']:.1%}",
            f"  Model ROC-AUC:     {m.get('roc_auc', 0):.4f}",
        ]
        return self._ok(response="\n".join(lines))

    # ------------------------------------------------------------------
    # Slash commands
    # ------------------------------------------------------------------

    def _slash(self, text: str, tl: str) -> dict[str, Any]:
        """Handle slash commands including /help, /curveball, /retrain, and /reload.

        Args:
            text: The original command string preserving original casing.
            tl: The lower-cased command string used for prefix matching.

        Returns:
            A dict containing the command output, or an error message for unrecognised commands.
        """
        if tl.startswith("/help"):
            return self._ok(response=_HELP_TEXT)
        if tl.startswith("/curveball"):
            parts = tl.split()
            if len(parts) == 2 and parts[1] == "list":
                return self._ok(response=self.orch.send("curveball", "list").get("report", ""))
            return self._ok(response=self.orch.send("curveball", "run", {"command": text}).get("report", ""))
        if tl == "/retrain":
            self.orch.state.is_ready = False
            self.orch.send("data_pipeline", "rebuild_features")
            self.orch.send("model", "train_or_load", {"force_retrain": True})
            self.orch.send("model", "explain_all")
            self.orch._build_interpreter()
            self.orch.state.is_ready = True
            m = self.orch.state.metrics
            excls = self.orch.config["model"].get("feature_exclusions", [])
            excl_note = f"  (excl: {', '.join(excls)})" if excls else ""
            return self._ok(response=(
                f"Model retrained{excl_note}.\n"
                f"  ROC-AUC={m.get('roc_auc', 0):.4f}  "
                f"Recall={m.get('recall', 0):.4f}  F1={m.get('f1', 0):.4f}"
            ))
        if tl == "/reload":
            self.orch.state.is_ready = False
            self.orch.send("data_pipeline", "load_all")
            self.orch.send("model", "train_or_load", {"force_retrain": True})
            self.orch.send("model", "explain_all")
            self.orch._build_interpreter()
            self.orch.state.is_ready = True
            return self._ok(response="Data reloaded and model retrained.")
        return self._ok(response="Unknown command. Type /help for available commands.")


# ------------------------------------------------------------------
# Module-level helpers (imported by tests)
# ------------------------------------------------------------------

def _extract_account_id(text: str) -> str | None:
    """Search text for an account ID matching the ACC… prefix or UUID format.

    Args:
        text: The raw input string to search.

    Returns:
        The matched account ID string in upper case, or None if no match is found.
    """
    m = _ACCOUNT_RE.search(text)
    return m.group(1).upper() if m else None


def _extract_top_n(text: str) -> int | None:
    """Search text for a number indicating how many results the user wants.

    Args:
        text: The raw input string to search (e.g. "top 5 accounts", "show me 3").

    Returns:
        The extracted integer count, or None if no numeric quantity is found.
    """
    m = _TOP_N_RE.search(text)
    if not m:
        return None
    val = m.group(1) or m.group(2) or m.group(3)
    return int(val) if val else None


_HELP_TEXT = """Available queries:

  Account-specific:
    "Why is ACC000123 predicted to churn?"
    "Top 3 factors for ACC000456"
    "What's the churn probability for ACC000789?"
    "Explain ACC001234"

  Portfolio:
    "Show top 5 accounts most likely to churn"
    "Which account has the lowest churn probability?"
    "Enterprise accounts at high risk"
    "Churn breakdown by region / by tier"
    "What's our MRR at risk?"
    "Which features are most important overall?"
    "Model performance"
    "Portfolio summary"

  Slash commands:
    /help                    — this message
    /curveball list          — list test scenarios
    /curveball <name>        — run a scenario
    /retrain                 — force model retrain
    /reload                  — reload data + retrain"""
