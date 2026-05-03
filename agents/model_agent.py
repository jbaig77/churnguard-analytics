"""ModelAgent — owns training, prediction, and SHAP explanations."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from agents.base_agent import BaseAgent
from core.model import ChurnModel
from core.explainer import Explainer

if TYPE_CHECKING:
    from agents.orchestrator import Orchestrator


class ModelAgent(BaseAgent):
    """Agent responsible for model lifecycle: training, prediction, and explainability.

    Wraps a :class:`~core.model.ChurnModel` for training and inference and an
    :class:`~core.explainer.Explainer` for SHAP-based feature attributions.
    All inputs are read from and all outputs written to the orchestrator's
    shared state so no other agent needs to import this class directly.
    """

    def __init__(self, orchestrator: "Orchestrator"):
        """Initialise the model agent with a churn model and explainer.

        Args:
            orchestrator: The central :class:`~agents.orchestrator.Orchestrator`
                          instance.  Its ``config`` must contain ``"model"``
                          and ``"project_root"`` keys consumed by
                          :class:`~core.model.ChurnModel` and
                          :class:`~core.explainer.Explainer`.
        """
        super().__init__("model")
        self.orch = orchestrator
        cfg = orchestrator.config
        self.churn_model = ChurnModel(cfg["model"], cfg["project_root"])
        self.explainer = Explainer(cfg["model"], cfg["project_root"])

    def handle(self, msg_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Dispatch an inbound model message to the appropriate handler.

        Args:
            msg_type: One of:

                      * ``"train_or_load"`` — load from cache or train fresh.
                      * ``"train"`` — always train from scratch.
                      * ``"explain_all"`` — compute SHAP values for all accounts.
                      * ``"explain_account"`` — explain a single account.
                      * ``"predict"`` — generate predictions for all accounts.
                      * ``"feature_importance"`` — return global feature importance.

            payload:  Parameter dict whose relevant keys depend on
                      ``msg_type``:

                      * ``"train_or_load"`` — ``force_retrain`` (bool, default
                        ``False``): skip the cache when ``True``.
                      * ``"explain_account"`` — ``account_id`` (str): target
                        account; ``top_n`` (int, default 5): factors to include.
                      * ``"feature_importance"`` — ``top_n`` (int, default 10):
                        features to return.

        Returns:
            A response dict from the delegated handler, or an error dict for
            unknown message types.
        """
        if msg_type == "train_or_load":
            return self._train_or_load(payload.get("force_retrain", False))
        if msg_type == "train":
            return self._train()
        if msg_type == "explain_all":
            return self._explain_all()
        if msg_type == "explain_account":
            return self._explain_account(payload.get("account_id", ""), payload.get("top_n", 5))
        if msg_type == "predict":
            return self._predict_all()
        if msg_type == "feature_importance":
            return self._feature_importance(payload.get("top_n", 10))
        return self._err(f"Unknown message type: {msg_type}")

    # ------------------------------------------------------------------

    def _train_or_load(self, force_retrain: bool) -> dict[str, Any]:
        """Load a cached model if available, otherwise train a new one.

        When ``force_retrain`` is ``False`` and a valid on-disk cache exists,
        the cached model is restored and predictions are generated immediately
        without touching the training data.  Otherwise delegates to
        :meth:`_train`.

        Args:
            force_retrain: When ``True``, skip the cache check and always
                           call :meth:`_train`.

        Returns:
            Ok response with keys ``"source"`` (``"cache"`` or ``"trained"``)
            and ``"metrics"`` (dict of evaluation scores).
        """
        if not force_retrain and self.churn_model.load_cache():
            self.orch.state.model = self.churn_model.model
            self.orch.state.feature_names = self.churn_model.feature_names
            self.orch.state.metrics = self.churn_model.metrics
            # Re-align state.features to the cached model's column set so the
            # explainer and predictions always see exactly what the model expects,
            # even if state.features was rebuilt from disk with a different exclusion list.
            if self.orch.state.features is not None and self.churn_model.feature_names:
                self.orch.state.features = self.orch.state.features.reindex(
                    columns=self.churn_model.feature_names, fill_value=0
                )
            self._predict_all()
            return self._ok(source="cache", metrics=self.churn_model.metrics)
        return self._train()

    def _train(self) -> dict[str, Any]:
        """Train the churn model on the current feature matrix and target series.

        Reads ``features`` and ``target`` from shared state, fits the
        :class:`~core.model.ChurnModel`, persists results back to shared
        state, and immediately generates predictions for all accounts.

        Returns:
            Ok response with keys:

            * ``"source"`` (str): always ``"trained"``.
            * ``"metrics"`` (dict): evaluation metrics produced during training
              (e.g. ROC-AUC, precision, recall).

            Returns an error response if ``features`` or ``target`` are absent
            from shared state, or if training raises an exception.
        """
        features = self.orch.state.features
        target = self.orch.state.target
        if features is None or target is None:
            return self._err("Features not available. Run data_pipeline first.")
        try:
            metrics = self.churn_model.train(features, target)
            self.orch.state.model = self.churn_model.model
            self.orch.state.feature_names = self.churn_model.feature_names
            self.orch.state.metrics = metrics
            self.orch.state.y_test_true = self.churn_model.y_test_true
            self.orch.state.y_test_scores = self.churn_model.y_test_scores
            self._predict_all()
            return self._ok(source="trained", metrics=metrics)
        except Exception as exc:
            return self._err(str(exc))

    def _predict_all(self) -> dict[str, Any]:
        """Generate churn predictions for every account in the feature matrix.

        Calls :meth:`~core.model.ChurnModel.predict` and stores the resulting
        DataFrame in ``orchestrator.state.predictions``.

        Returns:
            Ok response with key ``"accounts"`` (int): number of accounts
            scored.

            Returns an error response if the feature matrix is absent from
            shared state or if prediction raises an exception.
        """
        features = self.orch.state.features
        if features is None:
            return self._err("No features available.")
        try:
            preds = self.churn_model.predict(features)
            self.orch.state.predictions = preds
            return self._ok(accounts=len(preds))
        except Exception as exc:
            return self._err(str(exc))

    def _explain_all(self) -> dict[str, Any]:
        """Compute SHAP contribution values for all accounts and store them in state.

        Fits the :class:`~core.explainer.Explainer` against the trained model
        and the full feature matrix.  The resulting contribution array is
        written to ``orchestrator.state.shap_values`` for downstream use.

        Returns:
            Ok response with no additional keys on success.

            Returns an error response if the model or feature matrix is absent,
            or if the explainer raises an exception (logged at EXCEPTION level).
        """
        features = self.orch.state.features
        model = self.churn_model.model
        if features is None or model is None:
            return self._err("Model/features not available.")
        try:
            # Always align to the model's training columns before computing SHAP —
            # state.features may have more or fewer columns than the model expects
            # when exclusion settings changed between the load and the explain call.
            feat_cols = self.churn_model.feature_names
            if feat_cols:
                features = features.reindex(columns=feat_cols, fill_value=0)
            self.explainer.fit(model, features)
            self.orch.state.shap_values = self.explainer._contribs
            return self._ok()
        except Exception as exc:
            self.logger.exception("explain_all failed")
            return self._err(str(exc))

    def _explain_account(self, account_id: str, top_n: int = 5) -> dict[str, Any]:
        """Produce a human-readable SHAP explanation for a single account.

        Retrieves the predicted churn probability, fetches the top ``top_n``
        SHAP factors from the explainer, and formats them into a multi-line
        text block enriched with optional interpreter context lines.

        Args:
            account_id: Index value identifying the account in the predictions
                        DataFrame.
            top_n: Number of top contributing factors to include in the
                   explanation.  Defaults to 5.

        Returns:
            Ok response with keys:

            * ``"text"`` (str): formatted multi-line explanation string.
            * ``"factors"`` (list[dict]): raw factor dicts from the explainer,
              each containing ``"feature"``, ``"label"``, ``"direction"``, and
              ``"feature_value"``.
            * ``"probability"`` (float): raw churn probability for this account.

            Returns an error response if the account is not found in
            predictions or if SHAP values have not been computed yet.
        """
        preds = self.orch.state.predictions
        if preds is None or account_id not in preds.index:
            return self._err(f"Account '{account_id}' not found in predictions.")
        if self.explainer._contribs is None:
            return self._err("Explainer not ready — try /retrain.")
        prob = float(preds.loc[account_id, "churn_probability"])
        factors_result = self.explainer.explain_account(account_id, top_n=top_n)
        factors = factors_result.get("factors", [])

        # Build rich text with interpreter context lines
        interp = self.orch.state.config.get("_interpreter")
        text = self._build_explanation_text(account_id, prob, factors, interp)
        return self._ok(text=text, factors=factors, probability=prob)

    def _build_explanation_text(self, account_id: str, prob: float,
                                factors: list, interp) -> str:
        """Format an account's churn probability and SHAP factors as a display string.

        Determines the risk band from ``prob``, appends an optional
        plain-English probability interpretation line from the interpreter,
        then lists each factor with a direction arrow and optional contextual
        benchmark line.

        Args:
            account_id: Account identifier printed in the header line.
            prob:       Predicted churn probability in ``[0, 1]``.
            factors:    List of factor dicts from the explainer, each expected
                        to contain ``"direction"`` (``"increases"`` /
                        ``"decreases"``), ``"label"`` (str), ``"feature"``
                        (str), and ``"feature_value"`` (numeric or ``None``).
            interp:     Optional :class:`~core.interpreter.Interpreter`
                        instance used to add narrative context lines; pass
                        ``None`` to skip contextual enrichment.

        Returns:
            Multi-line string suitable for display in a terminal or TUI panel,
            with a header block showing the account ID, churn probability, and
            risk band, followed by a numbered list of contributing factors.
        """
        risk = "HIGH" if prob >= 0.70 else ("MEDIUM" if prob >= 0.40 else "LOW")
        lines = [
            f"Account {account_id}",
            f"Churn probability: {prob:.1%}  [{risk} RISK]",
        ]
        if interp:
            lines.append(interp.explain_probability(prob))
        lines.append("")
        lines.append(f"Top {len(factors)} contributing factors:")
        for i, f in enumerate(factors, 1):
            arrow = "▲" if f["direction"] == "increases" else "▼"
            fv = f["feature_value"]
            val_str = f" = {fv}" if fv is not None else ""
            lines.append(f"  {i}. {arrow} {f['label']}{val_str}")
            if interp and fv is not None:
                ctx = interp.context_line(f["feature"], fv)
                if ctx:
                    lines.append(ctx)
        return "\n".join(lines)

    def _feature_importance(self, top_n: int) -> dict[str, Any]:
        """Return a ranked list of the most important features globally.

        Prefers SHAP-based global importance from the explainer; falls back to
        the model's built-in feature importance (e.g. Gini impurity) if no
        SHAP values are available.

        Args:
            top_n: Maximum number of features to return, ordered by descending
                   importance.  Defaults to 10 when called via :meth:`handle`.

        Returns:
            Ok response with key ``"importance"`` containing a list of record
            dicts, each with at minimum ``"feature"`` (str) and ``"importance"``
            (float) keys.
        """
        df = self.explainer.global_importance(top_n=top_n)
        if df.empty:
            df = self.churn_model.feature_importance(top_n=top_n)
        return self._ok(importance=df.to_dict("records"))
