"""Central orchestrator — owns shared state, routes messages to agents.

Shared state lives here so agents can be replaced/added without
changing the data contract between them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class SharedState:
    """Single source of truth for all pipeline artifacts.

    An instance of this dataclass is owned by the :class:`Orchestrator` and
    mutated in-place as each pipeline stage completes.  Agents read and write
    to this object via ``orchestrator.state`` rather than passing data between
    themselves directly.

    Attributes:
        frames: Raw DataFrames keyed by source name (e.g. ``"accounts"``,
                ``"events"``), populated by the data-pipeline agent.
        features: Engineered feature matrix with one row per account, produced
                  by the feature-engineering step.
        target: Binary churn label series aligned to ``features``.
        predictions: DataFrame of model outputs indexed by account ID,
                     containing at minimum ``churn_probability`` and
                     ``risk_level`` columns.
        shap_values: SHAP contribution array produced by the explainer, shaped
                     ``(n_accounts, n_features)``.
        model: The fitted scikit-learn (or compatible) estimator object.
        feature_names: Ordered list of column names that correspond to the
                       columns of ``features`` and to the SHAP value axes.
        metrics: Evaluation metrics dict (e.g. ``{"roc_auc": 0.87, ...}``).
        config: Full application config dict, also used as a side-channel for
                transient objects like the ``_interpreter`` instance.
        is_ready: ``True`` once the full pipeline has completed successfully.
        status_message: Human-readable description of the current pipeline
                        stage, surfaced in the TUI status bar.
    """
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)   # raw sources
    features: pd.DataFrame | None = None
    target: pd.Series | None = None
    predictions: pd.DataFrame | None = None
    shap_values: np.ndarray | None = None
    model: Any = None
    feature_names: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    y_test_true: np.ndarray | None = None
    y_test_scores: np.ndarray | None = None
    config: dict[str, Any] = field(default_factory=dict)
    is_ready: bool = False
    status_message: str = "Not initialized"


class Orchestrator:
    """Central message bus and shared-state container for the agent pipeline.

    The orchestrator holds the single :class:`SharedState` instance, maintains
    a registry of named :class:`~agents.base_agent.BaseAgent` objects, and
    provides methods for sending targeted or broadcast messages.  It also owns
    the high-level pipeline lifecycle (``initialize`` / ``reload``) and
    dispatches status updates to registered UI callbacks.
    """

    def __init__(self, config: dict[str, Any]):
        """Set up the orchestrator with a pipeline configuration.

        Args:
            config: Application-wide configuration dict, typically loaded from
                    YAML/JSON.  Must contain sub-keys expected by the various
                    agents (e.g. ``"schema"``, ``"model"``, ``"project_root"``).
        """
        self.config = config
        self.state = SharedState(config=config)
        self._agents: dict[str, "BaseAgent"] = {}
        self._callbacks: list[Any] = []   # UI refresh callbacks

    # ------------------------------------------------------------------
    # Agent registry
    # ------------------------------------------------------------------

    def register(self, agent: "BaseAgent") -> None:
        """Add an agent to the registry so it can receive messages.

        Args:
            agent: Any :class:`~agents.base_agent.BaseAgent` subclass.  The
                   agent is stored under ``agent.name`` and will be reachable
                   via :meth:`send` and :meth:`broadcast` using that key.

        Returns:
            None.
        """
        self._agents[agent.name] = agent
        logger.debug("Registered agent: %s", agent.name)

    def send(self, recipient: str, msg_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Route a single message to a named agent and return its response.

        Args:
            recipient: Registration name of the target agent (e.g.
                       ``"data_pipeline"``).
            msg_type:  Operation token forwarded verbatim to the agent's
                       :meth:`~agents.base_agent.BaseAgent.handle` method.
            payload:   Optional parameter dict for the operation.  Defaults to
                       an empty dict when ``None``.

        Returns:
            The dict returned by the agent's ``handle`` method, or an error
            dict with ``"status": "error"`` if the recipient is unknown or if
            the agent raises an unhandled exception.
        """
        if recipient not in self._agents:
            return {"status": "error", "message": f"Unknown agent: {recipient}"}
        try:
            return self._agents[recipient].handle(msg_type, payload or {})
        except Exception as exc:
            logger.exception("Agent '%s' raised during '%s'", recipient, msg_type)
            return {"status": "error", "message": str(exc)}

    def broadcast(self, msg_type: str, payload: dict[str, Any] | None = None) -> dict[str, dict]:
        """Send the same message to every registered agent and collect responses.

        Args:
            msg_type: Operation token forwarded to all agents.
            payload:  Optional parameter dict shared across all agents.
                      Defaults to an empty dict when ``None``.

        Returns:
            Dict mapping each agent's registration name to the response dict
            it returned from :meth:`send`.
        """
        return {name: self.send(name, msg_type, payload) for name in self._agents}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, force_retrain: bool = False) -> None:
        """Run the full pipeline: load → engineer → train → explain.

        Executes each pipeline stage in order, updating ``state.status_message``
        and notifying callbacks before each stage.  Raises ``RuntimeError`` if
        data loading or model training fails so callers can surface the error
        without swallowing it.

        Args:
            force_retrain: When ``True``, bypass any on-disk model cache and
                           always train from scratch.  When ``False`` (default),
                           a cached model is used if one exists.

        Returns:
            None.  All artefacts are written to ``self.state`` in-place.

        Raises:
            RuntimeError: If the data-pipeline or model-training stage returns
                          a non-ok status.
        """
        self.state.status_message = "Loading data…"
        self._notify()
        resp = self.send("data_pipeline", "load_all")
        if resp.get("status") != "ok":
            raise RuntimeError(f"Data loading failed: {resp.get('message')}")

        self.state.status_message = "Training model…"
        self._notify()
        resp = self.send("model", "train_or_load", {"force_retrain": force_retrain})
        if resp.get("status") != "ok":
            raise RuntimeError(f"Model training failed: {resp.get('message')}")

        self.state.status_message = "Generating explanations…"
        self._notify()
        self.send("model", "explain_all")

        self.state.status_message = "Building interpreter…"
        self._notify()
        self._build_interpreter()

        self.state.status_message = "Ready"
        self.state.is_ready = True
        self._notify()
        n_accounts = len(self.state.predictions) if self.state.predictions is not None else 0
        logger.info("Orchestrator initialized. Accounts: %d | Features: %d",
                    n_accounts, len(self.state.feature_names))

    def _build_interpreter(self) -> None:
        """Instantiate a :class:`~core.interpreter.Interpreter` and stash it in state.

        Imports the interpreter lazily so the module is not required at startup.
        On failure the warning is logged and the pipeline continues without an
        interpreter (explanation context lines will simply be omitted).

        Returns:
            None.  On success, ``self.state.config["_interpreter"]`` is set to
            the newly created :class:`~core.interpreter.Interpreter` instance.
        """
        try:
            from core.interpreter import Interpreter
            interp = Interpreter(self.state.features, self.state.predictions)
            self.state.config["_interpreter"] = interp
        except Exception as exc:
            logger.warning("Could not build interpreter: %s", exc)

    def reload(self, force_retrain: bool = True) -> None:
        """Re-run the full pipeline (used by curveball agent).

        Marks the state as not-ready and delegates to :meth:`initialize`.  By
        default forces a retrain so that newly injected data sources are
        reflected in the model.

        Args:
            force_retrain: Passed through to :meth:`initialize`.  Defaults to
                           ``True`` so the model is always retrained on reload.

        Returns:
            None.
        """
        self.state.is_ready = False
        self.initialize(force_retrain=force_retrain)

    # ------------------------------------------------------------------
    # Status callbacks (for TUI refresh)
    # ------------------------------------------------------------------

    def add_status_callback(self, fn) -> None:
        """Register a callable that is invoked whenever the status message changes.

        Args:
            fn: Any callable that accepts a single ``str`` argument (the new
                status message).  Typically a TUI widget refresh function.

        Returns:
            None.
        """
        self._callbacks.append(fn)

    def _notify(self) -> None:
        """Call every registered status callback with the current status message.

        Exceptions raised by individual callbacks are silently swallowed so
        that a broken UI callback cannot interrupt the pipeline.

        Returns:
            None.
        """
        for fn in self._callbacks:
            try:
                fn(self.state.status_message)
            except Exception:
                pass
