"""CurveballAgent — introduces adversarial scenarios to test system resilience.

Each scenario modifies config, data, or schema in a controlled way, then
triggers a full pipeline reload and compares results. The goal is to verify
the system degrades gracefully rather than crashing.
"""
from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from agents.base_agent import BaseAgent
from curveballs.scenarios import SCENARIOS, Scenario

if TYPE_CHECKING:
    from agents.orchestrator import Orchestrator


class CurveballAgent(BaseAgent):
    def __init__(self, orchestrator: "Orchestrator"):
        """Store a reference to the orchestrator and initialise an empty scenario history list."""
        super().__init__("curveball")
        self.orch = orchestrator
        self._history: list[dict] = []

    def handle(self, msg_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Dispatch an incoming message to the appropriate curveball action handler.

        Args:
            msg_type: The action to perform; one of 'run', 'list', 'history', or 'reset'.
            payload: A dict of action parameters; 'run' messages may include a 'command' key.

        Returns:
            A response dict from the matched action handler, or an error dict for unknown types.
        """
        if msg_type == "run":
            return self._run(payload.get("command", ""))
        if msg_type == "list":
            return self._list()
        if msg_type == "history":
            return self._ok(history=self._history)
        if msg_type == "reset":
            return self._reset()
        return self._err(f"Unknown message type: {msg_type}")

    # ------------------------------------------------------------------

    def _run(self, command: str) -> dict[str, Any]:
        """Parse command and run the named (or random) scenario."""
        parts = command.strip().split()
        # parts: ['/curveball', 'scenario_name'] or ['/curveball', 'random']
        scenario_name = parts[1].lower() if len(parts) > 1 else "random"

        if scenario_name == "list":
            return self._list()

        if scenario_name == "random":
            scenario = random.choice(list(SCENARIOS.values()))
        elif scenario_name in SCENARIOS:
            scenario = SCENARIOS[scenario_name]
        else:
            available = ", ".join(SCENARIOS.keys())
            return self._ok(report=f"Unknown scenario '{scenario_name}'.\nAvailable: {available}")

        return self._execute(scenario)

    def _execute(self, scenario: Scenario) -> dict[str, Any]:
        """Snapshot state → apply scenario → rebuild features → retrain → compare → report → rollback."""
        self.logger.info("Running curveball: %s", scenario.name)

        # Snapshot pre-state
        pre_metrics = dict(self.orch.state.metrics)
        pre_features = len(self.orch.state.feature_names)
        pre_accounts = len(self.orch.state.predictions) if self.orch.state.predictions is not None else 0

        # Apply the scenario (modifies state/config in-place)
        apply_result = scenario.apply(self.orch)
        if not apply_result.get("ok", True):
            return self._ok(report=f"[CURVEBALL FAILED TO APPLY]\n{apply_result.get('reason','')}")

        # Rebuild features from the already-modified in-memory frames and retrain.
        # Using rebuild_features (not reload) so the disk read does not
        # overwrite the scenario's in-memory changes to state.frames.
        try:
            self.orch.send("data_pipeline", "rebuild_features")
            self.orch.send("model", "train_or_load", {"force_retrain": True})
            self.orch.send("model", "explain_all")
            self.orch.state.is_ready = True
            reload_ok = True
            error_msg = ""
        except Exception as exc:
            reload_ok = False
            error_msg = str(exc)

        # Compare post-state
        post_metrics = dict(self.orch.state.metrics)
        post_features = len(self.orch.state.feature_names)
        post_accounts = len(self.orch.state.predictions) if self.orch.state.predictions is not None else 0

        report = self._build_report(
            scenario, apply_result, reload_ok, error_msg,
            pre_metrics, post_metrics, pre_features, post_features,
            pre_accounts, post_accounts,
        )

        entry = {
            "scenario": scenario.name,
            "reload_ok": reload_ok,
            "pre_metrics": pre_metrics,
            "post_metrics": post_metrics,
            "report": report,
        }
        self._history.append(entry)

        # Roll back after the test so normal operation is restored.
        # force_retrain=True is required: the curveball's test run wrote a
        # scenario-trained model to the cache, so loading from cache would
        # restore the wrong model.  A fresh retrain on the original data
        # both restores correctness and writes the clean cache for next time.
        scenario.rollback(self.orch)
        self.orch.send("data_pipeline", "reload")
        self.orch.send("model", "train_or_load", {"force_retrain": True})
        self.orch.send("model", "explain_all")
        self.orch.state.is_ready = True

        return self._ok(report=report)

    def _list(self) -> dict[str, Any]:
        """Return a formatted list of all available curveball scenario names and their descriptions.

        Returns:
            A dict containing a 'report' string with every scenario name and description, plus usage instructions.
        """
        lines = ["Available curveball scenarios:\n"]
        for name, scenario in SCENARIOS.items():
            lines.append(f"  {name:<25} — {scenario.description}")
        lines.append("\nUsage: /curveball <name>  or  /curveball random")
        return self._ok(report="\n".join(lines))

    def _reset(self) -> dict[str, Any]:
        """Force a clean reload of the original data."""
        self.orch.send("data_pipeline", "reload")
        self.orch.send("model", "train_or_load", {"force_retrain": False})
        self.orch.send("model", "explain_all")
        self.orch.state.is_ready = True
        return self._ok(report="System reset to original data.")

    def _build_report(
        self, scenario, apply_result, reload_ok, error_msg,
        pre_m, post_m, pre_f, post_f, pre_a, post_a,
    ) -> str:
        """Format a structured text report comparing pre- and post-scenario pipeline metrics.

        Args:
            scenario: The Scenario object that was executed, providing name and description.
            apply_result: Dict returned by scenario.apply() indicating success and optional notes.
            reload_ok: Whether the pipeline reload succeeded without raising an exception.
            error_msg: Error string from a failed reload, or empty string if reload succeeded.
            pre_m: Metrics dict captured before the scenario was applied.
            post_m: Metrics dict captured after the scenario was applied.
            pre_f: Number of features before the scenario was applied.
            post_f: Number of features after the scenario was applied.
            pre_a: Number of accounts before the scenario was applied.
            post_a: Number of accounts after the scenario was applied.

        Returns:
            A multi-line string report summarising scenario impact on accounts, features, and model metrics.
        """
        lines = [
            f"╔══ CURVEBALL REPORT: {scenario.name} ══",
            f"║  Description: {scenario.description}",
            f"║  Applied:     {'✓ OK' if apply_result.get('ok', True) else '✗ FAILED'}",
            f"║  Reload:      {'✓ System survived' if reload_ok else f'✗ CRASHED: {error_msg}'}",
            "╠══ IMPACT ══",
            f"║  Accounts:    {pre_a} → {post_a}  (Δ {post_a - pre_a:+d})",
            f"║  Features:    {pre_f} → {post_f}  (Δ {post_f - pre_f:+d})",
        ]
        if pre_m and post_m:
            for key in ["roc_auc", "recall", "f1"]:
                pre_v = pre_m.get(key, 0)
                post_v = post_m.get(key, 0)
                delta = post_v - pre_v
                flag = "⚠" if abs(delta) > 0.05 else "✓"
                lines.append(f"║  {key.upper():<12} {pre_v:.4f} → {post_v:.4f}  (Δ {delta:+.4f}) {flag}")
        notes = apply_result.get("notes", "")
        if notes:
            lines.append(f"╠══ NOTES ══")
            for line in notes.splitlines():
                lines.append(f"║  {line}")
        lines.append("╚══ (System rolled back to original state) ══")
        return "\n".join(lines)
