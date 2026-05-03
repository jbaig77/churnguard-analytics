"""Feature exclusion settings dialog.

Lets the user toggle individual features out of the training pipeline
(e.g. to remove synthetic leakage) and trigger a live retrain.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.orchestrator import Orchestrator

BG      = "#1e1e2e"
PANEL   = "#2a2a3e"
SURFACE = "#313244"
TEXT    = "#cdd6f4"
BLUE    = "#89b4fa"
RED     = "#f38ba8"
ORANGE  = "#fab387"
GREEN   = "#a6e3a1"
GREY    = "#6c7086"
SUBTEXT = "#a6adc8"
YELLOW  = "#f9e2af"

FONT_SANS  = ("Segoe UI", 14)
FONT_BOLD  = ("Segoe UI", 14, "bold")
FONT_TITLE = ("Segoe UI", 16, "bold")
FONT_SMALL = ("Segoe UI", 12)
FONT_MONO  = ("Courier New", 12)

# Features the user can toggle off, with explanations
_EXCLUDABLE = [
    (
        "account_health_score",
        "⚠  Synthetic leakage",
        RED,
        "Churned accounts score 0–40, active accounts score 60–100 — a hard engineered gap with "
        "zero overlap. This single feature explains why AUC = 1.00. Removing it forces the model "
        "to learn from genuine behavioural signals.",
    ),
    (
        "risk_flag",
        "⚠  Potentially post-hoc",
        ORANGE,
        "Set by the CS team during quarterly reviews. If the flag is applied after churn intent "
        "is already known, it leaks the label. Removing it gives a more honest signal.",
    ),
]


def open_feature_settings(
    orchestrator: "Orchestrator",
    parent: tk.Tk,
    on_apply: Callable,
) -> None:
    """Open the feature settings dialog.

    Args:
        orchestrator: The running orchestrator; its config is read and written.
        parent: Owning ``tk.Tk`` root window.
        on_apply: Callable invoked (with no arguments) after the user clicks
            "Apply & Retrain", so the app can trigger the pipeline reload.
    """
    FeatureSettingsDialog(orchestrator, parent, on_apply).show()


class FeatureSettingsDialog:
    """Modal-style Toplevel for toggling feature exclusions and triggering retrains."""

    def __init__(self, orch: "Orchestrator", parent: tk.Tk, on_apply: Callable):
        self.orch     = orch
        self.on_apply = on_apply

        self.win = tk.Toplevel(parent)
        self.win.title("Feature Exclusion Settings")
        self.win.geometry("680x480")
        self.win.configure(bg=BG)
        self.win.resizable(False, False)
        self.win.grab_set()   # modal
        self._vars: dict[str, tk.BooleanVar] = {}
        self._build()

    def show(self) -> None:
        self.win.lift()
        self.win.focus_set()

    # ── Layout ────────────────────────────────────────────────────────

    def _build(self) -> None:
        current_exclusions = set(
            self.orch.config["model"].get("feature_exclusions", [])
        )

        # Header
        hdr = tk.Frame(self.win, bg=PANEL)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="Feature Exclusion Settings", font=FONT_TITLE,
                 bg=PANEL, fg=BLUE).pack(side=tk.LEFT, padx=12, pady=8)

        # Explanation
        exp = tk.Frame(self.win, bg=BG)
        exp.pack(fill=tk.X, padx=14, pady=(8, 4))
        tk.Label(
            exp,
            text="Toggle features off to remove them from the training pipeline. "
                 "Clicking 'Apply & Retrain' rebuilds the feature matrix and retrains "
                 "the model from scratch — no restart needed.",
            bg=BG, fg=SUBTEXT, font=FONT_SMALL,
            wraplength=640, justify=tk.LEFT,
        ).pack(anchor=tk.W)

        ttk.Separator(self.win, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=14, pady=6)

        # Feature rows
        for feature, badge, badge_color, description in _EXCLUDABLE:
            var = tk.BooleanVar(value=(feature in current_exclusions))
            self._vars[feature] = var
            self._add_feature_row(feature, badge, badge_color, description, var)

        ttk.Separator(self.win, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=14, pady=8)

        # Status label (updated while retraining)
        self._status_var = tk.StringVar(value="")
        tk.Label(self.win, textvariable=self._status_var,
                 bg=BG, fg=YELLOW, font=FONT_SMALL).pack(padx=14, anchor=tk.W)

        # Buttons
        btn_frame = tk.Frame(self.win, bg=BG)
        btn_frame.pack(fill=tk.X, padx=14, pady=(4, 12))

        ttk.Button(btn_frame, text="Apply & Retrain",
                   command=self._apply).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_frame, text="Cancel",
                   command=self.win.destroy).pack(side=tk.LEFT)

        # Report button (only meaningful after a retrain)
        ttk.Button(btn_frame, text="📄 Generate Report",
                   command=self._generate_report).pack(side=tk.RIGHT)

    def _add_feature_row(self, feature: str, badge: str, badge_color: str,
                         description: str, var: tk.BooleanVar) -> None:
        row = tk.Frame(self.win, bg=SURFACE, bd=0)
        row.pack(fill=tk.X, padx=14, pady=3, ipady=6)

        # Checkbox + feature name
        left = tk.Frame(row, bg=SURFACE)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8)

        tk.Checkbutton(
            left, variable=var, bg=SURFACE,
            activebackground=SURFACE, selectcolor=PANEL,
        ).pack(side=tk.LEFT)
        tk.Label(left, text=feature, bg=SURFACE, fg=TEXT,
                 font=FONT_MONO).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(left, text=badge, bg=SURFACE, fg=badge_color,
                 font=FONT_SMALL).pack(side=tk.LEFT)

        # Description
        tk.Label(row, text=description, bg=SURFACE, fg=SUBTEXT,
                 font=FONT_SMALL, wraplength=420, justify=tk.LEFT,
                 anchor=tk.NW).pack(side=tk.LEFT, padx=8, pady=(2, 2))

    # ── Actions ───────────────────────────────────────────────────────

    def _apply(self) -> None:
        """Write the new exclusion list to config and invoke the retrain callback."""
        exclusions = [f for f, var in self._vars.items() if var.get()]
        self.orch.config["model"]["feature_exclusions"] = exclusions

        label = (
            f"Excluding: {', '.join(exclusions)}" if exclusions
            else "No exclusions — using all features"
        )
        self._status_var.set(f"Retraining in background… watch the status bar below.")
        self.win.update_idletasks()

        self.on_apply(exclusions)

    def _generate_report(self) -> None:
        """Generate a technical report reflecting current exclusion settings."""
        import threading
        self._status_var.set("Generating report…")
        self.win.update_idletasks()

        def _run():
            try:
                from generate_report import build_report
                exclusions = [f for f, var in self._vars.items() if var.get()]
                path = build_report(self.orch, excluded_features=exclusions)
                self._status_var.set(f"✓ Saved: {path}")
            except Exception as exc:
                self._status_var.set(f"Error: {exc}")

        threading.Thread(target=_run, daemon=True).start()
