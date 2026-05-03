"""Standalone Model Performance window.

Opens as a tk.Toplevel with a metrics table and a 2×2 chart grid
(ROC, PR, confusion matrix, score distribution).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ui import plots as P

if TYPE_CHECKING:
    from agents.orchestrator import Orchestrator

BG      = "#1e1e2e"
PANEL   = "#2a2a3e"
SURFACE = "#313244"
TEXT    = "#cdd6f4"
BLUE    = "#89b4fa"
GREEN   = "#a6e3a1"
GREY    = "#6c7086"
SUBTEXT = "#a6adc8"

FONT_MONO  = ("Courier New", 13)
FONT_SANS  = ("Segoe UI", 14)
FONT_BOLD  = ("Segoe UI", 14, "bold")
FONT_TITLE = ("Segoe UI", 17, "bold")
FONT_SMALL = ("Segoe UI", 12)

_METRIC_META = [
    ("accuracy",    "Accuracy",      "% of all test accounts correctly classified"),
    ("precision",   "Precision",     "of flagged accounts, fraction that truly churn"),
    ("recall",      "Recall",        "fraction of actual churners that were caught"),
    ("f1",          "F1 Score",      "harmonic mean of precision and recall"),
    ("roc_auc",     "AUC (ROC)",     "probability model ranks churner above non-churner"),
    ("pr_auc",      "AUC (PR)",      "area under the precision-recall curve"),
    ("cv_auc_mean", "CV AUC (mean)", "cross-validated ROC-AUC across 5 folds"),
    ("cv_auc_std",  "CV AUC (±std)", "variation across folds — lower means more stable"),
]


def open_model_performance(orchestrator: "Orchestrator", parent: tk.Tk) -> None:
    """Create and raise the Model Performance window.

    Args:
        orchestrator: The running orchestrator whose ``state`` holds metrics and test data.
        parent: The main application Tk root, used as the Toplevel owner.
    """
    win = ModelPerformanceWindow(orchestrator, parent)
    win.show()


class ModelPerformanceWindow:
    """Tk.Toplevel window with detailed model diagnostics.

    Displays a metrics table and four diagnostic charts:
    ROC curve, Precision-Recall curve, Confusion Matrix,
    and Score Distribution by actual class.
    """

    def __init__(self, orch: "Orchestrator", parent: tk.Tk):
        """Initialise the window and build its contents immediately.

        Args:
            orch: The orchestrator instance whose ``state`` is read for metrics
                  and test-set predictions.
            parent: The owning ``tk.Tk`` root window.
        """
        self.orch = orch
        self.win = tk.Toplevel(parent)
        self.win.title("Model Performance — ChurnGuard Analytics")
        self.win.geometry("1120x820")
        self.win.configure(bg=BG)
        self.win.minsize(800, 600)
        self._apply_style()
        self._build()

    def show(self) -> None:
        """Bring the window to the front and give it keyboard focus."""
        self.win.lift()
        self.win.focus_set()

    # ── Style ─────────────────────────────────────────────────────────

    def _apply_style(self) -> None:
        s = ttk.Style(self.win)
        s.configure("Perf.TSeparator", background=GREY)

    # ── Layout ────────────────────────────────────────────────────────

    def _build(self) -> None:
        state = self.orch.state
        metrics = state.metrics
        y_true  = state.y_test_true
        y_scores = state.y_test_scores

        # Header bar
        header = tk.Frame(self.win, bg=PANEL)
        header.pack(fill=tk.X)
        tk.Label(header, text="Model Performance", font=FONT_TITLE,
                 bg=PANEL, fg=BLUE).pack(side=tk.LEFT, padx=12, pady=8)

        if not metrics:
            tk.Label(self.win, text="No model metrics available yet — run the pipeline first.",
                     bg=BG, fg=TEXT, font=FONT_SANS).pack(pady=40)
            return

        # Metrics table
        table_frame = tk.Frame(self.win, bg=BG)
        table_frame.pack(fill=tk.X, padx=14, pady=(10, 4))
        tk.Label(table_frame,
                 text="Evaluation Metrics  (held-out 20% test set, XGBoost threshold = 70%)",
                 font=FONT_BOLD, bg=BG, fg=BLUE).pack(anchor=tk.W, pady=(0, 6))
        self._build_metrics_table(table_frame, metrics)

        ttk.Separator(self.win, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=14, pady=6)

        # Charts
        if y_true is None or y_scores is None:
            tk.Label(
                self.win,
                text="Detailed charts require test-set predictions.\n"
                     "Type /retrain in the chat to regenerate them.",
                bg=BG, fg=SUBTEXT, font=FONT_SMALL, justify=tk.CENTER,
            ).pack(pady=30)
            return

        high_t = self.orch.config["model"]["model"].get("high_risk_threshold", 0.70)

        charts_frame = tk.Frame(self.win, bg=BG)
        charts_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        charts_frame.columnconfigure(0, weight=1)
        charts_frame.columnconfigure(1, weight=1)
        charts_frame.rowconfigure(0, weight=1)
        charts_frame.rowconfigure(1, weight=1)

        self._embed(P.roc_curve_chart(y_true, y_scores),                     charts_frame, 0, 0)
        self._embed(P.pr_curve_chart(y_true, y_scores),                      charts_frame, 0, 1)
        self._embed(P.confusion_matrix_chart(y_true, y_scores, high_t),      charts_frame, 1, 0)
        self._embed(P.score_distribution_chart(y_true, y_scores),            charts_frame, 1, 1)

    # ── Metrics table ─────────────────────────────────────────────────

    def _build_metrics_table(self, parent: tk.Frame, metrics: dict) -> None:
        """Render metrics as alternating-row table with value and plain-English description.

        Args:
            parent: The frame to pack table rows into.
            metrics: Dict of metric name → float value.
        """
        for i, (key, label, desc) in enumerate(_METRIC_META):
            val = metrics.get(key)
            if val is None:
                continue
            bg = SURFACE if i % 2 == 0 else BG
            row = tk.Frame(parent, bg=bg)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=f"  {label}", width=20, anchor=tk.W,
                     bg=bg, fg=BLUE, font=FONT_MONO).pack(side=tk.LEFT)
            tk.Label(row, text=f"  {val:.4f}", width=10, anchor=tk.W,
                     bg=bg, fg=TEXT, font=FONT_MONO).pack(side=tk.LEFT)
            tk.Label(row, text=f"  {desc}", anchor=tk.W,
                     bg=bg, fg=SUBTEXT, font=FONT_SMALL).pack(side=tk.LEFT, fill=tk.X)

    # ── Chart embedding ───────────────────────────────────────────────

    def _embed(self, fig_ax: tuple, parent: tk.Frame, row: int, col: int) -> None:
        """Place a matplotlib figure into the specified grid cell.

        Args:
            fig_ax: ``(Figure, Axes)`` tuple as returned by a ``plots.*`` function.
            parent: The grid container frame.
            row: Grid row index.
            col: Grid column index.
        """
        fig, _ = fig_ax
        cell = tk.Frame(parent, bg=BG)
        cell.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
        canvas = FigureCanvasTkAgg(fig, master=cell)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
