"""Data Diagnostics window — feature leakage detection and class-separability audit.

Opens as a tk.Toplevel with three tabs:
  1. Leakage Table   — every feature ranked by |r| with the churn label, leaky ones flagged red
  2. Correlation     — horizontal bar chart of top-25 features by |r|
  3. Distributions   — 3×2 histogram grid for the top-6 highest-correlation features
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ui import plots as P

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

FONT_MONO  = ("Courier New", 13)
FONT_SANS  = ("Segoe UI", 14)
FONT_BOLD  = ("Segoe UI", 14, "bold")
FONT_TITLE = ("Segoe UI", 17, "bold")
FONT_SMALL = ("Segoe UI", 12)
FONT_TREE  = ("Segoe UI", 12)


def open_diagnostics(orchestrator: "Orchestrator", parent: tk.Tk) -> None:
    """Create and raise the Data Diagnostics window.

    Args:
        orchestrator: The running orchestrator whose ``state`` holds the
            feature matrix and target series.
        parent: The owning ``tk.Tk`` root window.
    """
    win = DiagnosticsWindow(orchestrator, parent)
    win.show()


class DiagnosticsWindow:
    """Comprehensive feature leakage and class-separability audit window.

    Scans every engineered feature for correlation with the churn label
    (point-biserial r = Pearson r with binary target) and flags features
    whose churned and active distributions have zero overlap.
    """

    def __init__(self, orch: "Orchestrator", parent: tk.Tk):
        """Build the window immediately on construction.

        Args:
            orch: Orchestrator whose ``state.features`` and ``state.target``
                  are used for the analysis.
            parent: Owning ``tk.Tk`` root window.
        """
        self.orch = orch
        self.win = tk.Toplevel(parent)
        self.win.title("Data Diagnostics — Feature Leakage Audit")
        self.win.geometry("1200x860")
        self.win.configure(bg=BG)
        self.win.minsize(900, 640)
        self._apply_style()
        self._build()

    def show(self) -> None:
        """Bring the window to the front and give it keyboard focus."""
        self.win.lift()
        self.win.focus_set()

    # ── Style ─────────────────────────────────────────────────────────

    def _apply_style(self) -> None:
        s = ttk.Style(self.win)
        s.configure("Diag.Treeview",
                    background=PANEL, fieldbackground=PANEL,
                    foreground=TEXT, rowheight=26, font=FONT_TREE)
        s.configure("Diag.Treeview.Heading",
                    background=SURFACE, foreground=BLUE,
                    font=("Segoe UI", 12, "bold"))
        s.map("Diag.Treeview",
              background=[("selected", BLUE)], foreground=[("selected", BG)])
        s.configure("Diag.TNotebook",     background=PANEL)
        s.configure("Diag.TNotebook.Tab", background=SURFACE, foreground=SUBTEXT,
                    padding=[10, 4], font=FONT_SMALL)
        s.map("Diag.TNotebook.Tab",
              background=[("selected", BLUE)], foreground=[("selected", BG)])

    # ── Layout ────────────────────────────────────────────────────────

    def _build(self) -> None:
        state = self.orch.state
        if state.features is None or state.target is None:
            tk.Label(self.win, text="No feature data available — run the pipeline first.",
                     bg=BG, fg=TEXT, font=FONT_SANS).pack(pady=40)
            return

        analysis = self._run_analysis(state.features, state.target)
        leaky    = analysis[analysis["leaky"]]
        n_total  = len(analysis)
        n_leaky  = len(leaky)

        # ── Header ────────────────────────────────────────────────────
        header = tk.Frame(self.win, bg=PANEL)
        header.pack(fill=tk.X)
        tk.Label(header, text="Data Diagnostics — Feature Leakage Audit",
                 font=FONT_TITLE, bg=PANEL, fg=BLUE).pack(side=tk.LEFT, padx=12, pady=8)

        # ── Summary banner ────────────────────────────────────────────
        banner_bg = "#3a1e2e" if n_leaky > 0 else "#1e3a2e"
        banner_fg = RED      if n_leaky > 0 else GREEN
        banner = tk.Frame(self.win, bg=banner_bg)
        banner.pack(fill=tk.X, padx=0)

        icon = "⚠" if n_leaky > 0 else "✓"
        msg  = (
            f"  {icon}  {n_leaky} of {n_total} features have ZERO OVERLAP between "
            f"churned and active distributions — these single-handedly explain AUC ≈ 1.0.  "
            f"Additionally, {len(analysis[analysis['abs_r'] >= 0.70]) - n_leaky} features "
            f"have |r| ≥ 0.70 with the churn label (near-deterministic in the synthetic data).  "
            if n_leaky > 0 else
            f"  {icon}  No strictly zero-overlap features detected.  "
            f"{len(analysis[analysis['abs_r'] >= 0.50])} features have |r| ≥ 0.50."
        )
        tk.Label(banner, text=msg, bg=banner_bg, fg=banner_fg,
                 font=FONT_SANS, wraplength=1170, justify=tk.LEFT,
                 padx=8, pady=6).pack(anchor=tk.W)

        # ── Explanation text ──────────────────────────────────────────
        exp_frame = tk.Frame(self.win, bg=BG)
        exp_frame.pack(fill=tk.X, padx=14, pady=(6, 2))
        explanation = (
            "Point-biserial r = Pearson correlation between each feature and the binary churn label (0/1).  "
            "|r| > 0.8 = extremely strong.  "
            "Zero overlap means the feature's range for churned accounts does not intersect its range for active accounts — "
            "any threshold classifier would perfectly separate the classes on that feature alone.\n"
            "Note: features like cancellation_requested_count where active accounts always = 0 are also near-deterministic "
            "even though they technically 'overlap' at zero."
        )
        tk.Label(exp_frame, text=explanation, bg=BG, fg=SUBTEXT, font=FONT_SMALL,
                 wraplength=1160, justify=tk.LEFT).pack(anchor=tk.W)

        ttk.Separator(self.win, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=14, pady=4)

        # ── Notebook ──────────────────────────────────────────────────
        nb = ttk.Notebook(self.win, style="Diag.TNotebook")
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))

        t1 = tk.Frame(nb, bg=BG)
        t2 = tk.Frame(nb, bg=BG)
        t3 = tk.Frame(nb, bg=BG)
        nb.add(t1, text="  Leakage Table  ")
        nb.add(t2, text="  Correlation Chart  ")
        nb.add(t3, text="  Top Feature Distributions  ")

        self._build_table(t1, analysis)
        self._build_correlation_chart(t2, analysis)
        self._build_distribution_grid(t3, state.features, state.target, analysis)

    # ── Tab 1: Leakage table ──────────────────────────────────────────

    def _build_table(self, parent: tk.Frame, analysis: pd.DataFrame) -> None:
        """Treeview showing all features ranked by |r|, leaky rows in red.

        Args:
            parent: Container frame.
            analysis: Output of :meth:`_run_analysis`.
        """
        # Legend
        legend = tk.Frame(parent, bg=BG)
        legend.pack(fill=tk.X, padx=8, pady=(4, 2))
        tk.Label(legend, text="⚠ Red = zero overlap between churned/active distributions  |  "
                              "Sorted by |correlation| with churn label (descending)",
                 bg=BG, fg=SUBTEXT, font=FONT_SMALL).pack(anchor=tk.W)

        cols = ("feature", "abs_r", "r", "mean_ch", "mean_ac",
                "max_ch", "min_ac", "overlap", "flag")
        tree = ttk.Treeview(parent, columns=cols, show="headings",
                            selectmode="browse", style="Diag.Treeview")

        for col, label, width, anchor in [
            ("feature",  "Feature",         200, tk.W),
            ("abs_r",    "|r|",              58, tk.CENTER),
            ("r",        "r (signed)",       78, tk.CENTER),
            ("mean_ch",  "Mean (churned)",  110, tk.CENTER),
            ("mean_ac",  "Mean (active)",   110, tk.CENTER),
            ("max_ch",   "Max (churned)",   100, tk.CENTER),
            ("min_ac",   "Min (active)",    100, tk.CENTER),
            ("overlap",  "Overlap?",         70, tk.CENTER),
            ("flag",     "Flag",             60, tk.CENTER),
        ]:
            tree.heading(col, text=label)
            tree.column(col, width=width, anchor=anchor, minwidth=40)

        tree.tag_configure("leaky",  foreground=RED)
        tree.tag_configure("high",   foreground=ORANGE)
        tree.tag_configure("normal", foreground=TEXT)

        for _, row in analysis.iterrows():
            flag = "⚠ LEAKY" if row["leaky"] else (
                   "⚠ HIGH"  if row["abs_r"] >= 0.50 else "")
            tag = "leaky" if row["leaky"] else (
                  "high"  if row["abs_r"] >= 0.50 else "normal")
            overlap_str = "NO ⚠" if not row["has_overlap"] else "yes"
            tree.insert("", tk.END, tags=(tag,), values=(
                row["feature"],
                f"{row['abs_r']:.4f}",
                f"{row['r']:+.4f}",
                f"{row['mean_ch']:.3f}",
                f"{row['mean_ac']:.3f}",
                f"{row['max_ch']:.3f}",
                f"{row['min_ac']:.3f}",
                overlap_str,
                flag,
            ))

        vsb = ttk.Scrollbar(parent, orient=tk.VERTICAL,   command=tree.yview)
        hsb = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side=tk.RIGHT,  fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        tree.pack(fill=tk.BOTH, expand=True, padx=(8, 0), pady=(0, 0))

    # ── Tab 2: Correlation chart ──────────────────────────────────────

    def _build_correlation_chart(self, parent: tk.Frame, analysis: pd.DataFrame) -> None:
        fig, _ = P.leakage_correlation_chart(analysis)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    # ── Tab 3: Distribution grid ──────────────────────────────────────

    def _build_distribution_grid(self, parent: tk.Frame,
                                  features: pd.DataFrame,
                                  target: pd.Series,
                                  analysis: pd.DataFrame) -> None:
        top6 = analysis.head(6)["feature"].tolist()
        tk.Label(parent,
                 text=("Top 6 features by |r|.  Red = churned accounts, green = active.  "
                       "Non-overlapping distributions indicate the model is being handed the answer."),
                 bg=BG, fg=SUBTEXT, font=FONT_SMALL,
                 wraplength=1160, justify=tk.LEFT).pack(anchor=tk.W, padx=8, pady=(4, 2))

        fig, _ = P.leaky_feature_grid(features, target, top6)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    # ── Analysis ──────────────────────────────────────────────────────

    @staticmethod
    def _run_analysis(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Compute point-biserial correlation and overlap stats for every feature.

        Args:
            features: Engineered feature matrix indexed by account ID.
            target: Binary churn label Series aligned to ``features``.

        Returns:
            DataFrame with one row per feature, sorted descending by
            ``abs_r``, containing columns: ``feature``, ``r``, ``abs_r``,
            ``mean_ch``, ``mean_ac``, ``max_ch``, ``min_ac``,
            ``has_overlap``, ``leaky``.
        """
        records = []
        for col in features.columns:
            series = features[col].dropna()
            common = series.index.intersection(target.index)
            if len(common) < 10:
                continue
            x = series.loc[common].values.astype(float)
            y = target.loc[common].values.astype(float)

            r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-8 else 0.0

            x_ch = x[y == 1]
            x_ac = x[y == 0]
            has_overlap = True
            if len(x_ch) > 0 and len(x_ac) > 0:
                has_overlap = not (x_ch.max() < x_ac.min() or x_ac.max() < x_ch.min())

            records.append({
                "feature":     col,
                "r":           round(r, 4),
                "abs_r":       round(abs(r), 4),
                "mean_ch":     round(float(x_ch.mean()), 4) if len(x_ch) > 0 else float("nan"),
                "mean_ac":     round(float(x_ac.mean()), 4) if len(x_ac) > 0 else float("nan"),
                "max_ch":      round(float(x_ch.max()),  4) if len(x_ch) > 0 else float("nan"),
                "min_ac":      round(float(x_ac.min()),  4) if len(x_ac) > 0 else float("nan"),
                "has_overlap": has_overlap,
                "leaky":       not has_overlap,
            })

        return (pd.DataFrame(records)
                .sort_values("abs_r", ascending=False)
                .reset_index(drop=True))
