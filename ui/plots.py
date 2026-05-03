"""Matplotlib chart factories.

Uses matplotlib.figure.Figure (not plt.subplots) so figures can be
created safely in background threads. The UI embeds them via
FigureCanvasTkAgg on the main thread.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

BG     = "#1e1e2e"
PANEL  = "#2a2a3e"
TEXT   = "#cdd6f4"
RED    = "#f38ba8"
ORANGE = "#fab387"
GREEN  = "#a6e3a1"
BLUE   = "#89b4fa"
GREY   = "#6c7086"

_RC = {
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": GREY, "axes.labelcolor": TEXT,
    "text.color": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    "grid.color": GREY, "grid.alpha": 0.3,
    "font.family": "monospace",
}


def _fig(w: float, h: float) -> tuple:
    """Create a themed Figure and Axes pair that is safe to call from any thread.

    Applies the module-level ``_RC`` style context (dark background,
    monospace font, muted grid) and wires up spine and tick colours so
    every chart produced by this module shares a consistent look.

    Args:
        w: Figure width in inches.
        h: Figure height in inches.

    Returns:
        A ``(Figure, Axes)`` tuple where the Figure uses
        ``matplotlib.figure.Figure`` (not ``plt.figure``) and the single
        Axes subplot has the panel background colour and themed spines.
    """
    import matplotlib as mpl
    with mpl.rc_context(_RC):
        fig = Figure(figsize=(w, h), facecolor=BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(GREY)
        ax.tick_params(colors=TEXT)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
    return fig, ax


def churn_distribution(predictions: pd.DataFrame) -> tuple:
    """Plot a histogram of churn probabilities colour-coded by risk tier.

    Bins are coloured blue (low, < 20 %), orange (watch zone, 20–70 %), or red
    (high, >= 70 %). Vertical dashed lines mark the low and high
    thresholds. An annotation in the top-right quadrant shows the count of
    high-risk accounts.

    Args:
        predictions: DataFrame that must contain a ``churn_probability``
            column of float values in [0, 1], one row per account.

    Returns:
        A ``(Figure, Axes)`` tuple containing the rendered histogram,
        ready to be embedded in the UI via ``FigureCanvasTkAgg``.
    """
    fig, ax = _fig(6, 3.5)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.15)

    probs = predictions["churn_probability"].values
    bins  = np.linspace(0, 1, 101)
    counts, edges = np.histogram(probs, bins=bins)
    colors = [RED if e >= 0.70 else (ORANGE if e >= 0.20 else BLUE) for e in edges[:-1]]

    ax.bar(edges[:-1], counts, width=(edges[1]-edges[0])*0.9,
           color=colors, align="edge", alpha=0.85)
    ax.axvline(0.20, color=ORANGE, linestyle="--", linewidth=1, alpha=0.8,
               label="Low threshold (20%)")
    ax.axvline(0.70, color=RED,    linestyle="--", linewidth=1, alpha=0.8,
               label="High threshold (70%)")

    ax.set_xlabel("Churn Probability  (0% = no risk,  100% = certain churn)", color=TEXT)
    ax.set_ylabel("Number of accounts", color=TEXT)
    ax.set_title("Churn Probability Distribution", color=TEXT, pad=8)
    ax.grid(axis="y")
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor=GREY, labelcolor=TEXT)

    high = int((probs >= 0.70).sum())
    total = len(probs)
    if counts.max() > 0:
        ax.annotate(f"{high} high-risk ({high/total:.0%})", xy=(0.85, counts.max() * 0.85),
                    color=RED, fontsize=8, ha="center")
    ax.annotate("Each bar = accounts in a 1% probability range",
                xy=(0.01, 0.97), xycoords="axes fraction",
                color=GREY, fontsize=7, va="top")
    return fig, ax


def feature_importance(importance_df: pd.DataFrame, top_n: int = 12) -> tuple:
    """Plot a horizontal bar chart of the top-N most important features by mean SHAP value.

    The top three bars are highlighted in blue; the remainder are rendered
    in grey. Each bar is labelled with its numeric importance value to four
    decimal places.

    Args:
        importance_df: DataFrame with at least a ``feature`` column (str)
            and an ``importance`` column (float, mean absolute SHAP value),
            sorted descending by importance. An optional ``label`` column
            may supply human-readable display names; if absent the raw
            ``feature`` values are used.
        top_n: Maximum number of features to display. Defaults to 12.

    Returns:
        A ``(Figure, Axes)`` tuple containing the rendered bar chart,
        ready to be embedded in the UI via ``FigureCanvasTkAgg``.
    """
    df = importance_df.head(top_n).copy()
    labels = df.get("label", df["feature"]).tolist()
    values = df["importance"].tolist()

    fig, ax = _fig(6, max(3.0, top_n * 0.38))
    fig.subplots_adjust(left=0.42, right=0.95, top=0.92, bottom=0.10)

    y = list(range(len(labels) - 1, -1, -1))
    bar_colors = [BLUE if i < 3 else GREY for i in range(len(labels))]
    bars = ax.barh(y, values, color=bar_colors, alpha=0.85, height=0.68)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean |SHAP value|  (longer bar = stronger predictor of churn across all accounts)", color=TEXT)
    ax.set_title("Global Feature Importance (SHAP)", color=TEXT, pad=8)
    ax.grid(axis="x")

    max_v = values[0] if values else 1
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max_v * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7, color=TEXT)
    return fig, ax


def account_factors(factors: list[dict], account_id: str, prob: float) -> tuple:
    """Plot a waterfall-style horizontal bar chart of per-feature SHAP contributions for a single account.

    Bars extending to the right (positive SHAP) are coloured red to
    indicate features that increase churn risk; bars extending to the left
    (negative SHAP) are coloured green to indicate features that decrease
    risk. If no factors are provided a placeholder message is shown
    instead.

    Args:
        factors: List of factor dicts, each containing:
            - ``"label"`` (str): Human-readable feature name. Labels
              longer than 35 characters are truncated.
            - ``"contribution"`` (float): SHAP value in log-odds space.
              Positive values increase predicted churn probability.
        account_id: The account identifier displayed in the chart title.
        prob: The model's predicted churn probability for this account,
            expressed as a float in [0, 1]. Used in the title and to
            derive the risk tier label (LOW / MEDIUM / HIGH).

    Returns:
        A ``(Figure, Axes)`` tuple containing the rendered chart, ready to
        be embedded in the UI via ``FigureCanvasTkAgg``.
    """
    if not factors:
        fig, ax = _fig(5, 3)
        ax.text(0.5, 0.5, "No factors available", ha="center", va="center",
                transform=ax.transAxes, color=TEXT)
        return fig, ax

    labels = [f["label"][:35] for f in factors]
    values = [f["contribution"] for f in factors]
    colors = [RED if v > 0 else GREEN for v in values]

    fig, ax = _fig(6, max(3.0, len(factors) * 0.5 + 1.5))
    fig.subplots_adjust(left=0.44, right=0.97, top=0.88, bottom=0.10)

    y = list(range(len(labels) - 1, -1, -1))
    ax.barh(y, values, color=colors, alpha=0.85, height=0.65)
    ax.axvline(0, color=GREY, linewidth=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("← reduces churn risk          SHAP contribution          raises churn risk →", color=TEXT)
    risk = "HIGH" if prob >= 0.70 else "LOW"
    ax.set_title(f"{account_id}  —  {prob:.1%} churn probability [{risk}]", color=TEXT, pad=8)
    ax.grid(axis="x")
    ax.annotate("Bar length = how strongly this feature pushes the prediction for this account",
                xy=(0.5, 0.01), xycoords="axes fraction",
                color=GREY, fontsize=7, ha="center", va="bottom")

    red_p   = mpatches.Patch(color=RED,   label="↑ Increases churn risk")
    green_p = mpatches.Patch(color=GREEN, label="↓ Decreases churn risk")
    ax.legend(handles=[red_p, green_p], fontsize=7, facecolor=PANEL,
              edgecolor=GREY, labelcolor=TEXT, loc="lower right")
    return fig, ax


def roc_curve_chart(y_true: np.ndarray, y_scores: np.ndarray) -> tuple:
    """Plot a ROC curve with AUC annotation and a random-classifier baseline.

    Args:
        y_true: Binary ground-truth labels (0 = retained, 1 = churned).
        y_scores: Predicted churn probabilities in [0, 1].

    Returns:
        A ``(Figure, Axes)`` tuple ready for embedding via ``FigureCanvasTkAgg``.
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    fig, ax = _fig(5, 4)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.14)
    ax.plot(fpr, tpr, color=BLUE, linewidth=2, label=f"Model (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate  (healthy accounts incorrectly flagged)", color=TEXT)
    ax.set_ylabel("True Positive Rate  (churned accounts correctly caught)", color=TEXT)
    ax.set_title("ROC Curve", color=TEXT, pad=8)
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GREY, labelcolor=TEXT)
    ax.grid(True)
    return fig, ax


def pr_curve_chart(y_true: np.ndarray, y_scores: np.ndarray) -> tuple:
    """Plot a Precision-Recall curve with average-precision annotation.

    Args:
        y_true: Binary ground-truth labels (0 = retained, 1 = churned).
        y_scores: Predicted churn probabilities in [0, 1].

    Returns:
        A ``(Figure, Axes)`` tuple ready for embedding via ``FigureCanvasTkAgg``.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    baseline = float(y_true.mean())

    fig, ax = _fig(5, 4)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.14)
    ax.plot(recall, precision, color=GREEN, linewidth=2, label=f"Model (AP = {ap:.4f})")
    ax.axhline(baseline, color=GREY, linestyle="--", linewidth=1,
               label=f"Random ({baseline:.2f})")
    ax.set_xlabel("Recall  (fraction of churned accounts caught)", color=TEXT)
    ax.set_ylabel("Precision  (of flagged accounts, fraction that truly churn)", color=TEXT)
    ax.set_title("Precision-Recall Curve", color=TEXT, pad=8)
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GREY, labelcolor=TEXT)
    ax.grid(True)
    return fig, ax


def confusion_matrix_chart(y_true: np.ndarray, y_scores: np.ndarray,
                           threshold: float = 0.70) -> tuple:
    """Plot a 2×2 confusion matrix heatmap for a given decision threshold.

    Args:
        y_true: Binary ground-truth labels (0 = retained, 1 = churned).
        y_scores: Predicted churn probabilities in [0, 1].
        threshold: Decision threshold applied to convert probabilities to labels.
                   Defaults to 0.70 (high-risk tier boundary).

    Returns:
        A ``(Figure, Axes)`` tuple ready for embedding via ``FigureCanvasTkAgg``.
    """
    from sklearn.metrics import confusion_matrix as sk_cm
    y_pred = (y_scores >= threshold).astype(int)
    cm = sk_cm(y_true, y_pred)

    fig, ax = _fig(4.5, 3.8)
    fig.subplots_adjust(left=0.18, right=0.95, top=0.85, bottom=0.18)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.2)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nRetained", "Predicted\nChurned"], color=TEXT)
    ax.set_yticklabels(["Actual\nRetained", "Actual\nChurned"], color=TEXT)
    ax.set_title(f"Confusion Matrix  (threshold = {threshold:.0%})", color=TEXT, pad=8)
    for i in range(2):
        for j in range(2):
            cell_color = "black" if cm[i, j] > cm.max() * 0.55 else TEXT
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color=cell_color, fontsize=15, fontweight="bold")
    return fig, ax


def leakage_correlation_chart(analysis_df: "pd.DataFrame") -> tuple:
    """Horizontal bar chart of feature correlations with the churn label.

    Bars are coloured red for zero-overlap (leaky) features and blue for
    normal features.  Vertical dashed lines mark |r| = 0.5 and |r| = 0.8.

    Args:
        analysis_df: DataFrame with columns ``"feature"``, ``"abs_r"``, and
            ``"leaky"`` (bool), sorted descending by ``abs_r``.

    Returns:
        A ``(Figure, Axes)`` tuple ready for embedding via ``FigureCanvasTkAgg``.
    """
    top = analysis_df.head(25).iloc[::-1].copy()   # reverse so highest is at top
    labels = top["feature"].tolist()
    values = top["abs_r"].tolist()
    colors = [RED if leaky else BLUE for leaky in top["leaky"].tolist()]

    fig, ax = _fig(5.5, max(4.0, len(labels) * 0.30 + 1.0))
    fig.subplots_adjust(left=0.40, right=0.96, top=0.92, bottom=0.08)

    y = list(range(len(labels)))
    ax.barh(y, values, color=colors, alpha=0.85, height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("|Point-biserial r|  (Pearson correlation with binary churn label)", color=TEXT)
    ax.set_title("Feature Correlation with Churn Label  (top 25)", color=TEXT, pad=8)
    ax.axvline(0.5, color=ORANGE, linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(0.8, color=RED,    linestyle="--", linewidth=1, alpha=0.8)
    ax.annotate("|r|=0.5", xy=(0.502, len(labels) * 0.02), color=ORANGE, fontsize=7)
    ax.annotate("|r|=0.8", xy=(0.802, len(labels) * 0.02), color=RED,    fontsize=7)
    ax.grid(axis="x", alpha=0.3)

    red_p   = mpatches.Patch(color=RED,  label="⚠ Zero overlap (leaky)")
    blue_p  = mpatches.Patch(color=BLUE, label="Overlapping distributions")
    ax.legend(handles=[red_p, blue_p], fontsize=7, facecolor=PANEL, edgecolor=GREY,
              labelcolor=TEXT, loc="lower right")
    return fig, ax


def leaky_feature_grid(features_df: "pd.DataFrame", target: "pd.Series",
                       feature_names: list[str]) -> tuple:
    """Grid of dual-class histograms for the top leakage-risk features.

    Up to 6 features are shown in a 3×2 grid.  Each subplot overlays
    churned (red) and retained (green) account distributions for that
    feature so the degree of class separation is immediately visible.

    Args:
        features_df: Feature matrix DataFrame indexed by account ID.
        target: Binary churn target Series aligned to ``features_df``.
        feature_names: Ordered list of feature column names to plot
            (highest leakage risk first).  At most 6 are shown.

    Returns:
        A ``(Figure, None)`` tuple ready for embedding via
        ``FigureCanvasTkAgg``.  The Axes tuple is ``None`` because the
        figure contains multiple subplots.
    """
    n = min(len(feature_names), 6)
    if n == 0:
        fig, ax = _fig(5, 3)
        ax.text(0.5, 0.5, "No features to display", ha="center", va="center",
                transform=ax.transAxes, color=TEXT)
        return fig, ax

    rows, cols = (n + 1) // 2, 2
    fig = Figure(figsize=(10, rows * 3.0 + 0.5), facecolor=BG)
    fig.subplots_adjust(hspace=0.50, wspace=0.28, left=0.08, right=0.97,
                        top=0.94, bottom=0.06)

    import matplotlib as mpl
    for i, fname in enumerate(feature_names[:6]):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(GREY)
        ax.tick_params(colors=TEXT, labelsize=7)
        ax.xaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)

        with mpl.rc_context(_RC):
            col = features_df[fname].fillna(features_df[fname].median())
            common = col.index.intersection(target.index)
            x = col.loc[common]
            y = target.loc[common]
            retained = x[y == 0]
            churned  = x[y == 1]

            lo = float(x.min()); hi = float(x.max())
            bins = np.linspace(lo, hi, 35) if hi > lo else 10
            ax.hist(retained.values, bins=bins, alpha=0.60, color=GREEN,
                    label=f"Retained (n={len(retained):,})")
            ax.hist(churned.values,  bins=bins, alpha=0.70, color=RED,
                    label=f"Churned  (n={len(churned):,})")
            ax.set_title(fname[:32], color=TEXT, fontsize=8, pad=4)
            ax.legend(fontsize=6, facecolor=PANEL, edgecolor=GREY, labelcolor=TEXT)
            ax.grid(axis="y", alpha=0.3, color=GREY)

    return fig, None


def score_distribution_chart(y_true: np.ndarray, y_scores: np.ndarray) -> tuple:
    """Plot overlapping score histograms split by actual class (churned vs retained).

    Args:
        y_true: Binary ground-truth labels (0 = retained, 1 = churned).
        y_scores: Predicted churn probabilities in [0, 1].

    Returns:
        A ``(Figure, Axes)`` tuple ready for embedding via ``FigureCanvasTkAgg``.
    """
    fig, ax = _fig(5, 4)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.14)
    bins = np.linspace(0, 1, 26)
    retained = y_scores[y_true == 0]
    churned  = y_scores[y_true == 1]
    ax.hist(retained, bins=bins, alpha=0.65, color=GREEN, label=f"Retained  (n={len(retained):,})")
    ax.hist(churned,  bins=bins, alpha=0.65, color=RED,   label=f"Churned   (n={len(churned):,})")
    ax.axvline(0.70, color=RED,    linestyle="--", linewidth=1.2, alpha=0.8, label="High threshold (70%)")
    ax.axvline(0.20, color=ORANGE, linestyle="--", linewidth=1.2, alpha=0.8, label="Low threshold (20%)")
    ax.set_xlabel("Predicted Churn Probability", color=TEXT)
    ax.set_ylabel("Number of accounts", color=TEXT)
    ax.set_title("Score Distribution by Actual Class", color=TEXT, pad=8)
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GREY, labelcolor=TEXT)
    ax.grid(axis="y")
    return fig, ax
