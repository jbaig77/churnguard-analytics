"""ChurnGuard Analytics — tkinter desktop app.

Native desktop window: copy-paste, embedded charts, plain-English
interpretation of every number.
"""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk
from typing import Any, TYPE_CHECKING

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ui import plots as P
from ui.model_performance_window import open_model_performance
from ui.diagnostics_window import open_diagnostics
from ui.feature_settings_dialog import open_feature_settings

if TYPE_CHECKING:
    from agents.orchestrator import Orchestrator

# ── Colour palette ────────────────────────────────────────────────────
BG       = "#1e1e2e"
PANEL    = "#2a2a3e"
SURFACE  = "#313244"
TEXT     = "#cdd6f4"
SUBTEXT  = "#a6adc8"
RED      = "#f38ba8"
ORANGE   = "#fab387"
GREEN    = "#a6e3a1"
BLUE     = "#89b4fa"
YELLOW   = "#f9e2af"
BORDER   = "#45475a"

FONT_MONO  = ("Courier New", 16)
FONT_SANS  = ("Segoe UI", 16) if tk.TkVersion else ("Helvetica", 16)
FONT_BOLD  = ("Segoe UI", 16, "bold")
FONT_TITLE = ("Segoe UI", 19, "bold")
FONT_SMALL = ("Segoe UI", 14)


class ChurnGuardApp:
    def __init__(self, orchestrator: "Orchestrator"):
        """Wire up the orchestrator, initialize state variables, and build the full Tk window layout.

        Args:
            orchestrator: Orchestrator instance that owns the pipeline agents and shared state.
        """
        self.orch = orchestrator
        self._last_factors: list[dict] = []
        self._last_account: str = ""
        self._last_prob: float = 0.0

        # ── Root window ───────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("ChurnGuard Analytics")
        self.root.geometry("1200x760")
        self.root.configure(bg=BG)
        self.root.minsize(900, 600)
        self._apply_style()

        # ── Layout ────────────────────────────────────────────────────
        self._build_toolbar()
        pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=BG,
                              sashwidth=4, sashrelief=tk.FLAT, sashpad=0)
        pane.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        left = self._build_dashboard(pane)
        right = self._build_chat(pane)
        pane.add(left,  minsize=380, width=460)
        pane.add(right, minsize=400)

        self._build_statusbar()

        # ── Keyboard shortcuts ────────────────────────────────────────
        self.root.bind("<Control-r>", lambda _: self._refresh_dashboard())
        self.root.bind("<Return>",    lambda _: self._on_send())

    def run(self) -> None:
        """Start the background pipeline initialisation thread and enter the Tkinter main loop."""
        self._set_status("Initializing pipeline…")
        threading.Thread(target=self._init_pipeline, daemon=True).start()
        self.root.mainloop()

    # ─────────────────────────────────────────────────────────────────
    # Style
    # ─────────────────────────────────────────────────────────────────

    def _apply_style(self) -> None:
        """Configure ttk.Style with the dark Catppuccin palette for all widget types."""
        s = ttk.Style(self.root)
        s.theme_use("clam")
        s.configure(".",          background=BG,    foreground=TEXT,  font=FONT_SANS)
        s.configure("TFrame",     background=BG)
        s.configure("TLabel",     background=BG,    foreground=TEXT)
        s.configure("TButton",    background=SURFACE, foreground=TEXT, padding=(8, 4))
        s.map("TButton", background=[("active", BLUE)], foreground=[("active", BG)])
        s.configure("TNotebook",              background=PANEL, tabmargins=[2, 4, 0, 0])
        s.configure("TNotebook.Tab",          background=SURFACE, foreground=SUBTEXT,
                    padding=[12, 4], font=FONT_SMALL)
        s.map("TNotebook.Tab",
              background=[("selected", BLUE)], foreground=[("selected", BG)])
        s.configure("Treeview",       background=PANEL, fieldbackground=PANEL,
                    foreground=TEXT,  rowheight=30, font=FONT_SMALL)
        s.configure("Treeview.Heading", background=SURFACE, foreground=BLUE,
                    font=("Segoe UI", 14, "bold"))
        s.map("Treeview", background=[("selected", BLUE)], foreground=[("selected", BG)])
        s.configure("TSeparator", background=BORDER)

    # ─────────────────────────────────────────────────────────────────
    # Toolbar
    # ─────────────────────────────────────────────────────────────────

    def _build_toolbar(self) -> None:
        """Create the top toolbar frame with the title label and Refresh/Help buttons."""
        bar = tk.Frame(self.root, bg=PANEL, height=40)
        bar.pack(fill=tk.X, padx=0, pady=0)
        tk.Label(bar, text="ChurnGuard Analytics", font=FONT_TITLE,
                 bg=PANEL, fg=BLUE).pack(side=tk.LEFT, padx=12)
        ttk.Button(bar, text="⟳ Refresh",    command=self._refresh_dashboard).pack(side=tk.RIGHT, padx=4, pady=6)
        ttk.Button(bar, text="? Help",       command=self._show_help).pack(side=tk.RIGHT, padx=0, pady=6)
        ttk.Button(bar, text="📊 Model",     command=self._show_model_performance).pack(side=tk.RIGHT, padx=4, pady=6)
        ttk.Button(bar, text="🔍 Diagnose",  command=self._show_diagnostics).pack(side=tk.RIGHT, padx=4, pady=6)
        ttk.Button(bar, text="⚙ Features",  command=self._show_feature_settings).pack(side=tk.RIGHT, padx=4, pady=6)

    # ─────────────────────────────────────────────────────────────────
    # Left panel — Dashboard
    # ─────────────────────────────────────────────────────────────────

    def _build_dashboard(self, parent) -> tk.Frame:
        """Build the left dashboard panel containing Overview, Charts, and Risk Table notebook tabs.

        Args:
            parent: The PanedWindow that will host the returned frame.

        Returns:
            The outer tk.Frame that holds the notebook.
        """
        frame = tk.Frame(parent, bg=BG)

        nb = ttk.Notebook(frame)
        nb.pack(fill=tk.BOTH, expand=True)
        self._nb = nb

        # Tab 1 — Overview
        t1 = tk.Frame(nb, bg=BG)
        nb.add(t1, text="Overview")
        self._stats_text = self._scrollable_text(t1)  # _scrollable_text packs itself

        # Tab 2 — Charts
        t2 = tk.Frame(nb, bg=BG)
        nb.add(t2, text="Charts")
        self._chart_nb = ttk.Notebook(t2)
        self._chart_nb.pack(fill=tk.BOTH, expand=True)

        self._chart_dist_frame  = tk.Frame(self._chart_nb, bg=BG)
        self._chart_fi_frame    = tk.Frame(self._chart_nb, bg=BG)
        self._chart_acct_frame  = tk.Frame(self._chart_nb, bg=BG)
        self._chart_nb.add(self._chart_dist_frame,  text="Distribution")
        self._chart_nb.add(self._chart_fi_frame,    text="Feature Importance")
        self._chart_nb.add(self._chart_acct_frame,  text="Account Detail")

        # Tab 3 — Risk Table
        t3 = tk.Frame(nb, bg=BG)
        nb.add(t3, text="Risk Table")
        self._build_risk_table(t3)

        return frame

    def _build_risk_table(self, parent) -> None:
        """Create the Treeview risk table with a tier filter bar, sortable column headings, and row-select binding.

        Adds radio buttons above the table for High Risk / Medium / Low Risk / All tiers.
        The selected filter is stored in ``self._risk_filter`` and applied without re-querying
        the model — it just re-slices the already-loaded predictions.

        Args:
            parent: The tk.Frame that will contain the filter bar, Treeview, and scrollbar.
        """
        # ── Filter bar ────────────────────────────────────────────────
        filter_frame = tk.Frame(parent, bg=BG)
        filter_frame.pack(fill=tk.X, padx=6, pady=(4, 2))

        tk.Label(filter_frame, text="Show:", bg=BG, fg=SUBTEXT,
                 font=FONT_SMALL).pack(side=tk.LEFT, padx=(0, 8))

        self._risk_filter = tk.StringVar(value="high")
        for label, val, fg in [
            ("High Risk",  "high",   RED),
            ("Low Risk",   "low",    GREEN),
        ]:
            tk.Radiobutton(
                filter_frame, text=label, variable=self._risk_filter, value=val,
                command=self._on_risk_filter_change,
                bg=BG, fg=fg, selectcolor=SURFACE,
                activebackground=BG, activeforeground=fg,
                font=FONT_SMALL,
            ).pack(side=tk.LEFT, padx=4)

        # ── Treeview ──────────────────────────────────────────────────
        cols = ("account_id", "prob", "risk", "tier", "mrr")
        tree = ttk.Treeview(parent, columns=cols, show="headings", selectmode="browse")
        self._risk_tree = tree

        for col, label, width in [
            ("account_id", "Account",  110),
            ("prob",       "Churn %",   70),
            ("risk",       "Risk",       60),
            ("tier",       "Tier",       90),
            ("mrr",        "MRR/mo",     75),
        ]:
            tree.heading(col, text=label,
                         command=lambda c=col: self._sort_tree(c))
            tree.column(col, width=width, anchor=tk.CENTER)

        tree.tag_configure("high",   foreground=RED)
        tree.tag_configure("medium", foreground=ORANGE)
        tree.tag_configure("low",    foreground=GREEN)

        vsb = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)
        tree.bind("<<TreeviewSelect>>", self._on_row_select)

        self._all_risk_rows: list[dict] = []  # populated by _refresh_risk_table

    # ─────────────────────────────────────────────────────────────────
    # Right panel — Chat
    # ─────────────────────────────────────────────────────────────────

    def _build_chat(self, parent) -> tk.Frame:
        """Build the right chat panel with a read-only log, colour tags, and an input entry with Send button.

        Args:
            parent: The PanedWindow that will host the returned frame.

        Returns:
            The outer tk.Frame that holds the chat log and input row.
        """
        frame = tk.Frame(parent, bg=BG)

        tk.Label(frame, text="CHAT  (text is selectable — Ctrl+C to copy)",
                 font=FONT_SMALL, bg=BG, fg=SUBTEXT).pack(anchor=tk.W, padx=6, pady=(4, 0))

        # Chat log — Text with DISABLED state so it's read-only but selectable
        self._chat = tk.Text(
            frame,
            bg=PANEL, fg=TEXT, insertbackground=TEXT,
            font=FONT_MONO, wrap=tk.WORD,
            relief=tk.FLAT, padx=8, pady=6,
            state=tk.DISABLED,
            cursor="xterm",   # text cursor so selection is obvious
        )
        self._chat.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # Tags for colour-coded output
        for tag, kwargs in [
            ("header",   {"font": ("Courier New", 11, "bold"), "foreground": BLUE}),
            ("high",     {"foreground": RED}),
            ("medium",   {"foreground": ORANGE}),
            ("low",      {"foreground": GREEN}),
            ("context",  {"foreground": SUBTEXT, "font": ("Courier New", 9)}),
            ("query",    {"foreground": YELLOW, "font": ("Courier New", 10, "bold")}),
            ("warn",     {"foreground": ORANGE}),
            ("bold",     {"font": ("Courier New", 10, "bold")}),
            ("dim",      {"foreground": SUBTEXT}),
        ]:
            self._chat.tag_configure(tag, **kwargs)

        # Input row
        inp_frame = tk.Frame(frame, bg=BG)
        inp_frame.pack(fill=tk.X, padx=6, pady=(0, 6))

        self._entry = tk.Entry(
            inp_frame, bg=SURFACE, fg=TEXT, insertbackground=TEXT,
            font=FONT_MONO, relief=tk.FLAT,
        )
        self._entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6, padx=(0, 6))
        self._entry.bind("<Return>", lambda _: self._on_send())

        ttk.Button(inp_frame, text="Send", command=self._on_send).pack(side=tk.RIGHT)

        return frame

    # ─────────────────────────────────────────────────────────────────
    # Status bar
    # ─────────────────────────────────────────────────────────────────

    def _build_statusbar(self) -> None:
        """Add the bottom status bar with a StringVar-backed label to the root window."""
        bar = tk.Frame(self.root, bg=PANEL, height=22)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        self._status_var = tk.StringVar(value="Initializing…")
        tk.Label(bar, textvariable=self._status_var,
                 font=FONT_SMALL, bg=PANEL, fg=SUBTEXT, anchor=tk.W).pack(
            side=tk.LEFT, padx=8)

    def _set_status(self, msg: str) -> None:
        """Update the status bar text.

        Args:
            msg: The message string to display in the status bar.
        """
        self._status_var.set(msg)

    # ─────────────────────────────────────────────────────────────────
    # Pipeline init (background thread)
    # ─────────────────────────────────────────────────────────────────

    def _init_pipeline(self) -> None:
        """Background thread: call orchestrator.initialize() then schedule _on_ready on the main thread."""
        def on_status(msg):
            self.root.after(0, self._set_status, msg)
        self.orch.add_status_callback(on_status)
        try:
            force = self.orch.config.get("_force_retrain", False)
            self.orch.initialize(force_retrain=force)
            self.root.after(0, self._on_ready)
        except Exception as exc:
            self.root.after(0, self._chat_write, f"Initialization error: {exc}\n", "high")

    def _on_ready(self) -> None:
        """Update the status bar and seed the chat with welcome text after pipeline initialisation completes."""
        n = len(self.orch.state.predictions) if self.orch.state.predictions is not None else 0
        self._set_status(f"Ready  |  {n:,} accounts  |  Model: XGBoost  |  Ctrl+R to refresh")
        self._refresh_dashboard()
        self._chat_write("ChurnGuard Analytics Assistant\n", "header")
        self._chat_write("Data loaded and model ready. Try:\n", "dim")
        self._chat_write('  "Why is account ACC000123 predicted to churn?"\n', "dim")
        self._chat_write('  "Show top 10 at-risk accounts"\n', "dim")
        self._chat_write('  Type /help for all commands\n\n', "dim")

    # ─────────────────────────────────────────────────────────────────
    # Dashboard refresh
    # ─────────────────────────────────────────────────────────────────

    def _refresh_dashboard(self) -> None:
        """Spawn a background thread to refresh all dashboard panels if the pipeline is ready."""
        if not self.orch.state.is_ready:
            return
        threading.Thread(target=self._do_refresh, daemon=True).start()

    def _do_refresh(self) -> None:
        """Background thread: call all four refresh sub-methods and report any exceptions to the status bar."""
        try:
            self._refresh_stats()
            self._refresh_risk_table()
            self._refresh_dist_chart()
            self._refresh_fi_chart()
        except Exception as exc:
            self.root.after(0, self._set_status, f"Refresh error: {exc}")

    def _refresh_stats(self) -> None:
        """Fetch analytics summary and model metrics, format them as tagged lines, and schedule _update_stats_text."""
        resp = self.orch.send("analytics", "summary")
        if resp.get("status") != "ok":
            return
        metrics = self.orch.state.metrics
        interp = self.orch.state.config.get("_interpreter")
        churn_rate = float(self.orch.state.target.mean()) if self.orch.state.target is not None else 0
        lines: list[tuple[str, str]] = []   # (text, tag)

        lines += [("═══ PORTFOLIO OVERVIEW ═══\n\n", "header")]
        total = resp["total_accounts"]
        high = resp["high_risk_count"]
        med  = resp["medium_risk_count"]
        low  = resp["low_risk_count"]
        avg  = resp["avg_churn_prob"]

        lines += [(f"  Total accounts:         {total:,}\n", "")]
        lines += [(f"  ⛔ High risk  (≥70%):   {high:,}  ({high/total:.1%})\n", "high")]
        lines += [(f"  ✓  Low risk   (<20%):   {low:,}  ({low/total:.1%})\n", "low")]
        lines += [(f"\n  Avg churn probability:   {avg:.1%}\n", "")]

        if interp:
            for ln in interp.explain_summary(resp):
                lines += [(f"  {ln}\n", "context")]

        lines += [("\n═══ MODEL PERFORMANCE ═══\n\n", "header")]
        if metrics:
            auc = metrics.get("roc_auc", 0)
            lines += [(f"  AUC (ROC):   {auc:.4f}", "")]
            if auc >= 0.99:
                lines += [("  ← see note below\n", "warn")]
            else:
                lines += [("\n", "")]
            lines += [(f"  AUC (PR):    {metrics.get('pr_auc', 0):.4f}\n", "")]
            lines += [(f"  Recall:      {metrics.get('recall', 0):.4f}\n", "")]
            lines += [(f"  Precision:   {metrics.get('precision', 0):.4f}\n", "")]
            lines += [(f"  F1:          {metrics.get('f1', 0):.4f}\n", "")]
            lines += [(f"  CV AUC:      {metrics.get('cv_auc_mean', 0):.4f}"
                       f" ± {metrics.get('cv_auc_std', 0):.4f}\n", "")]

            if interp:
                lines += [("\n", "")]
                for ln in interp.explain_metrics(metrics, churn_rate):
                    lines += [(f"  {ln}\n", "context")]

        self.root.after(0, self._update_stats_text, lines)

    def _update_stats_text(self, lines: list[tuple[str, str]]) -> None:
        """Replace the Overview tab text with the provided list of (text, tag) tuples.

        Args:
            lines: List of (text, tag) tuples where tag is a colour tag name or empty string.
        """
        t = self._stats_text
        t.configure(state=tk.NORMAL)
        t.delete("1.0", tk.END)
        for text, tag in lines:
            if tag:
                t.insert(tk.END, text, tag)
            else:
                t.insert(tk.END, text)
        t.configure(state=tk.DISABLED)

    def _refresh_risk_table(self) -> None:
        """Read all predictions from shared state, join account metadata, cache rows, then apply the current tier filter.

        Reads directly from ``orch.state.predictions`` rather than calling the analytics
        agent, so all tiers are available for instant filter switching without re-querying.
        """
        preds = self.orch.state.predictions
        accounts_df = self.orch.state.frames.get("accounts")
        if preds is None:
            return

        import pandas as pd
        extra_cols = [c for c in ["subscription_tier", "mrr"]
                      if accounts_df is not None and c in accounts_df.columns]
        if extra_cols:
            df = preds.join(accounts_df.set_index("account_id")[extra_cols], how="left")
        else:
            df = preds.copy()

        df = df.sort_values("churn_probability", ascending=False)
        rows = []
        for acc_id, row in df.iterrows():
            rows.append({
                "account_id": acc_id,
                "churn_probability": float(row["churn_probability"]),
                "risk_level": str(row["risk_level"]),
                "subscription_tier": row.get("subscription_tier"),
                "mrr": row.get("mrr"),
            })
        self._all_risk_rows = rows
        self.root.after(0, self._apply_risk_filter)

    def _populate_risk_table(self, accounts: list[dict]) -> None:
        """Clear and repopulate the Treeview with account rows coloured by risk tier.

        Args:
            accounts: List of account dicts each containing churn_probability, risk_level, and related fields.
        """
        tree = self._risk_tree
        for row in tree.get_children():
            tree.delete(row)
        for acc in accounts:
            prob  = acc["churn_probability"]
            risk  = acc["risk_level"].upper()
            tier  = acc.get("subscription_tier") or "—"
            mrr   = acc.get("mrr")
            mrr_s = f"${mrr:,.0f}" if isinstance(mrr, (int, float)) else "—"
            tag   = risk.lower()
            tree.insert("", tk.END,
                        values=(acc["account_id"], f"{prob:.1%}", risk, tier, mrr_s),
                        tags=(tag,))

    def _apply_risk_filter(self) -> None:
        """Filter ``self._all_risk_rows`` by the selected tier and repopulate the table.

        Called on initial load and whenever a filter radio button is clicked.
        High / Medium / Low show only that tier; All shows every account.
        Results are capped at ``top_n_accounts`` from config to keep the UI responsive.
        """
        if not self._all_risk_rows:
            return
        tier = self._risk_filter.get()
        high_t = self.orch.config["model"]["model"].get("high_risk_threshold", 0.70)
        low_t  = self.orch.config["model"]["model"].get("medium_risk_threshold", 0.20)
        if tier == "high":
            rows = [r for r in self._all_risk_rows if r["churn_probability"] >= high_t]
        else:  # low
            rows = [r for r in self._all_risk_rows if r["churn_probability"] < low_t]

        top_n = self.orch.config["model"]["dashboard"]["top_n_accounts"]
        self._populate_risk_table(rows[:top_n])

    def _on_risk_filter_change(self) -> None:
        """Respond to a filter radio button click by re-applying the tier filter immediately."""
        self._apply_risk_filter()

    def _refresh_dist_chart(self) -> None:
        """Regenerate the churn-probability histogram and embed it in the Distribution tab."""
        preds = self.orch.state.predictions
        if preds is None:
            return
        fig, _ = P.churn_distribution(preds)
        self.root.after(0, self._embed_chart, fig, self._chart_dist_frame)

    def _refresh_fi_chart(self) -> None:
        """Fetch feature importance from the model agent, generate the chart, and embed it in the Feature Importance tab."""
        top_n = self.orch.config["model"]["dashboard"]["top_n_features"]
        resp = self.orch.send("model", "feature_importance", {"top_n": top_n})
        if resp.get("status") != "ok":
            return
        import pandas as pd
        df = pd.DataFrame(resp["importance"])
        fig, _ = P.feature_importance(df, top_n)
        self.root.after(0, self._embed_chart, fig, self._chart_fi_frame)

    def _embed_chart(self, fig, frame: tk.Frame) -> None:
        """Destroy existing widgets in frame, create a FigureCanvasTkAgg for fig, and pack it.

        Args:
            fig: The matplotlib Figure to embed.
            frame: The tk.Frame that will host the canvas widget.
        """
        for widget in frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ─────────────────────────────────────────────────────────────────
    # Chat
    # ─────────────────────────────────────────────────────────────────

    def _on_send(self) -> None:
        """Read the entry text, clear it, echo the query to the chat log, and spawn a background handler thread."""
        text = self._entry.get().strip()
        if not text:
            return
        self._entry.delete(0, tk.END)
        self._chat_write(f"\n▶  {text}\n", "query")
        threading.Thread(target=self._handle_query, args=(text,), daemon=True).start()

    def _handle_query(self, text: str) -> None:
        """Background thread: send text to the conversation agent and schedule _display_response on the main thread.

        Args:
            text: The raw query string entered by the user.
        """
        resp = self.orch.send("conversation", "query", {"text": text})
        response  = resp.get("response", resp.get("message", "No response."))
        factors   = resp.get("factors", [])
        account_id = resp.get("account_id", "")
        prob       = resp.get("probability", 0.0)

        self.root.after(0, self._display_response, response, factors, account_id, prob, text)

    def _display_response(self, response: str, factors: list,
                          account_id: str, prob: float, query: str) -> None:
        """Write the formatted response to the chat, update the account chart if factors were returned, and refresh after state-changing commands.

        Args:
            response: The text response from the conversation agent to display.
            factors: List of SHAP feature-factor dicts; may be empty if no account was resolved.
            account_id: The account identifier associated with the response, or empty string.
            prob: The churn probability for the account, or 0.0 if not applicable.
            query: The original user query, used to detect state-changing commands.
        """
        self._write_formatted(response)
        self._chat_write("\n", "")

        # If we got account factors, update the account chart tab
        if factors and account_id:
            self._last_factors  = factors
            self._last_account  = account_id
            self._last_prob     = prob
            self._update_account_chart(factors, account_id, prob)
            self._nb.select(0)   # go back to overview tab
            self._chart_nb.select(2)   # account detail sub-tab

        # Refresh dashboard after state-changing commands
        if any(c in query.lower() for c in ["/retrain", "/reload", "/curveball"]):
            self._refresh_dashboard()

    def _write_formatted(self, text: str) -> None:
        """Write response with syntax colouring based on content patterns."""
        import re
        for line in text.split("\n"):
            ll = line.lower()
            # Probability percentages
            tag = ""
            if re.search(r'\b(high risk|high\]|🚨)', ll):
                tag = "high"
            elif re.search(r'\b(medium risk|medium\]|⚠)', ll):
                tag = "medium"
            elif re.search(r'\b(low risk|low\]|✓)', ll):
                tag = "low"
            elif ll.startswith("  →"):
                tag = "context"
            elif "═══" in line or line.startswith("top ") or line.startswith("account "):
                tag = "bold"
            elif line.startswith("  note:") or "⚠ note:" in ll:
                tag = "warn"

            self._chat_write(line + "\n", tag)

    def _chat_write(self, text: str, tag: str = "") -> None:
        """Append text to the chat log with an optional colour tag, then scroll to the bottom.

        Args:
            text: The string to append to the chat log.
            tag: The name of a configured colour tag to apply, or empty string for default styling.
        """
        self._chat.configure(state=tk.NORMAL)
        if tag:
            self._chat.insert(tk.END, text, tag)
        else:
            self._chat.insert(tk.END, text)
        self._chat.configure(state=tk.DISABLED)
        self._chat.see(tk.END)

    # ─────────────────────────────────────────────────────────────────
    # Account detail chart
    # ─────────────────────────────────────────────────────────────────

    def _update_account_chart(self, factors: list, account_id: str, prob: float) -> None:
        """Generate the per-account SHAP waterfall chart and schedule its embedding in the Account Detail tab.

        Args:
            factors: List of feature-factor dicts used to build the waterfall chart.
            account_id: The account identifier shown in the chart title.
            prob: The churn probability displayed on the chart.
        """
        fig, _ = P.account_factors(factors, account_id, prob)
        self.root.after(0, self._embed_chart, fig, self._chart_acct_frame)
        self.root.after(0, lambda: self._nb.select(0))

    # ─────────────────────────────────────────────────────────────────
    # Actions
    # ─────────────────────────────────────────────────────────────────

    def _on_row_select(self, _event) -> None:
        """Respond to Treeview row clicks by pre-filling the entry with a churn-explanation query and sending it."""
        sel = self._risk_tree.selection()
        if not sel:
            return
        vals = self._risk_tree.item(sel[0])["values"]
        if vals:
            account_id = str(vals[0])
            self._entry.delete(0, tk.END)
            self._entry.insert(0, f"Why is {account_id} predicted to churn?")
            self._on_send()

    def _sort_tree(self, col: str) -> None:
        """Sort all Treeview rows by the given column, attempting numeric sort then falling back to string sort.

        Args:
            col: The column identifier to sort by.
        """
        tree = self._risk_tree
        data = [(tree.set(k, col), k) for k in tree.get_children("")]
        try:
            data.sort(key=lambda x: float(x[0].strip("%$,")) if x[0].strip("%$,") else 0,
                      reverse=True)
        except ValueError:
            data.sort(reverse=True)
        for i, (_, k) in enumerate(data):
            tree.move(k, "", i)

    def _show_model_performance(self) -> None:
        """Open the Model Performance window with metrics table and diagnostic charts."""
        open_model_performance(self.orch, self.root)

    def _show_diagnostics(self) -> None:
        """Open the Data Diagnostics window with the feature leakage audit."""
        open_diagnostics(self.orch, self.root)

    def _show_feature_settings(self) -> None:
        """Open the Feature Exclusion Settings dialog."""
        open_feature_settings(self.orch, self.root, self._on_exclusions_applied)

    def _on_exclusions_applied(self, exclusions: list) -> None:
        """Callback from the settings dialog: retrain in a background thread."""
        threading.Thread(
            target=self._retrain_with_exclusions,
            args=(exclusions,),
            daemon=True,
        ).start()

    def _retrain_with_exclusions(self, exclusions: list) -> None:
        """Background thread: rebuild features with new exclusions, retrain, refresh."""
        label = f"Excluded: {', '.join(exclusions)}" if exclusions else "all features"
        self.root.after(0, self._set_status, f"Retraining ({label})…")
        self.orch.state.is_ready = False
        try:
            r = self.orch.send("data_pipeline", "rebuild_features")
            if r.get("status") != "ok":
                raise RuntimeError(f"Feature rebuild failed: {r.get('message')}")

            r = self.orch.send("model", "train_or_load", {"force_retrain": True})
            if r.get("status") != "ok":
                raise RuntimeError(f"Training failed: {r.get('message')}")

            # Mark ready and refresh NOW so metrics are visible before slow SHAP step.
            self.orch.state.is_ready = True
            n = len(self.orch.state.predictions) if self.orch.state.predictions is not None else 0
            self.root.after(0, self._set_status, f"Trained  |  {n:,} accounts  |  computing explanations…")
            self.root.after(0, self._refresh_dashboard)

            self.orch.send("model", "explain_all")
            self.orch._build_interpreter()

            msg = f"Ready  |  {n:,} accounts  |  {label}"
            self.root.after(0, self._set_status, msg)
            self.root.after(0, self._refresh_dashboard)
        except Exception as exc:
            self.orch.state.is_ready = True
            self.root.after(0, self._set_status, f"Retrain error: {exc}")
            self.root.after(0, self._refresh_dashboard)

    def _show_help(self) -> None:
        """Pre-fill the entry with '/help' and send it."""
        self._entry.delete(0, tk.END)
        self._entry.insert(0, "/help")
        self._on_send()

    # ─────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────

    def _scrollable_text(self, parent, **kwargs) -> tk.Text:
        """Create a read-only tk.Text with a vertical scrollbar and pre-configured colour tags.

        Args:
            parent: The parent widget that will contain the scrollable text frame.
            **kwargs: Additional keyword arguments forwarded to the tk.Text constructor.

        Returns:
            The configured tk.Text widget.
        """
        frame = tk.Frame(parent, bg=BG)
        frame.pack(fill=tk.BOTH, expand=True)
        txt = tk.Text(frame, bg=PANEL, fg=TEXT, font=FONT_SMALL,
                      wrap=tk.WORD, relief=tk.FLAT, padx=8, pady=6,
                      state=tk.DISABLED, cursor="xterm", **kwargs)
        vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=txt.yview)
        txt.configure(yscrollcommand=vsb.set)
        txt.tag_configure("header",  font=("Segoe UI", 10, "bold"), foreground=BLUE)
        txt.tag_configure("high",    foreground=RED)
        txt.tag_configure("medium",  foreground=ORANGE)
        txt.tag_configure("low",     foreground=GREEN)
        txt.tag_configure("context", foreground=SUBTEXT, font=FONT_SMALL)
        txt.tag_configure("warn",    foreground=ORANGE)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        return txt
