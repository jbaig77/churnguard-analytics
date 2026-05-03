"""ChurnGuard Analytics — entry point.

Usage:
    python main.py                  # launch TUI (trains or loads cached model)
    python main.py --retrain        # force model retrain, then launch TUI
    python main.py --headless       # run pipeline only (no TUI), useful for CI/testing
    python main.py --curveball list # list available curveball scenarios
    python main.py --curveball <name>  # run a single curveball from the CLI
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so all imports resolve regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))

from core.config import load_config
from agents.orchestrator import Orchestrator
from agents.data_pipeline_agent import DataPipelineAgent
from agents.model_agent import ModelAgent
from agents.analytics_agent import AnalyticsAgent
from agents.conversation_agent import ConversationAgent
from agents.curveball_agent import CurveballAgent


def setup_logging(level: str = "WARNING") -> None:
    """Configure the root logger with a timestamped format.

    Args:
        level: Logging level name as a string (e.g. ``"DEBUG"``,
            ``"INFO"``, ``"WARNING"``). Case-insensitive. Unrecognised
            values fall back to ``WARNING``.

    Returns:
        None.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.WARNING),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_orchestrator(config: dict) -> Orchestrator:
    """Instantiate the Orchestrator and register all application agents.

    Creates one instance of each agent (DataPipeline, Model, Analytics,
    Conversation, Curveball) and registers them with the orchestrator so
    they can receive inter-agent messages.

    Args:
        config: Application config dict as returned by ``load_config()``.

    Returns:
        A fully initialised ``Orchestrator`` with all five agents
        registered and ready to handle messages.
    """
    orch = Orchestrator(config)
    orch.register(DataPipelineAgent(orch))
    orch.register(ModelAgent(orch))
    orch.register(AnalyticsAgent(orch))
    orch.register(ConversationAgent(orch))
    orch.register(CurveballAgent(orch))
    return orch


def run_tui(orch: Orchestrator, force_retrain: bool) -> None:
    """Launch the Tkinter-based terminal UI and block until it is closed.

    Sets the ``_force_retrain`` flag on the orchestrator config so the
    UI's initialisation sequence can decide whether to retrain the model
    or load a cached one.

    Args:
        orch: Fully registered ``Orchestrator`` instance.
        force_retrain: When ``True``, the UI will retrain the model from
            scratch rather than loading a previously cached artifact.

    Returns:
        None. Blocks until the UI window is closed.
    """
    from ui.app import ChurnGuardApp
    orch.config["_force_retrain"] = force_retrain
    app = ChurnGuardApp(orch)
    app.run()


def run_headless(orch: Orchestrator, force_retrain: bool) -> None:
    """Run the full analytics pipeline without a UI and print a portfolio summary.

    Initialises the orchestrator (loading data, training or loading the
    model), queries the analytics agent for a portfolio summary, and
    prints key metrics to the terminal using Rich formatting.

    Args:
        orch: Fully registered ``Orchestrator`` instance.
        force_retrain: When ``True``, forces the model to be retrained
            from scratch rather than restored from cache.

    Returns:
        None. All output is written to stdout via the Rich console.
    """
    from rich.console import Console
    console = Console()
    console.print("[bold cyan]ChurnGuard Analytics — headless mode[/bold cyan]")

    def on_status(msg: str) -> None:
        """Print a single pipeline status message to the console.

        Args:
            msg: Status message string emitted by the orchestrator.

        Returns:
            None.
        """
        console.print(f"  [dim]{msg}[/dim]")

    orch.add_status_callback(on_status)
    orch.initialize(force_retrain=force_retrain)

    resp = orch.send("analytics", "summary")
    if resp.get("status") == "ok":
        console.print(f"\n[bold]Portfolio summary:[/bold]")
        console.print(f"  Total accounts:   {resp['total_accounts']:,}")
        console.print(f"  High risk (>=70%): {resp['high_risk_count']:,}")
        console.print(f"  Avg churn prob:   {resp['avg_churn_prob']:.1%}")
        m = resp.get("metrics", {})
        console.print(f"  ROC-AUC:          {m.get('roc_auc', 0):.4f}")
        console.print(f"  Recall:           {m.get('recall', 0):.4f}")


def run_curveball_cli(orch: Orchestrator, scenario_name: str) -> None:
    """Execute a single curveball scenario from the command line and print its report.

    Initialises the orchestrator with the cached model (no retrain), then
    dispatches the named scenario to the curveball agent and prints the
    resulting report to the terminal.

    Args:
        orch: Fully registered ``Orchestrator`` instance.
        scenario_name: Name of the curveball scenario to run, or
            ``"list"`` to have the agent enumerate available scenarios.

    Returns:
        None. The scenario report is printed to stdout via the Rich
        console.
    """
    from rich.console import Console
    console = Console()

    orch.initialize(force_retrain=False)
    command = f"/curveball {scenario_name}"
    resp = orch.send("curveball", "run", {"command": command})
    console.print(resp.get("report", resp.get("message", "")))


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate run mode.

    Supports three mutually exclusive modes:
    - Default (no flags): launches the interactive TUI.
    - ``--headless``: runs the pipeline and prints a summary with no UI.
    - ``--curveball SCENARIO``: runs one curveball scenario and exits.

    The ``--retrain`` flag is honoured in TUI and headless modes to force
    a fresh model training cycle. ``--log-level`` controls verbosity for
    all modes.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="ChurnGuard Analytics")
    parser.add_argument("--retrain", action="store_true", help="Force model retrain")
    parser.add_argument("--headless", action="store_true", help="Run pipeline without TUI")
    parser.add_argument("--curveball", metavar="SCENARIO", help="Run a curveball scenario")
    parser.add_argument("--log-level", default="WARNING", help="Logging level (default: WARNING)")
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = load_config()
    orch = build_orchestrator(config)

    if args.curveball:
        run_curveball_cli(orch, args.curveball)
    elif args.headless:
        run_headless(orch, force_retrain=args.retrain)
    else:
        run_tui(orch, force_retrain=args.retrain)


if __name__ == "__main__":
    main()
