"""DataPipelineAgent — owns data loading and feature engineering."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from agents.base_agent import BaseAgent
from core.data_loader import DataLoader
from core.feature_engineer import FeatureEngineer

if TYPE_CHECKING:
    from agents.orchestrator import Orchestrator


class DataPipelineAgent(BaseAgent):
    """Agent responsible for ingesting raw data and producing the feature matrix.

    On construction a :class:`~core.data_loader.DataLoader` and a
    :class:`~core.feature_engineer.FeatureEngineer` are instantiated from the
    schema configuration.  Both are replaced in-place when a new data source is
    added dynamically via the ``"add_source"`` message.

    Results (raw frames, feature matrix, and target series) are written
    directly to the orchestrator's shared state so downstream agents can
    consume them without knowing about this agent.
    """

    def __init__(self, orchestrator: "Orchestrator"):
        """Initialise the data-pipeline agent with loader and engineer instances.

        Args:
            orchestrator: The central :class:`~agents.orchestrator.Orchestrator`
                          whose config provides the schema definition and whose
                          state receives the pipeline outputs.
        """
        super().__init__("data_pipeline")
        self.orch = orchestrator
        cfg = orchestrator.config
        self.loader = DataLoader(cfg["schema"])
        self.engineer = FeatureEngineer(cfg["schema"])

    def handle(self, msg_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a data-pipeline message to the appropriate handler.

        Args:
            msg_type: One of:

                      * ``"load_all"`` — load all configured sources and
                        engineer features.
                      * ``"reload"`` — alias for ``"load_all"``.
                      * ``"add_source"`` — register a new data source and
                        re-run the full pipeline.

            payload:  Parameter dict used only by ``"add_source"``:

                      * ``source_name`` (str): registration key for the new
                        source.
                      * ``source_def`` (dict): source definition object to
                        insert into the schema config.

        Returns:
            A response dict from the delegated handler, or an error dict for
            unknown message types.
        """
        if msg_type == "load_all":
            return self._load_all()
        if msg_type == "reload":
            return self._load_all()
        if msg_type == "rebuild_features":
            return self._rebuild_features()
        if msg_type == "add_source":
            return self._add_source(payload)
        return self._err(f"Unknown message type: {msg_type}")

    # ------------------------------------------------------------------

    def _load_all(self) -> dict[str, Any]:
        """Load every configured data source, engineer features, and update shared state.

        Calls :meth:`~core.data_loader.DataLoader.load_all` to obtain raw
        DataFrames, then passes them through
        :meth:`~core.feature_engineer.FeatureEngineer.build` to produce the
        aligned feature matrix and target series.  Both artefacts are stored
        on ``orchestrator.state``.

        Returns:
            Ok response with keys:

            * ``accounts`` (int): number of rows in the feature matrix.
            * ``features`` (int): number of feature columns.
            * ``sources`` (list[str]): names of all successfully loaded source
              frames.

            Returns an error response if any exception occurs during loading
            or feature engineering.
        """
        try:
            frames = self.loader.load_all()
            self.orch.state.frames = frames

            target_cfg  = self.orch.config["schema"]["target"]
            exclusions  = self.orch.config["model"].get("feature_exclusions", [])
            features, target = self.engineer.build(frames, target_cfg, exclude=exclusions)

            self.orch.state.features = features
            self.orch.state.target = target
            self.logger.info(
                "Pipeline complete: %d accounts, %d features",
                len(features), len(features.columns)
            )
            return self._ok(
                accounts=len(features),
                features=len(features.columns),
                sources=list(frames.keys()),
            )
        except Exception as exc:
            return self._err(str(exc))

    def _rebuild_features(self) -> dict[str, Any]:
        """Re-run feature engineering on the current in-memory frames without reloading from disk.

        Used by the curveball agent after a scenario has modified ``state.frames``
        in-place, so the model can retrain on the modified data without the disk
        reload overwriting the scenario's changes.

        Returns:
            Ok response with keys ``accounts`` (int) and ``features`` (int),
            or an error response if ``state.frames`` is empty or feature
            engineering raises an exception.
        """
        frames = self.orch.state.frames
        if not frames:
            return self._err("No frames in state — run load_all first.")
        try:
            target_cfg = self.orch.config["schema"]["target"]
            exclusions = self.orch.config["model"].get("feature_exclusions", [])
            features, target = self.engineer.build(frames, target_cfg, exclude=exclusions)
            self.orch.state.features = features
            self.orch.state.target = target
            return self._ok(accounts=len(features), features=len(features.columns))
        except Exception as exc:
            return self._err(str(exc))

    def _add_source(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Dynamically add a new data source (Phase 2 curveball hook).

        Inserts the source definition into the live schema config, then
        rebuilds the :class:`~core.data_loader.DataLoader` and
        :class:`~core.feature_engineer.FeatureEngineer` from scratch before
        triggering a full pipeline reload.

        Args:
            payload: Dict containing:

                     * ``source_name`` (str): key under which the new source
                       will be registered in ``schema["data_sources"]``.
                     * ``source_def`` (dict): source definition object
                       (format must match what :class:`~core.data_loader.DataLoader`
                       expects).

        Returns:
            The response from :meth:`_load_all` after the new source has been
            added, or an error response if ``source_name`` or ``source_def``
            are missing from ``payload``.
        """
        source_name = payload.get("source_name")
        source_def = payload.get("source_def")
        if not source_name or not source_def:
            return self._err("add_source requires 'source_name' and 'source_def'")
        self.orch.config["schema"]["data_sources"][source_name] = source_def
        self.loader = DataLoader(self.orch.config["schema"])
        self.engineer = FeatureEngineer(self.orch.config["schema"])
        return self._load_all()
