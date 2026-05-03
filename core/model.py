"""XGBoost churn model: training, evaluation, prediction, caching."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class ChurnModel:
    """XGBoost-based binary classifier that predicts customer churn probability.

    Handles the full modelling lifecycle: sanitising features, training with
    class-imbalance correction, cross-validated AUC evaluation, threshold-based
    risk bucketing, feature-importance reporting, and joblib/YAML caching so
    that an already-trained model can be reloaded without re-fitting.

    Attributes:
        cfg: The ``"model"`` sub-dict from the model config YAML, containing
            hyperparameters, thresholds, CV settings, and cache paths.
        root: Absolute ``pathlib.Path`` to the project root directory.
        model: The fitted ``XGBClassifier`` instance, or ``None`` before
            training.
        feature_names: Ordered list of feature column names recorded at
            training time; used to align inference inputs.
        metrics: Dict of evaluation metrics populated after ``train()``
            completes (accuracy, precision, recall, f1, roc_auc, pr_auc,
            cv_auc_mean, cv_auc_std).
        label_encoders: Dict mapping categorical column names to their fitted
            ``LabelEncoder`` instances; persisted so that the same encoding
            is applied at inference time.
    """

    def __init__(self, model_cfg: dict[str, Any], project_root: str):
        """Initialise the model wrapper with configuration and paths.

        Args:
            model_cfg: The full model configuration dict as returned by
                ``load_config()["model"]``.  Must contain a ``"model"`` key
                whose value is a dict with at minimum a ``"params"`` sub-dict
                of XGBoost hyperparameters.
            project_root: Absolute path string to the repository root, used
                to resolve relative cache file paths specified in the config.
        """
        self.cfg = model_cfg["model"]
        self.root = Path(project_root)
        self.model: XGBClassifier | None = None
        self.feature_names: list[str] = []
        self.metrics: dict[str, float] = {}
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.y_test_true: np.ndarray | None = None
        self.y_test_scores: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
        """Fit the XGBoost classifier and evaluate it on a held-out test split.

        Sanitises ``features`` (infinite values, NaNs, categorical encoding),
        performs a stratified train/test split, fits the classifier with
        automatic ``scale_pos_weight`` to handle class imbalance, computes
        cross-validated AUC scores, evaluates standard classification metrics
        on the test set, and optionally serialises the trained artefacts to
        disk via :meth:`_save_cache`.

        Args:
            features: A ``pd.DataFrame`` where each row is one account and
                each column is an engineered feature.  The index must be
                compatible with ``target.index``.
            target: A binary ``pd.Series`` (0 = retained, 1 = churned) aligned
                with ``features`` by index.

        Returns:
            A dict containing the following float metrics (all rounded to 4
            decimal places):

            - ``"accuracy"``     — fraction of correctly classified test examples.
            - ``"precision"``    — positive-predictive value on the test set.
            - ``"recall"``       — true-positive rate on the test set.
            - ``"f1"``           — harmonic mean of precision and recall.
            - ``"roc_auc"``      — area under the ROC curve.
            - ``"pr_auc"``       — area under the precision-recall curve.
            - ``"cv_auc_mean"``  — mean ROC-AUC across stratified CV folds.
            - ``"cv_auc_std"``   — standard deviation of CV fold AUC scores.
        """
        X = features.copy()
        y = target.loc[X.index]

        self.feature_names = list(X.columns)
        X = self._sanitize(X)

        seed = self.cfg.get("random_seed", 42)
        test_size = self.cfg.get("test_size", 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

        pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum() + 1e-6)
        params = {**self.cfg.get("params", {}), "scale_pos_weight": pos_weight,
                  "random_state": seed, "use_label_encoder": False, "verbosity": 0}

        self.model = XGBClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # CV AUC
        cv_aucs = self._cv_auc(X, y, seed)
        proba = self.model.predict_proba(X_test)[:, 1]
        preds = (proba >= self.cfg.get("high_risk_threshold", 0.5)).astype(int)

        self.metrics = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "recall": round(recall_score(y_test, preds, zero_division=0), 4),
            "f1": round(f1_score(y_test, preds, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
            "pr_auc": round(average_precision_score(y_test, proba), 4),
            "cv_auc_mean": round(float(np.mean(cv_aucs)), 4),
            "cv_auc_std": round(float(np.std(cv_aucs)), 4),
        }
        self.y_test_true = y_test.values
        self.y_test_scores = proba
        logger.info("Model trained | %s", self.metrics)
        if self.cfg.get("cache", {}).get("enabled", True):
            self._save_cache(features)
        return self.metrics

    def _cv_auc(self, X: np.ndarray, y: pd.Series, seed: int) -> list[float]:
        """Compute per-fold ROC-AUC scores using stratified k-fold cross-validation.

        Trains a fresh ``XGBClassifier`` on each training fold (with the
        same class-imbalance correction used in :meth:`train`) and evaluates
        it on the corresponding validation fold.  The number of folds is read
        from ``cfg["cross_validation_folds"]`` (default 5).

        Args:
            X: Feature matrix as a ``pd.DataFrame`` (after sanitisation),
                indexed to match ``y``.
            y: Binary target ``pd.Series`` (0 = retained, 1 = churned).
            seed: Integer random seed forwarded to both ``StratifiedKFold``
                and each fold's ``XGBClassifier`` for reproducibility.

        Returns:
            A list of ``float`` ROC-AUC scores, one per fold, in fold order.
        """
        folds = self.cfg.get("cross_validation_folds", 5)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        aucs = []
        for tr_idx, val_idx in skf.split(X, y):
            pos_w = float((y.iloc[tr_idx] == 0).sum()) / float((y.iloc[tr_idx] == 1).sum() + 1e-6)
            params = {**self.cfg.get("params", {}), "scale_pos_weight": pos_w,
                      "random_state": seed, "use_label_encoder": False, "verbosity": 0}
            m = XGBClassifier(**params)
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx], verbose=False)
            p = m.predict_proba(X.iloc[val_idx])[:, 1]
            aucs.append(roc_auc_score(y.iloc[val_idx], p))
        return aucs

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Score accounts and assign a risk tier for each.

        Aligns ``features`` to the columns seen during training (filling any
        missing columns with 0), sanitises the data, and runs the trained
        classifier to obtain churn probabilities.  Each account is then
        assigned a ``risk_level`` of ``"high"``, ``"medium"``, or ``"low"``
        based on the thresholds from the model config.

        Args:
            features: A ``pd.DataFrame`` of feature values, one row per
                account.  The index is expected to contain account IDs and is
                preserved in the output.

        Returns:
            A ``pd.DataFrame`` indexed by ``account_id`` with two columns:

            - ``"churn_probability"`` (float) — predicted probability of churn
              in the range [0, 1].
            - ``"risk_level"`` (str) — ``"high"`` if probability >=
              ``high_risk_threshold`` (default 0.70), ``"medium"`` if >=
              ``medium_risk_threshold`` (default 0.40), else ``"low"``.

        Raises:
            RuntimeError: If :meth:`train` or :meth:`load_cache` has not been
                called before this method.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        X = features.reindex(columns=self.feature_names, fill_value=0)
        X = self._sanitize(X)
        proba = self.model.predict_proba(X)[:, 1]
        high_t = self.cfg.get("high_risk_threshold", 0.70)
        med_t = self.cfg.get("medium_risk_threshold", 0.40)
        risk_level = np.where(proba >= high_t, "high",
                     np.where(proba >= med_t, "medium", "low"))
        return pd.DataFrame(
            {"account_id": features.index, "churn_probability": proba, "risk_level": risk_level}
        ).set_index("account_id")

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Return the top features ranked by XGBoost's built-in importance scores.

        Uses the ``feature_importances_`` attribute of the fitted
        ``XGBClassifier``, which reports gain-based importance by default.

        Args:
            top_n: Maximum number of features to return, sorted descending by
                importance score.  Defaults to 20.

        Returns:
            A ``pd.DataFrame`` with columns:

            - ``"feature"`` (str) — feature column name.
            - ``"importance"`` (float) — XGBoost importance score.

            Rows are sorted from most to least important and the index is
            reset to 0-based integers.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")
        scores = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_names, "importance": scores})
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _save_cache(self, features: pd.DataFrame) -> None:
        """Serialise the trained model, feature matrix, and metadata to disk.

        Writes three files whose paths are resolved from the ``"cache"``
        sub-dict of the model config:

        1. ``model_path`` (joblib) — the ``XGBClassifier``, feature names,
           and metrics dict bundled in a single dict.
        2. ``features_path`` (joblib, ``.joblib`` extension forced) — the raw
           feature ``DataFrame`` passed to :meth:`train`, for later
           explainability use.
        3. ``meta_path`` (YAML) — human-readable summary of metrics and
           feature names.

        Parent directories are created automatically if they do not exist.

        Args:
            features: The original (pre-sanitisation) feature ``DataFrame``
                that was passed to :meth:`train`, saved so that the
                :class:`Explainer` can reuse it without re-running the
                feature-engineering pipeline.
        """
        cache = self.cfg.get("cache", {})
        model_path = self.root / cache.get("model_path", "models/churn_model.joblib")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names,
                     "metrics": self.metrics}, model_path)
        features_path = self.root / cache.get("features_path", "models/features.parquet")
        features_path = features_path.with_suffix(".joblib")
        joblib.dump(features, features_path)
        meta_path = self.root / cache.get("meta_path", "models/meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump({"metrics": self.metrics, "feature_names": self.feature_names}, f)
        logger.info("Model cached to %s", model_path)

    def load_cache(self) -> bool:
        """Attempt to restore a previously saved model from the joblib cache.

        Reads the path from ``cfg["cache"]["model_path"]`` and, if the file
        exists, deserialises the ``XGBClassifier``, feature names, and metrics
        into the current instance so that :meth:`predict` and
        :meth:`feature_importance` can be called without re-training.

        Returns:
            ``True`` if the cache was found and loaded successfully.
            ``False`` if caching is disabled in the config, the cache file
            does not exist, or deserialisation raised an exception (in which
            case a warning is logged and the instance is left unchanged).
        """
        cache = self.cfg.get("cache", {})
        if not cache.get("enabled", True):
            return False
        model_path = self.root / cache.get("model_path", "models/churn_model.joblib")
        if not model_path.exists():
            return False
        try:
            data = joblib.load(model_path)
            self.model = data["model"]
            self.feature_names = data["feature_names"]
            self.metrics = data["metrics"]
            logger.info("Loaded cached model from %s", model_path)
            return True
        except Exception as exc:
            logger.warning("Failed to load cached model: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _sanitize(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean a feature matrix so it is safe to pass to XGBoost.

        Performs three operations in order:

        1. Replaces ``+inf`` and ``-inf`` with ``NaN``.
        2. Fills remaining ``NaN`` values with the column-wise median
           (numeric columns only).
        3. Label-encodes every ``object`` or ``category`` column using a
           ``LabelEncoder`` stored in ``self.label_encoders``.  On first
           encounter a new encoder is fitted; on subsequent calls the
           existing encoder is reused, mapping any unseen category strings
           to the sentinel ``"__unknown__"``.

        Args:
            X: Feature ``pd.DataFrame`` to sanitise.  Modified in-place and
               also returned.

        Returns:
            The sanitised ``pd.DataFrame`` with all columns in numeric dtype,
            free of infinities and NaNs, and categorical columns integer-
            encoded.
        """
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True))
        for col in X.select_dtypes(include=["object", "category"]).columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                known = set(self.label_encoders[col].classes_)
                X[col] = X[col].apply(lambda v: v if v in known else "__unknown__")
                if "__unknown__" not in known:
                    self.label_encoders[col].classes_ = np.append(
                        self.label_encoders[col].classes_, "__unknown__"
                    )
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        return X
