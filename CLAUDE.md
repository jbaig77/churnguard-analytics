# ChurnGuard — Claude Code Interview Reference

This file loads automatically into every Claude Code session. Use it during the Phase 2 live interview to answer questions instantly without re-reading the codebase.

---

## 30-Second Orientation

```
CSV files → DataLoader → FeatureEngineer → XGBoost (ChurnModel) → TreeSHAP → UI
```

- All column names and data sources live in `config/schema.yaml` — never hardcoded
- All thresholds and hyperparameters live in `config/model_config.yaml`
- Agents communicate through `orch.send("agent_name", "message", {payload})` — see §Agent API
- `orch.state` is a dataclass holding all live runtime data (frames, features, predictions, metrics)
- The UI never touches data directly — it always goes through the orchestrator

**Run the pipeline headlessly (no UI, ~10s):**
```bash
python test_headless.py
```

**Run the app normally:**
```bash
python main.py
python main.py --retrain    # force retrain
python main.py --headless   # no UI, print summary
```

---

## Extension Point Map — Where to Change What

### 1. Add a new CSV data source

**Only `config/schema.yaml` needed** for numeric/boolean sources:

```yaml
# Add under data_sources:
nps_scores:
  path: "nps_scores.csv"          # file in data/ directory
  type: csv
  primary_key: account_id
  join_key: account_id            # how to join to accounts
  columns:
    account_id: account_id        # logical_name: raw_csv_column_name
    nps_score: nps_score
    survey_response_rate: survey_response_rate
```

The generic aggregator (`core/feature_engineer.py:FeatureEngineer._agg_generic`) handles it automatically — numeric columns → mean, bool strings → true-rate. Output columns prefixed `nps_scores__`.

**If you need custom aggregation** (e.g., weighted scores, percentile bucketing), add a method to `core/feature_engineer.py`:
```python
def _agg_nps_scores(self, df: pd.DataFrame, _name: str) -> pd.DataFrame:
    g = df.groupby("account_id")
    agg = pd.DataFrame(index=g.groups.keys())
    agg["nps_score_mean"] = g["nps_score"].mean()
    agg["nps_promoter_rate"] = g["nps_score"].apply(lambda x: (x >= 50).mean())
    agg.index.name = "account_id"
    return agg
```

Then trigger rebuild: `orch.send("data_pipeline", "rebuild_features")`

---

### 2. Rename a column in a CSV source

In `config/schema.yaml`, update the column mapping (logical_name → raw_csv_name):

```yaml
# Before: account_health_score: account_health_score
# After CSV renames the column to "health_index":
account_health_score: health_index
```

No code changes needed. DataLoader (`core/data_loader.py:DataLoader._apply_column_mapping`) handles it.

---

### 3. Redefine churn (change what counts as "churned")

In `config/schema.yaml`:

```yaml
target:
  source: accounts
  column: account_status
  positive_value: churned
  extra_positive_values: ["suspended"]   # add any extra positive labels here
```

Then rebuild: `orch.send("data_pipeline", "rebuild_features")` + retrain.

---

### 4. Exclude a feature from training (remove leakage)

Option A — via `config/model_config.yaml`:
```yaml
model:
  feature_exclusions: ["account_health_score", "risk_flag"]
```

Option B — programmatically:
```python
orch.config["model"]["feature_exclusions"] = ["account_health_score"]
orch.send("data_pipeline", "rebuild_features")
orch.send("model", "train_or_load", {"force_retrain": True})
orch.send("model", "explain_all")
orch._build_interpreter()
```

The exclusion is applied in `core/feature_engineer.py:FeatureEngineer.build()` at the `exclude` parameter.

---

### 5. Change risk tier thresholds

In `config/model_config.yaml`:
```yaml
model:
  high_risk_threshold: 0.70    # P(churn) >= this → "high"
  medium_risk_threshold: 0.20  # P(churn) >= this → "medium", else "low"
```

Takes effect on next `orch.send("model", "predict")` or full retrain.

---

### 6. Add a new chat query type

File: `agents/conversation_agent.py`

Find `_dispatch()` method and add a new condition in the if/elif chain:
```python
# Around line 120 — add before the final fallback
if "average mrr" in text and "at-risk" in text:
    return self._handle_avg_mrr_at_risk()
```

Add the handler method:
```python
def _handle_avg_mrr_at_risk(self) -> dict:
    preds = self.orch.state.predictions
    if preds is None:
        return self._ok(response="No predictions available yet.")
    at_risk = preds[preds["risk_tier"].isin(["high", "medium"])]
    avg_mrr = at_risk["mrr"].mean() if "mrr" in at_risk.columns else "n/a"
    return self._ok(response=f"Average MRR for at-risk accounts: ${avg_mrr:,.0f}")
```

---

### 7. Add a new slash command

File: `agents/conversation_agent.py`, in `_dispatch()`:
```python
if tl == "/export":
    return self._cmd_export()
```

```python
def _cmd_export(self) -> dict:
    import csv, io
    preds = self.orch.state.predictions
    if preds is None:
        return self._ok(response="No predictions to export.")
    top = preds.nlargest(100, "churn_probability")[["account_id", "churn_probability", "risk_tier"]]
    path = "exports/top_100_at_risk.csv"
    import os; os.makedirs("exports", exist_ok=True)
    top.to_csv(path, index=False)
    return self._ok(response=f"Exported top 100 at-risk accounts to {path}")
```

---

### 8. Swap the model type (e.g., LogisticRegression)

File: `core/model.py` — the `ChurnModel` class.

The XGBoost classifier is instantiated at line ~80:
```python
self._clf = xgb.XGBClassifier(**params)
```

To swap in LogisticRegression:
```python
from sklearn.linear_model import LogisticRegression
self._clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
```

Note: SHAP TreeExplainer will break for non-tree models. In `core/explainer.py`, the explainer uses:
```python
self._ex = model.get_booster().predict(dmat, pred_contribs=True)
```
For non-tree models, replace with `shap.LinearExplainer` or disable SHAP and return zeros.

---

### 9. Add a new derived feature to the account features

File: `core/feature_engineer.py:FeatureEngineer._build_account_features()` (~line 130)

Add a new column to `feats`:
```python
# MRR per seat
if "mrr" in acc.columns and "seats_active" in acc.columns:
    feats["mrr_per_seat"] = acc["mrr"] / acc["seats_active"].replace(0, np.nan)
```

---

### 10. Export predictions / add a button to the UI

File: `ui/app.py` — find the toolbar frame, add a button:
```python
ttk.Button(toolbar, text="⬇ Export", command=self._export_predictions).pack(side=tk.LEFT, padx=4)
```

Add the method:
```python
def _export_predictions(self) -> None:
    preds = self.orch.state.predictions
    if preds is None:
        return
    path = "exports/predictions.csv"
    import os; os.makedirs("exports", exist_ok=True)
    preds.to_csv(path, index=False)
    self._set_status(f"Exported predictions to {path}")
```

---

## Agent Message API

```python
orch.send("data_pipeline", "load_all")                          # reload CSVs from disk
orch.send("data_pipeline", "rebuild_features")                  # re-run feature engineering
orch.send("data_pipeline", "add_source", {"name": "nps", ...})  # add source at runtime

orch.send("model", "train_or_load", {"force_retrain": False})   # load cache or train
orch.send("model", "train")                                      # force train
orch.send("model", "explain_all")                                # compute all SHAP values
orch.send("model", "explain_account", {"account_id": "ACC001"}) # single account SHAP
orch.send("model", "feature_importance", {"top_n": 10})         # top N features
orch.send("model", "predict")                                    # re-score all accounts

orch.send("analytics", "top_risk", {"n": 10})                   # top N at-risk accounts
orch.send("analytics", "summary")                                # portfolio summary
orch.send("analytics", "account_detail", {"account_id": "X"})  # account details

orch.send("conversation", "query", {"text": "top 10 at-risk"})  # natural language query
```

---

## Shared State (`orch.state`)

```python
orch.state.frames         # dict[str, pd.DataFrame] — raw loaded DataFrames keyed by source name
orch.state.features       # pd.DataFrame — engineered feature matrix (account_id index)
orch.state.target         # pd.Series — binary 0/1 churn labels
orch.state.predictions    # pd.DataFrame — columns: account_id, churn_probability, risk_tier, ...
orch.state.shap_values    # np.ndarray shape (n_accounts, n_features)
orch.state.feature_names  # list[str] — feature column names matching shap_values columns
orch.state.metrics        # dict — roc_auc, f1, recall, precision, accuracy, pr_auc, cv_auc_mean
orch.state.model          # ChurnModel instance
orch.state.is_ready       # bool — False during retrain; gates dashboard refresh
```

---

## Curveball Playbook

The interviewer will likely pick from one of these patterns. Resolution time in brackets.

### A. "Here's a new CSV file — add it as a data source" [5 min]
1. Put the file in `data/`
2. Add a block to `config/schema.yaml` under `data_sources` (see §1 above)
3. Run `python main.py --retrain` or use the UI's `/reload` command
4. If custom aggregation needed: add `_agg_<source_name>` to `core/feature_engineer.py`

### B. "The column name in the CSV changed" [2 min]
1. Update `config/schema.yaml` → `data_sources → accounts → columns → <logical>: <new_raw_name>`
2. Reload data

### C. "Include suspended accounts as churned" [2 min]
1. Update `config/schema.yaml` → `target → extra_positive_values: ["suspended"]`
2. Retrain

### D. "Remove this feature — it looks like it's leaking" [2 min]
1. Add to `config/model_config.yaml` → `model → feature_exclusions: ["feature_name"]`
2. Or use the ⚙ Features button in the UI
3. Retrain

### E. "Can you add a query for X?" [10 min]
1. Add elif branch in `agents/conversation_agent.py:ConversationAgent._dispatch()`
2. Implement handler method returning `self._ok(response="...")`

### F. "Can you export predictions to CSV?" [5 min]
1. Add button in `ui/app.py` toolbar
2. Implement method calling `orch.state.predictions.to_csv(...)`

### G. "What happens to the model if we double the dataset?" [1 min]
- Already built: `/curveball scale_test` in the chat or `python main.py --curveball scale_test`

### H. "Show what happens if 30% of data goes missing" [1 min]
- Already built: `/curveball null_injection`

### I. "Change the high-risk threshold to 0.80" [1 min]
1. Update `config/model_config.yaml` → `model → high_risk_threshold: 0.80`
2. `orch.send("model", "predict")` or retrain

### J. "Add a new subscription tier we haven't seen" [1 min]
- Already built: `/curveball new_tier`

---

## Full Pipeline Sequence (for reference)

```python
# What orch.initialize() does, step by step:
orch.send("data_pipeline", "load_all")                           # 1. Load CSVs → orch.state.frames
orch.send("model", "train_or_load", {"force_retrain": False})    # 2. Train or load cache → orch.state.model, metrics
orch.send("model", "explain_all")                                # 3. TreeSHAP → orch.state.shap_values
orch._build_interpreter()                                         # 4. Wire up plain-English interpreter
orch.state.is_ready = True                                       # 5. Signal UI to refresh
```

---

## Key File Index

| File | Responsibility |
|------|---------------|
| `config/schema.yaml` | Data sources, column maps, churn target — **change here first** |
| `config/model_config.yaml` | Thresholds, hyperparameters, feature exclusions |
| `core/feature_engineer.py` | All feature derivation; add `_agg_<source>` for new sources |
| `core/model.py` | XGBoost training, prediction, SHAP integration |
| `core/data_loader.py` | CSV loading + column renaming from schema |
| `agents/orchestrator.py` | Shared state, message routing |
| `agents/data_pipeline_agent.py` | Handles `load_all`, `rebuild_features`, `add_source` |
| `agents/model_agent.py` | Handles `train_or_load`, `explain_all`, `feature_importance` |
| `agents/conversation_agent.py` | NL query parsing + slash commands |
| `curveballs/scenarios.py` | 7 adversarial test scenarios with apply/rollback |
| `ui/app.py` | Main tkinter window, dashboard, chat |
| `test_headless.py` | Fast end-to-end pipeline test, no UI required |
