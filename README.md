# ChurnGuard Analytics

Predictive churn analytics platform with a multi-agent Python backend and an interactive desktop UI (tkinter).

## Architecture

```
7 agents, each with a single responsibility:

Orchestrator           — owns shared state, routes messages between agents
DataPipelineAgent      — loads CSVs, engineers features (schema-driven, config-only changes for Phase 2)
ModelAgent             — XGBoost training, prediction, SHAP explanations
AnalyticsAgent         — summaries, top-risk lookups, account details
ConversationAgent      — parses natural-language queries, formats responses
CurveballAgent         — runs adversarial test scenarios, reports impact
```

All column names and data source paths live in `config/schema.yaml` — not in agent code. Adding a new Phase 2 data source requires only a YAML entry.

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python main.py
```

The first run trains the XGBoost model (~30–60 seconds). Subsequent runs load from cache in `models/`.

**Note:** `tkinter` is required and is included with most Python distributions. If it is missing on Linux, install it with `sudo apt-get install python3-tk`.

## Usage

### Launch the app
```bash
python main.py                   # normal startup (uses cached model if available)
python main.py --retrain         # force model retrain
python main.py --headless        # pipeline only, no UI (useful for testing)
```

### Dashboard controls
| Control | Action |
|---------|--------|
| `Ctrl+R` | Refresh dashboard |
| Click row in Risk Table | Explain that account in chat |
| `📊 Model` button | Open model performance window (ROC, PR, confusion matrix) |
| `🔍 Diagnose` button | Open feature leakage diagnostics |
| `⚙ Features` button | Exclude leaky features and retrain |

### Chat queries (natural language)
```
Why is ACC000123 predicted to churn?
What are the top 3 factors for ACC000456?
Churn probability for ACC000789
Show top 10 at-risk accounts
Which features are most important?
How is the model performing?
Portfolio summary
```

### Slash commands
```
/help                            — show all commands
/curveball list                  — list test scenarios
/curveball column_rename         — schema drift: CSV column renamed
/curveball new_source            — add synthetic NPS data source
/curveball churn_redefine        — expand churn to include suspended accounts
/curveball null_injection        — inject 30% nulls into key fields
/curveball new_tier              — introduce unseen 'platinum' tier
/curveball drop_column           — remove integration_count column
/curveball scale_test            — double dataset to ~10,000 accounts
/curveball random                — run a random scenario
/retrain                         — force model retrain
/reload                          — reload data from disk + retrain
```

### CLI curveball runner
```bash
python main.py --curveball list
python main.py --curveball column_rename
python main.py --curveball random
```

## Phase 2 Adaptation Guide

**New data source:** Add a block to `config/schema.yaml` under `data_sources`. No code changes needed for numeric/boolean sources. For custom aggregation, add a `_agg_<source_name>` method to `core/feature_engineer.py`.

**Rename a column:** Update the mapping in `config/schema.yaml` `columns:` section (`logical_name: raw_name`).

**Change churn definition:** Update `target.positive_value` or `target.extra_positive_values` in `config/schema.yaml`.

**New model type:** Implement a model class following the `ChurnModel` interface in `core/model.py`. Swap in `agents/model_agent.py`.

## Project Structure

```
quicksilver/
├── main.py                          # entry point
├── requirements.txt
├── generate_report.py               # Word document report builder
├── config/
│   ├── schema.yaml                  # data sources, column mappings, target definition
│   └── model_config.yaml            # hyperparameters, thresholds, cache paths
├── core/
│   ├── config.py                    # YAML loader
│   ├── data_loader.py               # schema-driven CSV loading
│   ├── feature_engineer.py          # feature engineering pipeline
│   ├── model.py                     # XGBoost training + prediction
│   ├── explainer.py                 # TreeSHAP explanations
│   └── interpreter.py               # plain-English metric/probability interpretation
├── agents/
│   ├── base_agent.py
│   ├── orchestrator.py              # shared state + message routing
│   ├── data_pipeline_agent.py
│   ├── model_agent.py
│   ├── analytics_agent.py
│   ├── conversation_agent.py
│   └── curveball_agent.py
├── curveballs/
│   └── scenarios.py                 # 7 adversarial test scenarios
├── ui/
│   ├── app.py                       # main tkinter window (dashboard + chat)
│   ├── plots.py                     # matplotlib chart helpers
│   ├── model_performance_window.py  # ROC/PR/confusion matrix window
│   ├── diagnostics_window.py        # feature leakage audit window
│   └── feature_settings_dialog.py   # feature exclusion + retrain dialog
├── data/                            # CSV input files (not committed)
│   ├── account_lifecycle_events.csv
│   ├── user_engagement_metrics.csv
│   └── support_interaction_history.csv
└── models/                          # saved model artifacts (auto-created on first run)
```

## Data

The three CSV files should be placed in the `data/` directory (or update `data_dir` in `config/schema.yaml`):
- `account_lifecycle_events.csv`
- `user_engagement_metrics.csv`
- `support_interaction_history.csv`

## Model

- **Algorithm:** XGBoost with `scale_pos_weight` for class imbalance
- **Validation:** 5-fold stratified cross-validation + 80/20 holdout
- **Explanations:** TreeSHAP via XGBoost's native `pred_contribs` (pre-computed for all accounts)
- **Target:** `account_status == 'churned'`

## Known Data Quality Issue

`account_health_score` in the source data has a hard-engineered gap (churned accounts: 0–40, active accounts: 60–100, zero overlap). This causes AUC = 1.00 when the feature is included. The diagnostics window (`🔍 Diagnose`) surfaces this, and the feature settings dialog (`⚙ Features`) lets you exclude it and retrain. The technical report covers this in detail.
