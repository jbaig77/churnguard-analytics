"""Headless integration test — runs the full pipeline without launching the UI.

Use this during Phase 2 to validate changes quickly after any modification:
    python test_headless.py

Exit codes: 0 = pass, 1 = fail.
"""
import sys
import time
import logging

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def main() -> int:
    t0 = time.time()
    print("ChurnGuard — headless pipeline test")
    print("=" * 50)

    try:
        from main import build_orchestrator
        from core.config import load_config
    except Exception as exc:
        print(f"FAIL  import error: {exc}")
        return 1

    # ── 1. Build orchestrator and load data ──────────────────────────
    print("Loading data…", end=" ", flush=True)
    try:
        config = load_config()
        orch = build_orchestrator(config)
        r = orch.send("data_pipeline", "load_all")
        assert r.get("status") == "ok", f"load_all failed: {r}"
        n_sources = len(orch.state.frames)
        n_accounts = len(orch.state.frames.get("accounts", []))
        print(f"OK  ({n_sources} sources, {n_accounts:,} accounts)")
    except Exception as exc:
        print(f"FAIL  {exc}")
        return 1

    # ── 2. Feature engineering ────────────────────────────────────────
    print("Engineering features…", end=" ", flush=True)
    try:
        r = orch.send("data_pipeline", "rebuild_features")
        assert r.get("status") == "ok", f"rebuild_features failed: {r}"
        n_features = len(orch.state.features.columns) if orch.state.features is not None else 0
        print(f"OK  ({n_features} features)")
    except Exception as exc:
        print(f"FAIL  {exc}")
        return 1

    # ── 3. Model training / cache load ───────────────────────────────
    print("Training / loading model…", end=" ", flush=True)
    try:
        r = orch.send("model", "train_or_load", {"force_retrain": False})
        assert r.get("status") == "ok", f"train_or_load failed: {r}"
        m = orch.state.metrics or {}
        excls = orch.config["model"].get("feature_exclusions", [])
        excl_note = f"  (excl: {', '.join(excls)})" if excls else ""
        print(
            f"OK{excl_note}\n"
            f"      AUC={m.get('roc_auc', 0):.4f}  "
            f"F1={m.get('f1', 0):.4f}  "
            f"Recall={m.get('recall', 0):.4f}  "
            f"Precision={m.get('precision', 0):.4f}"
        )
    except Exception as exc:
        print(f"FAIL  {exc}")
        return 1

    # ── 4. Predictions ────────────────────────────────────────────────
    print("Scoring accounts…", end=" ", flush=True)
    try:
        preds = orch.state.predictions
        if preds is None:
            raise RuntimeError("predictions not populated after train_or_load")
        risk_col = "risk_level" if "risk_level" in preds.columns else "risk_tier"
        risk_counts = preds[risk_col].value_counts().to_dict() if risk_col in preds.columns else {}
        print(
            f"OK  ({len(preds):,} accounts)  "
            + "  ".join(f"{k}={v}" for k, v in sorted(risk_counts.items()))
        )
    except Exception as exc:
        print(f"FAIL  {exc}")
        return 1

    # ── 5. Analytics agent ────────────────────────────────────────────
    print("Analytics agent…", end=" ", flush=True)
    try:
        r = orch.send("analytics", "top_risk", {"n": 5})
        assert r.get("status") == "ok", f"top_risk failed: {r}"
        top = r.get("accounts", [])
        print(f"OK  (top-5 returned {len(top)} records)")
    except Exception as exc:
        print(f"FAIL  {exc}")
        return 1

    # ── 6. Conversation agent ─────────────────────────────────────────
    print("Conversation agent…", end=" ", flush=True)
    try:
        r = orch.send("conversation", "query", {"text": "portfolio summary"})
        assert r.get("status") == "ok", f"conversation failed: {r}"
        print("OK")
    except Exception as exc:
        print(f"FAIL  {exc}")
        return 1

    # ── 7. Feature importance ─────────────────────────────────────────
    print("Feature importance…", end=" ", flush=True)
    try:
        r = orch.send("model", "feature_importance", {"top_n": 5})
        assert r.get("status") == "ok", f"feature_importance failed: {r}"
        importance = r.get("importance", r.get("features", []))
        top_feat = importance[0]["feature"] if importance else "n/a"
        print(f"OK  (top feature: {top_feat})")
    except Exception as exc:
        print(f"FAIL  {exc}")
        return 1

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("=" * 50)
    print(f"ALL CHECKS PASSED  ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
