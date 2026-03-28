"""Train and evaluate a local XGBoost baseline for one bank partition."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import FEATURE_COLS, TARGET_COL, XGBOOST_PARAMS, PARTITIONED_DIR
from utils.metrics import (
    compute_metrics, compute_metrics_at_threshold,
    find_optimal_threshold, save_metrics,
)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return only feature columns that exist in the dataframe."""
    return [c for c in FEATURE_COLS if c in df.columns]


def run_baseline(bank: str) -> None:
    src = PARTITIONED_DIR / f"bank_{bank}_engineered.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found — run feature_engineering.py first")

    print(f"Loading {src}...")
    df = pd.read_csv(src)

    feature_cols = get_feature_cols(df)
    X = df[feature_cols].fillna(0.0)
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}  Fraud rate test: {y_test.mean():.4f}")

    local_params = {
        **XGBOOST_PARAMS,
        "scale_pos_weight": int((y_train == 0).sum() / max((y_train == 1).sum(), 1)),
    }
    model = XGBClassifier(**local_params, early_stopping_rounds=30)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print(f"  Best iteration: {model.best_iteration}")

    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics at default threshold (0.5)
    y_pred_default = model.predict(X_test)
    metrics_default = compute_metrics(y_test.values, y_pred_default, y_prob)

    # Find optimal threshold and compute metrics there
    opt_thresh, opt_f1 = find_optimal_threshold(y_test.values, y_prob)
    metrics_optimal = compute_metrics_at_threshold(y_test.values, y_prob, opt_thresh)

    print(f"\nBank {bank.upper()} — Default threshold (0.5):")
    for k, v in metrics_default.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")

    print(f"\nBank {bank.upper()} — Optimal threshold ({opt_thresh:.2f}):")
    for k, v in metrics_optimal.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")
    print(f"  confusion_matrix: {metrics_optimal['confusion_matrix']}")

    # Save both metric sets
    combined = {
        "default_threshold": metrics_default,
        "optimal_threshold": metrics_optimal,
    }
    metrics_path = PARTITIONED_DIR / f"bank_{bank}_baseline_metrics.json"
    save_metrics(combined, metrics_path)
    print(f"\nMetrics saved → {metrics_path}")

    model_path = PARTITIONED_DIR / f"bank_{bank}_local_model.json"
    model.save_model(model_path)
    print(f"Model saved → {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True, choices=["a", "b", "c"])
    args = parser.parse_args()
    run_baseline(args.bank)


if __name__ == "__main__":
    main()
