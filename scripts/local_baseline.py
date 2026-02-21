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
from utils.metrics import compute_metrics, save_metrics


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

    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.values, y_pred, y_prob)

    print(f"\nBank {bank.upper()} baseline metrics:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")
    print(f"  confusion_matrix: {metrics['confusion_matrix']}")

    metrics_path = PARTITIONED_DIR / f"bank_{bank}_baseline_metrics.json"
    save_metrics(metrics, metrics_path)
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
