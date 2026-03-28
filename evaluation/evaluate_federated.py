"""Evaluate the global federated model against local baselines on all banks."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import FEATURE_COLS, TARGET_COL, XGBOOST_PARAMS, PARTITIONED_DIR, BANKS
from utils.metrics import (
    compute_metrics, compute_metrics_at_threshold,
    find_optimal_threshold, save_metrics,
)

DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "workspace" / "sim_run" / "server" / "global_fraud_model.json"


def load_global_model(model_path: Path, n_features: int) -> XGBClassifier:
    if not model_path.exists():
        raise FileNotFoundError(f"Global model not found at {model_path}")

    model = XGBClassifier(**XGBOOST_PARAMS)
    dummy_X = np.zeros((2, n_features), dtype=np.float32)
    dummy_y = np.array([0, 1], dtype=np.int32)
    model.fit(dummy_X, dummy_y)
    model.get_booster().load_model(str(model_path))
    return model


def evaluate_on_bank(model: XGBClassifier, bank: str) -> dict:
    src = PARTITIONED_DIR / f"bank_{bank}_engineered.csv"
    df = pd.read_csv(src)

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].fillna(0.0)
    y = df[TARGET_COL].astype(int)

    # Use the same split as local baseline for fair comparison
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    y_prob = model.predict_proba(X_test)[:, 1]

    # Default threshold
    y_pred_default = (y_prob >= 0.5).astype(int)
    metrics_default = compute_metrics(y_test.values, y_pred_default, y_prob)

    # Optimal threshold
    opt_thresh, _ = find_optimal_threshold(y_test.values, y_prob)
    metrics_optimal = compute_metrics_at_threshold(y_test.values, y_prob, opt_thresh)

    return {
        "default_threshold": metrics_default,
        "optimal_threshold": metrics_optimal,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=Path, default=DEFAULT_MODEL_PATH,
        help="Path to the global federated model",
    )
    args = parser.parse_args()

    # Determine feature count from bank_a
    df_sample = pd.read_csv(PARTITIONED_DIR / "bank_a_engineered.csv", nrows=5)
    feature_cols = [c for c in FEATURE_COLS if c in df_sample.columns]

    print(f"Loading global model from {args.model_path}...")
    model = load_global_model(args.model_path, len(feature_cols))

    all_results = {}
    for bank in BANKS:
        print(f"\nEvaluating on Bank {bank.upper()}...")
        results = evaluate_on_bank(model, bank)
        all_results[bank] = results

        d = results["default_threshold"]
        o = results["optimal_threshold"]
        print(f"  Default (0.50): F1={d['f1']:.4f}  AUC-PR={d['auc_pr']:.4f}  AUC-ROC={d['auc_roc']:.4f}  P={d['precision']:.4f}  R={d['recall']:.4f}")
        print(f"  Optimal ({o['threshold']:.2f}): F1={o['f1']:.4f}  AUC-PR={o['auc_pr']:.4f}  AUC-ROC={o['auc_roc']:.4f}  P={o['precision']:.4f}  R={o['recall']:.4f}")

    # Load local baselines for comparison
    print("\n" + "=" * 70)
    print("COLLABORATIVE UPLIFT COMPARISON (Optimal Thresholds)")
    print("=" * 70)
    print(f"{'Bank':<8} {'Metric':<10} {'Local':<10} {'Federated':<10} {'Uplift':<10}")
    print("-" * 48)

    for bank in BANKS:
        local_path = PARTITIONED_DIR / f"bank_{bank}_baseline_metrics.json"
        if local_path.exists():
            with open(local_path) as f:
                local = json.load(f)
            # Handle both old (flat) and new (nested) metric formats
            if "optimal_threshold" in local:
                local_opt = local["optimal_threshold"]
            else:
                local_opt = local

            fed_opt = all_results[bank]["optimal_threshold"]

            for metric in ["f1", "auc_pr", "auc_roc"]:
                local_val = local_opt[metric]
                fed_val = fed_opt[metric]
                uplift = fed_val - local_val
                sign = "+" if uplift >= 0 else ""
                print(f"  {bank.upper():<6} {metric:<10} {local_val:<10.4f} {fed_val:<10.4f} {sign}{uplift:.4f}")
            print()

    # Save federated metrics
    out_path = Path(__file__).parent / "federated_metrics.json"
    save_metrics(all_results, out_path)
    print(f"Federated metrics saved → {out_path}")


if __name__ == "__main__":
    main()
