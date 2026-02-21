"""SHAP analysis of the global federated model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import (
    FEATURE_COLS, TARGET_COL, XGBOOST_PARAMS, PARTITIONED_DIR
)

# Expected velocity features to appear in top SHAP importance
EXPECTED_TOP_FEATURES = {"tx_count_30m", "tx_count_2h", "amt_zscore"}

GLOBAL_MODEL_PATH = Path(__file__).parent.parent / "workspace" / "server" / "run_1" / "app_server" / "global_fraud_model.json"
OUTPUT_PLOT = Path(__file__).parent / "shap_summary.png"


def load_test_data(bank: str = "a") -> tuple[np.ndarray, np.ndarray, list[str]]:
    src = PARTITIONED_DIR / f"bank_{bank}_engineered.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found")

    df = pd.read_csv(src)
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].fillna(0.0)
    y = df[TARGET_COL].astype(int)

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_test.values, y_test.values, feature_cols


def load_global_model(feature_cols: list[str], X_sample: np.ndarray) -> XGBClassifier:
    """Load the global model trained by federated learning."""
    model = XGBClassifier(**XGBOOST_PARAMS)
    # Fit on a tiny dummy sample to initialise sklearn wrapper
    model.fit(X_sample[:2], np.array([0, 1]))
    if GLOBAL_MODEL_PATH.exists():
        model.get_booster().load_model(str(GLOBAL_MODEL_PATH))
        print(f"Loaded global model from {GLOBAL_MODEL_PATH}")
    else:
        print(f"WARNING: Global model not found at {GLOBAL_MODEL_PATH}")
        print("Using untrained model structure for SHAP demonstration.")
    return model


def run_shap_analysis(model: XGBClassifier, X_test: np.ndarray, feature_cols: list[str]) -> None:
    explainer = shap.TreeExplainer(model)

    # Use a sample of at most 2000 rows for speed
    sample = X_test[:2000] if len(X_test) > 2000 else X_test
    shap_values = explainer.shap_values(sample)

    # Identify top-10 features by mean |SHAP|
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:10]
    top_features = {feature_cols[i] for i in top_idx}

    print("\nTop-10 features by mean |SHAP|:")
    for rank, i in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {feature_cols[i]:30s}  {mean_abs_shap[i]:.6f}")

    found = EXPECTED_TOP_FEATURES & top_features
    missing = EXPECTED_TOP_FEATURES - top_features
    if missing:
        print(f"\nWARNING: Velocity/Z-score features not in top 10: {missing}")
    else:
        print(f"\nAll expected features in top 10: {found}")

    # Summary plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        sample,
        feature_names=feature_cols,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    print(f"\nSHAP summary plot saved → {OUTPUT_PLOT}")


def main() -> None:
    print("Loading test data (bank_a)...")
    X_test, _, feature_cols = load_test_data("a")

    model = load_global_model(feature_cols, X_test)
    run_shap_analysis(model, X_test, feature_cols)


if __name__ == "__main__":
    main()
