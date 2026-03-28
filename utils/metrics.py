import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    f1_score, average_precision_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score,
)

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Compute F1, AUC-PR, precision, recall, and confusion matrix."""
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_pr": float(average_precision_score(y_true, y_prob)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_samples": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
    }

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Find the threshold that maximises F1-score."""
    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.05, 0.96, 0.01):
        preds = (y_prob >= thresh).astype(int)
        f1 = float(f1_score(y_true, preds, zero_division=0))
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(thresh)
    return best_thresh, best_f1


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    """Compute all metrics at a specific probability threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": threshold,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_pr": float(average_precision_score(y_true, y_prob)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_samples": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
    }


def save_metrics(metrics_dict: dict, path: str | Path) -> None:
    """Save metrics dict as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
