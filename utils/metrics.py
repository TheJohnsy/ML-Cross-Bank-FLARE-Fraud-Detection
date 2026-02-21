import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    f1_score, average_precision_score, precision_score, recall_score, confusion_matrix
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
