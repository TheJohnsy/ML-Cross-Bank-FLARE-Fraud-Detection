"""Generate all result visualisation plots for the project report."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from config import PARTITIONED_DIR, BANKS

PLOTS_DIR = Path(__file__).parent.parent / "evaluation" / "plots"


def load_metrics() -> dict[str, dict]:
    metrics = {}
    for bank in BANKS:
        path = PARTITIONED_DIR / f"bank_{bank}_baseline_metrics.json"
        if path.exists():
            with open(path) as f:
                raw = json.load(f)
            # Handle nested format (default_threshold / optimal_threshold)
            if "optimal_threshold" in raw:
                metrics[bank] = raw["optimal_threshold"]
            else:
                metrics[bank] = raw
    return metrics


def plot_fraud_rate_distribution(metrics: dict[str, dict]) -> None:
    """Bar chart of fraud rates across banks."""
    banks = [f"Bank {b.upper()}" for b in metrics]
    fraud_rates = [m["n_fraud"] / m["n_samples"] * 100 for m in metrics.values()]
    colors = ["#4CAF50", "#FF9800", "#F44336"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(banks, fraud_rates, color=colors, edgecolor="white", linewidth=1.5)
    for bar, rate in zip(bars, fraud_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{rate:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_ylabel("Fraud Rate (%)", fontsize=12)
    ax.set_title("Non-IID Fraud Rate Distribution Across Banks", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(fraud_rates) * 1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fraud_rate_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fraud_rate_distribution.png")


def plot_metrics_comparison(metrics: dict[str, dict]) -> None:
    """Grouped bar chart comparing F1, AUC-PR, AUC-ROC across banks."""
    metric_names = ["f1", "auc_pr", "auc_roc", "precision", "recall"]
    labels = ["F1-Score", "AUC-PR", "AUC-ROC", "Precision", "Recall"]
    banks = list(metrics.keys())
    colors = ["#4CAF50", "#FF9800", "#F44336"]

    x = np.arange(len(metric_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, bank in enumerate(banks):
        values = [metrics[bank][m] for m in metric_names]
        bars = ax.bar(x + i * width, values, width, label=f"Bank {bank.upper()}",
                      color=colors[i], edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, rotation=0)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Local Baseline Metrics Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved metrics_comparison.png")


def plot_confusion_matrices(metrics: dict[str, dict]) -> None:
    """Side-by-side confusion matrices for all banks."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = ["Greens", "Oranges", "Reds"]

    for idx, (bank, m) in enumerate(metrics.items()):
        cm = np.array(m["confusion_matrix"])
        total = cm.sum()
        ax = axes[idx]

        im = ax.imshow(cm, interpolation="nearest", cmap=colors[idx], alpha=0.7)
        ax.set_title(f"Bank {bank.upper()}", fontsize=13, fontweight="bold")

        for i in range(2):
            for j in range(2):
                pct = cm[i, j] / total * 100
                ax.text(j, i, f"{cm[i, j]:,}\n({pct:.1f}%)", ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() * 0.5 else "black")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Legit", "Fraud"], fontsize=10)
        ax.set_yticklabels(["Legit", "Fraud"], fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Actual", fontsize=11)

    fig.suptitle("Confusion Matrices — Local Baselines", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved confusion_matrices.png")


def plot_data_distribution(metrics: dict[str, dict]) -> None:
    """Stacked bar chart of dataset sizes per bank (fraud vs legit)."""
    banks = [f"Bank {b.upper()}" for b in metrics]
    fraud_counts = [m["n_fraud"] for m in metrics.values()]
    legit_counts = [m["n_samples"] - m["n_fraud"] for m in metrics.values()]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(banks, legit_counts, label="Legitimate", color="#2196F3", edgecolor="white")
    ax.bar(banks, fraud_counts, bottom=legit_counts, label="Fraud", color="#F44336", edgecolor="white")

    for i, (l, f) in enumerate(zip(legit_counts, fraud_counts)):
        ax.text(i, l + f + 200, f"{l + f:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Number of Transactions (Test Set)", fontsize=12)
    ax.set_title("Dataset Size Distribution Across Banks", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "data_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved data_distribution.png")


def plot_shap_top_features() -> None:
    """Horizontal bar chart of SHAP top-10 features (hardcoded from results)."""
    features = [
        ("vg11_std", 0.780),
        ("C13", 0.566),
        ("TransactionAmt", 0.561),
        ("card2", 0.542),
        ("D1", 0.530),
        ("D2", 0.491),
        ("uid_C8_mean", 0.455),
        ("addr1", 0.431),
        ("card5", 0.391),
        ("card1", 0.377),
    ]
    names = [f[0] for f in reversed(features)]
    values = [f[1] for f in reversed(features)]

    # Color engineered features differently
    engineered = {"vg11_std", "uid_C8_mean"}
    colors = ["#F44336" if n in engineered else "#2196F3" for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", ha="left", va="center", fontsize=10)

    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("Top-10 Feature Importance (Global Federated Model)", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#F44336", label="Engineered Features"),
        Patch(facecolor="#2196F3", label="Original Features"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_top_features.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved shap_top_features.png")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()

    if not metrics:
        print("No baseline metrics found. Run local_baseline.py first.")
        return

    print("Generating plots...")
    plot_fraud_rate_distribution(metrics)
    plot_metrics_comparison(metrics)
    plot_confusion_matrices(metrics)
    plot_data_distribution(metrics)
    plot_shap_top_features()
    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
