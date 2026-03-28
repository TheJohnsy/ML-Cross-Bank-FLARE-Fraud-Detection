"""Streamlit dashboard for the Cross-Bank Federated Fraud Detection project."""
import json
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["figure.dpi"] = 200

BASE_DIR = Path(__file__).parent
PARTITIONED_DIR = BASE_DIR / "data" / "partitioned"
EVAL_DIR = BASE_DIR / "evaluation"
PLOTS_DIR = EVAL_DIR / "plots"
BANKS = ["a", "b", "c"]
BANK_LABELS = {"a": "Bank A", "b": "Bank B", "c": "Bank C"}
BANK_BG = {
    "a": "rgba(46,204,113,0.07)",
    "b": "rgba(52,152,219,0.07)",
    "c": "rgba(231,76,60,0.07)",
}
BANK_BORDER = {
    "a": "rgba(46,204,113,0.25)",
    "b": "rgba(52,152,219,0.25)",
    "c": "rgba(231,76,60,0.25)",
}
BANK_CMAP = {"a": "Greens", "b": "Oranges", "c": "Reds"}
BANK_BAR_COLOR = {"a": "#4CAF50", "b": "#FF9800", "c": "#F44336"}

st.set_page_config(page_title="Federated Fraud Detection", page_icon="F", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 10px 14px;
    }
    [data-testid="stMetric"] label {
        font-size: 0.78rem !important;
        color: rgba(255,255,255,0.55) !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; border-radius: 8px 8px 0 0; }
    h1 { margin-bottom: 0 !important; }
    h2 { margin-top: 1.2rem !important; }
    h3 { margin-top: 0.8rem !important; }
    .bank-container { border-radius: 12px; padding: 18px; margin-bottom: 8px; }
    .finding-card { border-radius: 0 8px 8px 0; padding: 14px 16px; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)


def bank_header(bank: str) -> None:
    st.markdown(
        f'<div class="bank-container" style="background:{BANK_BG[bank]}; '
        f'border: 1px solid {BANK_BORDER[bank]};"><strong>{BANK_LABELS[bank]}</strong></div>',
        unsafe_allow_html=True,
    )


# ── Data loaders ──
@st.cache_data
def load_bank_metrics():
    metrics = {}
    for bank in BANKS:
        path = PARTITIONED_DIR / f"bank_{bank}_baseline_metrics.json"
        if path.exists():
            with open(path) as f:
                raw = json.load(f)
            if "optimal_threshold" in raw:
                metrics[bank] = raw["optimal_threshold"]
                metrics[f"{bank}_default"] = raw["default_threshold"]
            else:
                metrics[bank] = raw
    return metrics


@st.cache_data
def load_partition_stats():
    stats = {}
    for bank in BANKS:
        path = PARTITIONED_DIR / f"bank_{bank}.csv"
        if path.exists():
            df = pd.read_csv(path, usecols=["isFraud", "TransactionAmt"])
            stats[bank] = {
                "total_rows": len(df),
                "fraud_count": int(df["isFraud"].sum()),
                "fraud_rate": float(df["isFraud"].mean()),
                "avg_amount": float(df["TransactionAmt"].mean()),
                "median_amount": float(df["TransactionAmt"].median()),
            }
    return stats


@st.cache_data
def load_federated_metrics():
    path = EVAL_DIR / "federated_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ── Plot helpers ──
def plot_single_confusion_matrix(cm: np.ndarray, title: str, cmap: str) -> plt.Figure:
    total = cm.sum()
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(cm, cmap=cmap, alpha=0.65)
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            ax.text(j, i, f"{cm[i, j]:,}\n({pct:.1f}%)", ha="center", va="center",
                    fontsize=13, fontweight="bold", color="black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit", "Fraud"], fontsize=11)
    ax.set_yticklabels(["Legit", "Fraud"], fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_single_bar(banks_data: dict, metric_key: str, title: str, ylabel: str, fmt: str = ".4f") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    names = [BANK_LABELS[b] for b in banks_data]
    vals = [banks_data[b][metric_key] for b in banks_data]
    colors = [BANK_BAR_COLOR[b] for b in banks_data]
    bars = ax.bar(names, vals, color=colors, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                f"{v:{fmt}}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_metrics_grouped(banks_data: dict) -> plt.Figure:
    metric_names = ["f1", "auc_pr", "auc_roc", "precision", "recall"]
    labels = ["F1", "AUC-PR", "AUC-ROC", "Precision", "Recall"]
    x = np.arange(len(metric_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, bank in enumerate(banks_data):
        values = [banks_data[bank][m] for m in metric_names]
        bars = ax.bar(x + i * width, values, width, label=BANK_LABELS[bank],
                      color=BANK_BAR_COLOR[bank], edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Local Baseline Metrics Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_shap_bars(features: list) -> plt.Figure:
    names = [f[0] for f in reversed(features)]
    values = [f[1] for f in reversed(features)]
    engineered = {f[0] for f in features if f[2]}
    colors = ["#E74C3C" if n in engineered else "#3498DB" for n in names]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", ha="left", va="center", fontsize=10)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("Top-10 Feature Importance (Global Federated Model)", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#E74C3C", label="Engineered"),
        Patch(facecolor="#3498DB", label="Original"),
    ], loc="lower right", fontsize=10)
    fig.tight_layout()
    return fig


# ── Header ──
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown("# Cross-Bank Federated Fraud Detection")
    st.markdown("*Privacy-preserving collaborative model training using* ***NVIDIA FLARE*** *and* ***XGBoost***")
with h2:
    st.markdown("")
    st.markdown("")
    total_stats = load_partition_stats()
    if total_stats:
        total_tx = sum(s["total_rows"] for s in total_stats.values())
        total_fraud = sum(s["fraud_count"] for s in total_stats.values())
        st.metric("Total Transactions", f"{total_tx:,}")
        st.metric("Total Fraud Cases", f"{total_fraud:,}")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Pipeline Overview", "Data & Partitions", "Local Baselines", "Federated Training", "SHAP Explainability",
])

# ═══════════════════════════════════════════════════════
# Tab 1
# ═══════════════════════════════════════════════════════
with tab1:
    st.markdown("### System Architecture")
    st.markdown("""
    This project simulates a real-world scenario where **three banks** want to collaboratively
    detect fraud without sharing their raw transaction data. Each bank trains locally on its
    own data, and a federated learning protocol coordinates the training across all banks.
    """)

    col1, col2, col3 = st.columns(3)
    phase_styles = [
        ("Phase 1: Local Foundations", "rgba(46,204,113,0.06)", "rgba(46,204,113,0.2)"),
        ("Phase 2: Federated Training", "rgba(52,152,219,0.06)", "rgba(52,152,219,0.2)"),
        ("Phase 3: Evaluation", "rgba(231,76,60,0.06)", "rgba(231,76,60,0.2)"),
    ]
    phase_content = [
        "- **Data partitioning** into 3 non-IID banks\n- **Feature engineering**: Z-scores, velocity, V-groups, UID aggregations\n- **Local XGBoost baselines** per bank",
        "- **Cyclic model passing** (no raw data shared)\n- Each bank **warm-starts** from the global model\n- **15 rounds** of collaborative training",
        "- **SHAP** feature importance analysis\n- **Local vs federated** comparison\n- **Threshold optimization**",
    ]
    for col, (title, bg, border), content in zip([col1, col2, col3], phase_styles, phase_content):
        with col:
            st.markdown(f'<div class="bank-container" style="background:{bg}; border:1px solid {border};"><h4>{title}</h4></div>', unsafe_allow_html=True)
            st.markdown(content)

    st.divider()
    st.markdown("### Feature Engineering Pipeline")
    st.info("All features are computed **locally within each bank's data boundary** -- no raw data crosses bank boundaries at any point.")

    fe_cols = st.columns(4)
    fe_data = [
        ("Z-Score Features", "1", "Per-card spending deviation: how unusual is this transaction compared to cardholder history?"),
        ("Velocity Features", "4", "Transaction count and sum within 30-min and 2-hr sliding windows preceding each transaction."),
        ("V-Group Summaries", "22", "339 Vesta columns reduced to 11 groups (by NaN pattern), each summarised as row-wise mean + std."),
        ("UID Aggregations", "15", "Client groups (card1+addr1) aggregated: 14 C-column means + transaction frequency. Group ID dropped."),
    ]
    for col, (label, val, caption) in zip(fe_cols, fe_data):
        with col:
            st.metric(label, val)
            st.caption(caption)

# ═══════════════════════════════════════════════════════
# Tab 2
# ═══════════════════════════════════════════════════════
with tab2:
    st.markdown("### Data Partitions")
    st.markdown("""
    The **IEEE-CIS Fraud Detection** dataset (590K transactions) is partitioned into 3 banks
    by grouping credit cards (`card1`). Cards with higher fraud rates are assigned disproportionately
    to Bank C, creating a **non-IID** split that simulates real-world conditions where different
    banks serve different customer populations.
    """)
    stats = load_partition_stats()

    if stats:
        cols = st.columns(3)
        for i, bank in enumerate(BANKS):
            s = stats[bank]
            with cols[i]:
                bank_header(bank)
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Transactions", f"{s['total_rows']:,}")
                    st.metric("Fraud Rate", f"{s['fraud_rate']:.2%}")
                    st.metric("Avg Amount", f"${s['avg_amount']:,.0f}")
                with m2:
                    st.metric("Fraud Cases", f"{s['fraud_count']:,}")
                    st.metric("Legit Rate", f"{1 - s['fraud_rate']:.2%}")
                    st.metric("Median Amount", f"${s['median_amount']:,.0f}")

        st.divider()
        st.markdown("#### Fraud Rate Distribution")
        st.caption("The non-IID split ensures each bank has a meaningfully different fraud prevalence.")
        fr_cols = st.columns(3)
        for i, bank in enumerate(BANKS):
            with fr_cols[i]:
                fig, ax = plt.subplots(figsize=(4, 3.5))
                rate = stats[bank]["fraud_rate"] * 100
                bar = ax.bar([BANK_LABELS[bank]], [rate], color=BANK_BAR_COLOR[bank], edgecolor="white", width=0.5)
                ax.text(0, rate + 0.08, f"{rate:.2f}%", ha="center", fontweight="bold", fontsize=13)
                ax.set_ylabel("Fraud Rate (%)")
                ax.set_ylim(0, max(s["fraud_rate"] for s in stats.values()) * 100 * 1.4)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()

        st.markdown("#### Dataset Size")
        st.caption("Bank C has the most data but also the hardest classification problem.")
        ds_cols = st.columns(3)
        for i, bank in enumerate(BANKS):
            with ds_cols[i]:
                s = stats[bank]
                legit = s["total_rows"] - s["fraud_count"]
                fig, ax = plt.subplots(figsize=(4, 3.5))
                ax.bar([BANK_LABELS[bank]], [legit], label="Legitimate", color="#3498DB", edgecolor="white", width=0.5)
                ax.bar([BANK_LABELS[bank]], [s["fraud_count"]], bottom=[legit], label="Fraud", color="#E74C3C", edgecolor="white", width=0.5)
                ax.text(0, legit + s["fraud_count"] + 1000, f"{s['total_rows']:,}", ha="center", fontweight="bold", fontsize=11)
                ax.set_ylabel("Transactions")
                ax.set_ylim(0, max(ss["total_rows"] for ss in stats.values()) * 1.15)
                ax.legend(fontsize=9)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()

# ═══════════════════════════════════════════════════════
# Tab 3
# ═══════════════════════════════════════════════════════
with tab3:
    st.markdown("### Local Baseline Results")
    st.markdown("""
    Each bank trains an **XGBoost classifier independently** on its own data. These baselines
    represent how well each bank can detect fraud using **only its local data** -- before any
    federated collaboration. All metrics below use the **optimal decision threshold**.
    """)
    metrics = load_bank_metrics()

    if metrics:
        with st.expander("What do these metrics mean?", expanded=False):
            st.markdown("""
            | Metric | What it measures | Why it matters |
            |--------|-----------------|----------------|
            | **F1-Score** | Harmonic mean of precision and recall | Balances false positives and missed fraud |
            | **AUC-PR** | Area under precision-recall curve | Model quality for imbalanced data |
            | **AUC-ROC** | Area under ROC curve | Ranking ability |
            | **Precision** | Of fraud alerts, how many are real? | Fewer false alarms |
            | **Recall** | Of actual fraud, how much caught? | Fewer missed cases |
            | **Threshold** | Probability cutoff | Tuned to maximize F1 |
            """)

        bank_metrics = {b: metrics[b] for b in BANKS if b in metrics}

        cols = st.columns(3)
        for i, bank in enumerate(BANKS):
            if bank not in metrics:
                continue
            m = metrics[bank]
            with cols[i]:
                bank_header(bank)
                st.metric("F1-Score", f"{m['f1']:.4f}")
                r1, r2 = st.columns(2)
                with r1:
                    st.metric("AUC-PR", f"{m['auc_pr']:.4f}")
                    st.metric("Precision", f"{m['precision']:.4f}")
                with r2:
                    st.metric("AUC-ROC", f"{m['auc_roc']:.4f}")
                    st.metric("Recall", f"{m['recall']:.4f}")
                if "threshold" in m:
                    st.metric("Optimal Threshold", f"{m['threshold']:.2f}")

        st.divider()
        st.markdown("#### Metrics Comparison")
        st.caption("Bank B achieves the best overall performance. Bank C has the hardest problem.")
        mc_cols = st.columns(3)
        metric_names = ["f1", "auc_pr", "auc_roc", "precision", "recall"]
        metric_labels = ["F1", "AUC-PR", "AUC-ROC", "Precision", "Recall"]
        for i, bank in enumerate(BANKS):
            if bank not in bank_metrics:
                continue
            m = bank_metrics[bank]
            with mc_cols[i]:
                fig, ax = plt.subplots(figsize=(4.5, 3.8))
                vals = [m[k] for k in metric_names]
                bars = ax.bar(metric_labels, vals, color=BANK_BAR_COLOR[bank], edgecolor="white", width=0.6)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
                ax.set_ylim(0, 1.08)
                ax.set_title(BANK_LABELS[bank], fontweight="bold", fontsize=12)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(axis="x", labelsize=9)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()

        st.divider()
        st.markdown("#### Confusion Matrices")
        with st.expander("How to read a confusion matrix", expanded=False):
            st.markdown("""
            | Position | Name | Meaning |
            |----------|------|---------|
            | **Top-left** | True Negatives | Legitimate correctly identified |
            | **Top-right** | False Positives | Legitimate incorrectly flagged |
            | **Bottom-left** | False Negatives | Fraud missed |
            | **Bottom-right** | True Positives | Fraud caught |
            """)

        cm_cols = st.columns(3)
        for i, bank in enumerate(BANKS):
            m = metrics[bank]
            with cm_cols[i]:
                cm = np.array(m["confusion_matrix"])
                fig = plot_single_confusion_matrix(cm, BANK_LABELS[bank], BANK_CMAP[bank])
                st.pyplot(fig)
                plt.close()

        st.divider()
        st.markdown("#### Detailed Metrics")
        rows = []
        for bank in BANKS:
            m = metrics[bank]
            rows.append({
                "Bank": BANK_LABELS[bank],
                "F1": f"{m['f1']:.4f}", "AUC-PR": f"{m['auc_pr']:.4f}",
                "AUC-ROC": f"{m['auc_roc']:.4f}", "Precision": f"{m['precision']:.4f}",
                "Recall": f"{m['recall']:.4f}", "Test Samples": f"{m['n_samples']:,}",
                "Fraud Count": f"{m['n_fraud']:,}",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Bank"), use_container_width=True)

# ═══════════════════════════════════════════════════════
# Tab 4
# ═══════════════════════════════════════════════════════
with tab4:
    st.markdown("### Federated Training")
    st.markdown("""
    In federated learning, banks collaborate to train a shared model **without exchanging raw data**.
    Only model updates (XGBoost tree structures) are communicated. We use **cyclic training** --
    the standard approach for tree-based models in federated settings.
    """)

    with st.expander("Why cyclic training?", expanded=False):
        st.markdown("""
        **FedAvg** averages model parameters -- suitable for neural networks but **incompatible
        with XGBoost** (tree structures can't be averaged). Cyclic training passes the model
        sequentially through each bank, each warm-starting and adding new trees. This is
        NVIDIA's recommended approach for XGBoost FL.
        """)

    st.code("Round N:   Server --> Bank A --> Bank B --> Bank C --> Server\n                    (train)    (train)    (train)", language=None)

    st.divider()
    st.markdown("### Training Configurations")
    st.caption("Two configurations tested. Each local baseline has 400 trees trained on its own data.")

    cfg1, cfg2 = st.columns(2)
    with cfg1:
        st.markdown(
            '<div class="bank-container" style="background:rgba(255,255,255,0.03); '
            'border:1px solid rgba(255,255,255,0.1);"><strong>Config 1: Conservative</strong></div>',
            unsafe_allow_html=True,
        )
        c1a, c1b, c1c = st.columns(3)
        with c1a: st.metric("Rounds", "10")
        with c1b: st.metric("Trees/Round", "20")
        with c1c: st.metric("Total Trees", "600")
        st.caption("Each bank contributes ~200 trees -- half of its local model's 400.")

    with cfg2:
        st.markdown(
            '<div class="bank-container" style="background:rgba(46,204,113,0.06); '
            'border:1px solid rgba(46,204,113,0.2);"><strong>Config 2: Increased Budget</strong></div>',
            unsafe_allow_html=True,
        )
        c2a, c2b, c2c = st.columns(3)
        with c2a: st.metric("Rounds", "15")
        with c2b: st.metric("Trees/Round", "50")
        with c2c: st.metric("Total Trees", "2,250")
        st.caption("Each bank contributes ~750 trees -- nearly 2x its local model's 400.")

    model_path = BASE_DIR / "workspace" / "sim_run" / "server" / "global_fraud_model.json"
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        st.success(f"Global model trained successfully -- Config 2 ({size_mb:.1f} MB, ~2,250 trees)")

    st.divider()
    st.markdown("### Collaborative Uplift: Federated vs Local")
    st.markdown("**Collaborative uplift** = federated metric minus local metric. Positive = federation helped.")

    st.markdown("#### Config 1: 600 Trees")
    config1_data = {
        "a": {"f1": 0.7432, "auc_pr": 0.7617, "auc_roc": 0.9562},
        "b": {"f1": 0.7079, "auc_pr": 0.7492, "auc_roc": 0.9587},
        "c": {"f1": 0.5885, "auc_pr": 0.6127, "auc_roc": 0.9316},
    }
    rows1 = []
    for bank in BANKS:
        if bank in metrics:
            local_f1 = metrics[bank]["f1"]
            fed_f1 = config1_data[bank]["f1"]
            rows1.append({
                "Bank": BANK_LABELS[bank],
                "Local F1": f"{local_f1:.4f}", "Fed F1": f"{fed_f1:.4f}",
                "F1 Uplift": f"{fed_f1 - local_f1:+.4f}",
                "Fed AUC-PR": f"{config1_data[bank]['auc_pr']:.4f}",
                "Fed AUC-ROC": f"{config1_data[bank]['auc_roc']:.4f}",
            })
    if rows1:
        st.dataframe(pd.DataFrame(rows1).set_index("Bank"), use_container_width=True)
    st.caption("All banks show negative uplift -- federated model is capacity-starved with ~200 trees per bank.")

    st.markdown("#### Config 2: 2,250 Trees")
    fed_metrics = load_federated_metrics()
    if fed_metrics and metrics:
        rows2 = []
        for bank in BANKS:
            if bank in fed_metrics and bank in metrics:
                local_f1 = metrics[bank]["f1"]
                local_auc_pr = metrics[bank]["auc_pr"]
                fd = fed_metrics[bank]
                fd = fd.get("optimal_threshold", fd)
                rows2.append({
                    "Bank": BANK_LABELS[bank],
                    "Local F1": f"{local_f1:.4f}", "Fed F1": f"{fd['f1']:.4f}",
                    "F1 Uplift": f"{fd['f1'] - local_f1:+.4f}",
                    "Fed AUC-PR": f"{fd['auc_pr']:.4f}",
                    "AUC-PR Uplift": f"{fd['auc_pr'] - local_auc_pr:+.4f}",
                })
        if rows2:
            st.dataframe(pd.DataFrame(rows2).set_index("Bank"), use_container_width=True)
        st.caption("Gap narrowed 60-70%. Bank A's AUC-PR flipped positive (+0.014) -- data-scarce bank benefits most.")

    # Uplift comparison chart
    st.markdown("#### Uplift Comparison: Config 1 vs Config 2")
    up_cols = st.columns(3)
    for i, bank in enumerate(BANKS):
        with up_cols[i]:
            if bank in metrics and bank in config1_data and fed_metrics and bank in fed_metrics:
                local_f1 = metrics[bank]["f1"]
                c1_uplift = config1_data[bank]["f1"] - local_f1
                fd = fed_metrics[bank].get("optimal_threshold", fed_metrics[bank])
                c2_uplift = fd["f1"] - local_f1

                fig, ax = plt.subplots(figsize=(4.5, 3.5))
                bars = ax.bar(["Config 1\n(600 trees)", "Config 2\n(2,250 trees)"],
                              [c1_uplift, c2_uplift],
                              color=["#E74C3C" if c1_uplift < 0 else "#2ECC71",
                                     "#E74C3C" if c2_uplift < 0 else "#2ECC71"],
                              edgecolor="white", width=0.5)
                for bar, v in zip(bars, [c1_uplift, c2_uplift]):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + (0.003 if v >= 0 else -0.008),
                            f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top",
                            fontsize=11, fontweight="bold")
                ax.axhline(y=0, color="white", linewidth=0.5, alpha=0.3)
                ax.set_ylabel("F1 Uplift")
                ax.set_title(BANK_LABELS[bank], fontweight="bold")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()

    st.divider()
    st.markdown("### Key Findings")
    f1c, f2c = st.columns(2)
    f3c, f4c = st.columns(2)
    findings = [
        (f1c, "Tree Budget is the Bottleneck", "600 to 2,250 trees dramatically narrowed the uplift gap across all banks.", "#3498db"),
        (f2c, "Data-Scarce Banks Benefit Most", "Bank A (1.9% fraud) achieved positive AUC-PR uplift from federation.", "#2ecc71"),
        (f3c, "Non-IID Creates Residual Gap", "Heterogeneous data causes each bank to pull the model toward its distribution.", "#f39c12"),
        (f4c, "Generalisation Potential", "The federated model has seen fraud from all banks and may generalise better to new populations.", "#9b59b6"),
    ]
    for col, title, desc, color in findings:
        with col:
            st.markdown(
                f'<div class="finding-card" style="background:rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08); '
                f'border-left:3px solid {color};">'
                f'<strong>{title}</strong><br/><small>{desc}</small></div>',
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════
# Tab 5
# ═══════════════════════════════════════════════════════
with tab5:
    st.markdown("### SHAP Feature Importance Analysis")
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** measures how much each feature contributes to
    individual predictions. Higher mean |SHAP| = more influence on fraud/legitimate decisions.
    Features in **red** are engineered; **blue** are original.
    """)

    shap_features = [
        ("vg11_std", 0.780, True), ("C13", 0.566, False),
        ("TransactionAmt", 0.561, False), ("card2", 0.542, False),
        ("D1", 0.530, False), ("D2", 0.491, False),
        ("uid_C8_mean", 0.455, True), ("addr1", 0.431, False),
        ("card5", 0.391, False), ("card1", 0.377, False),
    ]

    st.markdown("#### Top-10 Features by Mean |SHAP|")

    def _plot_shap_half(features_slice, title):
        names = [f[0] for f in reversed(features_slice)]
        values = [f[1] for f in reversed(features_slice)]
        engineered = {f[0] for f in features_slice if f[2]}
        colors = ["#E74C3C" if n in engineered else "#3498DB" for n in names]
        fig, ax = plt.subplots(figsize=(5, 3.8))
        bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, values):
            ax.text(val + 0.008, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", ha="left", va="center", fontsize=9.5)
        ax.set_xlabel("Mean |SHAP Value|", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        max_val = max(f[1] for f in shap_features)
        ax.set_xlim(0, max_val * 1.2)
        fig.tight_layout()
        return fig

    from matplotlib.patches import Patch
    sh_col1, sh_col2 = st.columns(2)
    with sh_col1:
        fig = _plot_shap_half(shap_features[:5], "Rank 1-5")
        st.pyplot(fig)
        plt.close()
    with sh_col2:
        fig = _plot_shap_half(shap_features[5:], "Rank 6-10")
        st.pyplot(fig)
        plt.close()

    st.caption("Red = engineered features, Blue = original dataset features.")

    st.divider()
    st.markdown("#### Key Insights")
    i1, i2, i3 = st.columns(3)
    insights = [
        (i1, "#1: vg11_std", "V-column group 11 std (V284-V321) is the top feature -- validates our dimensionality reduction for 339 Vesta columns.", "rgba(231,76,60,0.06)", "rgba(231,76,60,0.2)"),
        (i2, "#7: uid_C8_mean", "Avg C8 per client group (card1+addr1) -- confirms aggregating stats by client group captures fraud signal.", "rgba(231,76,60,0.06)", "rgba(231,76,60,0.2)"),
        (i3, "Card & Address", "card1 (#10), card2 (#4), addr1 (#8) are the building blocks of our client group identifier.", "rgba(52,152,219,0.06)", "rgba(52,152,219,0.2)"),
    ]
    for col, title, desc, bg, border in insights:
        with col:
            st.markdown(
                f'<div class="bank-container" style="background:{bg}; border:1px solid {border};">'
                f'<strong>{title}</strong><br/><small>{desc}</small></div>',
                unsafe_allow_html=True,
            )

    st.divider()
    with st.expander("What are these features?", expanded=False):
        feat_info = pd.DataFrame([
            {"Feature": "vg11_std", "Description": "Std dev of V-column group 11 (V284-V321, Vesta risk scores)", "Type": "Engineered"},
            {"Feature": "C13", "Description": "Transaction counting feature (Vesta, undisclosed)", "Type": "Original"},
            {"Feature": "TransactionAmt", "Description": "Dollar amount of the transaction", "Type": "Original"},
            {"Feature": "card2", "Description": "Card-related (likely last 4 digits or bank ID)", "Type": "Original"},
            {"Feature": "D1", "Description": "Days since credit card account opened", "Type": "Original"},
            {"Feature": "D2", "Description": "Days since a related event (undisclosed)", "Type": "Original"},
            {"Feature": "uid_C8_mean", "Description": "Avg C8 across transactions by same client group", "Type": "Engineered"},
            {"Feature": "addr1", "Description": "Regional billing address code", "Type": "Original"},
            {"Feature": "card5", "Description": "Card-related (country or issuing bank)", "Type": "Original"},
            {"Feature": "card1", "Description": "Card issuer ID -- primary key for client grouping", "Type": "Original"},
        ])
        st.dataframe(feat_info.set_index("Feature"), use_container_width=True)

    st.markdown("#### SHAP Summary Plot")
    st.caption("Each dot = one transaction. Color = feature value (red=high, blue=low). Position = SHAP impact (right=fraud, left=legitimate).")
    shap_path = EVAL_DIR / "shap_summary.png"
    if shap_path.exists():
        _, img_col, _ = st.columns([1, 2, 1])
        with img_col:
            st.image(str(shap_path), use_container_width=True)

    st.divider()
    st.markdown("#### Feature Importance Table")
    tbl_left, tbl_right = st.columns(2)
    with tbl_left:
        feat_df1 = pd.DataFrame([
            {"Rank": i + 1, "Feature": f[0], "|SHAP|": f"{f[1]:.3f}",
             "Type": "Engineered" if f[2] else "Original"}
            for i, f in enumerate(shap_features[:5])
        ])
        st.dataframe(feat_df1.set_index("Rank"), use_container_width=True)
    with tbl_right:
        feat_df2 = pd.DataFrame([
            {"Rank": i + 6, "Feature": f[0], "|SHAP|": f"{f[1]:.3f}",
             "Type": "Engineered" if f[2] else "Original"}
            for i, f in enumerate(shap_features[5:])
        ])
        st.dataframe(feat_df2.set_index("Rank"), use_container_width=True)
