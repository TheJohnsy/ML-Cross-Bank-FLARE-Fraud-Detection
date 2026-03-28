# Cross-Bank Federated Transaction Fraud Detection

This project implements a privacy-preserving, federated transaction fraud detection system using NVIDIA FLARE (NVFlare) and XGBoost. It enables financial institutions to collaboratively train high-performance models while keeping sensitive raw transaction data strictly local to address regulatory and data-sharing constraints.

## Goals and Success Criteria

The project is benchmarked against specific performance and security milestones.

### Performance Benchmarks

| Goal Name | Criteria | Rationale |
| :--- | :--- | :--- |
| **Global Model Quality** | Maximize F1-Score on the global, held-out test set. | Balances Precision and Recall in highly imbalanced fraud datasets. |
| **Collaborative Uplift** | FL Model must significantly outperform individual local data silo models. | Quantifies the value derived from multi-party collaboration. |
| **Feature Utility** | Locally computed time-windowed (velocity) features drive the model's predictive power. | Validates that locally computed feature engineering is effective. |

### Privacy and Security Benchmarks

| Goal Name | Criteria | Rationale |
| :--- | :--- | :--- |
| **Privacy Assurance** | Successful execution of the Differential Privacy (DP) filter. | Provides formal assurance without complex custom cryptographic implementation. |
| **Secure Protocol** | Configuration of Secure Provisioning and Mutual TLS (mTLS) for all communication. | Ensures an enterprise-grade foundation for authentication and encryption. |

## Tech Stack

* **Federated Learning Framework**: NVIDIA FLARE (NVFlare) for secure, distributed orchestration.
* **ML Algorithm**: XGBoost for high-performance tabular classification.
* **Data Processing**: Pandas, NumPy for advanced local feature engineering.
* **Privacy Enhancement**: FLARE Privacy Filters for Differential Privacy (DP) implementation.
* **Model Interpretability**: SHAP to validate business logic and feature importance.
* **Development**: Python and Docker for consistent deployment across the federation.

## System Architecture and Workflow

The implementation follows a three-phase structure based on the Cross-Silo FL paradigm.

### Phase 1: Local Foundations

* **Data Partitioning**: Partitioning the IEEE-CIS Fraud Detection dataset into three non-uniform (non-IID) client datasets with varying fraud rates (Bank A: ~2%, Bank B: ~3.4%, Bank C: ~4%).
* **Feature Engineering**: Local calculation of Long-Term Profile Features (Z-scores), Short-Term Velocity Features (30min/2hr transaction counts), V-column group summaries, and client-group aggregated features — all computed within each bank's secure boundary.
* **Local Training**: Establishing initial baseline performance metrics using local XGBoost classifiers with dynamic class weighting.

### Phase 2: Federated Training

* **Adaptation**: Converting local routines into a FLARE-compatible ModelLearner using the Client API.
* **Cyclic Training**: Using a cyclic controller that passes the model serially from client to client, with each client warm-starting from the received global model — the standard approach for XGBoost federated learning.
* **Security**: Utilizing `provision.py` to generate secure Startup Kits and mTLS certificates.

### Phase 3: Privacy and Evaluation

* **Privacy Injection**: Differential Privacy filters configured in client settings to modify model updates before sharing.
* **Evaluation**: SHAP-based explainability analysis on the global federated model.
* **XAI Analysis**: Applying SHAP to interpret feature importance and validate the engineered features.

## Feature Engineering

Our feature engineering pipeline produces four categories of features, each computed locally within each bank's data boundary:

1. **Z-Score Profile Features**: Per-card spending deviation (`amt_zscore`) — turned out to be the single most important feature by SHAP analysis.
2. **Velocity Features**: Transaction count and sum within 30-minute and 2-hour sliding windows, computed without data leakage (current transaction excluded).
3. **V-Column Group Summaries**: The 339 Vesta-proprietary V-columns are grouped by shared NAN pattern (12 groups discovered via EDA), then summarised into per-group mean and standard deviation. Group V322–V339 was excluded after observing temporal inconsistency.
4. **Client-Group Aggregated Features**: We construct a client group identifier from `card1 + addr1`, then compute aggregated C-column statistics (mean) and transaction frequency per group. The group identifier itself is never used as a feature — only the aggregations are, which allows the model to generalise to unseen clients.

Categorical encoding uses a global mapping built across all bank partitions to ensure consistent integer assignments in the federated setting.

## Results

### Local Baselines (Optimal Threshold)

| Bank | Fraud Rate | Threshold | F1 | AUC-PR | AUC-ROC | Precision | Recall |
|------|-----------|-----------|-----|--------|---------|-----------|--------|
| A | 1.9% | 0.85 | 0.764 | 0.786 | 0.964 | 0.914 | 0.657 |
| B | 3.4% | 0.80 | 0.769 | 0.825 | 0.972 | 0.816 | 0.726 |
| C | 4.1% | 0.80 | 0.704 | 0.750 | 0.960 | 0.765 | 0.651 |

### Federated vs Local Comparison

| Bank | Local F1 | Federated F1 | Uplift |
|------|----------|--------------|--------|
| A | 0.764 | 0.743 | -0.021 |
| B | 0.769 | 0.708 | -0.061 |
| C | 0.704 | 0.589 | -0.115 |

The federated model did not achieve collaborative uplift over local baselines. This is attributed to the non-IID data distribution and limited per-round tree budget in cyclic training — a well-documented challenge in federated learning for heterogeneous data.

### SHAP Feature Importance (Global Model)

Top-10 features by mean |SHAP| on the federated global model:

| Rank | Feature | Mean |SHAP| |
|------|---------|--------------|
| 1 | vg11_std | 0.780 |
| 2 | C13 | 0.566 |
| 3 | TransactionAmt | 0.561 |
| 4 | card2 | 0.542 |
| 5 | D1 | 0.530 |
| 6 | D2 | 0.491 |
| 7 | uid_C8_mean | 0.455 |
| 8 | addr1 | 0.431 |
| 9 | card5 | 0.391 |
| 10 | card1 | 0.377 |

Key observations:
- The V-column group summary `vg11_std` ranks **#1**, validating our dimensionality reduction approach for the 339 Vesta-proprietary features.
- Client-group aggregated feature `uid_C8_mean` appears at #7, confirming that grouping clients and computing aggregate statistics provides signal.
- Card and address features (`card1`, `card2`, `addr1`) are highly ranked — these are the building blocks of the client group identifier.

## Contributors

* **Zohar Sahar** - 315144840
* **Yonatan Harel** - 208742593


### Docker Infrastructure Diagram
<img width="804" height="490" alt="image" src="https://github.com/user-attachments/assets/8382db20-ea84-4459-9feb-de1d70cddd27" />

## How to Run

### 1. Place dataset files
```
data/raw/train_transaction.csv
data/raw/train_identity.csv
```

### 2. Partition into 3 bank silos
```bash
python scripts/partition_data.py
```

### 3. Build global category mappings
```bash
python scripts/build_category_maps.py
```

### 4. Engineer features (run for each bank)
```bash
python scripts/feature_engineering.py --bank a
python scripts/feature_engineering.py --bank b
python scripts/feature_engineering.py --bank c
```

### 5. Train and evaluate local baselines
```bash
python scripts/local_baseline.py --bank a
python scripts/local_baseline.py --bank b
python scripts/local_baseline.py --bank c
```
Metrics saved to `data/partitioned/bank_{a,b,c}_baseline_metrics.json`.

### 6. Provision NVFlare startup kits
```bash
nvflare provision -p provision/project.yml -w workspace
```

### 7. Launch federated training (simulator)
```bash
nvflare simulator jobs/fraud_fl -w workspace/sim_run -n 3 -t 1 -c site-1,site-2,site-3
```
Runs 10 cyclic FL rounds across 3 simulated bank clients.

### 8. Run SHAP analysis on global model
```bash
python evaluation/shap_analysis.py --model-path workspace/sim_run/server/global_fraud_model.json
```
Saves `evaluation/shap_summary.png` and validates engineered features in top-10 SHAP importance.
