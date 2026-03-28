# How to Run

## Prerequisites

- Python 3.10 (required for NVFlare)
- IEEE-CIS Fraud Detection dataset

## Steps

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
Runs 15 cyclic FL rounds across 3 simulated bank clients.

### 8. Evaluate federated model
```bash
python evaluation/evaluate_federated.py --model-path workspace/sim_run/server/global_fraud_model.json
```

### 9. Run SHAP analysis on global model
```bash
python evaluation/shap_analysis.py --model-path workspace/sim_run/server/global_fraud_model.json
```
Saves `evaluation/shap_summary.png` and validates engineered features in top-10 SHAP importance.

### 10. Generate result plots
```bash
python scripts/generate_plots.py
```

### 11. Launch dashboard
```bash
pip install streamlit
streamlit run dashboard.py
```
