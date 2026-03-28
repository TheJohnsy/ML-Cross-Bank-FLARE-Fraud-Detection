# How to Run

## Prerequisites

### conda
Used to manage a Python 3.10 environment (NVFlare does not support Python 3.13+).

**macOS (Homebrew):**
```bash
brew install --cask miniconda
conda init zsh  # or: conda init bash
```

**Linux:**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

**Windows:**
Download and run the installer from https://docs.conda.io/en/latest/miniconda.html

### Docker (optional)
Only needed if you want to verify the containerised infrastructure with mTLS.

**macOS:**
```bash
brew install --cask docker
```

**Linux:**
```bash
sudo apt-get install docker.io docker-compose
```

**Windows:** Download Docker Desktop from https://www.docker.com/products/docker-desktop

## Environment Setup

### 1. Create a Python 3.10 environment
```bash
conda create -n flare python=3.10 -y
conda activate flare
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
Installs NVFlare, XGBoost, pandas, scikit-learn, SHAP, matplotlib, and Streamlit.

### 3. Install OpenMP (macOS only)
XGBoost requires OpenMP on macOS. If you get a `libxgboost.dylib could not be loaded` error:
```bash
brew install libomp
```

## Data Setup

### 4. Download the IEEE-CIS Fraud Detection dataset
Download from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data) and place the training files in `data/raw/`:
```
data/raw/train_transaction.csv
data/raw/train_identity.csv
```
Create the directory if it doesn't exist:
```bash
mkdir -p data/raw
```

## Phase 1: Local Foundations

### 5. Partition dataset into 3 bank silos
Splits the merged dataset into 3 non-IID partitions (Bank A, B, C) with different fraud rates, simulating separate financial institutions.
```bash
python scripts/partition_data.py
```
Output: `data/partitioned/bank_a.csv`, `bank_b.csv`, `bank_c.csv`

### 6. Build global category mappings
Reads all 3 partitions and creates a shared integer encoding for categorical columns, ensuring consistent encoding across banks in the federated setting.
```bash
python scripts/build_category_maps.py
```
Output: `data/partitioned/category_maps.json`

### 7. Engineer features for each bank
Computes per-card Z-scores, velocity features (30-min and 2-hr transaction windows), V-column group summaries, and client-group aggregated features. All features are computed locally within each bank's data.
```bash
python scripts/feature_engineering.py --bank a
python scripts/feature_engineering.py --bank b
python scripts/feature_engineering.py --bank c
```
Output: `data/partitioned/bank_{a,b,c}_engineered.csv`

Note: Velocity feature computation may take several minutes per bank due to the sliding window calculation.

### 8. Train and evaluate local baselines
Trains an XGBoost classifier independently on each bank's data, finds the optimal decision threshold, and saves metrics and the trained model.
```bash
python scripts/local_baseline.py --bank a
python scripts/local_baseline.py --bank b
python scripts/local_baseline.py --bank c
```
Output:
- `data/partitioned/bank_{a,b,c}_baseline_metrics.json` (F1, AUC-PR, AUC-ROC, precision, recall at both default and optimal thresholds)
- `data/partitioned/bank_{a,b,c}_local_model.json` (saved XGBoost models)

## Phase 2: Federated Training

### 9. Provision NVFlare workspace
Generates secure startup kits with mTLS certificates for the server and all 3 client sites.
```bash
nvflare provision -p provision/project.yml -w workspace
```
Output: `workspace/fraud_fl/prod_00/` containing server and site-1/2/3 startup kits.

### 10. Run federated training
Launches the NVFlare simulator, which runs 15 rounds of cyclic training across 3 simulated bank clients. Each bank warm-starts from the received model and adds 50 new XGBoost trees per round.
```bash
nvflare simulator jobs/fraud_fl -w workspace/sim_run -n 3 -t 1 -c site-1,site-2,site-3
```
Output: `workspace/sim_run/server/global_fraud_model.json` (the trained global federated model)

Note: This takes approximately 5-10 minutes depending on hardware.

## Phase 3: Evaluation

### 11. Evaluate federated model against local baselines
Loads the global federated model and evaluates it on all 3 banks' test sets (using the same train/test split as the local baselines). Prints a collaborative uplift comparison table.
```bash
python evaluation/evaluate_federated.py --model-path workspace/sim_run/server/global_fraud_model.json
```
Output: `evaluation/federated_metrics.json`

### 12. Run SHAP explainability analysis
Computes SHAP feature importance values on the global model using Bank A's test data. Identifies the top-10 most influential features and generates a summary plot.
```bash
python evaluation/shap_analysis.py --model-path workspace/sim_run/server/global_fraud_model.json
```
Output: `evaluation/shap_summary.png`

### 13. Generate result plots
Creates visualisation plots for the report: fraud rate distribution, metrics comparison, confusion matrices, dataset size distribution, and SHAP feature importance bar chart.
```bash
python scripts/generate_plots.py
```
Output: `evaluation/plots/*.png`

### 14. Launch interactive dashboard
Starts a Streamlit web dashboard that displays all results, metrics, plots, and the federated training analysis interactively.
```bash
streamlit run dashboard.py
```
Opens at `http://localhost:8501`.

## Optional: Docker Infrastructure

The Docker setup demonstrates the production deployment architecture with mTLS authentication. It is not required for running the training (the simulator above handles that).

### Build and verify Docker containers
```bash
docker-compose build
docker-compose up
```
This starts the NVFlare server and 3 bank clients in separate containers. You should see all 3 clients connect and authenticate successfully. Stop with `Ctrl+C`.

## Project Structure

```
scripts/
  partition_data.py          # Step 5: Split data into 3 banks
  build_category_maps.py     # Step 6: Global category encoding
  feature_engineering.py     # Step 7: Feature computation per bank
  local_baseline.py          # Step 8: Train local XGBoost models
  generate_plots.py          # Step 13: Generate report plots
jobs/fraud_fl/               # NVFlare federated learning job definition
evaluation/
  evaluate_federated.py      # Step 11: Compare federated vs local
  shap_analysis.py           # Step 12: SHAP feature importance
dashboard.py                 # Step 14: Interactive results dashboard
config.py                    # Global configuration (features, params, paths)
```
