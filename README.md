# Cross-Bank Federated Transaction Fraud Detection

This project implements a privacy-preserving, federated transaction fraud detection system using **NVIDIA FLARE (NVFlare)** and **XGBoost**. It enables financial institutions to collaboratively train high-performance models while keeping sensitive raw transaction data strictly local.

## Tech Stack

| Category | Tool / Library | Role in Project |
| --- | --- | --- |
| **Federated Learning** | **NVIDIA FLARE** | Core SDK for secure, distributed orchestration and job management.

 |
| **ML Algorithm** | **XGBoost** | High-performance classifier for tabular transaction data.

 |
| **Data Processing** | **Pandas, NumPy, RAPIDS** | Used for local feature engineering and data manipulation.

 |
| **Privacy** | **FLARE Privacy Filters** | Implements Differential Privacy (DP) for model updates.

 |
| **Interpretability** | **SHAP** | Validates business logic and feature importance.

 |
| **Deployment** | **Python, Docker** | Primary language and containerization for consistent deployment.

 |

---

## System Architecture

The system follows a **Horizontal Federated Learning** paradigm. Each bank (client) possesses the same feature set but different customer populations. The central server orchestrates the training rounds without ever seeing raw data.

* 
**FL Server:** Orchestrates the training lifecycle, manages mTLS certificates, and aggregates model updates using the **FedAvg** algorithm.


* 
**FL Clients (Banks):** Execute local training on private datasets and apply **Differential Privacy (DP)** filters before sending updates to the server.



---

## Data & Feature Engineering

We utilize the **IEEE-CIS Fraud Detection** dataset, partitioned into non-IID silos to simulate varying institutional fraud profiles.

### 1. Short-Term Velocity Features

To capture immediate fraudulent behavior, each client computes local features before training:

* 
**Transaction Frequency:** Count of transactions per `cardID` in the last **30 minutes** and **2 hours**.


* 
**Volumetric Metrics:** Sum of transaction amounts for a specific user within the same time windows.



### 2. Local Benchmarking

Before federating, each bank trains a local **XGBoost** model to establish a baseline. Our success metric is the **Collaborative Uplift**: the increase in **F1-Score** when using the global federated model compared to isolated local training.

---

## Security & Privacy Configuration

### 1. Secure Provisioning (mTLS)

The system is hardened using **Mutual TLS (mTLS)**. Every participant (server, clients, and admin) must possess a Startup Kit containing certificates signed by a private Root CA.

### 2. Differential Privacy (DP) Filters

To protect against model inversion, we apply privacy filters on the client side to modify updates before sharing.

**Example `config_fed_client.json` snippet:**

```json
"task_result_filters": [
    {
        "tasks": ["train"],
        "filters": [
            {
                "path": "nvflare.app_common.filters.svt_privacy.SVTPrivacy",
                "args": {
                    "fraction": 0.1,
                    "epsilon": 0.1
                }
            }
        ]
    }
]

```

---

## Implementation Roadmap

### Stage 1: The Simulator (Development)

Rapidly iterate on the `ModelLearner` using the NVFlare Simulator to debug feature engineering and aggregation logic.

```bash
nvflare simulator jobs/fraud_job -w workspace -n 3 -t 3

```

### Stage 2: Dockerized Distributed System (Final Deployment)

To meet the requirement for an independent server-side update, we deploy as independent containers using `docker-compose`.

* 
**Server Container:** Performs global model aggregation.


* 
**Client Containers:** Maintain private data silos and perform local training.



---

## 📈 Evaluation & XAI

Upon completion, the system executes the **Cross-Site Evaluator** to perform final model validation on an external test set. Finally, we apply **SHAP** to interpret model predictions and validate the utility of the engineered velocity features.

---

## 👥 Contributors

* 
**Zohar Sahar** - 315144840 


* 
**Yonatan Harel** - 208742593
