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
* **Data Processing**: Pandas, NumPy, and NVIDIA RAPIDS for advanced local feature engineering.
* **Privacy Enhancement**: FLARE Privacy Filters for Differential Privacy (DP) implementation.
* **Model Interpretability**: SHAP to validate business logic and feature importance.
* **Development**: Python and Docker for consistent deployment across the federation.

## System Architecture and Workflow

The implementation follows a three-phase structure based on the Cross-Silo FL paradigm.

### Phase 1: Local Foundations

* **Data Partitioning**: Partitioning a synthetic fraud dataset into three non-uniform (non-IID) client datasets.
* **Feature Engineering**: Local calculation of Long-Term Profile Features (Z-scores) and Short-Term Velocity Features (30min/2hr transaction counts) within each bank's secure boundary.
* **Local Training**: Establishing initial baseline performance metrics using local XGBoost classifiers.

### Phase 2: Federated Training

* **Adaptation**: Converting local routines into a FLARE-compatible ModelLearner using the Client API.
* **Aggregation**: Using the FedAvg (Federated Averaging) algorithm for robust model aggregation.
* **Security**: Utilizing provision.py to generate secure Startup Kits and mTLS certificates.

### Phase 3: Privacy and Evaluation

* **Privacy Injection**: Integrating Differential Privacy filters into client configurations to modify model updates before sharing.
* **Evaluation**: Running the Cross Site Evaluator workflow for final validation on external test sets.
* **XAI Analysis**: Applying SHAP to interpret feature importance and validate the engineered velocity features.

## Contributors

* **Zohar Sahar** - 315144840
* **Yonatan Harel** - 208742593
