### Phase 1: Local Foundations

#### 1. Data Partitioning: Creating Simulated Bank Silos
The goal is to transform a centralized dataset into three distinct "Banks" (Bank A, Bank B, and Bank C) to simulate a federated environment.

* **Data Acquisition and Merging:** Download the IEEE-CIS Fraud Detection dataset and merge the train_transaction.csv and train_identity.csv files on the TransactionID column.
* **Identify Partitioning Key:** Select a feature that naturally represents different entities, such as card1 (Card Issuer ID) or addr1 (Regional address), to simulate real-world institutional boundaries.
* **Execute Non-IID Split:** * Group the unique values of your partitioning key.
    * Randomly assign these groups to three separate dataframes.
* **Verification:** Ensure that each "Bank" has a different fraud rate (e.g., Bank A has 2% fraud while Bank B has 5%) to simulate the non-uniform distribution (Non-IID) required for the project.
* **Secure Storage:** Save these as three separate files (bank_a.csv, bank_b.csv, bank_c.csv) to ensure they are handled as private data.

#### 2. Feature Engineering: Developing Velocity and Profile Metrics
This step must be performed locally on each client's dataset to mimic the "secure boundary" of a bank.

* **Long-Term Profile Features:**
    * **Spending Z-Scores:** Calculate the mean and standard deviation of TransactionAmt for each card ID over the entire dataset.
    * For each transaction, compute how many standard deviations the current amount is from that card's historical average (the Z-score).
* **Short-Term Velocity Features:**
    * **Time Normalization:** Use the TransactionDT (timedelta) to create a rolling time index.
    * **30-Minute Windows:** Count the number of transactions and sum the transaction amounts for each user in the 30 minutes preceding the current transaction.
    * **2-Hour Windows:** Perform the same calculation for a 2-hour window.
* **Encoding:** Convert categorical variables into numerical formats suitable for XGBoost (e.g., using label encoding or category typing).

#### 3. Local Training: Establishing the Performance Baseline
Before starting federated learning, you must know how well a bank can detect fraud using only its own data.

* **Train/Test Split:** On each bank's local dataset, split the data into 80% training and 20% testing sets.
* **Model Configuration:** Initialize an XGBoost classifier with standard parameters (e.g., n_estimators=100, max_depth=6).
* **Baseline Training:** Train the model on the local training set for each bank independently.
* **Performance Metrics Collection:**
    * Predict fraud on the local 20% test set.
    * **Record Results:** Calculate and save the F1-Score and AUC-PR for each bank.

**Note:** These scores will serve as the benchmark to measure the "Collaborative Uplift" once you move to Phase 2.
