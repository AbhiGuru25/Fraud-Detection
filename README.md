# Fraud Transaction Detection 💳

## Objective
Build a system to classify whether a simulated financial transaction is fraudulent or legitimate. This involves analyzing spending habits, terminal limits, and time-based metrics.

## Dataset
This simulated dataset exhibits realistic imbalance and replicates three main fraud scenarios:
1.  Immediate obvious fraud (amount > 220).
2.  Compromised terminals phishing data.
3.  Card-not-present fraud from leaked customer credentials.

## Features
*   **Machine Learning Model:** A robust Random Forest approach utilizing engineered aggregate features (e.g. daily average amounts, transaction frequencies) to establish trust baselines and spot anomalous deviations.
*   **Web Interface:** An interactive Streamlit app serving as a fraud evaluation dashboard where manual inputs flag potentially suspicious traits.

## How to Run Locally

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## Files in this Repository
*   `Fraud_Detection.ipynb`: Data engineering, parsing dates, grouping transactions, and Random Forest tuning.
*   `app.py`: The Streamlit evaluation UI.
*   `fraud_detection_model.pkl`: Model binary.
*   `fraud_scaler.pkl`: The StandardScaler serialization for standardizing incoming numeric inputs.
*   `Fraud_Detection_Report.pdf`: Extensive PDF results analysis.
*   `requirements.txt`: Dependencies specification.
