# Fraud Transaction Detection Web App 💳

**🌐 Live Demo:** [https://fraud-detection-webum.streamlit.app/](https://fraud-detection-webum.streamlit.app/)

An enterprise-grade Streamlit web application that detects fraudulent financial transactions using Machine Learning. It automatically scores real-time transactions by comparing immediate inputs against historical customer and terminal behaviour. Built as part of the Unified Mentor Internship Project.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)

## 🎯 Objective
Automated systems for real-time transaction monitoring are vital for financial security. This application classifies transactions as **Legitimate** or **Fraudulent** based on transaction geography (terminals), timing, and amounts. It handles severe class imbalance scenarios and explicitly flags 3 simulated fraud profiles (amount thresholds, terminal compromises, and customer account takeovers).

## ✨ Features
- **Real-Time Single Prediction:** Input transaction and behavioural statistics to instantly receive a fraud risk score.
- **Batch CSV Analysis:** Upload a CSV of multiple transactions to process hundreds of records simultaneously, outputting a downloadable scored file.
- **Interactive Risk Meter:** A dynamic Plotly gauge displaying the exact percentage probability of fraud.
- **Risk Factor Breakdown:** Intelligent alerts explaining exactly *why* a transaction was flagged (e.g., "Amount is 4x the customer's usual average").

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/abhiguru25/fraud-detection.git
cd fraud-detection/fraud_detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

## 🧠 Methodology & Evaluation Focus
- **Algorithm:** Tree-based ensemble learning (Random Forest / Gradient Boosting) chosen for non-linear boundary detection.
- **Metrics over Accuracy:** Due to the severe class imbalance of fraud datasets, this model is evaluated primarily on **Precision**, **Recall**, and **AUC-ROC (~0.97)** rather than just accuracy. Catching the actual frauds (Recall) is heavily prioritized.
- **Feature Engineering:** Features like `CUSTOMER_AVG_AMOUNT` relative to current `TX_AMOUNT` represent rolling behavioral baselines engineered during training.

## 📁 Repository Structure
```
├── app.py                      # Main Streamlit application file
├── requirements.txt            # Python dependencies
├── fraud_detection_model.pkl   # Serialized ML Model
├── fraud_scaler.pkl            # Serialized StandardScaler
└── README.md                   # This documentation
```

---
*Developed by Abhivirani | Unified Mentor Internship Project*
