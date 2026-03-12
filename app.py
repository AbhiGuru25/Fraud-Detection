import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Fraud Transaction Detection", page_icon="💳")

@st.cache_resource
def load_models():
    model_path = os.path.join(os.path.dirname(__file__), 'fraud_detection_model.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), 'fraud_scaler.pkl')
    
    model, scaler = None, None
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_models()

st.title("Fraud Transaction Detection 💳")
st.write("Enter the details of a financial transaction to predict if it is Legitimate or Fraudulent.")

with st.form("fraud_form"):
    st.subheader("Transaction Details")
    col1, col2 = st.columns(2)
    with col1:
        tx_amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=50.0, step=10.0)
        tx_hour = st.slider("Hour of Day (0-23)", 0, 23, 12)
        tx_day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
        
    with col2:
        terminal_avg_amount = st.number_input("Terminal Avg Amount ($)", min_value=0.0, value=60.0)
        terminal_tx_count = st.number_input("Terminal TX Count (last 28 days)", min_value=0, value=150)
        customer_avg_amount = st.number_input("Customer Avg Amount ($)", min_value=0.0, value=45.0)
        customer_tx_count = st.number_input("Customer TX Count (last 28 days)", min_value=0, value=25)

    submit = st.form_submit_button("Detect Fraud")

if submit:
    if model is None or scaler is None:
        st.error("Model or scaler files not found. Ensure they are trained and saved in this directory.")
    else:
        # Create input array matching the training features: 
        # ['TX_AMOUNT', 'TX_HOUR', 'TX_DAY_OF_WEEK', 'TERMINAL_AVG_AMOUNT', 'TERMINAL_TX_COUNT', 'CUSTOMER_AVG_AMOUNT', 'CUSTOMER_TX_COUNT']
        input_data = pd.DataFrame([{
            'TX_AMOUNT': tx_amount,
            'TX_HOUR': tx_hour,
            'TX_DAY_OF_WEEK': tx_day_of_week,
            'TERMINAL_AVG_AMOUNT': terminal_avg_amount,
            'TERMINAL_TX_COUNT': terminal_tx_count,
            'CUSTOMER_AVG_AMOUNT': customer_avg_amount,
            'CUSTOMER_TX_COUNT': customer_tx_count
        }])
        
        # Scale the features
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        if prediction == 1:
            st.error(f"🚨 **FRAUDULENT TRANSACTION DETECTED** 🚨")
            st.warning(f"Fraud Probability: {probability*100:.2f}%")
        else:
            st.success(f"✅ **LEGITIMATE TRANSACTION** ✅")
            st.info(f"Fraud Probability: {probability*100:.2f}%")
