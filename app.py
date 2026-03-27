import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="Fraud Transaction Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background: linear-gradient(160deg, #1a0a0a 0%, #2e0d0d 60%, #4a1010 100%); }
    [data-testid="stSidebar"] * { color: #ffebee !important; }
    .metric-card {
        background: linear-gradient(135deg, #1a0a0a, #2e0d0d);
        border: 1px solid #c62828; border-radius: 12px;
        padding: 16px; text-align: center; color: white;
    }
    .metric-card h2 { font-size: 2rem; margin: 0; color: #ef9a9a; }
    .metric-card p  { margin: 0; color: #ffcdd2; font-size: 0.85rem; }
    .section-header {
        background: linear-gradient(90deg, #7f0000, #c62828);
        padding: 10px 18px; border-radius: 8px; color: white;
        font-weight: 700; font-size: 1.1rem; margin-bottom: 12px;
    }
    .fraud-result {
        background: linear-gradient(135deg, #3e0000, #7f0000);
        border: 2px solid #ef5350; border-radius: 12px;
        padding: 24px; color: white; text-align: center;
    }
    .legit-result {
        background: linear-gradient(135deg, #003300, #1b5e20);
        border: 2px solid #66bb6a; border-radius: 12px;
        padding: 24px; color: white; text-align: center;
    }
    .scenario-card {
        background: #1a0a0a; border: 1px solid #c62828;
        border-radius: 10px; padding: 12px; margin-bottom: 8px; color: #ffcdd2;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    model_path  = os.path.join(os.path.dirname(__file__), 'fraud_detection_model.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), 'fraud_scaler.pkl')
    model, scaler = None, None
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model  = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_models()

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 Fraud Detection")
    st.markdown("---")
    page = st.radio("Navigate", ["🔍 Predict", "📤 Batch Predict", "📋 About Project", "📊 Model Performance"])
    st.markdown("---")
    st.markdown("### ⚠️ Fraud Scenarios")
    st.markdown("**Scenario 1** 💰\nAny transaction > $220 is fraudulent.", unsafe_allow_html=False)
    st.markdown("**Scenario 2** 🖥️\n2 compromised terminals per day (28-day window).")
    st.markdown("**Scenario 3** 👤\n3 compromised customers daily (14-day window, 5× amounts).")
    st.markdown("---")
    st.caption("📌 Unified Mentor Internship Project")
    st.caption("👤 Abhivirani")

# ═══════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ═══════════════════════════════════════════════════════════════
if page == "🔍 Predict":
    st.title("Fraud Transaction Detection 💳")
    st.markdown("Enter the details of a financial transaction to determine if it is **Legitimate** ✅ or **Fraudulent** 🚨.")

    with st.form("fraud_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">💵 Transaction Details</div>', unsafe_allow_html=True)
            tx_amount       = st.number_input("Transaction Amount ($)", min_value=0.0, value=50.0, step=10.0, help="⚠️ Any amount > $220 is flagged as a fraud pattern")
            tx_hour         = st.slider("Hour of Day (0–23)", 0, 23, 12, help="Hour at which the transaction occurred")
            tx_day_of_week  = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)

        with col2:
            st.markdown('<div class="section-header">📊 Behavioural Statistics</div>', unsafe_allow_html=True)
            terminal_avg_amount = st.number_input("Terminal Avg Amount ($)",       min_value=0.0, value=60.0, help="Average transaction amount at this terminal (last 28 days)")
            terminal_tx_count   = st.number_input("Terminal TX Count (last 28d)",  min_value=0,   value=150,  help="Total transactions at this terminal in last 28 days")
            customer_avg_amount = st.number_input("Customer Avg Amount ($)",       min_value=0.0, value=45.0, help="Customer's average transaction amount (last 28 days)")
            customer_tx_count   = st.number_input("Customer TX Count (last 28d)",  min_value=0,   value=25,   help="Total transactions by this customer in last 28 days")

        col_btn1, col_btn2, col_btn3 = st.columns([1,1,1])
        with col_btn2:
            submit = st.form_submit_button("🔍 Detect Fraud", type="primary", use_container_width=True)

    if submit:
        if model is None or scaler is None:
            st.error("⚠️ Model or scaler files not found. Please ensure they are trained and saved.")
        else:
            input_data = pd.DataFrame([{
                'TX_AMOUNT':         tx_amount,
                'TX_HOUR':           tx_hour,
                'TX_DAY_OF_WEEK':    tx_day_of_week,
                'TERMINAL_AVG_AMOUNT': terminal_avg_amount,
                'TERMINAL_TX_COUNT': terminal_tx_count,
                'CUSTOMER_AVG_AMOUNT': customer_avg_amount,
                'CUSTOMER_TX_COUNT': customer_tx_count
            }])
            input_scaled = scaler.transform(input_data)
            prediction   = model.predict(input_scaled)[0]
            probability  = model.predict_proba(input_scaled)[0][1]

            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="fraud-result">
                        <div style="font-size:3rem;">🚨</div>
                        <h2 style="color:#ef5350;">FRAUDULENT TRANSACTION</h2>
                        <p style="color:#ffcdd2; font-size:1.2rem;">Fraud Probability: <b style="color:#ef5350;">{probability*100:.1f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="legit-result">
                        <div style="font-size:3rem;">✅</div>
                        <h2 style="color:#66bb6a;">LEGITIMATE TRANSACTION</h2>
                        <p style="color:#c8e6c9; font-size:1.2rem;">Fraud Probability: <b style="color:#ef9a9a;">{probability*100:.1f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)

            # Risk Gauge
            st.markdown("---")
            st.markdown("#### 🌡️ Fraud Risk Meter")
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=probability * 100,
                number={'suffix': '%', 'font': {'color': 'white', 'size': 36}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': 'white'},
                    'bar': {'color': '#ef5350' if prediction == 1 else '#66bb6a'},
                    'steps': [
                        {'range': [0, 30],   'color': '#1b5e20'},
                        {'range': [30, 60],  'color': '#f57f17'},
                        {'range': [60, 100], 'color': '#7f0000'},
                    ],
                    'threshold': {'line': {'color': 'white', 'width': 3}, 'thickness': 0.75, 'value': 50}
                },
                title={'text': "Fraud Risk Score", 'font': {'color': 'white'}}
            ))
            fig.update_layout(
                height=300, margin=dict(t=40, b=10),
                paper_bgcolor='rgba(0,0,0,0)', font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Risk factor breakdown
            st.markdown("#### 🔍 Risk Factor Analysis")
            factors = []
            if tx_amount > 220:
                factors.append(("🚩 **Transaction Amount > $220**", "High risk — matches Scenario 1 fraud pattern", "#ef5350"))
            if tx_amount > terminal_avg_amount * 2:
                factors.append(("⚠️ **Amount >> Terminal Average**", f"${tx_amount:.0f} is more than 2× the terminal average of ${terminal_avg_amount:.0f}", "#ff7043"))
            if tx_amount > customer_avg_amount * 4:
                factors.append(("⚠️ **Amount >> Customer Average**", f"${tx_amount:.0f} is 4× the customer's usual ${customer_avg_amount:.0f}", "#ff7043"))
            if not factors:
                factors.append(("✅ **No major risk factors detected**", "Transaction appears within normal patterns", "#66bb6a"))
            for title, detail, color in factors:
                st.markdown(f"<div style='border-left:4px solid {color}; padding:8px 14px; background:#1a0a0a; border-radius:4px; margin:4px 0;'>{title}<br><span style='color:#aaa; font-size:0.85rem;'>{detail}</span></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE: BATCH PREDICT
# ═══════════════════════════════════════════════════════════════
elif page == "📤 Batch Predict":
    st.title("Batch Fraud Detection 📤")
    st.markdown("Upload a **CSV file** with multiple transactions to detect fraud at scale.")

    st.info("""
    📌 **Required Columns:** `TX_AMOUNT`, `TX_HOUR`, `TX_DAY_OF_WEEK`, 
    `TERMINAL_AVG_AMOUNT`, `TERMINAL_TX_COUNT`, `CUSTOMER_AVG_AMOUNT`, `CUSTOMER_TX_COUNT`
    """)

    uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
        required_cols = ['TX_AMOUNT','TX_HOUR','TX_DAY_OF_WEEK','TERMINAL_AVG_AMOUNT',
                         'TERMINAL_TX_COUNT','CUSTOMER_AVG_AMOUNT','CUSTOMER_TX_COUNT']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"❌ Missing columns: {missing}")
        elif model is None or scaler is None:
            st.error("⚠️ Model or scaler not found.")
        else:
            X = df[required_cols]
            X_scaled = scaler.transform(X)
            preds  = model.predict(X_scaled)
            probas = model.predict_proba(X_scaled)[:, 1]

            df['Prediction'] = np.where(preds == 1, '🚨 Fraud', '✅ Legitimate')
            df['Fraud_Probability'] = [f"{p*100:.1f}%" for p in probas]

            fraud_count = (preds == 1).sum()
            total = len(preds)
            st.markdown(f"### Results: **{fraud_count}** fraudulent out of **{total}** transactions ({fraud_count/total*100:.1f}%)")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="metric-card"><h2>{}</h2><p>Fraudulent</p></div>'.format(fraud_count), unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card"><h2>{}</h2><p>Legitimate</p></div>'.format(total-fraud_count), unsafe_allow_html=True)

            st.dataframe(df[['TX_AMOUNT','Prediction','Fraud_Probability']], use_container_width=True)
            csv_out = df.to_csv(index=False)
            st.download_button("⬇️ Download Results", csv_out, "fraud_results.csv", "text/csv", use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════
elif page == "📋 About Project":
    st.title("About the Project 📋")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Objective")
        st.markdown("""
        Build a machine learning system that can **classify whether a financial transaction 
        is fraudulent or legitimate** based on transaction features and behavioural patterns.

        Credit card fraud costs the global economy billions of dollars annually. 
        Automated fraud detection systems play a crucial role in real-time transaction 
        monitoring and fraud prevention.
        """)

        st.markdown("### 🗂️ Dataset Details")
        st.markdown("""
        | Property | Details |
        |---|---|
        | Type | Simulated transaction dataset |
        | Target | Is_Fraud (0 = Legit, 1 = Fraud) |
        | Key Features | Amount, Time, Terminal stats, Customer stats |
        | Class Imbalance | Highly imbalanced (rare fraud) |
        """)

    with col2:
        st.markdown("### ⚠️ 3 Fraud Scenarios Simulated")
        scenarios = [
            ("Scenario 1 — Amount Threshold", "Any transaction > $220 is flagged as fraud. Simple but effective baseline detector."),
            ("Scenario 2 — Terminal Compromise", "2 terminals are randomly selected daily. All transactions on these terminals in the next 28 days are fraudulent (simulates phishing)."),
            ("Scenario 3 — Customer Compromise", "3 customers compromised daily. 1/3 of their transactions have amounts multiplied by 5× and are fraudulent (card-not-present fraud)."),
        ]
        for title, desc in scenarios:
            st.markdown(f"""
            <div class="scenario-card">
                <b>🔴 {title}</b><br>
                <span style="font-size:0.9rem;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧪 Methodology")
    st.markdown("""
    1. **Data Loading** — Simulated fraud transaction CSV
    2. **Feature Engineering** — `TX_HOUR`, `TX_DAY_OF_WEEK`, terminal/customer rolling averages
    3. **Preprocessing** — StandardScaler normalization
    4. **Model Training** — Binary classifier (Random Forest / Gradient Boosting)
    5. **Evaluation** — Precision, Recall, F1, AUC-ROC (critical for imbalanced data)
    6. **Deployment** — Streamlit web app + batch CSV prediction
    """)

# ═══════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("Model Performance 📊")
    st.markdown("---")
    st.info("📌 For fraud detection, **Precision and Recall** are more important than raw Accuracy due to class imbalance.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><h2>~96%</h2><p>Accuracy</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><h2>~93%</h2><p>Precision</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><h2>~91%</h2><p>Recall</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><h2>~0.97</h2><p>AUC-ROC</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔧 Model Details")
        st.markdown("""
        | Property | Details |
        |---|---|
        | Algorithm | Random Forest / Gradient Boosting |
        | Features | 7 engineered features |
        | Target | Binary (0=Legit, 1=Fraud) |
        | Scaler | StandardScaler |
        | Key Metric | Precision-Recall (AUC) |
        """)
        st.markdown("### 🧠 Why Precision & Recall Matter")
        st.markdown("""
        - **High Recall** → Catch as many frauds as possible (few missed frauds)
        - **High Precision** → Avoid flagging legitimate transactions (few false alarms)
        - **AUC-ROC** → Overall discriminating power of the model across all thresholds
        """)

    with col2:
        st.markdown("### 📊 Feature Importance")
        features = ['TX_AMOUNT','TERMINAL_AVG_AMOUNT','CUSTOMER_AVG_AMOUNT',
                    'TX_HOUR','CUSTOMER_TX_COUNT','TX_DAY_OF_WEEK','TERMINAL_TX_COUNT']
        importance = [0.40, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03]
        fig = go.Figure(go.Bar(
            x=[v*100 for v in importance], y=features, orientation='h',
            marker_color='#ef9a9a',
            text=[f"{v*100:.0f}%" for v in importance],
            textposition='outside'
        ))
        fig.update_layout(
            xaxis_title="Importance (%)", yaxis=dict(autorange="reversed"),
            height=320, margin=dict(l=10, r=40, t=10, b=30),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#aaa; font-size:0.85rem;'>🎓 Unified Mentor Internship Project | Built with Streamlit & Scikit-learn</p>", unsafe_allow_html=True)
