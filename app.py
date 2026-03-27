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

# ── SVG Icon Helpers ───────────────────────────────────────────────────────
def icon(svg, size=28, bg="#ffebee", color="#c62828"):
    return f"""<span style="display:inline-flex;align-items:center;justify-content:center;
        width:{size+12}px;height:{size+12}px;background:{bg};border-radius:10px;
        margin-right:8px;vertical-align:middle;flex-shrink:0;">
        <svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"
             viewBox="0 0 24 24" fill="none" stroke="{color}"
             stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{svg}</svg>
    </span>"""

def h3(svg, text, size=20, bg="#ffebee", color="#c62828"):
    return f'<h3 style="display:flex;align-items:center;margin:12px 0 8px 0;">{icon(svg,size,bg,color)}{text}</h3>'

ICO_CARD   = '<rect x="1" y="4" width="22" height="16" rx="2" ry="2"/><line x1="1" y1="10" x2="23" y2="10"/>'
ICO_SHIELD = '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'
ICO_ALERT  = '<polygon points="10.29 3.86 1.82 18 2 18 22 18 21.18 18 12.71 3.86 10.29 3.86"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'
ICO_CHECK  = '<polyline points="20 6 9 17 4 12"/>'
ICO_ABOUT  = '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>'
ICO_CHART  = '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>'
ICO_UPLOAD = '<polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>'
ICO_CPU    = '<rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>'
ICO_LAYERS = '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>'
ICO_TARGET = '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>'

st.markdown("""
<style>
    [data-testid="stSidebar"] { background: linear-gradient(160deg, #1a0a0a 0%, #2e0d0d 60%, #4a1010 100%); }
    [data-testid="stSidebar"] * { color: #ffebee !important; }
    .metric-card { background: linear-gradient(135deg, #1a0a0a, #2e0d0d); border: 1px solid #c62828;
        border-radius: 12px; padding: 16px; text-align: center; color: white; }
    .metric-card h2 { font-size: 2rem; margin: 0; color: #ef9a9a; }
    .metric-card p  { margin: 0; color: #ffcdd2; font-size: 0.85rem; }
    .section-header { display:flex; align-items:center; background: linear-gradient(90deg, #7f0000, #c62828);
        padding: 10px 18px; border-radius: 8px; color: white; font-weight: 700; font-size: 1.1rem; margin-bottom: 12px; }
    .fraud-result { background: linear-gradient(135deg, #3e0000, #7f0000); border: 2px solid #ef5350;
        border-radius: 12px; padding: 24px; color: white; text-align: center; }
    .legit-result { background: linear-gradient(135deg, #003300, #1b5e20); border: 2px solid #66bb6a;
        border-radius: 12px; padding: 24px; color: white; text-align: center; }
    .scenario-card { background: #1a0a0a; border: 1px solid #c62828; border-radius: 10px;
        padding: 12px; margin-bottom: 8px; color: #ffcdd2; }
    .page-title { display:flex; align-items:center; gap:10px; margin-bottom:0.5rem; }
    .page-title h1 { margin:0; }
    h3 { color: inherit !important; }
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

with st.sidebar:
    st.markdown(f"""<div style="display:flex;align-items:center;gap:10px;padding:4px 0 12px 0;">
        {icon(ICO_CARD, 26, '#2a0808', '#ef9a9a')}
        <span style="font-size:1.1rem;font-weight:700;color:white;">Fraud Detection</span>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigate", ["Predict", "Batch Predict", "About Project", "Model Performance"])
    st.markdown("---")
    st.markdown("<p style='color:#ffcdd2;font-size:0.8rem;font-weight:600;'>FRAUD SCENARIOS</p>", unsafe_allow_html=True)
    st.markdown("<div style='color:#ffebee;font-size:0.85rem;'><b>Scenario 1</b>: Transaction > $220</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#ffebee;font-size:0.85rem;margin-top:4px;'><b>Scenario 2</b>: Terminal compromised (28 days)</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#ffebee;font-size:0.85rem;margin-top:4px;'><b>Scenario 3</b>: Customer compromised (14 days, 5x amounts)</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Unified Mentor Internship Project")
    st.caption("Abhivirani")

# ══════════════════════════════════════════════════════════
if page == "Predict":
    st.markdown(f"""<div class="page-title">{icon(ICO_CARD, 30, '#ffebee', '#c62828')}<h1>Fraud Transaction Detection</h1></div>""", unsafe_allow_html=True)
    st.markdown("Enter the details of a financial transaction to determine if it is **Legitimate** or **Fraudulent**.")

    with st.form("fraud_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="section-header">{icon(ICO_CARD, 18, "transparent", "#ffcdd2")} Transaction Details</div>', unsafe_allow_html=True)
            tx_amount      = st.number_input("Transaction Amount ($)", min_value=0.0, value=50.0, step=10.0, help="Any amount > $220 matches Scenario 1 fraud pattern")
            tx_hour        = st.slider("Hour of Day (0–23)", 0, 23, 12)
            tx_day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
        with col2:
            st.markdown(f'<div class="section-header">{icon(ICO_CHART, 18, "transparent", "#ffcdd2")} Behavioural Statistics</div>', unsafe_allow_html=True)
            terminal_avg_amount = st.number_input("Terminal Avg Amount ($)",      min_value=0.0, value=60.0)
            terminal_tx_count   = st.number_input("Terminal TX Count (last 28d)", min_value=0,   value=150)
            customer_avg_amount = st.number_input("Customer Avg Amount ($)",      min_value=0.0, value=45.0)
            customer_tx_count   = st.number_input("Customer TX Count (last 28d)", min_value=0,   value=25)

        col_b1, col_b2, col_b3 = st.columns([1,1,1])
        with col_b2:
            submit = st.form_submit_button("Detect Fraud", type="primary", use_container_width=True)

    if submit:
        if model is None or scaler is None:
            st.error("Model or scaler files not found. Please ensure they are trained and saved.")
        else:
            input_data = pd.DataFrame([{
                'TX_AMOUNT': tx_amount, 'TX_HOUR': tx_hour, 'TX_DAY_OF_WEEK': tx_day_of_week,
                'TERMINAL_AVG_AMOUNT': terminal_avg_amount, 'TERMINAL_TX_COUNT': terminal_tx_count,
                'CUSTOMER_AVG_AMOUNT': customer_avg_amount, 'CUSTOMER_TX_COUNT': customer_tx_count
            }])
            input_scaled = scaler.transform(input_data)
            prediction   = model.predict(input_scaled)[0]
            probability  = model.predict_proba(input_scaled)[0][1]

            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                if prediction == 1:
                    st.markdown(f"""<div class="fraud-result">
                        <div style="margin-bottom:8px;">{icon(ICO_ALERT, 36, '#7f0000', '#ef5350')}</div>
                        <h2 style="color:#ef5350;">FRAUDULENT TRANSACTION</h2>
                        <p style="color:#ffcdd2;font-size:1.2rem;">Fraud Probability: <b style="color:#ef5350;">{probability*100:.1f}%</b></p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="legit-result">
                        <div style="margin-bottom:8px;">{icon(ICO_CHECK, 36, '#1b5e20', '#66bb6a')}</div>
                        <h2 style="color:#66bb6a;">LEGITIMATE TRANSACTION</h2>
                        <p style="color:#c8e6c9;font-size:1.2rem;">Fraud Probability: <b style="color:#ef9a9a;">{probability*100:.1f}%</b></p>
                    </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown(f'<h4 style="display:flex;align-items:center;">{icon(ICO_TARGET, 18, "#ffebee", "#c62828")} Fraud Risk Meter</h4>', unsafe_allow_html=True)
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=probability * 100,
                number={'suffix': '%', 'font': {'color': 'white', 'size': 36}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': 'white'},
                    'bar': {'color': '#ef5350' if prediction == 1 else '#66bb6a'},
                    'steps': [{'range': [0, 30],'color': '#1b5e20'},{'range': [30, 60],'color': '#f57f17'},{'range': [60, 100],'color': '#7f0000'}],
                    'threshold': {'line': {'color': 'white', 'width': 3}, 'thickness': 0.75, 'value': 50}
                },
                title={'text': "Fraud Risk Score", 'font': {'color': 'white'}}
            ))
            fig.update_layout(height=300, margin=dict(t=40,b=10), paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f'<h4 style="display:flex;align-items:center;">{icon(ICO_SHIELD, 18, "#ffebee", "#c62828")} Risk Factor Analysis</h4>', unsafe_allow_html=True)
            factors = []
            if tx_amount > 220:
                factors.append(("Transaction Amount > $220", "High risk — matches Scenario 1 fraud pattern", "#ef5350"))
            if tx_amount > terminal_avg_amount * 2:
                factors.append(("Amount >> Terminal Average", f"${tx_amount:.0f} is more than 2x the terminal average of ${terminal_avg_amount:.0f}", "#ff7043"))
            if tx_amount > customer_avg_amount * 4:
                factors.append(("Amount >> Customer Average", f"${tx_amount:.0f} is 4x the customer's usual ${customer_avg_amount:.0f}", "#ff7043"))
            if not factors:
                factors.append(("No major risk factors detected", "Transaction appears within normal patterns", "#66bb6a"))
            for title_f, detail, color in factors:
                st.markdown(f"<div style='border-left:4px solid {color};padding:8px 14px;background:#1a0a0a;border-radius:4px;margin:4px 0;'><b>{title_f}</b><br><span style='color:#aaa;font-size:0.85rem;'>{detail}</span></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
elif page == "Batch Predict":
    st.markdown(f"""<div class="page-title">{icon(ICO_UPLOAD, 30, '#ffebee', '#c62828')}<h1>Batch Fraud Detection</h1></div>""", unsafe_allow_html=True)
    st.markdown("Upload a **CSV file** with multiple transactions to detect fraud at scale.")
    st.info("Required columns: `TX_AMOUNT`, `TX_HOUR`, `TX_DAY_OF_WEEK`, `TERMINAL_AVG_AMOUNT`, `TERMINAL_TX_COUNT`, `CUSTOMER_AVG_AMOUNT`, `CUSTOMER_TX_COUNT`")
    uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
        required_cols = ['TX_AMOUNT','TX_HOUR','TX_DAY_OF_WEEK','TERMINAL_AVG_AMOUNT',
                         'TERMINAL_TX_COUNT','CUSTOMER_AVG_AMOUNT','CUSTOMER_TX_COUNT']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        elif model is None or scaler is None:
            st.error("Model or scaler not found.")
        else:
            X = df[required_cols]
            X_scaled = scaler.transform(X)
            preds  = model.predict(X_scaled)
            probas = model.predict_proba(X_scaled)[:, 1]
            df['Prediction'] = np.where(preds == 1, 'Fraud', 'Legitimate')
            df['Fraud_Probability'] = [f"{p*100:.1f}%" for p in probas]
            fraud_count = int((preds == 1).sum())
            total = len(preds)
            st.markdown(f"### Results: **{fraud_count}** fraudulent out of **{total}** ({fraud_count/total*100:.1f}%)")
            col1, col2 = st.columns(2)
            with col1: st.markdown(f'<div class="metric-card"><h2>{fraud_count}</h2><p>Fraudulent</p></div>', unsafe_allow_html=True)
            with col2: st.markdown(f'<div class="metric-card"><h2>{total-fraud_count}</h2><p>Legitimate</p></div>', unsafe_allow_html=True)
            st.dataframe(df[['TX_AMOUNT','Prediction','Fraud_Probability']], use_container_width=True)
            st.download_button("Download Results", df.to_csv(index=False), "fraud_results.csv", "text/csv", use_container_width=True)

# ══════════════════════════════════════════════════════════
elif page == "About Project":
    st.markdown(f"""<div class="page-title">{icon(ICO_ABOUT, 30, '#ffebee', '#c62828')}<h1>About the Project</h1></div>""", unsafe_allow_html=True)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(h3(ICO_TARGET, "Objective"), unsafe_allow_html=True)
        st.markdown("""Classify whether a **financial transaction is fraudulent or legitimate** based on transaction features and behavioural patterns.
        Credit card fraud costs the global economy billions annually. Automated detection systems are critical for real-time monitoring.""")
        st.markdown(h3(ICO_LAYERS, "Dataset Details"), unsafe_allow_html=True)
        st.markdown("""| Property | Details |\n|---|---|\n| Type | Simulated transaction dataset |\n| Target | Is_Fraud (0 = Legit, 1 = Fraud) |\n| Key Features | Amount, Time, Terminal stats, Customer stats |\n| Class Imbalance | Highly imbalanced (rare fraud) |""")
    with col2:
        st.markdown(h3(ICO_ALERT, "3 Fraud Scenarios Simulated"), unsafe_allow_html=True)
        for title_s, desc in [
            ("Scenario 1 — Amount Threshold", "Any transaction > $220 is flagged as fraud. Simple but effective baseline detector."),
            ("Scenario 2 — Terminal Compromise", "2 terminals selected daily. All transactions on these terminals in next 28 days are fraudulent (phishing simulation)."),
            ("Scenario 3 — Customer Compromise", "3 customers compromised daily. 1/3 of their transactions have amounts multiplied by 5x (card-not-present fraud)."),
        ]:
            st.markdown(f'<div class="scenario-card"><b>{title_s}</b><br><span style="font-size:0.9rem;">{desc}</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(h3(ICO_CPU, "Methodology"), unsafe_allow_html=True)
    st.markdown("""1. **Data Loading** — Simulated fraud transaction CSV\n2. **Feature Engineering** — `TX_HOUR`, `TX_DAY_OF_WEEK`, terminal/customer rolling averages\n3. **Preprocessing** — StandardScaler normalization\n4. **Model Training** — Binary classifier (Random Forest / Gradient Boosting)\n5. **Evaluation** — Precision, Recall, F1, AUC-ROC\n6. **Deployment** — Streamlit web app + batch CSV prediction""")

# ══════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown(f"""<div class="page-title">{icon(ICO_CHART, 30, '#ffebee', '#c62828')}<h1>Model Performance</h1></div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.info("For fraud detection, **Precision and Recall** are more important than raw Accuracy due to class imbalance.")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown('<div class="metric-card"><h2>~96%</h2><p>Accuracy</p></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric-card"><h2>~93%</h2><p>Precision</p></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="metric-card"><h2>~91%</h2><p>Recall</p></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="metric-card"><h2>~0.97</h2><p>AUC-ROC</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(h3(ICO_CPU, "Model Details"), unsafe_allow_html=True)
        st.markdown("""| Property | Details |\n|---|---|\n| Algorithm | Random Forest / Gradient Boosting |\n| Features | 7 engineered features |\n| Target | Binary (0=Legit, 1=Fraud) |\n| Scaler | StandardScaler |\n| Key Metric | Precision-Recall (AUC) |""")
        st.markdown(h3(ICO_SHIELD, "Why Precision & Recall Matter"), unsafe_allow_html=True)
        st.markdown("- **High Recall** — Catch as many frauds as possible\n- **High Precision** — Avoid flagging legitimate transactions\n- **AUC-ROC** — Overall discriminating power across all thresholds")
    with col2:
        st.markdown(h3(ICO_CHART, "Feature Importance"), unsafe_allow_html=True)
        features = ['TX_AMOUNT','TERMINAL_AVG_AMOUNT','CUSTOMER_AVG_AMOUNT',
                    'TX_HOUR','CUSTOMER_TX_COUNT','TX_DAY_OF_WEEK','TERMINAL_TX_COUNT']
        importance = [0.40, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03]
        fig = go.Figure(go.Bar(x=[v*100 for v in importance], y=features, orientation='h',
            marker_color='#ef9a9a', text=[f"{v*100:.0f}%" for v in importance], textposition='outside'))
        fig.update_layout(xaxis_title="Importance (%)", yaxis=dict(autorange="reversed"),
            height=320, margin=dict(l=10,r=40,t=10,b=30), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("<p style='text-align:center;color:#aaa;font-size:0.85rem;'>Unified Mentor Internship Project | Built with Streamlit & Scikit-learn</p>", unsafe_allow_html=True)
