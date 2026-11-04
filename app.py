import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------------------------------------------
# 1. Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="IoT Cybersecurity Threat Detector",
    page_icon="🛡️",
    layout="wide"
)

st.title(" Real-Time IoT Network Intrusion Detection System")
st.markdown("""
This interactive Streamlit dashboard utilizes a trained **Random Forest Machine Learning Classifier** 
to evaluate real-time network telemetry flow metrics and detect malicious network intrusions (such as **DDoS, Mirai, DoS, Recon, Spoofing**, or **Normal** traffic).
""")

st.divider()

# -----------------------------------------------------------------------------
# 2. Load Pipeline Artifacts
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load('iot_cyber_model.pkl')
    scaler = joblib.load('iot_scaler.pkl')
    columns = joblib.load('iot_model_columns.pkl')
    return model, scaler, columns

try:
    model, scaler, saved_columns = load_artifacts()
    st.sidebar.success("✅ Model Pipeline Loaded Successfully!")
except Exception as e:
    st.error(f" Error loading model files. Make sure 'iot_cyber_model.pkl', 'iot_scaler.pkl', and 'iot_model_columns.pkl' are in the same directory.")
    st.stop()

# Target Label Mapping (Matches LabelEncoder in training)
LABEL_MAPPING = {
    0: "Brute Force",
    1: "DDoS",
    2: "DoS",
    3: "Mirai Botnet",
    4: "Normal Traffic",
    5: "Reconnaissance (Scan)",
    6: "Spoofing"
}

# -----------------------------------------------------------------------------
# 3. Sidebar Input Form (Network Packet Controls)
# -----------------------------------------------------------------------------
st.sidebar.header(" Live Packet Header Telemetry")

flow_duration = st.sidebar.number_input("Flow Duration (ms)", min_value=0.0, max_value=86400.0, value=120.0, step=10.0)
header_length = st.sidebar.number_input("Header Length (bytes)", min_value=0.0, max_value=1500.0, value=54.0, step=2.0)
protocol_type = st.sidebar.selectbox("Protocol Type", options=[6, 17, 1], format_func=lambda x: {6: "TCP (6)", 17: "UDP (17)", 1: "ICMP (1)"}[x])
rate = st.sidebar.number_input("Packet Rate (packets/sec)", min_value=0.0, max_value=250000.0, value=142.5, step=10.0)

st.sidebar.subheader("TCP Flags & Counts")
syn_flag = st.sidebar.selectbox("SYN Flag Number", options=[0, 1], index=1)
fin_flag = st.sidebar.selectbox("FIN Flag Number", options=[0, 1], index=0)
ack_count = st.sidebar.number_input("ACK Flag Count", min_value=0, max_value=100, value=2)

# -----------------------------------------------------------------------------
# 4. Main Panel — Prediction & Telemetry Summary
# -----------------------------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Packet Summary")
    input_data = {
        'flow_duration': flow_duration,
        'Header_Length': header_length,
        'Protocol Type': protocol_type,
        'Rate': rate,
        'fin_flag_number': fin_flag,
        'syn_flag_number': syn_flag,
        'ack_count': ack_count
    }
    input_df = pd.DataFrame([input_data])
    st.dataframe(input_df, use_container_width=True)

with col2:
    st.subheader("🎯 Threat Evaluation Result")
    
    # Align input to model training columns
    input_reindexed = input_df.reindex(columns=saved_columns, fill_value=0)
    input_scaled = scaler.transform(input_reindexed)
    
    if st.button(" Analyze Network Packet", type="primary", use_container_width=True):
        # Model Prediction
        pred_code = model.predict(input_scaled)[0]
        pred_label = LABEL_MAPPING.get(pred_code, f"Class {pred_code}")
        
        # Class Probabilities
        probs = model.predict_proba(input_scaled)[0]
        confidence = np.max(probs) * 100
        
        st.divider()
        
        if pred_label == "Normal Traffic":
            st.success(f"### Status: SAFE\n**Classification:** {pred_label}\n**Confidence Score:** {confidence:.2f}%")
        else:
            st.error(f"### Status: THREAT DETECTED ⚠️\n**Classification:** {pred_label}\n**Confidence Score:** {confidence:.2f}%")
            
        st.write("#### Class Probabilities Breakdown:")
        prob_df = pd.DataFrame({
            'Attack Type': [LABEL_MAPPING.get(i, f"Class {i}") for i in range(len(probs))],
            'Probability': probs
        }).sort_values(by='Probability', ascending=False)
        
        st.bar_chart(prob_df.set_index('Attack Type'))

# -----------------------------------------------------------------------------
# 5. Footer Information
# -----------------------------------------------------------------------------
st.divider()
st.caption("IoT Cybersecurity Intrusion Detection System | Powered by Scikit-Learn & Streamlit")