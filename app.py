import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from tensorflow.keras.models import load_model

# ─── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="ECG Anomaly Detection",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2a2d3e);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #3a3f5c;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 4px;
    }
    .normal-badge {
        background-color: #064e3b;
        color: #6ee7b7;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .abnormal-badge {
        background-color: #7f1d1d;
        color: #fca5a5;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stAlert { border-radius: 10px; }
    div[data-testid="stMetricValue"] { font-size: 2rem; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODEL ────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model     = load_model("ecg_autoencoder.keras")
    scaler    = joblib.load("scaler.pkl")
    threshold = joblib.load("threshold.pkl")
    return model, scaler, threshold

model, scaler, threshold = load_artifacts()

# ─── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=80)
    st.title("ECG Anomaly\nDetector")
    st.markdown("---")

    st.markdown("### Model Info")
    st.info(f"""
    **Architecture:** Autoencoder  
    **Input Shape:** 187 features  
    **Latent Dim:** 32  
    **Threshold:** `{threshold:.5f}`  
    **ROC-AUC:** 0.817  
    **Accuracy:** 72%
    """)

    st.markdown("---")
    st.markdown("### Settings")
    custom_threshold = st.slider(
        "Anomaly Threshold",
        min_value=0.001,
        max_value=0.020,
        value=float(threshold),
        step=0.0001,
        format="%.4f",
        help="Lower = more sensitive to anomalies"
    )

    show_reconstruction = st.checkbox("Show reconstruction overlay", value=True)
    max_beats_to_show   = st.slider("Max beats to visualize", 3, 20, 10)

    st.markdown("---")
    st.caption("ECG Anomaly Detection | Deep Learning Project")
    st.warning("For educational purposes only. Not a medical diagnostic tool.")

# ─── HEADER ────────────────────────────────────────────────────
st.markdown("# ECG Anomaly Detection System")
st.markdown("##### Powered by Autoencoder — Unsupervised Anomaly Detection")
st.markdown("---")

# ─── FILE UPLOAD ───────────────────────────────────────────────
st.markdown("### Upload ECG Data")
st.markdown("Upload a CSV file where **each row = one heartbeat** with **187 signal values**.")

uploaded_file = st.file_uploader(
    "Drop your ECG CSV file here",
    type=["csv"],
    help="Each row should contain 187 numeric values representing one ECG beat"
)

# ─── SAMPLE DATA BUTTON ────────────────────────────────────────
col1, col2 = st.columns([1, 5])
with col1:
    use_sample = st.button("Use Sample Data", use_container_width=True)

# ─── INITIALIZE DATA ───────────────────────────────────────────
data = None

if use_sample:
    st.info("Using test_ecg_mixed.csv for demo purposes.")
    try:
        df_sample = pd.read_csv("test_ecg_mixed.csv", header=None)
        if df_sample.shape[1] == 188:
            df_sample = df_sample.iloc[:, :-1]
        data = df_sample.values
        st.success(f"Loaded {len(data)} sample beats (mixed normal + abnormal)")
    except:
        st.error("test_ecg_mixed.csv not found. Please upload a file manually.")

# ─── PROCESS UPLOADED FILE ─────────────────────────────────────
elif uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, header=None)

        # Drop label column if present
        if df.shape[1] == 188:
            df = df.iloc[:, :-1]
            st.caption("Label column detected and removed automatically.")

        if df.shape[1] != 187:
            st.error(f" Expected 187 columns, got {df.shape[1]}. Please check your file.")
        else:
            data = df.values
            st.success(f"Loaded **{len(data)} heartbeats** successfully.")
    except Exception as e:
        st.error(f"Error reading file: {e}")

# ─── RUN DETECTION ─────────────────────────────────────────────
if data is not None:
    with st.spinner("Running anomaly detection..."):

        # Scale & reconstruct
        X_scaled       = scaler.transform(data)
        X_reconstructed = model.predict(X_scaled, verbose=0)

        # Compute per-beat reconstruction error
        errors = np.mean(np.power(X_scaled - X_reconstructed, 2), axis=1)

        # Classify using selected threshold
        predictions = (errors > custom_threshold).astype(int)
        labels      = ["🔴 ABNORMAL" if p == 1 else "🟢 NORMAL" for p in predictions]

        n_total    = len(predictions)
        n_abnormal = int(predictions.sum())
        n_normal   = n_total - n_abnormal
        pct        = (n_abnormal / n_total) * 100

    # ── SUMMARY BANNER ─────────────────────────────────────────
    st.markdown("---")

    if n_abnormal == 0:
        st.success(f"## All {n_total} beats appear NORMAL — No anomalies detected.")
    elif pct > 60:
        st.error(f"## {n_abnormal} out of {n_total} beats flagged as ABNORMAL ({pct:.1f}%)")
    else:
        st.warning(f"## {n_abnormal} out of {n_total} beats flagged as ABNORMAL ({pct:.1f}%)")

    # ── METRIC CARDS ───────────────────────────────────────────
    st.markdown("### Summary")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Total Beats", n_total, )
    with m2:
        st.metric("Normal Beats", n_normal, delta=None)
    with m3:
        st.metric("Abnormal Beats", n_abnormal, delta=f"{pct:.1f}% of total",
                  delta_color="inverse")
    with m4:
        avg_err = float(np.mean(errors))
        st.metric("Avg Reconstruction Error", f"{avg_err:.5f}")

    st.markdown("---")

    # ── ERROR DISTRIBUTION CHART ───────────────────────────────
    st.markdown("### Reconstruction Error Distribution")

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=errors,
        nbinsx=50,
        name="Reconstruction Error",
        marker_color="#3b82f6",
        opacity=0.75
    ))
    fig_dist.add_vline(
        x=custom_threshold,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"Threshold = {custom_threshold:.5f}",
        annotation_position="top right",
        annotation_font_color="red"
    )
    fig_dist.update_layout(
        template="plotly_dark",
        xaxis_title="MSE Reconstruction Error",
        yaxis_title="Count",
        height=300,
        margin=dict(t=20, b=40),
        showlegend=False
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # ── ECG BEAT VISUALIZATIONS ────────────────────────────────
    st.markdown(f"### ECG Beat Analysis (showing first {min(max_beats_to_show, n_total)} beats)")

    beats_to_show = min(max_beats_to_show, n_total)

    for i in range(beats_to_show):
        original      = X_scaled[i]
        reconstructed = X_reconstructed[i]
        error         = errors[i]
        is_abnormal   = predictions[i] == 1

        color      = "#ef4444" if is_abnormal else "#22c55e"
        status     = "🔴 ABNORMAL" if is_abnormal else "🟢 NORMAL"
        fill_color = "rgba(239,68,68,0.3)" if is_abnormal else "rgba(34,197,94,0.3)"

        with st.expander(f"Beat #{i+1} — {status} | Error: {error:.5f}", expanded=(i < 3)):

            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("ECG Signal vs Reconstruction", "Absolute Error per Time Step"),
                vertical_spacing=0.15,
                row_heights=[0.6, 0.4]
            )

            time_steps = list(range(187))

            # Top: Original vs Reconstructed
            fig.add_trace(go.Scatter(
                x=time_steps, y=original.tolist(),
                mode='lines', name='Original ECG',
                line=dict(color='#60a5fa', width=1.5)
            ), row=1, col=1)

            if show_reconstruction:
                fig.add_trace(go.Scatter(
                    x=time_steps, y=reconstructed.tolist(),
                    mode='lines', name='Reconstructed',
                    line=dict(color='#f59e0b', width=1.5, dash='dash')
                ), row=1, col=1)

            # Threshold line on top plot
            fig.add_hline(y=custom_threshold, line_dash="dot",
                          line_color="gray", opacity=0.4, row=1, col=1)

            # Bottom: Absolute error filled
            abs_error = np.abs(original - reconstructed).tolist()
            fig.add_trace(go.Scatter(
                x=time_steps, y=abs_error,
                fill='tozeroy',
                fillcolor=fill_color,
                mode='lines',
                line=dict(color=color, width=1),
                name='|Error|'
            ), row=2, col=1)

            fig.update_layout(
                template="plotly_dark",
                height=420,
                margin=dict(t=40, b=20, l=40, r=20),
                legend=dict(orientation="h", y=1.12),
                showlegend=True
            )
            fig.update_xaxes(title_text="Time Steps", row=2, col=1)
            fig.update_yaxes(title_text="Amplitude",  row=1, col=1)
            fig.update_yaxes(title_text="|Error|",    row=2, col=1)

            st.plotly_chart(fig, use_container_width=True)

    # ── RESULTS TABLE ──────────────────────────────────────────
    st.markdown("### Full Results Table")

    results_df = pd.DataFrame({
        "Beat #"               : range(1, n_total + 1),
        "Reconstruction Error" : [f"{e:.6f}" for e in errors],
        "Status"               : labels,
        "Error vs Threshold"   : [f"{e - custom_threshold:+.6f}" for e in errors]
    })

    def highlight_rows(row):
        if "ABNORMAL" in row["Status"]:
            return ['background-color: rgba(239,68,68,0.15)'] * len(row)
        return ['background-color: rgba(34,197,94,0.08)'] * len(row)

    st.dataframe(
        results_df.style.apply(highlight_rows, axis=1),
        use_container_width=True,
        height=400
    )

    # ── DOWNLOAD RESULTS ───────────────────────────────────────
    st.markdown("### Export Results")
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="ecg_anomaly_results.csv",
        mime="text/csv",
        use_container_width=False
    )

# ─── EMPTY STATE ───────────────────────────────────────────────
else:
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; padding: 60px 20px; color: #6b7280;'>
        <div style='font-size: 4rem;'>🫀</div>
        <h3 style='color: #9ca3af;'>Upload an ECG CSV file to begin analysis</h3>
        <p>Each row = one heartbeat &nbsp;|&nbsp; 187 signal values per row</p>
        <p>Or click <b>Use Sample Data</b> to see a demo</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### About This Project")
st.markdown("""
<div style='background: linear-gradient(135deg, #1e2130, #2a2d3e); 
            border-radius: 12px; padding: 24px; border: 1px solid #3a3f5c;'>

<p style='color:#e2e8f0; font-size:1rem; font-weight:600; margin-bottom:12px;'>
ECG Anomaly Detection using Autoencoders</p>

<p style='color:#94a3b8; margin: 6px 0;'><b style='color:#cbd5e1;'>Model:</b> Dense Autoencoder (187 → 128 → 64 → 32 → 64 → 128 → 187)</p>
<p style='color:#94a3b8; margin: 6px 0;'><b style='color:#cbd5e1;'>Training:</b> Trained exclusively on normal ECG beats (unsupervised)</p>
<p style='color:#94a3b8; margin: 6px 0;'><b style='color:#cbd5e1;'>Detection:</b> Anomalies flagged via MSE reconstruction error threshold</p>
<p style='color:#94a3b8; margin: 6px 0;'><b style='color:#cbd5e1;'>Dataset:</b> PTB Diagnostic ECG Database (PTBDB)</p>
<p style='color:#94a3b8; margin: 6px 0;'><b style='color:#cbd5e1;'>Accuracy:</b> 72% &nbsp;|&nbsp; <b style='color:#cbd5e1;'>ROC-AUC:</b> 0.817</p>
<p style='color:#94a3b8; margin: 6px 0;'><b style='color:#cbd5e1;'>Threshold:</b> 80th percentile of normal reconstruction errors</p>

<div style='margin-top:16px; padding: 12px 16px; 
            border-left: 3px solid #3b82f6; background: rgba(59,130,246,0.08); 
            border-radius: 0 8px 8px 0;'>
<p style='color:#93c5fd; font-style:italic; margin:0;'>
"A heartbeat the model has never seen before will be poorly reconstructed — 
revealing itself through high reconstruction error."</p>
</div>

</div>
""", unsafe_allow_html=True)