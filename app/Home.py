
from pathlib import Path
import json

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics"

COMPARISON_FILE = METRICS_PATH / "forecasting_comparison.csv"
ANOMALY_FILE = METRICS_PATH / "gru_anomaly_results.csv"
SYNTHETIC_FILE = METRICS_PATH / "synthetic_metrics.json"

st.set_page_config(
    page_title="Electricity Intelligence",
    page_icon="⚡",
    layout="wide"
)


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

@st.cache_data
def load_data():
    comparison = pd.read_csv(COMPARISON_FILE)

    anomalies = pd.read_csv(ANOMALY_FILE)
    anomalies["timestamp"] = pd.to_datetime(anomalies["timestamp"])

    anomalies["is_anomaly"] = (
        anomalies["is_anomaly"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )

    with open(SYNTHETIC_FILE, "r") as file:
        synthetic = json.load(file)

    return comparison, anomalies, synthetic


try:
    comparison, anomalies, synthetic = load_data()
except Exception as error:
    st.error(f"Could not load project data: {error}")
    st.stop()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚡ Electricity Intelligence")
st.sidebar.caption("Deep Learning | Anomaly Detection")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Consumption Forecasting",
        "Anomaly Detection",
        "Model Comparison",
        "Synthetic Experiment"
    ]
)

st.sidebar.divider()
st.sidebar.caption("LSTM • GRU • Residual Analysis")


# --------------------------------------------------
# COMMON VALUES
# --------------------------------------------------

gru_row = comparison.loc[comparison["Model"] == "GRU"].iloc[0]

gru_mae = float(gru_row["MAE (kW)"])
gru_rmse = float(gru_row["RMSE (kW)"])
gru_r2 = float(gru_row["R2"])

threshold = float(anomalies["absolute_residual"].quantile(0.99))
anomaly_count = int(anomalies["is_anomaly"].sum())
total_observations = len(anomalies)

# Prefer the calibrated threshold stored in synthetic metrics,
# if available, rather than recalculating it from test data.
threshold = float(synthetic.get("threshold_kw", threshold))


# --------------------------------------------------
# PAGE 1: OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.title("Electricity Consumption Intelligence")
    st.markdown(
        "### Household electricity forecasting and anomaly monitoring"
    )

    st.write(
        "An LSTM and GRU based forecasting system that identifies "
        "unusual electricity consumption patterns through prediction residuals."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Test Observations", f"{total_observations:,}")
    col2.metric("Flagged Intervals", f"{anomaly_count:,}")
    col3.metric("GRU MAE", f"{gru_mae:.4f} kW")
    col4.metric("Detection Threshold", f"{threshold:.4f} kW")

    st.divider()

    st.subheader("Actual vs Predicted Consumption")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=anomalies["timestamp"],
        y=anomalies["actual_consumption"],
        name="Actual Consumption",
        mode="lines"
    ))

    fig.add_trace(go.Scatter(
        x=anomalies["timestamp"],
        y=anomalies["predicted_consumption"],
        name="GRU Prediction",
        mode="lines"
    ))

    fig.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Global Active Power (kW)",
        hovermode="x unified",
        template="plotly_dark",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Highest Residual Intervals")

    top_anomalies = (
        anomalies[anomalies["is_anomaly"] == True]
        .nlargest(10, "absolute_residual")
    )

    st.dataframe(top_anomalies, use_container_width=True)


# --------------------------------------------------
# PAGE 2: FORECASTING
# --------------------------------------------------

elif page == "Consumption Forecasting":

    st.title("Consumption Forecasting")

    st.write(
        "The GRU model uses the preceding 24 hours of electricity "
        "measurements to forecast the next 15-minute consumption interval."
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=anomalies["timestamp"],
        y=anomalies["actual_consumption"],
        name="Actual",
        mode="lines"
    ))

    fig.add_trace(go.Scatter(
        x=anomalies["timestamp"],
        y=anomalies["predicted_consumption"],
        name="GRU Predicted",
        mode="lines"
    ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Timestamp",
        yaxis_title="Global Active Power (kW)",
        hovermode="x unified",
        height=550
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("GRU Forecasting Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", f"{gru_mae:.4f} kW")
    c2.metric("RMSE", f"{gru_rmse:.4f} kW")
    c3.metric("R² Score", f"{gru_r2:.4f}")


# --------------------------------------------------
# PAGE 3: ANOMALY DETECTION
# --------------------------------------------------

elif page == "Anomaly Detection":

    st.title("Residual-Based Anomaly Detection")

    st.write(
        "An interval is flagged when its absolute prediction residual "
        "exceeds the threshold calibrated using validation data."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Threshold", f"{threshold:.4f} kW")
    c2.metric("Flagged Intervals", f"{anomaly_count:,}")
    c3.metric(
        "Flagged Percentage",
        f"{(anomaly_count / total_observations) * 100:.2f}%"
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=anomalies["timestamp"],
        y=anomalies["actual_consumption"],
        name="Actual Consumption",
        mode="lines"
    ))

    flagged = anomalies[anomalies["is_anomaly"] == True]

    fig.add_trace(go.Scatter(
        x=flagged["timestamp"],
        y=flagged["actual_consumption"],
        name="Flagged Intervals",
        mode="markers",
        marker=dict(color="red", size=7)
    ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Timestamp",
        yaxis_title="Global Active Power (kW)",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected Candidate Anomalies")

    st.dataframe(
        flagged.sort_values(
            "absolute_residual",
            ascending=False
        ),
        use_container_width=True
    )

    st.info(
        "Flagged intervals indicate unusual deviations from model "
        "predictions. They are candidates for investigation, not proof "
        "of electricity wastage or equipment faults."
    )


# --------------------------------------------------
# PAGE 4: MODEL COMPARISON
# --------------------------------------------------

elif page == "Model Comparison":

    st.title("Model Performance Comparison")

    st.write(
        "Comparison of persistence baseline, LSTM and GRU forecasting models."
    )

    st.dataframe(comparison, use_container_width=True)

    st.subheader("MAE and RMSE Comparison")

    metric_data = comparison.melt(
        id_vars="Model",
        value_vars=["MAE (kW)", "RMSE (kW)"],
        var_name="Metric",
        value_name="Error (kW)"
    )

    fig = px.bar(
        metric_data,
        x="Model",
        y="Error (kW)",
        color="Metric",
        barmode="group",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("R² Score Comparison")

    fig2 = go.Figure()

    fig2.add_trace(go.Bar(
        x=comparison["Model"],
        y=comparison["R2"],
        text=comparison["R2"].map(lambda x: f"{x:.4f}"),
        textposition="outside"
    ))

    fig2.update_layout(
        title="R² Score Comparison",
        xaxis_title="Model",
        yaxis_title="R² Score",
        template="plotly_dark"
    )

    st.plotly_chart(fig2, use_container_width=True)


# --------------------------------------------------
# PAGE 5: SYNTHETIC EXPERIMENT
# --------------------------------------------------

elif page == "Synthetic Experiment":

    st.title("Synthetic Anomaly Detection Experiment")

    st.write(
        "Controlled experiment evaluating whether the residual-based "
        "detector identifies artificially injected positive consumption spikes."
    )

    injected = int(synthetic["injected_anomalies"])
    detected = int(synthetic["detected_anomalies"])
    missed = int(synthetic["missed_anomalies"])
    detection_rate = float(synthetic["detection_rate_percent"])
    spike_magnitude = float(synthetic["spike_magnitude_kw"])

    c1, c2, c3 = st.columns(3)

    c1.metric("Injected Spikes", injected)
    c2.metric("Detected Spikes", detected)
    c3.metric("Detection Rate", f"{detection_rate:.1f}%")

    st.divider()

    c4, c5, c6 = st.columns(3)

    c4.metric("Missed Spikes", missed)
    c5.metric("Spike Magnitude", f"{spike_magnitude:.2f} kW")
    c6.metric("Threshold", f"{threshold:.4f} kW")

    st.subheader("Experiment Interpretation")

    st.write(
        f"The experiment injected {injected} artificial positive spikes "
        f"of {spike_magnitude:.2f} kW into the test consumption values. "
        f"The detector identified {detected} of them, giving a detection "
        f"rate of {detection_rate:.1f}%."
    )

    st.warning(
        "This is a controlled synthetic sensitivity experiment. "
        "It does not represent real-world anomaly detection accuracy "
        "because the dataset does not provide ground-truth anomaly labels."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Electricity Intelligence | Deep Learning Project | "
    "LSTM and GRU Based Forecasting"
)
