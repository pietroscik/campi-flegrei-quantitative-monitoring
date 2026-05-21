import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Campi Flegrei Monitoring", layout="wide")

# -----------------------------
# LOAD DATA
# -----------------------------

@st.cache_data
def load_data():
    catalog = pd.read_csv("../data/processed/catalog_clean.csv")
    bval = pd.read_csv("../data/processed/b_value_rolling.csv")
    etas = pd.read_csv("../data/processed/etas_output.csv")
    unrest = pd.read_csv("../data/processed/unrest_index.csv")
    warning = pd.read_csv("../data/processed/early_warning_system.csv")

    catalog["time"] = pd.to_datetime(catalog["time"])
    bval["time"] = pd.to_datetime(bval["time"])
    etas["time"] = pd.to_datetime(etas["time"])
    unrest["time"] = pd.to_datetime(unrest["time"])
    warning["time"] = pd.to_datetime(warning["time"])

    return catalog, bval, etas, unrest, warning


catalog, bval, etas, unrest, warning = load_data()

# -----------------------------
# SIDEBAR FILTER
# -----------------------------

st.sidebar.title("Filters")

min_mag = st.sidebar.slider("Min Magnitude", 0.0, 5.0, 0.0)

filtered = catalog[catalog["magnitude"] >= min_mag]

# -----------------------------
# KPI SECTION
# -----------------------------

st.title("🌋 Campi Flegrei Quantitative Monitoring")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Events", len(filtered))
col2.metric("Max Magnitude", float(filtered["magnitude"].max()))
col3.metric("Latest b-value", float(bval["b_value"].iloc[-1]))
col4.metric("Alert Level", warning["state"].iloc[-1])

# -----------------------------
# SEISMICITY PLOT
# -----------------------------

st.subheader("Seismicity Timeline")

fig1 = px.scatter(
    filtered,
    x="time",
    y="magnitude",
    title="Events over Time",
    opacity=0.6
)

st.plotly_chart(fig1, use_container_width=True)

# -----------------------------
# B-VALUE
# -----------------------------

st.subheader("Rolling b-value")

fig2 = px.line(
    bval,
    x="time",
    y="b_value",
    title="Gutenberg-Richter b-value"
)

st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# ETAS
# -----------------------------

st.subheader("ETAS Intensity")

fig3 = px.line(
    etas,
    x="time",
    y="lambda_etas",
    title="ETAS Conditional Intensity"
)

st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# UNREST INDEX
# -----------------------------

st.subheader("Unrest Index (Multi-Signal)")

fig4 = px.line(
    unrest,
    x="time",
    y="unrest_index",
    title="Integrated Unrest Index"
)

st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# WARNING STATES
# -----------------------------

st.subheader("Early Warning System")

st.dataframe(warning.tail(20))

alert_count = warning["alert_flag"].sum()

st.warning(f"Active Alerts: {alert_count}")
