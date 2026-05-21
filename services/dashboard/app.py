import streamlit as st
import requests
import plotly.express as px

st.title("🌋 CF REAL OBSERVATORY LIVE")

# -------------------------
# LIVE STATUS FROM API
# -------------------------

status = requests.get("http://localhost:8000/status").json()

st.metric("State", status["state"])
st.metric("Alert", status["alert_flag"])
st.metric("Unrest Index", status["unrest_index"])

# -------------------------
# LIVE CHART (UNREST)
# -------------------------

unrest = requests.get("http://localhost:8000/unrest").json()

fig = px.line(unrest, x="time", y="unrest_index", title="Live Unrest Index")

st.plotly_chart(fig, use_container_width=True)
