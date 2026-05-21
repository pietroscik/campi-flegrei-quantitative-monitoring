import streamlit as st
import pandas as pd

st.title("🚨 Alert Center")

df = pd.read_csv("../data/processed/early_warning_system.csv")

df["time"] = pd.to_datetime(df["time"])

latest = df.tail(1)

st.metric("Current State", latest["state"].values[0])
st.metric("Alert Flag", int(latest["alert_flag"].values[0]))

st.subheader("Recent Activity")

st.dataframe(df.tail(30))

# ALERT LOGIC VISUAL
critical = df[df["state"] == "CRITICAL"]

st.warning(f"Critical Events: {len(critical)}")
