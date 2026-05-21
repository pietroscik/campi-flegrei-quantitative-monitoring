import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🧮 ETAS Advanced View")

df = pd.read_csv("../data/processed/etas_output.csv")
df["time"] = pd.to_datetime(df["time"])

fig = px.scatter(
    df,
    x="time",
    y="lambda_etas",
    color="etas_alert",
    title="ETAS Intensity + Alerts"
)

st.plotly_chart(fig, use_container_width=True)
