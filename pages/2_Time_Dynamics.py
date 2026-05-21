import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📈 Dynamic System Evolution")

bval = pd.read_csv("../data/processed/b_value_rolling.csv")
etas = pd.read_csv("../data/processed/etas_output.csv")
unrest = pd.read_csv("../data/processed/unrest_index.csv")

bval["time"] = pd.to_datetime(bval["time"])
etas["time"] = pd.to_datetime(etas["time"])
unrest["time"] = pd.to_datetime(unrest["time"])

fig = px.line(bval, x="time", y="b_value", title="b-value evolution")
st.plotly_chart(fig, use_container_width=True)

fig2 = px.line(etas, x="time", y="lambda_etas", title="ETAS intensity")
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.line(unrest, x="time", y="unrest_index", title="Unrest Index")
st.plotly_chart(fig3, use_container_width=True)
