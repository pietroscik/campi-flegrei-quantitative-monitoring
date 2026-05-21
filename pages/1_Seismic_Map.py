import streamlit as st
import pandas as pd
import pydeck as pdk

st.title("🌍 Seismicity Map - Campi Flegrei")

df = pd.read_csv("../data/processed/catalog_clean.csv")

df = df.dropna(subset=["latitude", "longitude"])

layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position='[longitude, latitude]',
    get_radius=50,
    get_color='[255, 80, 80]',
    pickable=True
)

view_state = pdk.ViewState(
    latitude=40.85,
    longitude=14.14,
    zoom=10,
    pitch=30
)

st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
