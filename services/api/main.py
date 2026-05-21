from fastapi import FastAPI
import pandas as pd
import time

app = FastAPI(title="CF Observatory API")

def load_latest():
    return pd.read_csv("../data/processed/early_warning_system.csv")


@app.get("/status")
def status():
    df = load_latest()
    latest = df.tail(1).to_dict(orient="records")[0]
    return latest


@app.get("/unrest")
def unrest():
    df = pd.read_csv("../data/processed/unrest_index.csv")
    return df.tail(200).to_dict(orient="records")


@app.get("/etas")
def etas():
    df = pd.read_csv("../data/processed/etas_output.csv")
    return df.tail(200).to_dict(orient="records")
