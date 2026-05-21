import pandas as pd
import numpy as np

# -----------------------------
# Rolling Z-score anomaly
# -----------------------------

def compute_zscore(series, window=50):

    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    z = (series - rolling_mean) / rolling_std

    return z


# -----------------------------
# Anomaly detection (quantile-based)
# -----------------------------

def compute_quantile_anomalies(series, q_low=0.05, q_high=0.95):

    low = series.quantile(q_low)
    high = series.quantile(q_high)

    anomalies = (series < low) | (series > high)

    return anomalies.astype(int)


# -----------------------------
# Combined anomaly score
# -----------------------------

def anomaly_score(df, window=50):

    df = df.copy()

    df["b_zscore"] = compute_zscore(df["b_value"], window=window)

    df["anomaly_z"] = (np.abs(df["b_zscore"]) > 2).astype(int)

    df["anomaly_q"] = compute_quantile_anomalies(df["b_value"])

    # score finale (0–2)
    df["anomaly_score"] = df["anomaly_z"] + df["anomaly_q"]

    return df


# -----------------------------
# Pipeline
# -----------------------------

def run_anomaly(input_path="data/processed/b_value_rolling.csv"):

    df = pd.read_csv(input_path)
    df["time"] = pd.to_datetime(df["time"])

    df = anomaly_score(df, window=50)

    df.to_csv("data/processed/b_value_anomalies.csv", index=False)

    print("[OK] anomaly dataset saved -> data/processed/b_value_anomalies.csv")
    print(df["anomaly_score"].value_counts())

    return df


if _name_ == "_main_":
    run_anomaly()
