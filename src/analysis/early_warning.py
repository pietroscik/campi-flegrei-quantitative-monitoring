import pandas as pd
import numpy as np

# -----------------------------
# Threshold dinamici (robusti)
# -----------------------------

def compute_thresholds(series):

    q25 = series.quantile(0.25)
    q50 = series.quantile(0.50)
    q75 = series.quantile(0.75)
    q90 = series.quantile(0.90)

    return {
        "low": q25,
        "baseline": q50,
        "attention": q75,
        "alert": q90
    }


# -----------------------------
# Classification function
# -----------------------------

def classify_unrest(series, thresholds):

    conditions = [
        series <= thresholds["baseline"],
        (series > thresholds["baseline"]) & (series <= thresholds["attention"]),
        (series > thresholds["attention"]) & (series <= thresholds["alert"]),
        series > thresholds["alert"]
    ]

    labels = ["NORMAL", "ELEVATED", "HIGH", "CRITICAL"]

    return np.select(conditions, labels)


# -----------------------------
# Rolling stability check
# -----------------------------

def compute_persistence(series, window=7):

    return series.rolling(window).apply(lambda x: (x > 0).mean(), raw=True)


# -----------------------------
# Main pipeline
# -----------------------------

def run_warning_system(input_path="data/processed/unrest_index.csv"):

    df = pd.read_csv(input_path)
    df["time"] = pd.to_datetime(df["time"])

    thresholds = compute_thresholds(df["unrest_index"])

    df["state"] = classify_unrest(df["unrest_index"], thresholds)

    # persistence del segnale (stabilità dell'anomalia)
    df["persistence_7d"] = compute_persistence(df["unrest_index"])

    # alert flag finale (robusto)
    df["alert_flag"] = np.where(
        (df["state"] == "CRITICAL") &
        (df["persistence_7d"] > 0.6),
        1, 0
    )

    df.to_csv("data/processed/early_warning_system.csv", index=False)

    print("[OK] Early warning system saved -> data/processed/early_warning_system.csv")

    print(df["state"].value_counts())

    return df


if _name_ == "_main_":
    run_warning_system()
