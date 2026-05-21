import pandas as pd
import numpy as np

# -----------------------------
# SIGNAL 1: seismic rate
# -----------------------------

def compute_seismic_rate(df, freq="D"):

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")

    rate = df.resample(freq).size()
    rate = rate.rename("seismic_rate")

    return rate


# -----------------------------
# SIGNAL 2: b-value alignment
# -----------------------------

def align_bvalue(b_df):

    b_df = b_df.copy()
    b_df["time"] = pd.to_datetime(b_df["time"])
    b_df = b_df.set_index("time")

    return b_df["b_value"].rename("b_value")


# -----------------------------
# SIGNAL 3: uplift (RITE / GNSS)
# -----------------------------

def align_uplift(uplift_df):

    uplift_df = uplift_df.copy()
    uplift_df["time"] = pd.to_datetime(uplift_df["time"])
    uplift_df = uplift_df.set_index("time")

    return uplift_df["uplift"].rename("uplift")


# -----------------------------
# Normalization
# -----------------------------

def normalize(series):

    return (series - series.mean()) / series.std()


# -----------------------------
# Multi-signal fusion index
# -----------------------------

def build_unrest_index(seismic_rate, b_value, uplift):

    df = pd.concat([seismic_rate, b_value, uplift], axis=1)

    df = df.dropna()

    df["rate_n"] = normalize(df["seismic_rate"])
    df["b_n"] = normalize(df["b_value"])
    df["uplift_n"] = normalize(df["uplift"])

    # indice composito (pesi iniziali teorici)
    df["unrest_index"] = (
        0.4 * df["rate_n"] +
        0.3 * (-df["b_n"]) +   # b-value basso = stress ↑
        0.3 * df["uplift_n"]
    )

    return df


# -----------------------------
# PIPELINE
# -----------------------------

def run_multisignal(
    catalog_path="data/processed/catalog_clean.csv",
    b_path="data/processed/b_value_rolling.csv",
    uplift_path="data/external/uplift.csv"
):

    catalog = pd.read_csv(catalog_path)
    b_df = pd.read_csv(b_path)
    uplift_df = pd.read_csv(uplift_path)

    seismic_rate = compute_seismic_rate(catalog)
    b_value = align_bvalue(b_df)
    uplift = align_uplift(uplift_df)

    df = build_unrest_index(seismic_rate, b_value, uplift)

    df.to_csv("data/processed/unrest_index.csv")

    print("[OK] Multi-signal index saved -> data/processed/unrest_index.csv")

    return df


if _name_ == "_main_":
    run_multisignal()
