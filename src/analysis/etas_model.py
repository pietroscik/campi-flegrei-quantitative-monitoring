import pandas as pd
import numpy as np

# -----------------------------
# ETAS kernel function
# -----------------------------

def kernel(t, ti, m, M0=0.0, K=0.5, alpha=1.0, c=0.01, p=1.2):

    return K * np.exp(alpha * (m - M0)) * (t - ti + c) ** (-p)


# -----------------------------
# Intensity function λ(t)
# -----------------------------

def compute_intensity(df, M0=0.0, K=0.5, alpha=1.0, c=0.01, p=1.2):

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    times = df["time"].values
    mags = df["magnitude"].values

    lambda_t = []

    for i in range(len(df)):

        t = times[i]

        intensity = 0.0

        # background rate (μ)
        mu = 0.1

        for j in range(i):

            ti = times[j]

            dt = (t - ti).astype('timedelta64[s]').astype(float) / 86400.0  # days

            if dt <= 0:
                continue

            intensity += kernel(dt, 0, mags[j], M0, K, alpha, c, p)

        lambda_t.append(mu + intensity)

    df["lambda_etas"] = lambda_t

    return df


# -----------------------------
# Normalization / alert signal
# -----------------------------

def etas_signal(df):

    df["etas_z"] = (df["lambda_etas"] - df["lambda_etas"].mean()) / df["lambda_etas"].std()

    df["etas_alert"] = (df["etas_z"] > 2).astype(int)

    return df


# -----------------------------
# PIPELINE
# -----------------------------

def run_etas(input_path="data/processed/catalog_clean.csv"):

    df = pd.read_csv(input_path)

    df["time"] = pd.to_datetime(df["time"])

    df = compute_intensity(df)

    df = etas_signal(df)

    df.to_csv("data/processed/etas_output.csv", index=False)

    print("[OK] ETAS model saved -> data/processed/etas_output.csv")
    print(df["etas_alert"].value_counts())

    return df


if __name__ == "__main__":
    run_etas()
