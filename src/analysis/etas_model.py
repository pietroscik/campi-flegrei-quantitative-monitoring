import pandas as pd
import numpy as np
import os

# -----------------------------
# ETAS kernel function
# -----------------------------

def kernel(t, ti, m, M0=0.0, K=0.5, alpha=1.0, c=0.01, p=1.2):

    return K * np.exp(alpha * (m - M0)) * (t - ti + c) ** (-p)


# -----------------------------
# Intensity function λ(t)
# -----------------------------

def compute_intensity(df, M0=None, mu=0.1, K=0.5, alpha=1.0, c=0.01, p=1.2):

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    # Convert to numeric days since first event
    t0 = df["time"].iloc[0]
    times_days = (df["time"] - t0).dt.total_seconds().values / 86400.0
    mags = df["magnitude"].values
    
    if M0 is None:
        M0 = np.min(mags)

    lambda_t = np.zeros(len(df))
    
    # Precompute productivity term
    A = K * np.exp(alpha * (mags - M0))

    for i in range(len(df)):
        if i == 0:
            lambda_t[i] = mu
        else:
            # Vectorized calculation for all previous events j < i
            dt = times_days[i] - times_days[:i]
            lambda_t[i] = mu + np.sum(A[:i] * (dt + c) ** (-p))

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

def run_etas(input_path="data/processed/catalog_clean.csv", params_path="data/processed/etas_params.csv"):

    df = pd.read_csv(input_path)

    df["time"] = pd.to_datetime(df["time"])

    # Carica i parametri ottimizzati da MLE se esistono
    if os.path.exists(params_path):
        params = pd.read_csv(params_path).iloc[0].to_dict()
        df = compute_intensity(
            df, 
            mu=params.get('mu', 0.1),
            K=params.get('K', 0.5), 
            alpha=params.get('alpha', 1.0), 
            c=params.get('c', 0.01), 
            p=params.get('p', 1.2)
        )
    else:
        df = compute_intensity(df)

    df = etas_signal(df)

    df.to_csv("data/processed/etas_output.csv", index=False)

    print("[OK] ETAS model saved -> data/processed/etas_output.csv")
    print(df["etas_alert"].value_counts())

    return df


if __name__ == "__main__":
    run_etas()
