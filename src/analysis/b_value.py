import pandas as pd
import numpy as np

# -----------------------------
# Gutenberg-Richter b-value
# b = log10(e) / (mean(M) - M0)
# -----------------------------

def compute_b_value(magnitudes, m0=0.0):
    magnitudes = np.array(magnitudes)
    magnitudes = magnitudes[magnitudes >= m0]

    if len(magnitudes) < 10:
        return np.nan

    mean_m = np.mean(magnitudes)

    if mean_m == m0:
        return np.nan

    b = (np.log10(np.e)) / (mean_m - m0)
    return b


# -----------------------------
# Rolling b-value
# -----------------------------

def rolling_b_value(df, window=100, m0=0.0):

    df = df.sort_values("time").reset_index(drop=True)

    b_values = []
    times = []

    mags = df["magnitude"].values
    t = df["time"].values

    for i in range(window, len(df)):
        window_mags = mags[i-window:i]

        b = compute_b_value(window_mags, m0=m0)

        b_values.append(b)
        times.append(t[i])

    return pd.DataFrame({
        "time": times,
        "b_value": b_values
    })


# -----------------------------
# Global b-value
# -----------------------------

def global_b_value(df, m0=0.0):
    return compute_b_value(df["magnitude"], m0=m0)


# -----------------------------
# Pipeline runner
# -----------------------------

def run_b_analysis(input_path="data/processed/catalog_clean.csv"):

    df = pd.read_csv(input_path)
    df["time"] = pd.to_datetime(df["time"])

    global_b = global_b_value(df)

    rolling = rolling_b_value(df, window=100)

    rolling.to_csv("data/processed/b_value_rolling.csv", index=False)

    print(f"[OK] Global b-value: {global_b:.4f}")
    print(f"[OK] Rolling series saved: data/processed/b_value_rolling.csv")

    return global_b, rolling


if __name__ == "__main__":
    run_b_analysis()
