import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

# =========================
# LOAD PREDICTIONS
# =========================

df = pd.read_csv("data/processed/probabilistic_risk_score.csv")
df["time"] = pd.to_datetime(df["time"])

etas = pd.read_csv("data/processed/etas_output.csv")
etas["time"] = pd.to_datetime(etas["time"])

# =========================
# GROUND TRUTH (7-day horizon)
# =========================

event_times = etas[etas["magnitude"] >= 1.5]["time"]

def label_event(t):
    return np.any((event_times > t) & (event_times <= t + pd.Timedelta(days=7)))

df["event"] = df["time"].apply(label_event).astype(int)

# =========================
# RAW PROBABILITY
# =========================

p_raw = df["p_event"].values
y = df["event"].values

# =========================
# ISOTONIC CALIBRATION
# =========================

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(p_raw, y)

df["p_calibrated"] = iso.transform(p_raw)

# =========================
# METRICS BEFORE / AFTER
# =========================

brier_raw = brier_score_loss(y, p_raw)
brier_cal = brier_score_loss(y, df["p_calibrated"])

print("\n=== FINAL CALIBRATION ===")

print("Brier (raw):", brier_raw)
print("Brier (calibrated):", brier_cal)

# =========================
# CALIBRATION SHIFT
# =========================

print("\n=== DISTRIBUTION SHIFT ===")
print("Mean raw prob:", p_raw.mean())
print("Mean calibrated prob:", df["p_calibrated"].mean())
print("Observed event rate:", y.mean())

# =========================
# HIGH RISK CHECK
# =========================

thr = np.quantile(df["p_calibrated"], 0.95)

high = df[df["p_calibrated"] >= thr]

print("\n=== HIGH RISK REGIME ===")
print("High-risk event rate:", high["event"].mean())
print("High-risk mean raw:", high["p_event"].mean())

# =========================
# SAVE OUTPUT
# =========================

df[["time", "p_event", "p_calibrated"]].to_csv(
    "data/processed/probabilistic_risk_score_calibrated.csv",
    index=False
)

print("\nSaved: probabilistic_risk_score_calibrated.csv")
