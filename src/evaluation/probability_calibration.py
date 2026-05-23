import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("data/processed/probabilistic_risk_score.csv")
df["time"] = pd.to_datetime(df["time"])

etas = pd.read_csv("data/processed/etas_output.csv")
etas["time"] = pd.to_datetime(etas["time"])

# =========================
# GROUND TRUTH (same window logic)
# =========================

events = etas[etas["magnitude"] >= 1.5]["time"]

def label_event(t):
    return np.any((events > t) & (events <= t + pd.Timedelta(days=7)))

df["event"] = df["time"].apply(label_event).astype(int)

# =========================
# BRIER SCORE
# =========================

bs = brier_score_loss(df["event"], df["p_event"])

print("\n=== CALIBRATION METRICS ===")
print("Brier Score:", bs)

# =========================
# CALIBRATION CURVE
# =========================

prob_true, prob_pred = calibration_curve(
    df["event"],
    df["p_event"],
    n_bins=10,
    strategy="quantile"
)

# =========================
# PLOT
# =========================

plt.figure()
plt.plot(prob_pred, prob_true, marker="o", label="Model")
plt.plot([0,1],[0,1], linestyle="--", label="Perfect calibration")
plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.title("Calibration Curve - Seismic Risk Model")
plt.legend()
plt.tight_layout()

plt.savefig("results/calibration_curve.png")

# =========================
# SUMMARY DIAGNOSIS
# =========================

print("\n=== RELIABILITY DIAGNOSIS ===")

bias = df["p_event"].mean() - df["event"].mean()

print("Mean predicted probability:", df["p_event"].mean())
print("Observed event rate:", df["event"].mean())
print("Calibration bias:", bias)

if bias > 0:
    print("⚠️ Model is OVERCONFIDENT")
else:
    print("⚠️ Model is UNDERCONFIDENT or conservative")

print("\nCalibration curve saved to: results/calibration_curve.png")
