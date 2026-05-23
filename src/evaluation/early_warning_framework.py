import pandas as pd
import numpy as np
from scipy import stats

# =========================
# LOAD DATA
# =========================

unrest = pd.read_csv("data/processed/unrest_index.csv")
dl = pd.read_csv("data/processed/dl_anomalies.csv")
cp = pd.read_csv("data/processed/changepoint_output.csv")
etas = pd.read_csv("data/processed/etas_output.csv")

unrest["time"] = pd.to_datetime(unrest["time"])
dl["time"] = pd.to_datetime(dl["time"])
cp["time"] = pd.to_datetime(cp["time"])
etas["time"] = pd.to_datetime(etas["time"])

# =========================
# GROUND TRUTH EVENTS
# =========================

events = etas[etas["magnitude"] >= 1.5][["time"]].copy()
events["event"] = 1

df = unrest.merge(events, on="time", how="left")
df["event"] = df["event"].fillna(0)

df = df.merge(dl[["time","dl_is_anomaly"]], on="time", how="left")
df = df.merge(cp[["time","changepoint"]], on="time", how="left")

df = df.fillna(0)
df = df.sort_values("time")

# =========================
# 1. BUILD COMPOSITE EARLY WARNING SCORE
# =========================

# normalize signals
df["u_norm"] = (df["unrest_index"] - df["unrest_index"].mean()) / df["unrest_index"].std()
df["dl_norm"] = df["dl_is_anomaly"]
df["cp_norm"] = df["changepoint"]

# weights (can be optimized later)
w1, w2, w3 = 0.5, 0.3, 0.2

df["ews_score"] = (
    w1 * df["u_norm"] +
    w2 * df["dl_norm"] +
    w3 * df["cp_norm"]
)

# =========================
# 2. THRESHOLD DEFINITION
# =========================

theta = df["ews_score"].quantile(0.9)

df["alert"] = (df["ews_score"] >= theta).astype(int)

# =========================
# 3. VALIDATION METRICS
# =========================

print("\n=== EARLY WARNING FRAMEWORK ===")

# event detection
tp = len(df[(df["alert"] == 1) & (df["event"] == 1)])
fp = len(df[(df["alert"] == 1) & (df["event"] == 0)])
fn = len(df[(df["alert"] == 0) & (df["event"] == 1)])

precision = tp / (tp + fp + 1e-9)
recall = tp / (tp + fn + 1e-9)

print("Precision:", precision)
print("Recall:", recall)

# =========================
# 4. FALSE ALARM RATE
# =========================

far = fp / (fp + len(df[df["event"] == 0]))
print("False Alarm Rate:", far)

# =========================
# 5. LEAD TIME ESTIMATE (SIMPLE)
# =========================

lead_times = []

for _, ev in df[df["event"] == 1].iterrows():
    t = ev["time"]
    pre = df[df["time"] < t]

    alerts = pre[pre["alert"] == 1]

    if len(alerts) > 0:
        t_alert = alerts["time"].iloc[-1]
        lead = (t - t_alert).days
        lead_times.append(lead)

if len(lead_times) > 0:
    print("\nLead time mean:", np.mean(lead_times))
    print("Lead time median:", np.median(lead_times))

# =========================
# SUMMARY
# =========================

print("\n=== SUMMARY ===")
print("✔ Composite Early Warning System built")
print("✔ Precision/Recall computed")
print("✔ False alarm rate estimated")
print("✔ Lead time estimated")
