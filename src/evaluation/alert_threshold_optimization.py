import pandas as pd
import numpy as np

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("data/processed/operational_alert_system.csv")
df["time"] = pd.to_datetime(df["time"])

events = pd.read_csv("data/processed/etas_output.csv")
events["time"] = pd.to_datetime(events["time"])
events = events[events["magnitude"] >= 1.5]

event_times = events["time"].values

# =========================
# EVENT LABELING
# =========================

def label_event(t, window=7):
    return np.any((event_times > t) & (event_times <= t + np.timedelta64(window, "D")))

df["event"] = df["time"].apply(label_event).astype(int)

# =========================
# THRESHOLD GRID (CRITICAL PART)
# =========================

thresholds = np.linspace(0.5, 0.99, 30)

results = []

for thr in thresholds:

    df["alert"] = (df["p_calibrated"] >= thr).astype(int)

    TP = ((df["alert"] == 1) & (df["event"] == 1)).sum()
    FP = ((df["alert"] == 1) & (df["event"] == 0)).sum()
    FN = ((df["alert"] == 0) & (df["event"] == 1)).sum()

    FAR = FP / max((FP + TP), 1)
    MISS = FN / max((FN + TP), 1)

    # lead time proxy (simplified)
    lead_times = []

    for t in event_times:
        past = df[df["time"] <= t]
        alerts = past[past["alert"] == 1]

        if len(alerts) > 0:
            lead_times.append((t - alerts["time"].max()).days)

    LEAD = np.mean(lead_times) if len(lead_times) > 0 else 0

    # objective function
    J = MISS + FAR - 0.01 * LEAD

    results.append([thr, FAR, MISS, LEAD, J])

res = pd.DataFrame(results, columns=["threshold", "FAR", "MISS", "LEAD", "J"])

# =========================
# OPTIMAL THRESHOLD
# =========================

best = res.loc[res["J"].idxmin()]

print("\n=== OPTIMAL ALERT THRESHOLD ===")
print(best)

# =========================
# PARETO FRONTIER
# =========================

print("\n=== PARETO INSIGHT ===")
pareto = res[(res["FAR"] <= 0.2) & (res["MISS"] <= 0.5)]
print(pareto.head())

# =========================
# SAVE
# =========================

res.to_csv("data/processed/threshold_optimization_grid.csv", index=False)

print("\nSaved: threshold_optimization_grid.csv")
