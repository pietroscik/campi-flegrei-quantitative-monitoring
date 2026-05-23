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
# THRESHOLD GRID
# =========================

thresholds = np.linspace(0.5, 0.99, 40)

results = []

# =========================
# EVALUATION LOOP
# =========================

for thr in thresholds:

    df["alert"] = (df["p_calibrated"] >= thr).astype(int)

    TP = ((df["alert"] == 1) & (df["event"] == 1)).sum()
    FP = ((df["alert"] == 1) & (df["event"] == 0)).sum()
    FN = ((df["alert"] == 0) & (df["event"] == 1)).sum()

    FAR = FP / max(FP + TP, 1)
    MISS = FN / max(FN + TP, 1)

    # LEAD TIME
    lead_times = []

    for t in event_times:
        past = df[df["time"] <= t]
        alerts = past[past["alert"] == 1]

        if len(alerts) > 0:
            lead_times.append((t - alerts["time"].max()).days)

    LEAD = np.mean(lead_times) if len(lead_times) > 0 else 0

    results.append([thr, FAR, MISS, LEAD])

res = pd.DataFrame(results, columns=["threshold", "FAR", "MISS", "LEAD"])

# =========================
# PARETO FRONTIER
# =========================

def is_dominated(row, df):
    return np.any(
        (df["FAR"] <= row["FAR"]) &
        (df["MISS"] <= row["MISS"]) &
        (df["LEAD"] >= row["LEAD"]) &
        ((df[["FAR","MISS","LEAD"]] != row[["FAR","MISS","LEAD"]]).any(axis=1))
    )

pareto = []

for i, row in res.iterrows():
    if not is_dominated(row, res):
        pareto.append(row)

pareto = pd.DataFrame(pareto)

print("\n=== PARETO FRONTIER ===")
print(pareto.sort_values("FAR"))

# =========================
# POLICY LAYER
# =========================

print("\n=== POLICY SELECTION ===")

# hard constraints (operational realism)
MAX_FAR = 0.10
MAX_MISS = 0.40

feasible = pareto[
    (pareto["FAR"] <= MAX_FAR) &
    (pareto["MISS"] <= MAX_MISS)
]

if len(feasible) == 0:
    print("⚠️ No feasible solution under constraints")
    best = pareto.sort_values("MISS").iloc[0]
else:
    # choose best lead time among feasible
    best = feasible.sort_values("LEAD", ascending=False).iloc[0]

print("\n=== SELECTED OPERATING POINT ===")
print(best)

# =========================
# SAVE OUTPUTS
# =========================

pareto.to_csv("data/processed/pareto_frontier.csv", index=False)

pd.DataFrame([best]).to_csv(
    "data/processed/optimal_policy_point.csv",
    index=False
)

print("\nSaved: pareto_frontier.csv + optimal_policy_point.csv")
