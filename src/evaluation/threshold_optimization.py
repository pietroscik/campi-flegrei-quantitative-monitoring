import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score

# =========================
# LOAD DATA
# =========================

unrest = pd.read_csv("data/processed/unrest_index.csv")
etas = pd.read_csv("data/processed/etas_output.csv")

unrest["time"] = pd.to_datetime(unrest["time"])
etas["time"] = pd.to_datetime(etas["time"])

# =========================
# GROUND TRUTH
# =========================

events = etas[etas["magnitude"] >= 1.5][["time"]].copy()
events["event"] = 1

df = unrest.merge(events, on="time", how="left")
df["event"] = df["event"].fillna(0)

df = df.sort_values("time")

# =========================
# SCORE
# =========================

df["score"] = df["unrest_index"]

# =========================
# THRESHOLD GRID
# =========================

thresholds = np.linspace(
    df["score"].quantile(0.50),
    df["score"].quantile(0.99),
    50
)

results = []

# =========================
# EVALUATION FUNCTION
# =========================

def evaluate(theta):
    df["alert"] = (df["score"] >= theta).astype(int)

    tp = len(df[(df["alert"] == 1) & (df["event"] == 1)])
    fp = len(df[(df["alert"] == 1) & (df["event"] == 0)])
    fn = len(df[(df["alert"] == 0) & (df["event"] == 1)])

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    far = fp / (len(df[df["event"] == 0]) + 1e-9)

    # F1
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    # Skill function (custom)
    skill = f1 - 0.5 * far

    return precision, recall, far, f1, skill

# =========================
# GRID SEARCH
# =========================

for theta in thresholds:
    p, r, far, f1, skill = evaluate(theta)
    results.append([theta, p, r, far, f1, skill])

res = pd.DataFrame(results, columns=[
    "threshold", "precision", "recall", "far", "f1", "skill"
])

best = res.loc[res["skill"].idxmax()]

# =========================
# OUTPUT
# =========================

print("\n=== OPTIMAL THRESHOLD ===")
print("theta*:", best["threshold"])

print("\n=== PERFORMANCE ===")
print("Precision:", best["precision"])
print("Recall:", best["recall"])
print("FAR:", best["far"])
print("F1:", best["f1"])
print("Skill:", best["skill"])
