import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# =========================
# LOAD DATA
# =========================

unrest = pd.read_csv("data/processed/unrest_index.csv")
etas = pd.read_csv("data/processed/etas_output.csv")
dl = pd.read_csv("data/processed/dl_anomalies.csv")
cp = pd.read_csv("data/processed/changepoint_output.csv")

unrest["time"] = pd.to_datetime(unrest["time"])
etas["time"] = pd.to_datetime(etas["time"])

dl["time"] = pd.to_datetime(dl["time"])
cp["time"] = pd.to_datetime(cp["time"])

# =========================
# SORT
# =========================

unrest = unrest.sort_values("time")

# =========================
# BUILD FUTURE EVENT LABEL (CRITICAL FIX)
# =========================

window_days = 7

event_times = etas[etas["magnitude"] >= 1.5]["time"]

def has_future_event(t):
    return np.any((event_times > t) & (event_times <= t + pd.Timedelta(days=window_days)))

unrest["event"] = unrest["time"].apply(has_future_event).astype(int)

# =========================
# MERGE SIGNALS
# =========================

df = unrest.merge(dl[["time","dl_is_anomaly"]], on="time", how="left")
df = df.merge(cp[["time","changepoint"]], on="time", how="left")

df = df.fillna(0)

# =========================
# CHECK CLASS BALANCE
# =========================

print("\n=== CLASS DISTRIBUTION ===")
print(df["event"].value_counts())

if df["event"].sum() == 0:
    raise ValueError("Still no events → increase window_days or check timestamps")

# =========================
# FEATURES
# =========================

X = df[["unrest_index", "dl_is_anomaly", "changepoint"]].copy()

X["unrest_index"] = (
    X["unrest_index"] - X["unrest_index"].mean()
) / (X["unrest_index"].std() + 1e-9)

y = df["event"].values

# =========================
# MODEL
# =========================

model = LogisticRegression(class_weight="balanced")
model.fit(X, y)

df["p_event"] = model.predict_proba(X)[:, 1]

# =========================
# RESULTS
# =========================

print("\n=== PROBABILISTIC MODEL ===")

print("Event rate:", y.mean())

print("\nCoefficients:")
for name, c in zip(X.columns, model.coef_[0]):
    print(name, round(c, 4))

print("\nRisk stats:")
print("Mean P(event):", df["p_event"].mean())
print("Max P(event):", df["p_event"].max())

# =========================
# HIGH RISK REGIME
# =========================

thr = df["p_event"].quantile(0.95)

high = df[df["p_event"] >= thr]

print("\nHigh-risk mean unrest:", high["unrest_index"].mean())
df[["time", "p_event"]].to_csv(
    "data/processed/probabilistic_risk_score.csv",
    index=False
)
