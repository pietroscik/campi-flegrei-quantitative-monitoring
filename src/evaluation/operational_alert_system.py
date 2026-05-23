import pandas as pd
import numpy as np

# =========================
# LOAD CALIBRATED DATA
# =========================

df = pd.read_csv("data/processed/probabilistic_risk_score_calibrated.csv")
df["time"] = pd.to_datetime(df["time"])

unrest = pd.read_csv("data/processed/unrest_index.csv")
unrest["time"] = pd.to_datetime(unrest["time"])

dl = pd.read_csv("data/processed/dl_anomalies.csv")
dl["time"] = pd.to_datetime(dl["time"])

cp = pd.read_csv("data/processed/changepoint_output.csv")
cp["time"] = pd.to_datetime(cp["time"])

# =========================
# MERGE SIGNALS
# =========================

df = df.merge(unrest[["time","unrest_index"]], on="time", how="left")
df = df.merge(dl[["time","dl_is_anomaly"]], on="time", how="left")
df = df.merge(cp[["time","changepoint"]], on="time", how="left")

df = df.fillna(0)
df = df.sort_values("time")

# =========================
# TREND COMPUTATION
# =========================

df["unrest_trend"] = df["unrest_index"].diff().rolling(3).mean()
df["trend_signal"] = (df["unrest_trend"] > 0).astype(int)

# =========================
# CORE THRESHOLDS
# =========================

p_yellow = df["p_calibrated"].quantile(0.80)
p_red = df["p_calibrated"].quantile(0.95)

# =========================
# PERSISTENCE WINDOWS
# =========================

window = 3

df["high_risk"] = (df["p_calibrated"] >= p_yellow).astype(int)
df["red_risk"] = (df["p_calibrated"] >= p_red).astype(int)

df["persist_yellow"] = df["high_risk"].rolling(window).sum() >= window
df["persist_red"] = df["red_risk"].rolling(window).sum() >= window

# =========================
# MULTI-SIGNAL CONFIRMATION
# =========================

df["multi_signal"] = (
    (df["dl_is_anomaly"] > 0) |
    (df["changepoint"] > 0) |
    (df["trend_signal"] > 0)
).astype(int)

# =========================
# FINAL ALERT LOGIC
# =========================

df["alert_level"] = "GREEN"

df.loc[
    (df["persist_yellow"]) & (df["multi_signal"]),
    "alert_level"
] = "YELLOW"

df.loc[
    (df["persist_red"]) & (df["multi_signal"]),
    "alert_level"
] = "RED"

# override rule (extreme risk)
df.loc[
    df["p_calibrated"] > p_red * 1.2,
    "alert_level"
] = "RED"

# =========================
# SUMMARY
# =========================

print("\n=== OPERATIONAL ALERT SYSTEM ===")

print("\nAlert distribution:")
print(df["alert_level"].value_counts())

print("\nRisk by level:")
print(df.groupby("alert_level")["p_calibrated"].mean())

# =========================
# STABILITY CHECK
# =========================

print("\n=== SYSTEM STABILITY ===")

switches = (df["alert_level"] != df["alert_level"].shift()).sum()

print("State transitions:", switches)

# =========================
# OUTPUT
# =========================

df[["time","p_calibrated","alert_level"]].to_csv(
    "data/processed/operational_alert_system.csv",
    index=False
)

print("\nSaved: operational_alert_system.csv")
