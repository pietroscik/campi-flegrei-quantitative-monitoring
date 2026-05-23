import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================

alerts = pd.read_csv("data/processed/operational_alert_system.csv")
alerts["time"] = pd.to_datetime(alerts["time"])

events = pd.read_csv("data/processed/etas_output.csv")
events["time"] = pd.to_datetime(events["time"])

# =========================
# EVENT DEFINITION
# =========================

events = events[events["magnitude"] >= 1.5]
event_times = events["time"].values

def has_event_in_window(t, window_days=7):
    return np.any((event_times > t) & (event_times <= t + np.timedelta64(window_days, "D")))

alerts["event"] = alerts["time"].apply(has_event_in_window).astype(int)

# =========================
# ALERT FLAGS
# =========================

alerts["is_alert"] = alerts["alert_level"].isin(["YELLOW", "RED"]).astype(int)
alerts["is_red"] = (alerts["alert_level"] == "RED").astype(int)

# =========================
# 1. FALSE ALARM RATE
# =========================

false_alarms = alerts[(alerts["is_alert"] == 1) & (alerts["event"] == 0)]
total_alerts = alerts["is_alert"].sum()

FAR = len(false_alarms) / max(total_alerts, 1)

# =========================
# 2. MISS RATE
# =========================

missed_events = alerts[(alerts["event"] == 1) & (alerts["is_alert"] == 0)]
total_events = alerts["event"].sum()

MISS = len(missed_events) / max(total_events, 1)

# =========================
# 3. HIT RATE
# =========================

hits = alerts[(alerts["event"] == 1) & (alerts["is_alert"] == 1)]

HIT_RATE = len(hits) / max(total_events, 1)

# =========================
# 4. TIME-IN-ALERT RATIO
# =========================

TAR = alerts["is_alert"].mean()

# =========================
# 5. LEAD TIME ANALYSIS
# =========================

lead_times_red = []
lead_times_yellow = []

for t in event_times:
    past = alerts[alerts["time"] <= t]

    red_alerts = past[past["is_red"] == 1]
    yellow_alerts = past[past["alert_level"] == "YELLOW"]

    if len(red_alerts) > 0:
        lead_times_red.append((t - red_alerts["time"].max()).days)

    if len(yellow_alerts) > 0:
        lead_times_yellow.append((t - yellow_alerts["time"].max()).days)

# =========================
# 6. SKILL SCORE (PEIRCE)
# =========================

POD = HIT_RATE
POFD = FAR

SKILL = POD - POFD

# =========================
# PRINT SUMMARY
# =========================

print("\n=== OPERATIONAL VALIDATION REPORT ===")

print("\n--- CONFUSION STRUCTURE ---")
print("Total events:", total_events)
print("Total alerts:", total_alerts)

print("\n--- METRICS ---")
print("False Alarm Rate (FAR):", FAR)
print("Miss Rate:", MISS)
print("Hit Rate:", HIT_RATE)
print("Time-in-Alert Ratio:", TAR)
print("Skill Score (Peirce):", SKILL)

print("\n--- LEAD TIME ---")
print("Mean RED lead time:", np.mean(lead_times_red) if lead_times_red else None)
print("Median RED lead time:", np.median(lead_times_red) if lead_times_red else None)

print("Mean YELLOW lead time:", np.mean(lead_times_yellow) if lead_times_yellow else None)
print("Median YELLOW lead time:", np.median(lead_times_yellow) if lead_times_yellow else None)

# =========================
# 7. PLOTS (PAPER-GRADE)
# =========================

plt.figure()
plt.plot(alerts["time"], alerts["is_alert"], label="Alert (binary)")
plt.plot(alerts["time"], alerts["event"], label="Event", alpha=0.7)
plt.legend()
plt.title("Alert vs Event Timeline")
plt.tight_layout()
plt.savefig("results/alert_event_timeline.png")

plt.figure()
plt.hist(lead_times_red, bins=20)
plt.title("RED Lead Time Distribution")
plt.tight_layout()
plt.savefig("results/lead_time_red.png")

plt.figure()
plt.hist(lead_times_yellow, bins=20)
plt.title("YELLOW Lead Time Distribution")
plt.tight_layout()
plt.savefig("results/lead_time_yellow.png")

# =========================
# 8. SAVE METRICS
# =========================

metrics = {
    "FAR": FAR,
    "MISS_RATE": MISS,
    "HIT_RATE": HIT_RATE,
    "TAR": TAR,
    "SKILL": SKILL,
    "MEAN_RED_LEAD": np.mean(lead_times_red) if lead_times_red else None,
    "MEAN_YELLOW_LEAD": np.mean(lead_times_yellow) if lead_times_yellow else None
}

pd.DataFrame([metrics]).to_csv(
    "data/processed/operational_validation_metrics.csv",
    index=False
)

print("\nSaved: operational_validation_metrics.csv + figures")
