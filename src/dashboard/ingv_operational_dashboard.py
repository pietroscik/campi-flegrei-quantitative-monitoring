from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# BASE PATH (CRITICAL FIX)
# =========================

BASE_DIR = Path(__file__).resolve().parents[2]

# =========================
# LOAD DATA (ROBUST)
# =========================

alerts = pd.read_csv(BASE_DIR / "data/processed/early_warning_system.csv")
alerts["time"] = pd.to_datetime(alerts["time"])

events = pd.read_csv(BASE_DIR / "data/processed/etas_output.csv")
events["time"] = pd.to_datetime(events["time"])
events = events[events["magnitude"] >= 1.5]

event_times = events["time"].values

# =========================
# EVENT LABELING
# =========================

def label_event(t, window=7):
    return np.any((event_times > t) & (event_times <= t + np.timedelta64(window, "D")))

alerts["event"] = alerts["time"].apply(label_event).astype(int)

# ALERT FLAG (from real pipeline)
alerts["is_alert"] = alerts["alert_flag"].astype(int)

# =========================
# STREAMLIT UI
# =========================

st.title("🧭 Campi Flegrei – Operational Early Warning System")

# =========================
# CURRENT STATE
# =========================

latest = alerts.iloc[-1]

st.subheader("📍 Current System State")

col1, col2, col3 = st.columns(3)

col1.metric("State", latest.get("state", "UNKNOWN"))
col2.metric("Unrest Index", round(latest.get("unrest_index", 0), 3))
col3.metric("Alert Flag", int(latest["alert_flag"]))

st.metric("Calibrated Risk", round(latest.get("dl_anomaly_score", 0), 3))

# =========================
# RISK EVOLUTION
# =========================

st.subheader("📈 Risk Evolution")

fig, ax = plt.subplots()

if "dl_anomaly_score" in alerts.columns:
    ax.plot(alerts["time"], alerts["dl_anomaly_score"], label="DL anomaly score")

ax.plot(alerts["time"], alerts["unrest_index"], label="Unrest index")

ax.scatter(
    alerts[alerts["alert_flag"] == 1]["time"],
    alerts[alerts["alert_flag"] == 1]["unrest_index"],
    color="red",
    label="Alerts"
)

ax.legend()
st.pyplot(fig)

# =========================
# ALERT DISTRIBUTION
# =========================

st.subheader("🚦 Alert Distribution")

fig2, ax2 = plt.subplots()

alerts["state"].value_counts().plot(kind="bar", ax=ax2)

ax2.set_title("System States")
st.pyplot(fig2)

# =========================
# EVENT VS ALERT
# =========================

st.subheader("🌋 Event vs Alert Alignment")

fig3, ax3 = plt.subplots()

ax3.plot(alerts["time"], alerts["event"], label="Events")
ax3.plot(alerts["time"], alerts["is_alert"], label="Alerts", alpha=0.7)

ax3.legend()
st.pyplot(fig3)

# =========================
# OPERATIONAL METRICS
# =========================

TP = ((alerts["is_alert"] == 1) & (alerts["event"] == 1)).sum()
FP = ((alerts["is_alert"] == 1) & (alerts["event"] == 0)).sum()
FN = ((alerts["is_alert"] == 0) & (alerts["event"] == 1)).sum()

FAR = FP / max(FP + TP, 1)
MISS = FN / max(FN + TP, 1)

st.subheader("📊 Operational Metrics")

st.metric("False Alarm Rate (FAR)", round(FAR, 3))
st.metric("Miss Rate", round(MISS, 3))

# =========================
# LEAD TIME ESTIMATION
# =========================

lead_times = []

for t in event_times:
    past = alerts[alerts["time"] <= t]
    alert_times = past[past["is_alert"] == 1]["time"]

    if len(alert_times) > 0:
        lead_times.append((t - alert_times.max()).days)

if len(lead_times) > 0:
    st.metric("Mean Lead Time (days)", round(np.mean(lead_times), 2))
    st.metric("Median Lead Time (days)", round(np.median(lead_times), 2))

# =========================
# INTERPRETATION LAYER
# =========================

st.subheader("🧠 System Interpretation")

state = latest.get("state", "UNKNOWN")

if state == "GREEN":
    st.success("Low unrest regime – background activity")

elif state == "YELLOW":
    st.warning("Elevated unrest – monitoring recommended")

elif state == "RED":
    st.error("High-risk regime – escalation suggested")

else:
    st.info("System state undefined or transitional")

# =========================
# FOOTER
# =========================

st.caption(
    "Research prototype – probabilistic early warning system for volcanic monitoring (non-operational use)"
)
