import pandas as pd
import numpy as np

# =========================
# LOAD DATA
# =========================

unrest = pd.read_csv("data/processed/unrest_index.csv")
etas = pd.read_csv("data/processed/etas_output.csv")

unrest["time"] = pd.to_datetime(unrest["time"])
etas["time"] = pd.to_datetime(etas["time"])

# =========================
# DEFINE EVENTS
# =========================

events = etas[etas["magnitude"] >= 1.5].copy()
events = events.sort_values("time")

signal = unrest.sort_values("time")

# =========================
# PARAMS
# =========================

threshold = signal["unrest_index"].quantile(0.9)

lead_times = []

# =========================
# LEAD TIME COMPUTATION
# =========================

for _, ev in events.iterrows():
    t_event = ev["time"]

    # signal BEFORE event
    pre = signal[signal["time"] < t_event]

    # find first exceedance of threshold
    exceed = pre[pre["unrest_index"] >= threshold]

    if len(exceed) > 0:
        t_signal = exceed["time"].iloc[-1]  # last high signal before event
        lead = (t_event - t_signal).total_seconds() / 86400  # days
        lead_times.append(lead)

# =========================
# RESULTS
# =========================

lead_times = np.array(lead_times)

print("\n=== LEAD TIME ANALYSIS ===")

if len(lead_times) > 0:
    print("Events with signal:", len(lead_times))
    print("Mean lead time (days):", np.mean(lead_times))
    print("Median lead time (days):", np.median(lead_times))
    print("Std lead time:", np.std(lead_times))

    # =========================
    # SIGNIFICANCE TEST
    # =========================

    # H0: lead time = 0
    t_stat, p_val = stats = __import__("scipy").stats.ttest_1samp(lead_times, 0)

    print("\n=== SIGNIFICANCE TEST ===")
    print("t-stat:", t_stat)
    print("p-value:", p_val)

else:
    print("No lead times detected → threshold too strict or signal weak")
