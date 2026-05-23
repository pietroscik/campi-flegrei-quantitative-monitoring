import numpy as np
from scipy import stats

# =========================
# BOOTSTRAP TEST (ROBUSTO)
# =========================

normal = df[df["is_alert"] == 0]["unrest_index"].dropna().values
alert = df[df["is_alert"] == 1]["unrest_index"].dropna().values

def bootstrap_diff(a, b, n=10000):
    combined = np.concatenate([a, b])
    obs = np.mean(a) - np.mean(b)

    diffs = []
    for _ in range(n):
        np.random.shuffle(combined)
        new_a = combined[:len(a)]
        new_b = combined[len(a):]
        diffs.append(np.mean(new_a) - np.mean(new_b))

    p_value = np.mean(np.abs(diffs) >= np.abs(obs))
    return obs, p_value

if len(alert) > 5:
    diff, p = bootstrap_diff(normal, alert)
    print("\n=== BOOTSTRAP TEST ===")
    print(f"Difference: {diff:.4f}, p-value: {p:.6f}")
else:
    print("\n⚠️ ALERT sample too small → skipping parametric tests")
# =========================
# TEST 2: ALERT SYSTEM RANDOMNESS CHECK
# =========================

# test if alerts differ from random expectation
alert_rate = df["alert_flag"].mean()
expected = np.mean(df["is_alert"])

z = (alert_rate - expected) / np.sqrt(expected * (1 - expected) / len(df))

p_z = 2 * (1 - stats.norm.cdf(abs(z)))

print("\n=== ALERT SYSTEM vs GROUND TRUTH ===")
print(f"Alert rate: {alert_rate:.4f}")
print(f"Expected: {expected:.4f}")
print(f"Z-score: {z:.4f}, p-value: {p_z:.6f}")

# =========================
# TEST 3: CORRELATION UNREST vs ALERT
# =========================

corr, p_corr = stats.pointbiserialr(df["alert_flag"], df["unrest_index"])

print("\n=== CORRELATION ===")
print(f"Correlation: {corr:.4f}, p-value: {p_corr:.6f}")

# =========================
# SUMMARY
# =========================

print("\n=== SUMMARY ===")
print("If p < 0.05 => statistically significant relationship")
