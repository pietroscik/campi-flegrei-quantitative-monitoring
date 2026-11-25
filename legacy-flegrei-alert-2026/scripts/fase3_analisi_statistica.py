#!/usr/bin/env python3
"""
fase3_analisi_statistica.py - Fase 3 della CHECKLIST (Analisi Statistica Avanzata)

Eseguire dalla root del repository:
    python3 legacy-flegrei-alert-2026/scripts/fase3_analisi_statistica.py

Copre i task della CHECKLIST:
  3.1 change-point GNSS (PELT + Bai-Perron binary segmentation) con bootstrap CI (B=1000)
  3.2 rolling variance e autocorrelazione lag-1 dell'indicatore CSD su uplift
  3.3 fit Weibull sugli IET del catalogo (verifica beta > 1)
  3.4 modello Hawkes/ETAS con confronto AIC vs Weibull
  3.7 propagazione Monte-Carlo dell'incertezza sulle soglie di stress (yield point)
  3.5/3.6 geochimica: bloccati in assenza di dati gas (task 2.4 manuale)
"""

import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

SEED = 42
BOOT_B = 1000
ROLL_WIN = 30
HAWKES_WINDOW_DAYS = 180
MAG_MIN = 0.0


def load_uplift():
    times, vals = [], []
    with open(RAW / "uplift.csv") as f:
        for r in csv.DictReader(f):
            times.append(datetime.fromisoformat(r["time"]))
            vals.append(float(r["uplift"]))
    return times, vals


def load_catalog():
    rows = []
    with open(PROCESSED / "catalog_clean_ingv_events.csv") as f:
        for r in csv.DictReader(f):
            rows.append((datetime.fromisoformat(r["time"]), float(r["magnitude"])))
    rows.sort()
    return rows


def pelt_mean(x, penalty):
    n = len(x)
    S = [0.0] * (n + 1)
    SQ = [0.0] * (n + 1)
    for i, v in enumerate(x):
        S[i + 1] = S[i] + v
        SQ[i + 1] = SQ[i] + v * v

    def cost(a, b):
        if b <= a:
            return 0.0
        s = S[b] - S[a]
        m = s / (b - a)
        return SQ[b] - SQ[a] - s * m

    F = [0.0] + [math.inf] * n
    last = [0] * (n + 1)
    R = [0]
    for t in range(1, n + 1):
        best = math.inf
        best_tau = 0
        for tau in R:
            c = F[tau] + cost(tau, t) + penalty
            if c < best:
                best = c
                best_tau = tau
        F[t] = best
        last[t] = best_tau
        R = [tau for tau in R if F[tau] + cost(tau, t) < F[t] + cost(t, t)]
        R.append(t)
    cps = []
    t = n
    while t > 0:
        cps.append(last[t])
        t = last[t]
    return sorted(c for c in cps if c > 0)


def binary_segmentation(x, penalty, min_seg=30):
    n = len(x)
    S = [0.0] * (n + 1)
    SQ = [0.0] * (n + 1)
    for i, v in enumerate(x):
        S[i + 1] = S[i] + v
        SQ[i + 1] = SQ[i] + v * v

    def sse(a, b):
        s = S[b] - S[a]
        return SQ[b] - SQ[a] - s * s / (b - a)

    def rec(a, b, out):
        if b - a < 2 * min_seg:
            return
        best_gain = 0.0
        best_k = None
        total = sse(a, b)
        for k in range(a + min_seg, b - min_seg + 1):
            gain = total - sse(a, k) - sse(k, b)
            if gain > best_gain:
                best_gain = gain
                best_k = k
        if best_k is not None and best_gain > penalty:
            out.append(best_k)
            rec(a, best_k, out)
            rec(best_k, b, out)

    cps = []
    rec(0, n, cps)
    return sorted(cps)


def piecewise_fit(x, cps):
    bounds = [0] + cps + [len(x)]
    fitted = [0.0] * len(x)
    means = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        m = sum(x[a:b]) / (b - a)
        means.append(m)
        for i in range(a, b):
            fitted[i] = m
    return fitted, means


def step_3_1(times, vals):
    diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    var = sum(d * d for d in diffs) / (2.0 * len(diffs))
    penalty = math.log(len(vals)) * max(var, 1e-12)
    cps_pelt = pelt_mean(vals, penalty)
    cps_bp = binary_segmentation(vals, penalty)
    fitted, means = piecewise_fit(vals, cps_pelt)
    resid = [v - f for v, f in zip(vals, fitted)]

    rng = random.Random(SEED)
    boot = {cp: [] for cp in cps_pelt}
    for _ in range(BOOT_B):
        xb = [fitted[i] + resid[rng.randrange(len(resid))] for i in range(len(vals))]
        for cp in pelt_mean(xb, penalty):
            for orig in cps_pelt:
                if abs(cp - orig) <= ROLL_WIN:
                    boot[orig].append(cp)
                    break

    def pct(v, q):
        if not v:
            return None
        s = sorted(v)
        return s[min(len(s) - 1, int(q * len(s)))]

    out = PROCESSED / "gnss_changepoints.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cp_index", "date", "mean_before", "mean_after", "ci_lower_idx", "ci_upper_idx", "ci_lower_date", "ci_upper_date", "bai_perron_cp"])
        bounds = [0] + cps_pelt + [len(vals)]
        for k, cp in enumerate(cps_pelt):
            b = boot[cp]
            lo, hi = pct(b, 0.025), pct(b, 0.975)
            w.writerow([
                cp, times[cp].date(), f"{means[k]:.4f}", f"{means[k + 1]:.4f}",
                lo if lo is not None else cp, hi if hi is not None else cp,
                times[lo if lo is not None else cp].date(), times[hi if hi is not None else cp].date(),
                ";".join(str(c) for c in cps_bp),
            ])
    print(f"  3.1 change-point PELT: {cps_pelt} | Bai-Perron: {cps_bp} | bootstrap CI (B={BOOT_B}) -> gnss_changepoints.csv")
    return cps_pelt, means, bounds, resid


def step_3_2(times, vals):
    du = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    csd = [0.0]
    for d in du:
        csd.append(csd[-1] + d * d)
    out = PROCESSED / "csd_gnss_rolling.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "csd", "rolling_var", "rolling_ac1"])
        for i in range(len(vals)):
            if i + 1 < ROLL_WIN:
                w.writerow([times[i].date(), f"{csd[i]:.6f}", "", ""])
                continue
            seg = csd[i + 1 - ROLL_WIN: i + 1]
            m = sum(seg) / ROLL_WIN
            rv = sum((v - m) ** 2 for v in seg) / ROLL_WIN
            a = seg[:-1]
            b = seg[1:]
            ma = sum(a) / len(a)
            mb = sum(b) / len(b)
            num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
            den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
            ac1 = num / den if den > 0 else ""
            w.writerow([times[i].date(), f"{csd[i]:.6f}", f"{rv:.8f}", f"{ac1:.4f}" if ac1 != "" else ""])
    print(f"  3.2 CSD rolling (win={ROLL_WIN}): var e ac1 lag-1 -> csd_gnss_rolling.csv ({len(vals)} righe)")


def weibull_mle(x):
    lnx = [math.log(v) for v in x]
    mean_lnx = sum(lnx) / len(x)

    def g(k):
        xk = [math.exp(k * l) for l in lnx]
        sxk = sum(xk)
        return sum(a * b for a, b in zip(xk, lnx)) / sxk - 1.0 / k - mean_lnx

    k = 1.0
    for _ in range(100):
        gk = g(k)
        h = 1e-6 * k
        g2 = (g(k + h) - gk) / h
        if abs(g2) < 1e-12:
            break
        step = gk / g2
        k -= step
        if k <= 0.01:
            k = 0.01
        if abs(step) < 1e-10:
            break
    xk = [math.exp(k * l) for l in lnx]
    lam = (sum(xk) / len(x)) ** (1.0 / k)
    ll = len(x) * (math.log(k) - k * math.log(lam)) + (k - 1) * sum(lnx) - sum(math.exp(k * l) / lam ** k for l in lnx)
    return k, lam, ll


def step_3_3(catalog):
    times = [t for t, _ in catalog]
    iets = [(times[i + 1] - times[i]).total_seconds() / 3600.0 for i in range(len(times) - 1)]
    iets = [v for v in iets if v > 0]
    k, lam, ll = weibull_mle(iets)
    aic_weibull = 4 - 2 * ll
    n = len(iets)
    lam_exp = sum(iets) / n
    ll_exp = n * (math.log(1.0 / lam_exp) - 1.0)
    aic_exp = 2 - 2 * ll_exp
    out = PROCESSED / "iet_weibull.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_iet", "weibull_shape_beta", "weibull_scale_lambda_h", "loglik_weibull", "aic_weibull", "aic_exponential", "beta_gt_1"])
        w.writerow([n, f"{k:.4f}", f"{lam:.4f}", f"{ll:.2f}", f"{aic_weibull:.2f}", f"{aic_exp:.2f}", k > 1.0])
    print(f"  3.3 Weibull IET: beta={k:.4f} (>1: {k > 1.0}), lambda={lam:.4f} h, AIC weibull={aic_weibull:.1f} vs exp={aic_exp:.1f} -> iet_weibull.csv")
    return aic_weibull


def hawkes_loglik(params, times, mags, T):
    mu, K, alpha, c, p = params
    if mu <= 0 or K < 0 or alpha <= 0 or c <= 0 or p <= 1.0001:
        return -1e10
    n = len(times)
    ll = 0.0
    for i in range(n):
        lam = mu
        for j in range(i):
            lam += K * math.exp(alpha * (mags[j] - MAG_MIN)) * (times[i] - times[j] + c) ** (-p)
        ll += math.log(lam)
    integ = mu * T
    for j in range(n):
        integ += K * math.exp(alpha * (mags[j] - MAG_MIN)) * ((T - times[j] + c) ** (1 - p) - c ** (1 - p)) / (1 - p)
    return ll - integ


def nelder_mead(f, x0, step=0.5, max_iter=300):
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        p = list(x0)
        p[i] += step
        simplex.append(p)
    vals = [f(p) for p in simplex]
    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: vals[i])
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        worst = simplex[n]
        refl = [centroid[j] + (centroid[j] - worst[j]) for j in range(n)]
        fr = f(refl)
        if vals[0] <= fr < vals[n - 1]:
            simplex[n] = refl
            vals[n] = fr
            continue
        if fr < vals[0]:
            exp = [centroid[j] + 2 * (refl[j] - centroid[j]) for j in range(n)]
            fe = f(exp)
            if fe < fr:
                simplex[n] = exp
                vals[n] = fe
            else:
                simplex[n] = refl
                vals[n] = fr
            continue
        contr = [centroid[j] + 0.5 * (worst[j] - centroid[j]) for j in range(n)]
        fc = f(contr)
        if fc < vals[n]:
            simplex[n] = contr
            vals[n] = fc
            continue
        best = simplex[0]
        for i in range(1, n + 1):
            simplex[i] = [best[j] + 0.5 * (simplex[i][j] - best[j]) for j in range(n)]
            vals[i] = f(simplex[i])
    order = sorted(range(n + 1), key=lambda i: vals[i])
    return simplex[order[0]], vals[order[0]]


def step_3_4(catalog, aic_weibull):
    last = catalog[-1][0]
    cut = last - timedelta(days=HAWKES_WINDOW_DAYS)
    win = [(t, m) for t, m in catalog if t >= cut]
    t0 = win[0][0]
    times = [(t - t0).total_seconds() / 86400.0 for t, _ in win]
    mags = [m for _, m in win]
    T = (win[-1][0] - t0).total_seconds() / 86400.0
    n = len(times)

    def negll(z):
        mu = math.exp(z[0])
        K = math.exp(z[1])
        alpha = math.exp(z[2])
        c = math.exp(z[3])
        p = 1.0 + math.exp(z[4])
        return -hawkes_loglik((mu, K, alpha, c, p), times, mags, T)

    x0 = [math.log(n / T), math.log(0.3), math.log(1.0), math.log(0.05), math.log(0.2)]
    best, nll = nelder_mead(negll, x0)
    mu, K, alpha, c = (math.exp(best[0]), math.exp(best[1]), math.exp(best[2]), math.exp(best[3]))
    p = 1.0 + math.exp(best[4])
    ll_hawkes = -nll
    aic_hawkes = 10 - 2 * ll_hawkes
    ll_pois = n * math.log(n / T) - n
    aic_pois = 2 - 2 * ll_pois

    iets_w = [(times[i + 1] - times[i]) * 24.0 for i in range(n - 1)]
    _, _, ll_w = weibull_mle(iets_w)
    aic_weibull_w = 4 - 2 * ll_w

    out = PROCESSED / "hawkes_etas_fit.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_days", "n_events", "mu", "K", "alpha", "c_days", "p", "loglik_hawkes", "aic_hawkes", "aic_poisson", "aic_weibull_same_window", "autoexcitation_confirmed"])
        w.writerow([HAWKES_WINDOW_DAYS, n, f"{mu:.6f}", f"{K:.6f}", f"{alpha:.4f}", f"{c:.6f}", f"{p:.4f}",
                    f"{ll_hawkes:.2f}", f"{aic_hawkes:.2f}", f"{aic_pois:.2f}", f"{aic_weibull_w:.2f}", aic_hawkes < aic_pois])
    print(f"  3.4 Hawkes/ETAS ({n} ev, {HAWKES_WINDOW_DAYS}d): mu={mu:.4f} K={K:.4f} alpha={alpha:.3f} c={c:.4f} p={p:.3f} | AIC hawkes={aic_hawkes:.1f} vs poisson={aic_pois:.1f} (stessa likelihood) vs weibull-IET={aic_weibull_w:.1f} (riferimento) -> hawkes_etas_fit.csv")


def step_3_7(times, vals, cps_pelt, means, bounds, resid):
    rng = random.Random(SEED + 1)
    if cps_pelt:
        last_cp = cps_pelt[-1]
        post = vals[last_cp:]
        pre = vals[:last_cp]
    else:
        last_cp = 0
        post = vals
        pre = vals
    yield_level = sum(post) / len(post)
    sigma_pre = math.sqrt(sum((v - sum(pre) / len(pre)) ** 2 for v in pre) / len(pre)) if len(pre) > 1 else 0.0
    if len(post) > 2:
        xs = list(range(len(post)))
        mx = sum(xs) / len(xs)
        my = sum(post) / len(post)
        drift = sum((x - mx) * (y - my) for x, y in zip(xs, post)) / sum((x - mx) ** 2 for x in xs)
    else:
        drift = 0.0
    sigma_res = math.sqrt(sum(r * r for r in resid) / len(resid)) if resid else 0.0
    margin = vals[-1] - yield_level

    yields_b, margins_b, exceed_b = [], [], []
    for _ in range(BOOT_B):
        yb = yield_level + rng.gauss(0, sigma_res / math.sqrt(len(post)))
        yields_b.append(yb)
        margins_b.append(vals[-1] - yb)
        proj = vals[-1] + drift * 30 + rng.gauss(0, sigma_res * math.sqrt(30))
        exceed_b.append(1 if proj > yb + 2 * sigma_pre else 0)

    def pct(v, q):
        s = sorted(v)
        return s[min(len(s) - 1, int(q * len(s)))]

    rows = [
        ("yield_level", yield_level, pct(yields_b, 0.025), pct(yields_b, 0.975)),
        ("margin_last_vs_yield", margin, pct(margins_b, 0.025), pct(margins_b, 0.975)),
        ("p_margin_positive", sum(1 for m in margins_b if m > 0) / BOOT_B, None, None),
        ("p_exceed_yield_plus_2sigma_in_30d", sum(exceed_b) / BOOT_B, None, None),
        ("post_cp_drift_per_day", drift, None, None),
        ("sigma_pre_cp", sigma_pre, None, None),
    ]
    out = PROCESSED / "montecarlo_stress_threshold.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "ci_lower_95", "ci_upper_95"])
        for name, v, lo, hi in rows:
            w.writerow([name, f"{v:.6f}", "" if lo is None else f"{lo:.6f}", "" if hi is None else f"{hi:.6f}"])
    print(f"  3.7 Monte-Carlo (B={BOOT_B}): yield={yield_level:.4f} [CI {pct(yields_b, 0.025):.4f}, {pct(yields_b, 0.975):.4f}], P(margin>0)={rows[2][1]:.3f}, P(exceed 30d)={rows[3][1]:.3f} -> montecarlo_stress_threshold.csv")


def step_3_5_3_6():
    gas_files = [p for p in list(PROCESSED.glob("*gas*")) + list(RAW.glob("*gas*")) + list(PROCESSED.glob("*geochem*")) + list(RAW.glob("*geochem*")) if p.name != "geochemistry_status.csv"]
    out = PROCESSED / "geochemistry_status.csv"
    status = "blocked: no gas data (task 2.4 manual - INGV-OV bollettini)" if not gas_files else f"{len(gas_files)} file gas presenti"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "status"])
        w.writerow(["3.5 Engle-Granger/Johansen Gas vs Deformazione", status])
        w.writerow(["3.6 Granger causality lag=7gg", status])
    print(f"  3.5/3.6 geochimica: {status} -> geochemistry_status.csv")


def main():
    print("Fase 3 - Analisi Statistica Avanzata (CHECKLIST)")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    times, vals = load_uplift()
    catalog = load_catalog()
    cps, means, bounds, resid = step_3_1(times, vals)
    step_3_2(times, vals)
    aic_weibull = step_3_3(catalog)
    step_3_4(catalog, aic_weibull)
    step_3_7(times, vals, cps, means, bounds, resid)
    step_3_5_3_6()
    print("Fase 3 completata (task 3.5/3.6 bloccati per assenza dati geochimici).")


if __name__ == "__main__":
    main()
