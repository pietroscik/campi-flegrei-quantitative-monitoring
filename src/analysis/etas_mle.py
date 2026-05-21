import pandas as pd
import numpy as np
from scipy.optimize import minimize

# -----------------------------
# ETAS intensity function (vectorized)
# -----------------------------

def intensity_vectorized(t, times, mags, mu, K, alpha, c, p, M0=0.0):
    """Vectorized intensity calculation for better performance."""
    
    lam = mu
    
    # Calculate dt for all previous events at once
    dt = t - times
    mask = dt > 0
    
    if np.any(mask):
        valid_dt = dt[mask]
        valid_mags = mags[mask]
        
        # Vectorized kernel calculation
        contributions = K * np.exp(alpha * (valid_mags - M0)) * (valid_dt + c) ** (-p)
        lam += np.sum(contributions)
    
    return lam


# -----------------------------
# Log-likelihood (optimized)
# -----------------------------

def log_likelihood(params, times, mags):

    mu, K, alpha, c, p = params

    if mu <= 0 or K <= 0 or c <= 0 or p <= 1:
        return np.inf

    n = len(times)

    ll = 0.0

    # likelihood sum over events (using vectorized intensity)
    for i in range(n):
        
        lam = intensity_vectorized(times[i], times[:i], mags[:i], mu, K, alpha, c, p)

        if lam <= 0:
            continue

        ll += np.log(lam)

    # integral approximation (discrete grid - reduced resolution for speed)
    t_min, t_max = times[0], times[-1]
    grid_size = min(100, n)  # Limit grid size
    grid = np.linspace(t_min, t_max, grid_size)

    integral = 0.0

    for t in grid:
        integral += intensity_vectorized(t, times, mags, mu, K, alpha, c, p)

    integral *= (t_max - t_min) / len(grid)

    return -(ll - integral)  # negative log-likelihood


# -----------------------------
# FIT ETAS MODEL
# -----------------------------

def fit_etas(df):

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    # convert time to numeric (days)
    t0 = df["time"].iloc[0]

    times = (df["time"] - t0).dt.total_seconds().values / 86400.0
    mags = df["magnitude"].values

    # initial guess
    x0 = [0.1, 0.5, 1.0, 0.01, 1.2]

    bounds = [
        (1e-4, 5.0),   # mu
        (1e-4, 5.0),   # K
        (0.1, 5.0),    # alpha
        (1e-4, 1.0),   # c
        (1.01, 3.0)    # p
    ]

    result = minimize(
        log_likelihood,
        x0,
        args=(times, mags),
        bounds=bounds,
        method="L-BFGS-B",
        options={'maxiter': 100}  # Limit iterations
    )

    return result


# -----------------------------
# PIPELINE
# -----------------------------

def run_mle(input_path="data/processed/catalog_clean.csv"):

    df = pd.read_csv(input_path)

    result = fit_etas(df)

    params = {
        "mu": result.x[0],
        "K": result.x[1],
        "alpha": result.x[2],
        "c": result.x[3],
        "p": result.x[4]
    }

    print("[OK] ETAS MLE completed")
    print(params)

    pd.DataFrame([params]).to_csv("data/processed/etas_params.csv", index=False)

    return params


if __name__ == "__main__":
    run_mle()
