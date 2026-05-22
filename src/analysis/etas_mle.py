import pandas as pd
import numpy as np
from scipy.optimize import minimize

# -----------------------------
# ETAS intensity function (vectorized)
# -----------------------------

def intensity_vectorized(t, times, mags, mu, K, alpha, c, p, M0=0.0):
    """Vectorized intensity calculation for better performance."""
    
    lam = mu
    
    # For a single t, calculate dt from all previous times
    dt = t - times
    # Assuming times array only includes previous events (dt > 0)
    contributions = K * np.exp(alpha * (mags - M0)) * (dt + c) ** (-p)
    lam += np.sum(contributions)
    
    return lam


# -----------------------------
# Log-likelihood (optimized)
# -----------------------------

def log_likelihood(params, times, mags, m0):
    """
    Optimized Log-Likelihood for the ETAS model.
    Replaces numerical integration with the exact analytical integral
    and optimizes the likelihood sum loop.
    """

    mu, K, alpha, c, p = params

    # Strict bounds to prevent invalid values breaking the optimizer
    if mu <= 0 or K <= 0 or alpha <= 0 or c <= 0 or p <= 1.0001:
        return 1e10  # Return large penalty instead of np.inf for L-BFGS-B

    n = len(times)
    
    # Precompute magnitude productivity terms (A_i) to avoid recalculating in the loop
    A = K * np.exp(alpha * (mags - m0))

    ll = 0.0

    # Likelihood sum over events
    # Start from 1, event 0 has intensity = mu
    for i in range(1, n):
        
        dt = times[i] - times[:i]
        lam = mu + np.sum(A[:i] * (dt + c) ** (-p))

        if lam <= 0:
            continue

        ll += np.log(lam)
        
    # Add the first event (only background rate applies)
    ll += np.log(mu)

    # Analytical Integral of the ETAS intensity
    # Exact integration avoids the computational overhead of the discrete grid approximation
    t_min, t_max = times[0], times[-1]
    
    # Integral of background rate
    integral = mu * (t_max - t_min)
    
    # Integral of triggered rate:
    # int_{t_i}^{t_max} A_i (t - t_i + c)^-p dt = (A_i / (p - 1)) * [c^(1-p) - (t_max - t_i + c)^(1-p)]
    term1 = c ** (1 - p)
    term2 = (t_max - times + c) ** (1 - p)
    integral += np.sum((A / (p - 1)) * (term1 - term2))

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
    m0 = np.min(mags)  # Minimum magnitude in catalog as M0

    # initial guess
    x0 = [0.1, 0.5, 1.0, 0.01, 1.2]

    bounds = [
        (1e-4, 5.0),   # mu
        (1e-4, 5.0),   # K
        (0.1, 5.0),    # alpha
        (1e-5, 1.0),   # c
        (1.001, 3.0)   # p
    ]

    result = minimize(
        log_likelihood,
        x0,
        args=(times, mags, m0),
        bounds=bounds,
        method="L-BFGS-B",
        options={'maxiter': 250, 'disp': True}
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
