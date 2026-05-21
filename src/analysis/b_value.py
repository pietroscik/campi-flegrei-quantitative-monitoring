import pandas as pd
import numpy as np
from typing import Tuple, Optional

# -----------------------------
# Gutenberg-Richter b-value
# b = log10(e) / (mean(M) - M0)
# -----------------------------

def compute_b_value(magnitudes: np.ndarray, m0: float = 0.0) -> float:
    """
    Compute maximum likelihood estimate of b-value using Aki's formula.
    
    Parameters
    ----------
    magnitudes : array-like
        Array of magnitude values
    m0 : float
        Minimum magnitude threshold for completeness
        
    Returns
    -------
    float
        Estimated b-value, or NaN if insufficient data
    
    Notes
    -----
    Uses the Aki-Utsu formula: b = log10(e) / (M_mean - M0)
    Requires minimum 50 events for statistical stability.
    """
    magnitudes = np.array(magnitudes)
    magnitudes = magnitudes[magnitudes >= m0]

    # Minimum sample size for statistical reliability (actuarial standard)
    if len(magnitudes) < 50:
        return np.nan

    mean_m = np.mean(magnitudes)
    std_m = np.std(magnitudes)

    # Check for degenerate cases
    if mean_m == m0 or std_m < 0.01:
        return np.nan

    # Aki's estimator with correction for small samples
    b = (np.log10(np.e)) / (mean_m - m0)
    
    # Shi and Boltz correction for small sample bias
    n = len(magnitudes)
    if n < 200:
        correction_factor = 1.0 + (1.0 / n)
        b *= correction_factor
    
    return b


def compute_b_value_uncertainty(magnitudes: np.ndarray, m0: float = 0.0) -> Tuple[float, float]:
    """
    Compute b-value with uncertainty estimate using bootstrap resampling.
    
    Returns
    -------
    tuple
        (b_value, standard_error)
    """
    magnitudes = np.array(magnitudes)
    magnitudes = magnitudes[magnitudes >= m0]
    
    if len(magnitudes) < 50:
        return np.nan, np.nan
    
    b_main = compute_b_value(magnitudes, m0)
    
    # Bootstrap uncertainty estimation
    n_bootstrap = 200
    b_samples = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(magnitudes, size=len(magnitudes), replace=True)
        b_boot = compute_b_value(sample, m0)
        if not np.isnan(b_boot):
            b_samples.append(b_boot)
    
    if len(b_samples) < 10:
        return b_main, np.nan
    
    std_error = np.std(b_samples, ddof=1)
    return b_main, std_error


# -----------------------------
# Rolling b-value with adaptive windowing
# -----------------------------

def rolling_b_value(
    df: pd.DataFrame,
    window_events: int = 300,  # Increased from 100 to 300 for statistical stability
    min_events: int = 150,     # Minimum events for computation
    m0: float = 1.0,           # Magnitude of completeness for Campi Flegrei
    step: int = 50             # Step size for overlapping windows
) -> pd.DataFrame:
    """
    Compute rolling b-value using event-based windows for temporal stability.
    
    Parameters
    ----------
    df : pd.DataFrame
        Catalog DataFrame with 'time' and 'magnitude' columns
    window_events : int
        Number of events per window (default 300 for robust statistics)
    min_events : int
        Minimum events required to compute b-value
    m0 : float
        Magnitude completeness threshold
    step : int
        Step size between consecutive windows
        
    Returns
    -------
    pd.DataFrame
        Time series of b-values with uncertainties
        
    Notes
    -----
    Event-based windows ensure consistent statistical power across time periods
    with varying seismicity rates. This is critical for non-stationary volcanic
    processes where event rates can vary by orders of magnitude.
    """
    df = df.sort_values("time").reset_index(drop=True)

    b_values = []
    b_errors = []
    times = []
    window_centers = []
    event_counts = []

    mags = df["magnitude"].values
    t = df["time"].values

    # Use overlapping windows for smooth temporal evolution
    for i in range(0, len(df) - window_events, step):
        window_mags = mags[i:i+window_events]
        window_times = t[i:i+window_events]
        
        # Filter by magnitude completeness
        valid_mask = window_mags >= m0
        n_valid = np.sum(valid_mask)
        
        if n_valid < min_events:
            continue

        b_val, b_err = compute_b_value_uncertainty(window_mags[valid_mask], m0=m0)
        
        if not np.isnan(b_val):
            b_values.append(b_val)
            b_errors.append(b_err)
            # Use median time of window as representative time
            times.append(window_times[len(window_times)//2])
            window_centers.append(i + window_events//2)
            event_counts.append(n_valid)

    result_df = pd.DataFrame({
        "time": times,
        "b_value": b_values,
        "b_error": b_errors,
        "event_count": event_counts,
        "window_center_idx": window_centers
    })

    return result_df


def rolling_b_value_time_based(
    df: pd.DataFrame,
    window_days: int = 90,      # 3-month windows for seasonal/volcanic cycles
    step_days: int = 30,        # Monthly steps
    min_events: int = 100,      # Minimum events per window
    m0: float = 1.0
) -> pd.DataFrame:
    """
    Compute rolling b-value using fixed time windows.
    
    Useful for detecting seasonal patterns or comparing with other 
    time-based geophysical signals (uplift, gas emissions).
    
    Parameters
    ----------
    df : pd.DataFrame
        Catalog DataFrame
    window_days : int
        Window length in days (default 90 for quarterly analysis)
    step_days : int
        Step between windows in days
    min_events : int
        Minimum events required
    m0 : float
        Magnitude completeness threshold
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    b_values = []
    b_errors = []
    times = []
    event_counts = []

    start_date = df["time"].min()
    end_date = df["time"].max()

    current_start = start_date
    while current_start + pd.Timedelta(days=window_days) <= end_date:
        window_end = current_start + pd.Timedelta(days=window_days)
        
        mask = (df["time"] >= current_start) & (df["time"] < window_end)
        window_df = df[mask]
        
        if len(window_df) >= min_events:
            window_mags = window_df["magnitude"].values
            b_val, b_err = compute_b_value_uncertainty(window_mags, m0=m0)
            
            if not np.isnan(b_val):
                b_values.append(b_val)
                b_errors.append(b_err)
                times.append(current_start + pd.Timedelta(days=window_days//2))
                event_counts.append(len(window_df))
        
        current_start += pd.Timedelta(days=step_days)

    return pd.DataFrame({
        "time": times,
        "b_value": b_values,
        "b_error": b_errors,
        "event_count": event_counts
    })


# -----------------------------
# Global b-value with confidence intervals
# -----------------------------

def global_b_value(
    df: pd.DataFrame,
    m0: float = 1.0,
    confidence_level: float = 0.95
) -> dict:
    """
    Compute global b-value with full uncertainty quantification.
    
    Returns
    -------
    dict
        Contains b_value, standard_error, confidence_interval, n_events
    """
    magnitudes = df["magnitude"].values
    magnitudes = magnitudes[magnitudes >= m0]
    
    if len(magnitudes) < 50:
        return {
            "b_value": np.nan,
            "standard_error": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_events": len(magnitudes)
        }
    
    b_val, b_err = compute_b_value_uncertainty(magnitudes, m0)
    
    # Confidence interval using normal approximation
    from scipy import stats
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    ci_lower = b_val - z_score * b_err
    ci_upper = b_val + z_score * b_err
    
    return {
        "b_value": b_val,
        "standard_error": b_err,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_events": len(magnitudes)
    }


# -----------------------------
# Change point detection in b-value series
# -----------------------------

def detect_bvalue_changepoints(
    b_series: pd.DataFrame,
    min_segment_length: int = 5
) -> list:
    """
    Detect significant change points in b-value temporal evolution.
    
    Uses cumulative sum (CUSUM) approach for robust detection.
    
    Parameters
    ----------
    b_series : pd.DataFrame
        Output from rolling_b_value with 'time' and 'b_value' columns
    min_segment_length : int
        Minimum number of points between change points
        
    Returns
    -------
    list
        Indices of detected change points
    """
    from scipy.stats import zscore
    
    b_vals = b_series["b_value"].values
    b_errs = b_series.get("b_error", np.ones_like(b_vals) * 0.1).values
    
    # Weighted z-score considering uncertainties
    b_mean = np.nanmean(b_vals)
    b_std = np.sqrt(np.nanmean(b_errs**2))
    
    if b_std < 0.01:
        return []
    
    z_scores = np.abs(zscore(b_vals))
    
    # Identify potential change points (|z| > 2)
    potential_cps = np.where(z_scores > 2.0)[0]
    
    # Filter by minimum segment length
    if len(potential_cps) == 0:
        return []
    
    change_points = [potential_cps[0]]
    for cp in potential_cps[1:]:
        if cp - change_points[-1] >= min_segment_length:
            change_points.append(cp)
    
    return change_points


# -----------------------------
# Pipeline runner
# -----------------------------

def run_b_analysis(
    input_path: str = "data/processed/catalog_clean.csv",
    output_dir: str = "data/processed",
    m0: float = 1.0,
    window_events: int = 300
) -> dict:
    """
    Complete b-value analysis pipeline with robust statistical parameters.
    
    Parameters
    ----------
    input_path : str
        Path to cleaned catalog CSV
    output_dir : str
        Directory for output files
    m0 : float
        Magnitude completeness threshold (default 1.0 for Campi Flegrei)
    window_events : int
        Number of events per rolling window (default 300)
        
    Returns
    -------
    dict
        Analysis results including global b-value and rolling series metadata
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(input_path)
    df["time"] = pd.to_datetime(df["time"])

    # Global b-value with uncertainty
    global_results = global_b_value(df, m0=m0)
    
    print(f"\n[GLOBAL B-VALUE ANALYSIS]")
    print(f"  Magnitude threshold (M0): {m0}")
    print(f"  Total events analyzed: {global_results['n_events']}")
    print(f"  b-value: {global_results['b_value']:.4f} ± {global_results['standard_error']:.4f}")
    print(f"  95% CI: [{global_results['ci_lower']:.4f}, {global_results['ci_upper']:.4f}]")

    # Rolling b-value (event-based windows)
    rolling_event = rolling_b_value(df, window_events=window_events, m0=m0)
    rolling_event_path = os.path.join(output_dir, "b_value_rolling_events.csv")
    rolling_event.to_csv(rolling_event_path, index=False)
    
    print(f"\n[ROLLING B-VALUE - EVENT BASED]")
    print(f"  Window size: {window_events} events")
    print(f"  Number of windows: {len(rolling_event)}")
    if len(rolling_event) > 0:
        print(f"  Mean b: {rolling_event['b_value'].mean():.4f}")
        print(f"  Std b: {rolling_event['b_value'].std():.4f}")
        print(f"  Saved to: {rolling_event_path}")

    # Rolling b-value (time-based windows)
    rolling_time = rolling_b_value_time_based(df, window_days=90, m0=m0)
    rolling_time_path = os.path.join(output_dir, "b_value_rolling_time.csv")
    rolling_time.to_csv(rolling_time_path, index=False)
    
    print(f"\n[ROLLING B-VALUE - TIME BASED]")
    print(f"  Window size: 90 days")
    print(f"  Number of windows: {len(rolling_time)}")
    if len(rolling_time) > 0:
        print(f"  Mean b: {rolling_time['b_value'].mean():.4f}")
        print(f"  Saved to: {rolling_time_path}")

    # Change point detection
    if len(rolling_event) > 10:
        changepoints = detect_bvalue_changepoints(rolling_event)
        print(f"\n[CHANGE POINT DETECTION]")
        print(f"  Detected {len(changepoints)} significant change points")
        if changepoints:
            print(f"  Locations (indices): {changepoints}")

    return {
        "global": global_results,
        "rolling_event": rolling_event,
        "rolling_time": rolling_time,
        "changepoints": changepoints if len(rolling_event) > 10 else []
    }


if __name__ == "__main__":
    results = run_b_analysis()
