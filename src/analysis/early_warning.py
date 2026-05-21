import pandas as pd
import numpy as np
from typing import Dict, Tuple

# -----------------------------
# Threshold dinamici (robusti) con percentili statistici
# -----------------------------

def compute_thresholds(
    series: pd.Series,
    percentiles: Tuple[float, float, float, float] = (0.25, 0.50, 0.75, 0.90)
) -> Dict[str, float]:
    """
    Compute robust thresholds using percentile-based approach.
    
    Parameters
    ----------
    series : pd.Series
        Input time series
    percentiles : tuple
        Percentiles for threshold computation (low, baseline, attention, alert)
        
    Returns
    -------
    dict
        Dictionary with threshold values
        
    Notes
    -----
    Uses empirical percentiles rather than parametric assumptions,
    making the method robust to non-Gaussian distributions typical
    of volcanic unrest indicators.
    """
    q_low = series.quantile(percentiles[0])
    q_baseline = series.quantile(percentiles[1])
    q_attention = series.quantile(percentiles[2])
    q_alert = series.quantile(percentiles[3])

    return {
        "low": q_low,
        "baseline": q_baseline,
        "attention": q_attention,
        "alert": q_alert,
        "extreme": series.max()  # Add extreme level for context
    }


def compute_thresholds_robust(
    series: pd.Series,
    n_sigma: Tuple[float, float, float] = (1.5, 2.5, 3.5)
) -> Dict[str, float]:
    """
    Compute thresholds using median + MAD (robust to outliers).
    
    Parameters
    ----------
    series : pd.Series
        Input time series
    n_sigma : tuple
        Number of MAD deviations for each threshold level
        
    Returns
    -------
    dict
        Dictionary with threshold values based on robust statistics
    """
    median = series.median()
    mad = np.median(np.abs(series - median))
    
    # Convert MAD to sigma equivalent (for normal distribution: sigma ≈ 1.4826 * MAD)
    sigma_robust = 1.4826 * mad
    
    if sigma_robust < 1e-10:
        sigma_robust = series.std()
    
    return {
        "low": median - n_sigma[0] * sigma_robust,
        "baseline": median,
        "attention": median + n_sigma[0] * sigma_robust,
        "alert": median + n_sigma[1] * sigma_robust,
        "extreme": median + n_sigma[2] * sigma_robust
    }


# -----------------------------
# Classification function with persistence awareness
# -----------------------------

def classify_unrest(
    series: pd.Series,
    thresholds: Dict[str, float],
    method: str = "percentile"
) -> np.ndarray:
    """
    Classify unrest levels based on computed thresholds.
    
    Parameters
    ----------
    series : pd.Series
        Unrest index time series
    thresholds : dict
        Threshold dictionary from compute_thresholds
    method : str
        Classification method: 'percentile' or 'robust'
        
    Returns
    -------
    np.ndarray
        Array of classification labels
    """
    if method == "robust":
        # Use robust thresholds (median-based)
        conditions = [
            series <= thresholds["baseline"],
            (series > thresholds["baseline"]) & (series <= thresholds["attention"]),
            (series > thresholds["attention"]) & (series <= thresholds["alert"]),
            series > thresholds["alert"]
        ]
    else:
        # Use percentile-based thresholds
        conditions = [
            series <= thresholds["baseline"],
            (series > thresholds["baseline"]) & (series <= thresholds["attention"]),
            (series > thresholds["attention"]) & (series <= thresholds["alert"]),
            series > thresholds["alert"]
        ]

    labels = np.array(["NORMAL", "ELEVATED", "HIGH", "CRITICAL"])

    return np.select(conditions, labels, default="NORMAL")


# -----------------------------
# Rolling stability check with adaptive windows
# -----------------------------

def compute_persistence(
    series: pd.Series,
    window_days: int = 21,  # 3-week window for trend detection
    min_fraction: float = 0.6
) -> pd.Series:
    """
    Compute persistence of anomalous signal over rolling window.
    
    Parameters
    ----------
    series : pd.Series
        Input time series
    window_days : int
        Rolling window length in days (default 21 for 3-week trend)
    min_fraction : float
        Minimum fraction of days above threshold to consider persistent
        
    Returns
    -------
    pd.Series
        Persistence indicator (fraction of time above baseline)
        
    Notes
    -----
    A persistence value > 0.6 over 21 days indicates a sustained anomaly
    rather than transient noise, which is critical for volcanic monitoring.
    """
    # Compute rolling mean of positive anomalies
    rolling_mean = series.rolling(window=window_days, min_periods=7).mean()
    
    # Fraction of time above median
    median_val = series.median()
    above_median = (series > median_val).astype(int)
    persistence = above_median.rolling(
        window=window_days, 
        min_periods=7
    ).mean()
    
    return persistence.fillna(0.0)


def compute_trend_strength(
    series: pd.Series,
    window_days: int = 30
) -> pd.Series:
    """
    Compute trend strength using linear regression over rolling window.
    
    Parameters
    ----------
    series : pd.Series
        Input time series
    window_days : int
        Window for trend computation
        
    Returns
    -------
    pd.Series
        Trend slope (positive = increasing unrest)
    """
    def rolling_slope(x):
        if len(x) < 7:
            return np.nan
        x_vals = np.arange(len(x))
        try:
            slope = np.polyfit(x_vals, x, 1)[0]
            return slope
        except:
            return np.nan
    
    return series.rolling(window=window_days, min_periods=7).apply(
        rolling_slope, raw=True
    ).fillna(0.0)


# -----------------------------
# Multi-criteria alert system
# -----------------------------

def compute_alert_flag(
    df: pd.DataFrame,
    persistence_window: int = 21,
    trend_window: int = 30,
    persistence_threshold: float = 0.6,
    trend_threshold: float = 0.0
) -> pd.Series:
    """
    Compute final alert flag using multiple criteria.
    
    Criteria for alert:
    1. State is HIGH or CRITICAL
    2. Persistence > threshold (sustained anomaly)
    3. Positive trend (optional confirmation)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'unrest_index', 'state' columns
    persistence_window : int
        Window for persistence computation
    trend_window : int
        Window for trend computation
    persistence_threshold : float
        Minimum persistence for alert
    trend_threshold : float
        Minimum trend slope for alert confirmation
        
    Returns
    -------
    pd.Series
        Binary alert flag (1 = alert, 0 = no alert)
    """
    # Compute persistence
    persistence = compute_persistence(
        df["unrest_index"],
        window_days=persistence_window
    )
    
    # Compute trend
    trend = compute_trend_strength(
        df["unrest_index"],
        window_days=trend_window
    )
    
    # Multi-criteria alert
    alert_conditions = [
        df["state"].isin(["HIGH", "CRITICAL"]),
        persistence > persistence_threshold,
        trend > trend_threshold  # Optional: uncomment to require positive trend
    ]
    
    # Alert requires all conditions (can be relaxed)
    alert_flag = np.all(alert_conditions, axis=0).astype(int)
    
    return alert_flag


# -----------------------------
# Main pipeline with robust parameters
# -----------------------------

def run_warning_system(
    input_path: str = "data/processed/unrest_index.csv",
    output_dir: str = "data/processed",
    threshold_method: str = "percentile",
    persistence_window: int = 21,
    trend_window: int = 30
) -> pd.DataFrame:
    """
    Complete early warning system pipeline with robust statistical parameters.
    
    Parameters
    ----------
    input_path : str
        Path to unrest index CSV
    output_dir : str
        Directory for output files
    threshold_method : str
        Method for threshold computation: 'percentile' or 'robust'
    persistence_window : int
        Window for persistence computation (days)
    trend_window : int
        Window for trend computation (days)
        
    Returns
    -------
    pd.DataFrame
        Warning system output with classifications and alerts
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)
    
    # Handle both indexed and non-indexed time columns
    if 'time' in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
    elif df.index.name == 'time':
        df.index = pd.to_datetime(df.index)
    
    print("\n[EARLY WARNING SYSTEM]")
    print(f"  Input records: {len(df)}")
    print(f"  Time range: {df.index.min()} to {df.index.max()}")
    
    # Compute thresholds
    print(f"\n[THRESHOLD COMPUTATION - Method: {threshold_method}]")
    if threshold_method == "robust":
        thresholds = compute_thresholds_robust(df["unrest_index"])
        print(f"  Median: {df['unrest_index'].median():.4f}")
        print(f"  MAD: {np.median(np.abs(df['unrest_index'] - df['unrest_index'].median())):.4f}")
    else:
        thresholds = compute_thresholds(df["unrest_index"])
        print(f"  Percentiles:")
        print(f"    25th (low): {thresholds['low']:.4f}")
        print(f"    50th (baseline): {thresholds['baseline']:.4f}")
        print(f"    75th (attention): {thresholds['attention']:.4f}")
        print(f"    90th (alert): {thresholds['alert']:.4f}")
    
    print(f"    Extreme max: {thresholds.get('extreme', df['unrest_index'].max()):.4f}")

    # Classify unrest levels
    df["state"] = classify_unrest(
        df["unrest_index"],
        thresholds,
        method=threshold_method
    )

    print(f"\n[UNREST CLASSIFICATION]")
    state_counts = df["state"].value_counts()
    for state in ["NORMAL", "ELEVATED", "HIGH", "CRITICAL"]:
        count = state_counts.get(state, 0)
        pct = 100.0 * count / len(df)
        print(f"  {state}: {count} ({pct:.1f}%)")

    # Compute persistence (stability of anomaly)
    print(f"\n[PERSISTENCE ANALYSIS]")
    print(f"  Window: {persistence_window} days")
    df["persistence"] = compute_persistence(
        df["unrest_index"],
        window_days=persistence_window
    )
    
    # Compute trend strength
    print(f"  Trend window: {trend_window} days")
    df["trend"] = compute_trend_strength(
        df["unrest_index"],
        window_days=trend_window
    )

    # Compute final alert flag (multi-criteria)
    df["alert_flag"] = compute_alert_flag(
        df,
        persistence_window=persistence_window,
        trend_window=trend_window
    )
    
    n_alerts = df["alert_flag"].sum()
    print(f"\n[ALERT STATUS]")
    print(f"  Total alerts triggered: {n_alerts}")
    if n_alerts > 0:
        alert_dates = df[df["alert_flag"] == 1].index
        print(f"  Alert dates: {alert_dates.tolist()[:5]}{'...' if n_alerts > 5 else ''}")

    # Save results
    output_path = os.path.join(output_dir, "early_warning_system.csv")
    
    # Reset index for CSV export
    df_export = df.reset_index()
    df_export.to_csv(output_path, index=False)

    print(f"\n[OK] Early warning system saved -> {output_path}")
    print(f"    Output columns: {list(df_export.columns)}")

    return df_export


if __name__ == "__main__":
    df = run_warning_system()
