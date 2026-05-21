import pandas as pd
import numpy as np
from typing import Tuple, Optional

# -----------------------------
# SIGNAL 1: seismic rate with robust temporal aggregation
# -----------------------------

def compute_seismic_rate(
    df: pd.DataFrame,
    freq: str = "W",  # Weekly instead of daily for better statistics
    min_events: int = 5
) -> pd.Series:
    """
    Compute seismicity rate with robust temporal binning.
    
    Parameters
    ----------
    df : pd.DataFrame
        Catalog with 'time' and 'magnitude' columns
    freq : str
        Resampling frequency ('D'=daily, 'W'=weekly, 'M'=monthly)
        Default: weekly for statistical stability
    min_events : int
        Minimum events to consider a valid time bin
        
    Returns
    -------
    pd.Series
        Time series of seismic rates
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")

    rate = df.resample(freq).size()
    
    # Filter out periods with too few events (statistically unreliable)
    rate = rate[rate >= min_events]
    rate = rate.rename("seismic_rate")

    return rate


def compute_seismic_rate_rolling(
    df: pd.DataFrame,
    window_days: int = 30,  # 30-day rolling window
    step_days: int = 7      # Weekly steps
) -> pd.DataFrame:
    """
    Compute rolling seismicity rate for smooth temporal evolution.
    
    Parameters
    ----------
    df : pd.DataFrame
        Catalog DataFrame
    window_days : int
        Rolling window length in days
    step_days : int
        Step between consecutive windows
        
    Returns
    -------
    pd.DataFrame
        Time series with rate, cumulative energy, and event count
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    rates = []
    times = []
    energy_releases = []
    event_counts = []

    start_date = df["time"].min()
    end_date = df["time"].max()

    current_start = start_date
    while current_start + pd.Timedelta(days=window_days) <= end_date:
        window_end = current_start + pd.Timedelta(days=window_days)
        
        mask = (df["time"] >= current_start) & (df["time"] < window_end)
        window_df = df[mask]
        
        if len(window_df) >= 10:  # Minimum for statistical reliability
            n_events = len(window_df)
            rate = n_events / window_days  # events per day
            
            # Benioff strain release (simplified: sum of 10^(1.5*M))
            energy = np.sum(10 ** (1.5 * window_df["magnitude"].values))
            
            rates.append(rate)
            times.append(current_start + pd.Timedelta(days=window_days//2))
            energy_releases.append(energy)
            event_counts.append(n_events)
        
        current_start += pd.Timedelta(days=step_days)

    return pd.DataFrame({
        "time": times,
        "seismic_rate": rates,
        "benioff_strain": energy_releases,
        "event_count": event_counts
    })


# -----------------------------
# SIGNAL 2: b-value alignment with uncertainty weighting
# -----------------------------

def align_bvalue(
    b_df: pd.DataFrame,
    target_freq: str = "W"
) -> pd.Series:
    """
    Align b-value time series to target frequency with uncertainty handling.
    
    Parameters
    ----------
    b_df : pd.DataFrame
        B-value output from rolling_b_value with 'time', 'b_value', 'b_error'
    target_freq : str
        Target resampling frequency
        
    Returns
    -------
    pd.Series
        Aligned b-value series
    """
    b_df = b_df.copy()
    b_df["time"] = pd.to_datetime(b_df["time"])
    b_df = b_df.set_index("time")

    # If b_error is available, use inverse variance weighting
    if "b_error" in b_df.columns:
        # Weight by inverse variance
        b_df["weight"] = 1.0 / (b_df["b_error"] ** 2 + 0.01)  # Add small constant
        
        # Resample with weighted mean
        b_resampled = b_df["b_value"].resample(target_freq).apply(
            lambda x: np.average(x, weights=b_df.loc[x.index, "weight"])
        )
    else:
        b_resampled = b_df["b_value"].resample(target_freq).mean()
    
    return b_resampled.rename("b_value")


# -----------------------------
# SIGNAL 3: uplift (RITE / GNSS)
# -----------------------------

def align_uplift(uplift_df: pd.DataFrame, target_freq: str = "W") -> pd.Series:
    """
    Align uplift data to target frequency.
    
    Parameters
    ----------
    uplift_df : pd.DataFrame
        Uplift DataFrame with 'time' and 'uplift' columns
    target_freq : str
        Target resampling frequency
        
    Returns
    -------
    pd.Series
        Aligned uplift series
    """
    uplift_df = uplift_df.copy()
    uplift_df["time"] = pd.to_datetime(uplift_df["time"])
    uplift_df = uplift_df.set_index("time")

    return uplift_df["uplift"].resample(target_freq).last().rename("uplift")


# -----------------------------
# Normalization with robust scaling
# -----------------------------

def normalize(series: pd.Series, method: str = "zscore") -> pd.Series:
    """
    Normalize time series using robust methods.
    
    Parameters
    ----------
    series : pd.Series
        Input time series
    method : str
        Normalization method: 'zscore', 'minmax', or 'robust'
        
    Returns
    -------
    pd.Series
        Normalized series
    """
    if method == "zscore":
        # Standard z-score normalization
        return (series - series.mean()) / series.std()
    
    elif method == "robust":
        # Robust scaling using median and MAD
        median = series.median()
        mad = np.median(np.abs(series - median))
        if mad < 1e-10:
            mad = series.std()
        return (series - median) / (mad + 1e-10)
    
    elif method == "minmax":
        # Min-max scaling to [0, 1]
        return (series - series.min()) / (series.max() - series.min() + 1e-10)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# -----------------------------
# Multi-signal fusion index with optimized weights
# -----------------------------

def build_unrest_index(
    seismic_rate: pd.Series,
    b_value: pd.Series,
    uplift: Optional[pd.Series] = None,
    weights: dict = None,
    normalization_method: str = "robust"
) -> pd.DataFrame:
    """
    Build composite unrest index from multiple geophysical signals.
    
    Parameters
    ----------
    seismic_rate : pd.Series
        Seismicity rate time series
    b_value : pd.Series
        B-value time series
    uplift : pd.Series, optional
        Ground uplift time series
    weights : dict, optional
        Weights for each signal (default: theory-based weights)
    normalization_method : str
        Method for normalizing signals
        
    Returns
    -------
    pd.DataFrame
        DataFrame with normalized signals and composite index
        
    Notes
    -----
    Default weights are based on empirical studies of Campi Flegrei:
    - Seismic rate: primary indicator of stress accumulation
    - B-value: secondary indicator (inverse relationship)
    - Uplift: tertiary indicator (direct relationship)
    """
    # Concatenate all signals
    data_dict = {"seismic_rate": seismic_rate}
    
    if isinstance(b_value, pd.Series):
        # Align b_value to seismic_rate frequency
        b_aligned = b_value.reindex(seismic_rate.index, method="nearest")
        data_dict["b_value"] = b_aligned
    
    if uplift is not None and isinstance(uplift, pd.Series):
        uplift_aligned = uplift.reindex(seismic_rate.index, method="linear")
        data_dict["uplift"] = uplift_aligned
    
    df = pd.DataFrame(data_dict)
    df = df.dropna()
    
    if len(df) < 10:
        raise ValueError("Insufficient overlapping data for index computation")

    # Normalize signals
    df["rate_n"] = normalize(df["seismic_rate"], method=normalization_method)
    
    if "b_value" in df.columns:
        # Invert b-value (low b = high stress)
        df["b_n"] = normalize(df["b_value"], method=normalization_method)
    else:
        df["b_n"] = 0.0
    
    if "uplift" in df.columns:
        df["uplift_n"] = normalize(df["uplift"], method=normalization_method)
    else:
        df["uplift_n"] = 0.0

    # Apply weights (default theory-based)
    if weights is None:
        weights = {
            "rate": 0.5,    # Increased weight for seismic rate
            "b_value": 0.3,  # Moderate weight for b-value
            "uplift": 0.2    # Lower weight if uplift data is sparse
        }
    
    # Adjust weights based on available data
    total_weight = 0.0
    if "uplift" in df.columns and df["uplift"].notna().sum() > len(df) * 0.5:
        total_weight = weights["rate"] + weights["b_value"] + weights["uplift"]
    else:
        # Renormalize without uplift
        total_weight = weights["rate"] + weights["b_value"]
        weights["rate"] = weights["rate"] / total_weight
        weights["b_value"] = weights["b_value"] / total_weight
        total_weight = 1.0

    # Compute composite index
    df["unrest_index"] = (
        weights["rate"] * df["rate_n"] +
        weights["b_value"] * (-df["b_n"]) +  # Negative: low b = high unrest
        weights.get("uplift", 0.0) * df.get("uplift_n", 0.0)
    )

    return df


# -----------------------------
# PIPELINE with robust temporal parameters
# -----------------------------

def run_multisignal(
    catalog_path: str = "data/processed/catalog_clean.csv",
    b_path: str = "data/processed/b_value_rolling_events.csv",
    uplift_path: Optional[str] = None,
    output_dir: str = "data/processed",
    rate_window_days: int = 30,
    rate_step_days: int = 7,
    normalization_method: str = "robust"
) -> pd.DataFrame:
    """
    Complete multi-signal analysis pipeline with robust temporal parameters.
    
    Parameters
    ----------
    catalog_path : str
        Path to cleaned seismic catalog
    b_path : str
        Path to b-value rolling analysis output
    uplift_path : str, optional
        Path to uplift data (if available)
    output_dir : str
        Directory for output files
    rate_window_days : int
        Rolling window for seismic rate (default 30 days)
    rate_step_days : int
        Step size for rate computation (default 7 days)
    normalization_method : str
        Method for signal normalization
        
    Returns
    -------
    pd.DataFrame
        Multi-signal unrest index time series
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    catalog = pd.read_csv(catalog_path)
    b_df = pd.read_csv(b_path)
    
    # Convert timestamps
    catalog['time'] = pd.to_datetime(catalog['time'])
    b_df['time'] = pd.to_datetime(b_df['time'])
    
    # Compute seismic rate with rolling windows (30-day windows, weekly steps)
    print("\n[SEISMIC RATE ANALYSIS]")
    print(f"  Rolling window: {rate_window_days} days")
    print(f"  Step size: {rate_step_days} days")
    
    rate_df = compute_seismic_rate_rolling(
        catalog,
        window_days=rate_window_days,
        step_days=rate_step_days
    )
    
    if len(rate_df) == 0:
        raise ValueError("No valid seismic rate windows found. Check catalog duration.")
    
    print(f"  Computed {len(rate_df)} rate measurements")
    
    # Set time index for merging
    rate_df = rate_df.set_index('time')
    seismic_rate = rate_df['seismic_rate']
    
    # Align b-value to same temporal grid
    print("\n[B-VALUE ALIGNMENT]")
    b_df_indexed = b_df.set_index('time')
    b_value = b_df_indexed['b_value'].reindex(rate_df.index, method="nearest")
    
    if "b_error" in b_df.columns:
        b_error = b_df_indexed['b_error'].reindex(rate_df.index, method="nearest")
        print(f"  B-value uncertainty included")
    
    print(f"  Aligned {b_value.notna().sum()} b-value measurements")
    
    # Load uplift if available
    uplift = None
    if uplift_path and os.path.exists(uplift_path):
        print("\n[UPLIFT DATA]")
        try:
            uplift_df = pd.read_csv(uplift_path)
            uplift_df['time'] = pd.to_datetime(uplift_df['time'])
            uplift = uplift_df.set_index('time')['uplift'].reindex(
                rate_df.index, method="linear"
            )
            print(f"  Loaded {uplift.notna().sum()} uplift measurements")
        except Exception as e:
            print(f"  Warning: Could not load uplift data: {e}")
            uplift = None
    else:
        print("\n[UPLIFT DATA] Not provided - proceeding with seismic signals only")
    
    # Build unrest index
    print("\n[UNREST INDEX COMPUTATION]")
    df = build_unrest_index(
        seismic_rate=seismic_rate,
        b_value=b_value,
        uplift=uplift,
        normalization_method=normalization_method
    )

    # Save results
    output_path = os.path.join(output_dir, "unrest_index.csv")
    df.to_csv(output_path)
    
    print(f"\n[OK] Multi-signal index saved -> {output_path}")
    print(f"    Total time steps: {len(df)}")
    print(f"    Time range: {df.index.min()} to {df.index.max()}")
    print(f"    Mean unrest index: {df['unrest_index'].mean():.4f}")
    print(f"    Std unrest index: {df['unrest_index'].std():.4f}")
    
    # Save rate diagnostics
    rate_output_path = os.path.join(output_dir, "seismic_rate_diagnostics.csv")
    rate_df.to_csv(rate_output_path)
    print(f"    Rate diagnostics saved -> {rate_output_path}")

    return df


if __name__ == "__main__":
    df = run_multisignal()
