import numpy as np
import pandas as pd
from scipy import stats

# -----------------------------
# Changepoint Detection for Seismic Data
# -----------------------------

def detect_changepoints_cusum(series, threshold=5.0):
    """
    Detect changepoints using CUSUM (Cumulative Sum) method.
    
    Parameters:
    -----------
    series : array-like
        Time series data
    threshold : float
        Detection threshold
    
    Returns:
    --------
    changepoints : list
        Indices of detected changepoints
    """
    series = np.array(series)
    mean_val = np.mean(series)
    std_val = np.std(series)
    
    if std_val == 0:
        return []
    
    # Standardize
    z = (series - mean_val) / std_val
    
    # CUSUM calculation
    cusum_pos = np.zeros(len(series))
    cusum_neg = np.zeros(len(series))
    
    for i in range(1, len(series)):
        cusum_pos[i] = max(0, cusum_pos[i-1] + z[i] - 0.5)
        cusum_neg[i] = min(0, cusum_neg[i-1] + z[i] + 0.5)
    
    # Detect changepoints
    changepoints = []
    
    for i in range(1, len(series)):
        if cusum_pos[i] > threshold or abs(cusum_neg[i]) > threshold:
            changepoints.append(i)
            cusum_pos[i] = 0
            cusum_neg[i] = 0
    
    return changepoints


def detect_changepoints_pelt(series, penalty=3.0):
    """
    Simplified PELT-like changepoint detection using variance changes.
    
    Parameters:
    -----------
    series : array-like
        Time series data
    penalty : float
        Penalty for adding a changepoint
    
    Returns:
    --------
    changepoints : list
        Indices of detected changepoints
    """
    series = np.array(series)
    n = len(series)
    
    if n < 10:
        return []
    
    changepoints = []
    
    # Sliding window approach to detect variance changes
    window_size = min(50, n // 5)
    
    for i in range(window_size, n - window_size):
        left_var = np.var(series[i-window_size:i])
        right_var = np.var(series[i:i+window_size])
        
        # F-test for variance ratio
        if left_var > 0 and right_var > 0:
            var_ratio = max(left_var, right_var) / min(left_var, right_var)
            
            if var_ratio > penalty:
                changepoints.append(i)
    
    # Remove close changepoints (within window_size)
    filtered = []
    for cp in changepoints:
        if not filtered or cp - filtered[-1] >= window_size:
            filtered.append(cp)
    
    return filtered


def compute_segment_stats(series, changepoints):
    """
    Compute statistics for each segment defined by changepoints.
    
    Parameters:
    -----------
    series : array-like
        Time series data
    changepoints : list
        Indices of changepoints
    
    Returns:
    --------
    segments : list of dicts
        Statistics for each segment
    """
    series = np.array(series)
    points = [0] + changepoints + [len(series)]
    
    segments = []
    
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        segment = series[start:end]
        
        segments.append({
            "start": start,
            "end": end,
            "mean": np.mean(segment),
            "std": np.std(segment),
            "length": len(segment)
        })
    
    return segments


def run_changepoint_analysis(input_path="data/processed/catalog_clean.csv"):
    """
    Run changepoint detection on seismic catalog.
    """
    df = pd.read_csv(input_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    
    # Analyze magnitude time series
    mags = df["magnitude"].values
    
    # Detect changepoints
    cp_cusum = detect_changepoints_cusum(mags, threshold=5.0)
    cp_pelt = detect_changepoints_pelt(mags, penalty=3.0)
    
    # Combine results
    all_cp = sorted(set(cp_cusum + cp_pelt))
    
    # Compute segment statistics
    segments = compute_segment_stats(mags, all_cp)
    
    # Add changepoint flags to dataframe
    df["changepoint"] = 0
    for cp in all_cp:
        if cp < len(df):
            df.loc[cp, "changepoint"] = 1
    
    # Save results
    results = {
        "catalog": df,
        "changepoints": all_cp,
        "segments": segments
    }
    
    df.to_csv("data/processed/changepoint_output.csv", index=False)
    
    print("[OK] Changepoint analysis saved -> data/processed/changepoint_output.csv")
    print(f"[INFO] Detected {len(all_cp)} changepoints")
    
    for i, seg in enumerate(segments):
        print(f"  Segment {i+1}: events {seg['start']}-{seg['end']}, "
              f"mean M={seg['mean']:.2f}, std={seg['std']:.2f}")
    
    return results


if __name__ == "__main__":
    run_changepoint_analysis()