import numpy as np
import pandas as pd

# -----------------------------
# CSI (Critical Seismicity Index) Model
# -----------------------------

def compute_seismic_moment(magnitude):
    """
    Compute seismic moment from magnitude.
    
    Using the Hanks & Kanamori relationship:
    M0 = 10^(1.5*M + 9.1) in N·m
    
    Parameters:
    -----------
    magnitude : float or array-like
        Earthquake magnitude
    
    Returns:
    --------
    moment : float or array-like
        Seismic moment in N·m
    """
    magnitude = np.array(magnitude)
    
    # M0 = 10^(1.5*M + 9.1)
    moment = 10 ** (1.5 * magnitude + 9.1)
    
    return moment


def compute_csi(df, window=50, time_column="time", mag_column="magnitude"):
    """
    Compute Critical Seismicity Index (CSI).
    
    The CSI is based on the concept of accelerating seismicity before
    critical transitions. It combines seismic moment release with
    temporal clustering analysis.
    
    Parameters:
    -----------
    df : DataFrame
        Seismic catalog with time and magnitude columns
    window : int
        Rolling window size (number of events)
    time_column : str
        Name of the time column
    mag_column : str
        Name of the magnitude column
    
    Returns:
    --------
    csi_series : Series
        CSI time series
    """
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(time_column).reset_index(drop=True)
    
    # Compute seismic moment for each event
    df["moment"] = compute_seismic_moment(df[mag_column])
    
    csi_values = []
    times = []
    
    mags = df[mag_column].values
    moments = df["moment"].values
    times_arr = df[time_column].values
    
    for i in range(window, len(df)):
        window_mags = mags[i-window:i]
        window_moments = moments[i-window:i]
        window_times = times_arr[i-window:i]
        
        # Total moment release in window
        total_moment = np.sum(window_moments)
        
        # Event rate in window
        time_span = (window_times[-1] - window_times[0]).astype('timedelta64[s]').astype(float)
        if time_span > 0:
            event_rate = window / (time_span / 86400.0)  # events per day
        else:
            event_rate = 0
        
        # Magnitude variance (indicator of criticality)
        mag_variance = np.var(window_mags)
        
        # CSI formula: combination of moment release, rate, and variance
        # Higher CSI indicates approach to critical state
        csi = np.log10(total_moment + 1) * (1 + mag_variance) * np.log10(event_rate + 1)
        
        csi_values.append(csi)
        times.append(times_arr[i])
    
    return pd.Series(csi_values, index=times, name="csi")


def normalize_csi(csi_series):
    """
    Normalize CSI to 0-1 range using min-max scaling.
    
    Parameters:
    -----------
    csi_series : Series
        Raw CSI values
    
    Returns:
    --------
    normalized : Series
        Normalized CSI (0-1)
    """
    min_val = csi_series.min()
    max_val = csi_series.max()
    
    if max_val - min_val == 0:
        return pd.Series(0.5, index=csi_series.index)
    
    normalized = (csi_series - min_val) / (max_val - min_val)
    
    return normalized


def detect_critical_periods(csi_series, threshold=0.8):
    """
    Detect periods where CSI exceeds critical threshold.
    
    Parameters:
    -----------
    csi_series : Series
        CSI time series
    threshold : float
        Critical threshold (0-1)
    
    Returns:
    --------
    critical_mask : Series
        Boolean mask indicating critical periods
    """
    normalized = normalize_csi(csi_series)
    critical_mask = normalized > threshold
    
    return critical_mask


def run_csi_analysis(input_path="data/processed/catalog_clean.csv", window=50):
    """
    Run CSI analysis pipeline.
    """
    df = pd.read_csv(input_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    
    # Compute CSI
    csi = compute_csi(df, window=window)
    
    # Normalize
    csi_normalized = normalize_csi(csi)
    
    # Detect critical periods
    critical = detect_critical_periods(csi, threshold=0.8)
    
    # Create results dataframe
    results = pd.DataFrame({
        "time": csi.index,
        "csi": csi.values,
        "csi_normalized": csi_normalized.values,
        "critical": critical.astype(int)
    })
    
    # Save results
    results.to_csv("data/processed/csi_output.csv", index=False)
    
    print("[OK] CSI analysis saved -> data/processed/csi_output.csv")
    print(f"[INFO] Critical periods detected: {critical.sum()} out of {len(critical)}")
    
    if critical.sum() > 0:
        critical_times = results[critical]["time"]
        print(f"[INFO] Latest critical period: {critical_times.iloc[-1]}")
    
    return df, results


if __name__ == "__main__":
    run_csi_analysis()