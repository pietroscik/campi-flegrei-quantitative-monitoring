import numpy as np
import pandas as pd

# -----------------------------
# Benioff Strain Release Model
# -----------------------------

def compute_benioff_strain(magnitude):
    """
    Compute Benioff strain release from magnitude.
    
    The Benioff strain release is proportional to 10^(0.75*M).
    
    Parameters:
    -----------
    magnitude : float or array-like
        Earthquake magnitude
    
    Returns:
    --------
    strain : float or array-like
        Benioff strain release
    """
    magnitude = np.array(magnitude)
    
    # Benioff strain release: ε ∝ 10^(0.75*M)
    strain = 10 ** (0.75 * magnitude)
    
    return strain


def cumulative_benioff(df, time_column="time", mag_column="magnitude"):
    """
    Compute cumulative Benioff strain release over time.
    
    Parameters:
    -----------
    df : DataFrame
        Seismic catalog with time and magnitude columns
    time_column : str
        Name of the time column
    mag_column : str
        Name of the magnitude column
    
    Returns:
    --------
    df : DataFrame
        Catalog with cumulative strain column added
    """
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(time_column).reset_index(drop=True)
    
    # Compute strain for each event
    df["strain"] = compute_benioff_strain(df[mag_column])
    
    # Cumulative strain
    df["cumulative_strain"] = df["strain"].cumsum()
    
    return df


def rolling_benioff_rate(df, window=30, time_column="time", mag_column="magnitude"):
    """
    Compute rolling Benioff strain rate.
    
    Parameters:
    -----------
    df : DataFrame
        Seismic catalog
    window : int
        Rolling window size in days
    time_column : str
        Name of the time column
    mag_column : str
        Name of the magnitude column
    
    Returns:
    --------
    rate_series : Series
        Rolling strain rate time series
    """
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(time_column).reset_index(drop=True)
    
    # Compute strain
    df["strain"] = compute_benioff_strain(df[mag_column])
    
    # Set time index
    df = df.set_index(time_column)
    
    # Resample to daily and sum strain
    daily_strain = df["strain"].resample("D").sum()
    
    # Rolling sum
    rolling_rate = daily_strain.rolling(window=window, min_periods=1).sum()
    
    return rolling_rate


def fit_benioff_model(times, cumulative_strain):
    """
    Fit a linear model to cumulative Benioff strain vs time.
    
    This can be used to detect accelerating seismic release.
    
    Parameters:
    -----------
    times : array-like
        Time values (numeric)
    cumulative_strain : array-like
        Cumulative strain values
    
    Returns:
    --------
    slope : float
        Linear trend slope
    intercept : float
        Linear trend intercept
    r_squared : float
        Goodness of fit
    """
    times = np.array(times)
    cumulative_strain = np.array(cumulative_strain)
    
    if len(times) < 3:
        return np.nan, np.nan, np.nan
    
    # Linear regression
    coeffs = np.polyfit(times, cumulative_strain, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    
    # R-squared
    predicted = slope * times + intercept
    ss_res = np.sum((cumulative_strain - predicted) ** 2)
    ss_tot = np.sum((cumulative_strain - np.mean(cumulative_strain)) ** 2)
    
    if ss_tot == 0:
        r_squared = 1.0
    else:
        r_squared = 1 - (ss_res / ss_tot)
    
    return slope, intercept, r_squared


def run_benioff_analysis(input_path="data/processed/catalog_clean.csv"):
    """
    Run Benioff strain release analysis pipeline.
    """
    df = pd.read_csv(input_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    
    # Compute cumulative Benioff strain
    df = cumulative_benioff(df)
    
    # Compute rolling strain rate
    rolling_rate = rolling_benioff_rate(df, window=30)
    
    # Create results dataframe
    results = pd.DataFrame({
        "time": rolling_rate.index,
        "strain_rate": rolling_rate.values
    })
    
    # Fit linear trend to cumulative strain
    times_numeric = (df["time"] - df["time"].min()).dt.total_seconds() / 86400.0
    slope, intercept, r_squared = fit_benioff_model(
        times_numeric.values, 
        df["cumulative_strain"].values
    )
    
    # Save results
    results.to_csv("data/processed/benioff_output.csv", index=False)
    
    print("[OK] Benioff analysis saved -> data/processed/benioff_output.csv")
    print(f"[INFO] Linear trend: slope={slope:.4f}, R²={r_squared:.4f}")
    
    return df, results


if __name__ == "__main__":
    run_benioff_analysis()