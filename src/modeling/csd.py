import numpy as np
import pandas as pd
from scipy.optimize import minimize

# -----------------------------
# CSD (Coulomb Stress Drop) Model
# -----------------------------

def compute_csd(magnitude, area_km2=1.0):
    """
    Compute Coulomb Stress Drop from magnitude and rupture area.
    
    Parameters:
    -----------
    magnitude : float or array-like
        Earthquake magnitude
    area_km2 : float
        Rupture area in km²
    
    Returns:
    --------
    csd : float or array-like
        Coulomb Stress Drop in MPa
    """
    # Simplified formula: stress drop ~ 10^(1.5*M) / area
    # Using typical scaling relationships
    
    magnitude = np.array(magnitude)
    
    # Moment magnitude to seismic moment (N·m)
    # M0 = 10^(1.5*M + 9.1)
    M0 = 10 ** (1.5 * magnitude + 9.1)
    
    # Convert area to m²
    area_m2 = area_km2 * 1e6
    
    # Stress drop Δσ = (7/16) * M0 / r³ where r is characteristic radius
    # Simplified: Δσ ≈ M0 / area^(3/2)
    # Using simpler approximation for demonstration
    csd = 1e-6 * M0 / (area_m2 ** 1.5)  # Convert to MPa
    
    return csd


def rolling_csd(df, window=50, area_column=None):
    """
    Compute rolling average CSD over a window of events.
    
    Parameters:
    -----------
    df : DataFrame
        Catalog with magnitude column
    window : int
        Rolling window size (number of events)
    area_column : str, optional
        Column name for rupture area if available
    
    Returns:
    --------
    csd_series : Series
        Rolling CSD values
    """
    if area_column and area_column in df.columns:
        areas = df[area_column].values
    else:
        # Use magnitude-area scaling: log(A) = M - 4.0 (simplified)
        areas = 10 ** (df["magnitude"].values - 4.0)
    
    csd_values = []
    
    for i in range(window, len(df)):
        window_mags = df["magnitude"].iloc[i-window:i].values
        window_areas = areas[i-window:i]
        
        window_csd = compute_csd(window_mags, window_areas)
        csd_values.append(np.mean(window_csd))
    
    return pd.Series(csd_values, index=df.index[window:])


def run_csd_analysis(input_path="data/processed/catalog_clean.csv"):
    """
    Run CSD analysis pipeline.
    """
    df = pd.read_csv(input_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    
    # Compute CSD for each event (assuming typical area scaling)
    df["csd"] = compute_csd(df["magnitude"])
    
    # Compute rolling CSD
    df["csd_rolling"] = rolling_csd(df, window=50)
    
    # Save results
    df.to_csv("data/processed/csd_output.csv", index=False)
    
    print("[OK] CSD analysis saved -> data/processed/csd_output.csv")
    print(f"[INFO] Mean CSD: {df['csd'].mean():.4f} MPa")
    
    return df


if __name__ == "__main__":
    run_csd_analysis()