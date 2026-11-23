#!/usr/bin/env python3
"""
b_value.py - Analisi del parametro b di Gutenberg-Richter per i Campi Flegrei

Include:
- Stima del parametro b con metodo Maximum Likelihood (Aki-Utsu)
- Calcolo della Magnitude of Completeness (Mc) con metodo MAXC
- Analisi rolling con finestre adattive
- Stima dell'incertezza tramite bootstrap
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def compute_maxc(magnitudes: np.ndarray, bin_size: float = 0.1) -> float:
    """
    Compute the Magnitude of Completeness (Mc) using the Maximum Curvature (MAXC) method.

    Parameters
    ----------
    magnitudes : np.ndarray
        Array of magnitude values
    bin_size : float
        Bin size for the magnitude histogram

    Returns
    -------
    float
        Estimated completeness magnitude
    """
    if len(magnitudes) == 0:
        return np.nan

    min_mag = np.floor(np.min(magnitudes) / bin_size) * bin_size
    max_mag = np.ceil(np.max(magnitudes) / bin_size) * bin_size
    bins = np.arange(min_mag, max_mag + bin_size * 1.5, bin_size)
    hist, bin_edges = np.histogram(magnitudes, bins=bins)
    
    if len(hist) == 0:
        return np.nan
    
    max_idx = np.argmax(hist)
    mc_maxc = bin_edges[max_idx]
    return float(mc_maxc)


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

    # Minimum sample size for statistical reliability
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


def rolling_b_value(
    df: pd.DataFrame,
    window_events: int = 300,
    min_events: int = 150,
    m0: float = 1.0,
    dynamic_m0: bool = True,
    step: int = 50
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
    dynamic_m0 : bool
        If True, computes M0 dynamically per window using MAXC
    step : int
        Step size for overlapping windows

    Returns
    -------
    pd.DataFrame
        DataFrame with time, b_value, b_error, Mc, n_events
    """
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    results = []

    for i in range(0, len(df) - window_events + 1, step):
        window_df = df.iloc[i:i + window_events]
        magnitudes = window_df['magnitude'].values

        # Dynamic Mc estimation
        if dynamic_m0:
            m0_window = compute_maxc(magnitudes)
            if np.isnan(m0_window):
                m0_window = m0
        else:
            m0_window = m0

        # Compute b-value with uncertainty
        b_val, b_err = compute_b_value_uncertainty(magnitudes, m0_window)

        if not np.isnan(b_val):
            results.append({
                'time': window_df['time'].iloc[-1],
                'b_value': b_val,
                'b_error': b_err,
                'mc': m0_window,
                'n_events': len(magnitudes),
                'mean_magnitude': np.mean(magnitudes)
            })

    return pd.DataFrame(results)


def load_seismic_data():
    """Carica i dati sismici dalla cartella processed/."""
    processed_dir = Path("data/processed")

    # Cerca file sismici
    seismic_files = list(processed_dir.glob("*catalog*.csv"))
    
    if not seismic_files:
        # Prova con file generici
        seismic_files = list(processed_dir.glob("*seismic*.csv"))
    
    if not seismic_files:
        print("Nessun dato sismico trovato. Genero dati sintetici...")
        return None

    df = pd.concat([pd.read_csv(f) for f in seismic_files], ignore_index=True)
    
    if 'datetime' in df.columns:
        df['time'] = df['datetime']
    elif 'date' in df.columns and 'time' in df.columns:
        df['time'] = df['date'] + ' ' + df['time']
    
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    print(f"Caricati {len(df)} eventi sismici.")
    return df


def main():
    print("="*50)
    print("Flegrei Alert 2026 - Analisi b-value")
    print("="*50)

    df = load_seismic_data()

    if df is None:
        print("Eseguire prima simulate_data.py per generare dati di test.")
        return

    # Verifica colonne necessarie
    if 'magnitude' not in df.columns:
        print("Colonna 'magnitude' non trovata nei dati.")
        return

    print(f"\nIntervallo temporale: {df['time'].min()} - {df['time'].max()}")
    print(f"Magnitudo: {df['magnitude'].min():.1f} - {df['magnitude'].max():.1f}")

    # Calcola Mc globale
    mc_global = compute_maxc(df['magnitude'].values)
    print(f"\nMagnitude of Completeness (Mc): {mc_global:.2f}")

    # b-value globale
    b_global = compute_b_value(df['magnitude'].values, mc_global)
    print(f"b-value globale: {b_global:.3f}")

    # Rolling b-value
    print("\nCalcolo rolling b-value...")
    b_rolling = rolling_b_value(
        df,
        window_events=300,
        min_events=150,
        m0=mc_global,
        dynamic_m0=True,
        step=50
    )

    print(f"\nCalcolati {len(b_rolling)} valori di b-value rolling")
    print(f"b-value medio: {b_rolling['b_value'].mean():.3f}")
    print(f"b-value std: {b_rolling['b_value'].std():.3f}")
    print(f"b-value min: {b_rolling['b_value'].min():.3f}")
    print(f"b-value max: {b_rolling['b_value'].max():.3f}")

    # Salva output
    reports_dir = Path("data/processed")
    reports_dir.mkdir(parents=True, exist_ok=True)

    output_file = reports_dir / "b_value_rolling.csv"
    b_rolling.to_csv(output_file, index=False)
    print(f"\nOutput salvato in {output_file}")

    # Statistics summary
    print("\n" + "="*50)
    print("SUMMARY ANALISI B-VALUE")
    print("="*50)
    print(f"Eventi totali: {len(df)}")
    print(f"Magnitude of Completeness: {mc_global:.2f}")
    print(f"b-value globale: {b_global:.3f}")
    print(f"Finestre rolling analizzate: {len(b_rolling)}")
    print(f"Periodo coperto: {b_rolling['time'].min().strftime('%Y-%m-%d')} - {b_rolling['time'].max().strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    main()
