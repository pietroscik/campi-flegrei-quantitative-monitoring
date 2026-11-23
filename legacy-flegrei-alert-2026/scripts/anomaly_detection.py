#!/usr/bin/env python3
"""
anomaly_detection.py - Rilevamento anomalie nel parametro b-value

Include:
- Z-score method per anomaly detection
- Quantile-based method
- Combined anomaly score
- Rolling window statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def zscore_anomaly(series: pd.Series, window: int = 50, threshold: float = 2.0) -> pd.Series:
    """
    Detect anomalies using rolling Z-score method.

    Parameters
    ----------
    series : pd.Series
        Input time series (e.g., b-value)
    window : int
        Rolling window size for mean/std calculation
    threshold : float
        Z-score threshold for anomaly detection

    Returns
    -------
    pd.Series
        Boolean series indicating anomalies
    """
    rolling_mean = series.rolling(window=window, min_periods=10).mean()
    rolling_std = series.rolling(window=window, min_periods=10).std()

    zscore = (series - rolling_mean) / rolling_std.replace(0, np.nan)

    return np.abs(zscore) > threshold


def quantile_anomaly(series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
    """
    Detect anomalies using quantile-based method.

    Parameters
    ----------
    series : pd.Series
        Input time series
    lower : float
        Lower quantile threshold
    upper : float
        Upper quantile threshold

    Returns
    -------
    pd.Series
        Boolean series indicating anomalies
    """
    q_lower = series.quantile(lower)
    q_upper = series.quantile(upper)

    return (series < q_lower) | (series > q_upper)


def combined_anomaly_score(
    series: pd.Series,
    zscore_window: int = 50,
    zscore_threshold: float = 2.0,
    quantile_lower: float = 0.05,
    quantile_upper: float = 0.95
) -> pd.DataFrame:
    """
    Compute combined anomaly score from multiple methods.

    Parameters
    ----------
    series : pd.Series
        Input time series
    zscore_window : int
        Rolling window for Z-score
    zscore_threshold : float
        Z-score threshold
    quantile_lower : float
        Lower quantile for anomaly detection
    quantile_upper : float
        Upper quantile for anomaly detection

    Returns
    -------
    pd.DataFrame
        DataFrame with anomaly indicators and combined score
    """
    # Z-score anomaly
    zscore_anom = zscore_anomaly(series, zscore_window, zscore_threshold)

    # Quantile anomaly
    quantile_anom = quantile_anomaly(series, quantile_lower, quantile_upper)

    # Combined score (0-2 scale)
    combined_score = zscore_anom.astype(int) + quantile_anom.astype(int)

    result = pd.DataFrame({
        'time': series.index if hasattr(series.index, '__iter__') else range(len(series)),
        'value': series.values,
        'zscore_anomaly': zscore_anom.values,
        'quantile_anomaly': quantile_anom.values,
        'combined_score': combined_score.values
    })

    return result


def load_bvalue_data(filepath: str = "data/processed/b_value_rolling.csv") -> pd.DataFrame:
    """Carica i dati b-value dal file CSV."""
    path = Path(filepath)

    if not path.exists():
        print(f"File {filepath} non trovato.")
        return None

    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')

    print(f"Caricati {len(df)} valori di b-value.")
    return df


def main():
    print("="*50)
    print("Flegrei Alert 2026 - Rilevamento Anomalie b-value")
    print("="*50)

    df = load_bvalue_data()

    if df is None:
        print("Eseguire prima b_value.py per generare i dati.")
        return

    # Verifica colonna b_value
    if 'b_value' not in df.columns:
        print("Colonna 'b_value' non trovata.")
        return

    b_series = df['b_value']

    print(f"\nIntervallo temporale: {df.index.min()} - {df.index.max()}")
    print(f"b-value range: {b_series.min():.3f} - {b_series.max():.3f}")

    # Calcola anomalie
    print("\nCalcolo anomalie...")
    anomaly_results = combined_anomaly_score(
        b_series,
        zscore_window=50,
        zscore_threshold=2.0,
        quantile_lower=0.05,
        quantile_upper=0.95
    )

    anomaly_results['time'] = df.index

    # Summary statistiche
    n_zscore = anomaly_results['zscore_anomaly'].sum()
    n_quantile = anomaly_results['quantile_anomaly'].sum()
    n_combined = (anomaly_results['combined_score'] >= 2).sum()

    print(f"\nAnomalie rilevate:")
    print(f"  Z-score method: {n_zscore} ({100*n_zscore/len(anomaly_results):.1f}%)")
    print(f"  Quantile method: {n_quantile} ({100*n_quantile/len(anomaly_results):.1f}%)")
    print(f"  Entrambi i metodi: {n_combined} ({100*n_combined/len(anomaly_results):.1f}%)")

    # Salva output
    reports_dir = Path("data/processed")
    reports_dir.mkdir(parents=True, exist_ok=True)

    output_file = reports_dir / "b_value_anomalies.csv"
    anomaly_results.to_csv(output_file, index=False)
    print(f"\nOutput salvato in {output_file}")

    # Mostra periodi critici
    critical_periods = anomaly_results[anomaly_results['combined_score'] >= 2]
    
    if len(critical_periods) > 0:
        print("\n" + "="*50)
        print("PERIODI CRITICI (entrambi i metodi)")
        print("="*50)
        
        for idx, row in critical_periods.head(10).iterrows():
            print(f"  {row['time']}: b={row['value']:.3f}")


if __name__ == "__main__":
    main()
