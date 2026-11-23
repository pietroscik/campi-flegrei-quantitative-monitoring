#!/usr/bin/env python3
"""
early_warning.py - Sistema di Allerta Precoce per i Campi Flegrei

Include:
- Soglie dinamiche basate su percentili
- Classificazione livelli di allerta (GREEN, YELLOW, ORANGE, RED)
- Persistence check per ridurre falsi allarmi
- Calibrazione probabilistica del rischio
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


def compute_thresholds(
    series: pd.Series,
    percentiles: Tuple[float, float, float, float] = (0.25, 0.50, 0.75, 0.90),
    window_days: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute robust thresholds using percentile-based approach.

    Parameters
    ----------
    series : pd.Series
        Input time series
    percentiles : tuple
        Percentiles for threshold computation (low, baseline, attention, alert)
    window_days : int, optional
        If provided, computes rolling percentiles over the specified window.

    Returns
    -------
    dict
        Dictionary with threshold values (static or time-varying)
    """
    if window_days is not None:
        roller = series.rolling(f"{window_days}D", min_periods=1)
        q_low = roller.quantile(percentiles[0])
        q_baseline = roller.quantile(percentiles[1])
        q_attention = roller.quantile(percentiles[2])
        q_alert = roller.quantile(percentiles[3])
    else:
        q_low = series.quantile(percentiles[0])
        q_baseline = series.quantile(percentiles[1])
        q_attention = series.quantile(percentiles[2])
        q_alert = series.quantile(percentiles[3])

    return {
        "low": q_low,
        "baseline": q_baseline,
        "attention": q_attention,
        "alert": q_alert,
        "extreme": series.max()
    }


def classify_unrest(
    series: pd.Series,
    thresholds: Dict[str, Any],
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
    conditions = [
        series <= thresholds["baseline"],
        (series > thresholds["baseline"]) & (series <= thresholds["attention"]),
        (series > thresholds["attention"]) & (series <= thresholds["alert"]),
        series > thresholds["alert"]
    ]

    labels = np.array(["GREEN", "YELLOW", "ORANGE", "RED"])
    return np.select(conditions, labels, default="GREEN")


def compute_persistence(
    series: pd.Series,
    window_days: int = 21,
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
    """
    # Fraction of time above median
    median_val = series.median()
    above_median = (series > median_val).astype(int)
    persistence = above_median.rolling(
        window=window_days,
        min_periods=7
    ).mean()

    return persistence


def calibrate_probability(
    unrest_index: pd.Series,
    thresholds: Dict[str, Any],
    method: str = "empirical"
) -> pd.Series:
    """
    Calibrate unrest index to probability scale (0-1).

    Parameters
    ----------
    unrest_index : pd.Series
        Unrest index time series
    thresholds : dict
        Threshold dictionary
    method : str
        Calibration method: 'empirical' or 'logistic'

    Returns
    -------
    pd.Series
        Calibrated probability series
    """
    if method == "empirical":
        # Usa la posizione percentile come probabilità
        prob = unrest_index.rank(pct=True)

    elif method == "logistic":
        # Trasformazione logistica
        mean_val = unrest_index.mean()
        std_val = unrest_index.std()
        normalized = (unrest_index - mean_val) / std_val
        prob = 1 / (1 + np.exp(-normalized))

    else:
        raise ValueError(f"Unknown calibration method: {method}")

    return prob


def generate_alerts(
    unrest_df: pd.DataFrame,
    persistence_window: int = 21,
    persistence_threshold: float = 0.6
) -> pd.DataFrame:
    """
    Generate alert system output with persistence checks.

    Parameters
    ----------
    unrest_df : pd.DataFrame
        DataFrame with 'time' and 'unrest_index' columns
    persistence_window : int
        Window for persistence check (days)
    persistence_threshold : float
        Minimum persistence for confirmed alert

    Returns
    -------
    pd.DataFrame
        Alert system output with calibrated probabilities and flags
    """
    df = unrest_df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')

    # Compute thresholds
    thresholds = compute_thresholds(df['unrest_index'])

    # Classify alert levels
    df['alert_level'] = classify_unrest(df['unrest_index'], thresholds)

    # Compute persistence
    df['persistence'] = compute_persistence(
        df['unrest_index'],
        window_days=persistence_window
    )

    # Calibrate probability
    df['p_calibrated'] = calibrate_probability(
        df['unrest_index'],
        thresholds,
        method='empirical'
    )

    # Apply persistence check for critical alerts
    df['persistent_critical'] = (
        (df['alert_level'] == 'RED') &
        (df['persistence'] >= persistence_threshold)
    )

    # Binary flag for operational use
    df['alert_flag'] = (df['alert_level'].isin(['ORANGE', 'RED'])).astype(int)

    # Reset index for output
    df = df.reset_index()

    return df


def load_unrest_data(filepath: str = "data/processed/unrest_index.csv") -> pd.DataFrame:
    """Carica dati Unrest Index."""
    path = Path(filepath)

    if not path.exists():
        print(f"File {filepath} non trovato.")
        return None

    df = pd.read_csv(path)
    print(f"Caricati {len(df)} valori di Unrest Index.")
    return df


def main():
    print("="*50)
    print("Flegrei Alert 2026 - Sistema di Allerta Precoce")
    print("="*50)

    df = load_unrest_data()

    if df is None:
        print("Eseguire prima multi_signal_fusion.py per generare Unrest Index.")
        return

    print(f"\nIntervallo temporale: {df['time'].min()} - {df['time'].max()}")
    print(f"Unrest Index range: {df['unrest_index'].min():.3f} - {df['unrest_index'].max():.3f}")

    # Genera allerte
    print("\nGenerazione sistema di allerta...")
    alerts_df = generate_alerts(
        df,
        persistence_window=21,
        persistence_threshold=0.6
    )

    # Summary statistiche
    print("\nDistribuzione livelli di allerta:")
    for level in ['GREEN', 'YELLOW', 'ORANGE', 'RED']:
        count = (alerts_df['alert_level'] == level).sum()
        pct = 100 * count / len(alerts_df)
        print(f"  {level}: {count} ({pct:.1f}%)")

    # Persistenza
    n_persistent = alerts_df['persistent_critical'].sum()
    print(f"\nAllerte RED persistenti: {n_persistent}")

    # Probabilità calibrata
    print(f"\nProbabilità calibrata:")
    print(f"  Media: {alerts_df['p_calibrated'].mean():.3f}")
    print(f"  Max: {alerts_df['p_calibrated'].max():.3f}")

    # Salva output
    reports_dir = Path("data/processed")
    reports_dir.mkdir(parents=True, exist_ok=True)

    output_file = reports_dir / "early_warning_system.csv"
    alerts_df.to_csv(output_file, index=False)
    print(f"\nOutput salvato in {output_file}")

    # Mostra periodi critici
    critical = alerts_df[alerts_df['alert_level'] == 'RED']
    
    if len(critical) > 0:
        print("\n" + "="*50)
        print("PERIODI CRITICI (RED ALERT)")
        print("="*50)
        
        for idx, row in critical.head(10).iterrows():
            persistent_str = " [PERSISTENTE]" if row['persistent_critical'] else ""
            print(f"  {row['time']}: p={row['p_calibrated']:.3f}{persistent_str}")


if __name__ == "__main__":
    main()
