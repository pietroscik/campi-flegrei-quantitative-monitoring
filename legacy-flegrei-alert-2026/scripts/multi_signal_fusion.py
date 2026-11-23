#!/usr/bin/env python3
"""
multi_signal_fusion.py - Fusione multi-segnale per monitoraggio vulcanico

Include:
- Integrazione di sismicità, b-value e uplift GNSS
- Calcolo dell'Unrest Index composite
- Normalizzazione robusta dei segnali
- Allineamento temporale multi-frequenza
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def compute_seismic_rate(
    df: pd.DataFrame,
    freq: str = "W",
    min_events: int = 5
) -> pd.Series:
    """
    Compute seismicity rate with temporal binning.

    Parameters
    ----------
    df : pd.DataFrame
        Catalog with 'time' and 'magnitude' columns
    freq : str
        Resampling frequency ('D'=daily, 'W'=weekly, 'M'=monthly)
    min_events : int
        Minimum events to consider a valid time bin

    Returns
    -------
    pd.Series
        Time series of seismic rates
    """
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')

    rate = df.resample(freq).size()
    rate = rate[rate >= min_events]
    rate = rate.rename('seismic_rate')

    return rate


def align_bvalue(
    b_df: pd.DataFrame,
    target_freq: str = "W"
) -> pd.Series:
    """
    Align b-value time series to target frequency.

    Parameters
    ----------
    b_df : pd.DataFrame
        B-value output with 'time', 'b_value' columns
    target_freq : str
        Target resampling frequency

    Returns
    -------
    pd.Series
        Aligned b-value series
    """
    b_df = b_df.copy()
    b_df['time'] = pd.to_datetime(b_df['time'])
    b_df = b_df.set_index('time')

    b_resampled = b_df['b_value'].resample(target_freq).mean()
    return b_resampled.rename('b_value')


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
    if uplift_df is None or len(uplift_df) == 0:
        return pd.Series()

    uplift_df = uplift_df.copy()
    uplift_df['time'] = pd.to_datetime(uplift_df['time'])
    uplift_df = uplift_df.set_index('time')

    return uplift_df['uplift'].resample(target_freq).last().rename('uplift')


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
        return (series - series.mean()) / series.std()

    elif method == "minmax":
        return (series - series.min()) / (series.max() - series.min() + 1e-10)

    elif method == "robust":
        median = series.median()
        mad = np.median(np.abs(series - median))
        return (series - median) / (1.4826 * mad + 1e-10)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_unrest_index(
    seismic_rate: pd.Series,
    b_value: pd.Series,
    uplift: Optional[pd.Series] = None,
    weights: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Compute composite Unrest Index from multiple signals.

    UI(t) = w1 * Rate_norm(t) + w2 * [-b_norm(t)] + w3 * Uplift_norm(t)

    Parameters
    ----------
    seismic_rate : pd.Series
        Seismicity rate time series
    b_value : pd.Series
        B-value time series
    uplift : pd.Series, optional
        Ground uplift time series
    weights : dict
        Weights for each signal (default: {rate: 0.4, bvalue: 0.3, uplift: 0.3})

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized signals and composite index
    """
    if weights is None:
        weights = {'rate': 0.4, 'bvalue': 0.3, 'uplift': 0.3}

    # Align all series to common index
    common_index = seismic_rate.index.union(b_value.index)
    if uplift is not None and len(uplift) > 0:
        common_index = common_index.union(uplift.index)

    # Reindex and interpolate
    rate_aligned = seismic_rate.reindex(common_index).interpolate(method='time')
    b_aligned = b_value.reindex(common_index).interpolate(method='time')

    if uplift is not None and len(uplift) > 0:
        uplift_aligned = uplift.reindex(common_index).interpolate(method='time')
    else:
        uplift_aligned = None

    # Normalize signals
    rate_norm = normalize(rate_aligned.dropna(), method='zscore')
    b_norm = normalize(b_aligned.dropna(), method='zscore')

    # Invert b-value (low b = high stress)
    b_norm_inverted = -b_norm

    # Compute unrest index
    if uplift_aligned is not None and len(uplift_aligned) > 0:
        uplift_norm = normalize(uplift_aligned.dropna(), method='zscore')

        # Combine with available data
        ui_components = pd.DataFrame({
            'rate_norm': rate_norm,
            'b_norm_inv': b_norm_inverted,
            'uplift_norm': uplift_norm
        }, index=common_index)

        unrest_index = (
            weights['rate'] * ui_components['rate_norm'] +
            weights['bvalue'] * ui_components['b_norm_inv'] +
            weights['uplift'] * ui_components['uplift_norm']
        )
    else:
        # No uplift data: use only seismic signals
        ui_components = pd.DataFrame({
            'rate_norm': rate_norm,
            'b_norm_inv': b_norm_inverted
        }, index=common_index)

        # Renormalize weights
        total_weight = weights['rate'] + weights['bvalue']
        unrest_index = (
            (weights['rate'] / total_weight) * ui_components['rate_norm'] +
            (weights['bvalue'] / total_weight) * ui_components['b_norm_inv']
        )

    result = pd.DataFrame({
        'time': ui_components.index,
        'unrest_index': unrest_index.values,
        'rate_component': ui_components['rate_norm'].values,
        'bvalue_component': ui_components['b_norm_inv'].values
    })

    if uplift_aligned is not None and 'uplift_norm' in ui_components.columns:
        result['uplift_component'] = ui_components['uplift_norm'].values

    return result


def load_seismic_data() -> pd.DataFrame:
    """Carica dati sismici."""
    processed_dir = Path("data/processed")

    seismic_files = list(processed_dir.glob("*catalog*.csv"))
    if not seismic_files:
        seismic_files = list(processed_dir.glob("*seismic*.csv"))

    if not seismic_files:
        return None

    df = pd.concat([pd.read_csv(f) for f in seismic_files], ignore_index=True)

    if 'datetime' in df.columns:
        df['time'] = df['datetime']
    elif 'date' in df.columns and 'time' in df.columns:
        df['time'] = df['date'] + ' ' + df['time']

    df['time'] = pd.to_datetime(df['time'])
    return df


def load_bvalue_data() -> pd.DataFrame:
    """Carica dati b-value."""
    path = Path("data/processed/b_value_rolling.csv")
    if not path.exists():
        return None

    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'])
    return df


def load_uplift_data() -> Optional[pd.DataFrame]:
    """Carica dati di uplift (opzionale)."""
    # Prova percorsi diversi
    paths = [
        Path("data/external/uplift.csv"),
        Path("data/processed/gnss_uplift.csv"),
        Path("data/raw/uplift.csv")
    ]

    for path in paths:
        if path.exists():
            df = pd.read_csv(path)
            if 'date' in df.columns:
                df['time'] = df['date']
            elif 'datetime' in df.columns:
                df['time'] = df['datetime']

            df['time'] = pd.to_datetime(df['time'])
            return df

    print("Dati di uplift non trovati (opzionale).")
    return None


def main():
    print("="*50)
    print("Flegrei Alert 2026 - Fusione Multi-Segnale")
    print("="*50)

    # Carica dati
    print("\nCaricamento dati...")
    seismic_df = load_seismic_data()
    bvalue_df = load_bvalue_data()
    uplift_df = load_uplift_data()

    if seismic_df is None:
        print("Dati sismici non trovati. Eseguire prima simulate_data.py")
        return

    if bvalue_df is None:
        print("Dati b-value non trovati. Eseguire prima b_value.py")
        return

    print(f"  Sismicità: {len(seismic_df)} eventi")
    print(f"  b-value: {len(bvalue_df)} misure")
    if uplift_df is not None:
        print(f"  Uplift: {len(uplift_df)} misure")

    # Calcola segnali
    print("\nElaborazione segnali...")

    # Tasso sismico settimanale
    seismic_rate = compute_seismic_rate(seismic_df, freq="W", min_events=3)
    print(f"  Tasso sismico: {len(seismic_rate)} settimane")

    # Allinea b-value
    b_value_aligned = align_bvalue(bvalue_df, target_freq="W")
    print(f"  b-value allineato: {len(b_value_aligned)} settimane")

    # Allinea uplift (se disponibile)
    if uplift_df is not None:
        uplift_aligned = align_uplift(uplift_df, target_freq="W")
        print(f"  Uplift allineato: {len(uplift_aligned)} settimane")
    else:
        uplift_aligned = None
        print("  Uplift: non disponibile")

    # Calcola Unrest Index
    print("\nCalcolo Unrest Index...")
    unrest_result = compute_unrest_index(
        seismic_rate,
        b_value_aligned,
        uplift_aligned,
        weights={'rate': 0.4, 'bvalue': 0.3, 'uplift': 0.3}
    )

    print(f"\nUnrest Index calcolato per {len(unrest_result)} settimane")
    print(f"  Media: {unrest_result['unrest_index'].mean():.3f}")
    print(f"  Std Dev: {unrest_result['unrest_index'].std():.3f}")
    print(f"  Min: {unrest_result['unrest_index'].min():.3f}")
    print(f"  Max: {unrest_result['unrest_index'].max():.3f}")

    # Salva output
    reports_dir = Path("data/processed")
    reports_dir.mkdir(parents=True, exist_ok=True)

    output_file = reports_dir / "unrest_index.csv"
    unrest_result.to_csv(output_file, index=False)
    print(f"\nOutput salvato in {output_file}")

    # Summary
    print("\n" + "="*50)
    print("SUMMARY FUSIONE MULTI-SEGNALE")
    print("="*50)
    print(f"Periodo: {unrest_result['time'].min()} - {unrest_result['time'].max()}")
    print(f"Pesi utilizzati: rate={0.4}, b-value={0.3}, uplift={0.3}")

    # Identifica periodi di alto unrest
    high_unrest = unrest_result[unrest_result['unrest_index'] > unrest_result['unrest_index'].quantile(0.9)]
    if len(high_unrest) > 0:
        print(f"\nPeriodi di alto unrest (>90° percentile): {len(high_unrest)} settimane")


if __name__ == "__main__":
    main()
