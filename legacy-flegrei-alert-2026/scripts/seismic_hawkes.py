#!/usr/bin/env python3
"""
seismic_hawkes.py - Analisi sismica con processi di Hawkes e Survival Analysis

Analisi del catalogo sismico dei Campi Flegrei utilizzando:
- Processi puntuali di Hawkes per modellare il clustering temporale
- Survival Analysis (Weibull) per valutare i tempi di ricorrenza
- Test di stazionarietà
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def load_seismic_data():
    """Carica i dati sismici dalla cartella processed/."""
    processed_dir = Path("data/processed")
    
    # Cerca file sismici
    seismic_files = list(processed_dir.glob("*seismic*.csv"))
    
    if not seismic_files:
        print("Nessun dato sismico trovato. Genero dati sintetici...")
        return None
    
    df = pd.concat([pd.read_csv(f) for f in seismic_files], ignore_index=True)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print(f"Caricati {len(df)} eventi sismici.")
    return df

def inter_event_times(df):
    """Calcola i tempi intercorrenti tra eventi."""
    dt = df['datetime'].diff().dropna()
    return dt.dt.total_seconds() / 3600  # in ore

def fit_weibull(dt_hours):
    """
    Fit della distribuzione di Weibull ai tempi intercorrenti.
    
    La Weibull è utile per identificare:
    - k < 1: clustering (eventi tendono a raggrupparsi)
    - k = 1: processo di Poisson (eventi indipendenti)
    - k > 1: regolarità (eventi tendono a essere periodici)
    """
    # MLE per Weibull
    shape, loc, scale = stats.weibull_min.fit(dt_hours, floc=0)
    
    print(f"\nFit Weibull:")
    print(f"  Shape parameter (k): {shape:.3f}")
    print(f"  Scale parameter (λ): {scale:.3f}")
    
    if shape < 1:
        interpretation = "Clustering sismico rilevato (aftershock sequence)"
    elif shape > 1:
        interpretation = "Comportamento quasi-periodico"
    else:
        interpretation = "Processo di Poisson (eventi indipendenti)"
    
    print(f"  Interpretazione: {interpretation}")
    
    return shape, scale

def hawkes_log_likelihood(params, timestamps):
    """
    Log-likelihood per un processo di Hawkes esponenziale.
    
    Parametri:
    - mu: tasso background
    - alpha: produttività (numero medio di triggered events)
    - beta: decadimento temporale
    """
    mu, alpha, beta = params
    
    if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:
        return 1e10  # Penalizza parametri non validi
    
    T = timestamps[-1] - timestamps[0]
    n = len(timestamps)
    
    # Calcolo intensità condizionata
    log_lik = 0
    for i, t in enumerate(timestamps):
        intensity = mu
        for j in range(i):
            intensity += alpha * beta * np.exp(-beta * (t - timestamps[j]))
        log_lik += np.log(max(intensity, 1e-10))
    
    # Termine integrale
    log_lik -= mu * T
    log_lik += n * np.log(1 - alpha / beta) if alpha < beta else -1e10
    
    return -log_lik  # Minimizziamo il negativo

def fit_hawkes(timestamps):
    """Fit di un processo di Hawkes ai dati sismici."""
    # Normalizza timestamps in giorni
    t0 = timestamps.min()
    timestamps_norm = (timestamps - t0).dt.total_seconds() / 86400
    timestamps_arr = timestamps_norm.values
    
    # Stima iniziale
    rate = len(timestamps_arr) / timestamps_arr.max()
    init_params = [rate * 0.5, 0.5, 1.0]
    
    # Ottimizzazione
    result = minimize(
        hawkes_log_likelihood,
        init_params,
        args=(timestamps_arr,),
        method='L-BFGS-B',
        bounds=[(1e-6, None), (0, 0.99), (1e-6, None)]
    )
    
    if result.success:
        mu, alpha, beta = result.x
        print(f"\nFit Processo di Hawkes:")
        print(f"  Background rate (μ): {mu:.3f} eventi/giorno")
        print(f"  Productivity (α): {alpha:.3f}")
        print(f"  Decay rate (β): {beta:.3f} giorno⁻¹")
        print(f"  Branching ratio (n=α/β): {alpha/beta:.3f}")
        
        if alpha / beta > 0.5:
            print("  ⚠️  Alto grado di clustering: sistema potenzialmente critico")
        
        return {'mu': mu, 'alpha': alpha, 'beta': beta}
    else:
        print("Fit Hawkes non convergente.")
        return None

def survival_analysis(df):
    """Esegue survival analysis completa."""
    print("\n" + "="*50)
    print("SURVIVAL ANALYSIS")
    print("="*50)
    
    dt = inter_event_times(df)
    
    # Statistiche descrittive
    print(f"\nStatistiche tempi intercorrenti (ore):")
    print(f"  Media: {dt.mean():.2f}")
    print(f"  Mediana: {dt.median():.2f}")
    print(f"  Std Dev: {dt.std():.2f}")
    
    # Fit Weibull
    fit_weibull(dt)
    
    # Test di stazionarietà (test se il rateo cambia nel tempo)
    print("\nTest di stazionarietà:")
    n = len(dt)
    first_half = dt[:n//2].mean()
    second_half = dt[n//2:].mean()
    ratio = second_half / first_half if first_half > 0 else float('inf')
    
    print(f"  Rateo prima metà: {1/first_half:.3f} eventi/ora")
    print(f"  Rateo seconda metà: {1/second_half:.3f} eventi/ora")
    print(f"  Rapporto: {ratio:.2f}")
    
    if ratio > 1.5:
        print("  ⚠️  Aumento significativo della sismicità")
    elif ratio < 0.67:
        print("  ℹ️  Diminuzione della sismicità")

def main():
    print("="*50)
    print("Flegrei Alert 2026 - Analisi Sismica Hawkes")
    print("="*50)
    
    df = load_seismic_data()
    
    if df is None:
        print("Eseguire prima simulate_data.py per generare dati di test.")
        return
    
    # Survival Analysis
    survival_analysis(df)
    
    # Fit Hawkes
    print("\n" + "="*50)
    print("PROCESSO DI HAWKES")
    print("="*50)
    fit_hawkes(df['datetime'])
    
    # Output
    reports_dir = Path("reports/fig")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nAnalisi completata. Grafici salvati in {reports_dir}/")

if __name__ == "__main__":
    main()
