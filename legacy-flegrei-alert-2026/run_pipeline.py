#!/usr/bin/env python3
"""
run_pipeline.py - Pipeline completa per il monitoraggio dei Campi Flegrei

Questo script esegue l'intera pipeline di analisi utilizzando SOLO dati reali:
1. Download dati da fonti INGV certificate
2. Import e pulizia dati
3. Calcolo b-value con rolling window
4. Rilevamento anomalie (Z-score, quantili)
5. Fusione multi-segnale (sismicità, b-value, uplift GNSS)
6. Sistema di allerta precoce con soglie dinamiche
7. Generazione report e visualizzazioni

IMPORTANTE: Nessun dato sintetico viene generato o utilizzato.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_data_availability():
    """Verifica la presenza di dati reali nelle directory appropriate."""
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    
    print("\n[CHECK] Verifica disponibilità dati...")
    
    # Cerca dati sismici
    seismic_files = list(raw_dir.glob("*seismic*.csv")) + list(raw_dir.glob("*INGV*.csv"))
    if not seismic_files:
        print("  ⚠️  DATI SISMICI NON TROVATI")
        print("     Eseguire: python scripts/download_data.py")
        print("     Oppure scaricare manualmente da http://iside.rm.ingv.it/")
        return False
    else:
        print(f"  ✓ Dati sismici trovati: {len(seismic_files)} file")
    
    # Cerca dati GNSS (opzionali ma raccomandati)
    gnss_files = list(raw_dir.glob("*gnss*")) + list(raw_dir.glob("*GNSS*"))
    if gnss_files:
        print(f"  ✓ Dati GNSS trovati: {len(gnss_files)} file")
    else:
        print("  ⚠️  Dati GNSS non trovati (opzionali)")
    
    # Cerca dati geochimici (opzionali)
    gas_files = list(raw_dir.glob("*gas*")) + list(raw_dir.glob("*geochem*"))
    if gas_files:
        print(f"  ✓ Dati geochimici trovati: {len(gas_files)} file")
    else:
        print("  ⚠️  Dati geochimici non trovati (opzionali)")
    
    return True

def run_ingestion():
    """Esegue l'import e pulizia dei dati."""
    print("\n" + "=" * 60)
    print("[STEP 1] INGESTION E PULIZIA DATI")
    print("=" * 60)
    
    try:
        from src.data_ingestion import run_ingestion_pipeline, DataIngestionError
        
        catalog_path = Path("data/processed/cleaned_catalog.csv")
        df = run_ingestion_pipeline(data_dir="data/raw", output_path=str(catalog_path))
        
        print(f"\n✓ Catalogo pulito: {len(df)} eventi")
        print(f"  Periodo: {df['time'].min()} - {df['time'].max()}")
        
        # Normalizza nome colonna magnitudo (può essere 'mag' o 'magnitude')
        if 'mag' in df.columns and 'magnitude' not in df.columns:
            df = df.rename(columns={'mag': 'magnitude'})
        
        print(f"  Magnitudo: {df['magnitude'].min():.1f} - {df['magnitude'].max():.1f}")
        
        return df
    except Exception as e:
        print(f"\n✗ Errore nell'ingestion: {e}")
        return None

def run_bvalue_analysis(df):
    """Esegue l'analisi del b-value."""
    print("\n" + "=" * 60)
    print("[STEP 2] CALCOLO B-VALUE")
    print("=" * 60)
    
    try:
        from src.analysis.b_value import rolling_b_value, compute_maxc
        
        # Calcola Mc (magnitude of completeness)
        mags = df["magnitude"].values
        mc = compute_maxc(mags)
        print(f"\nMagnitude of completeness (Mc): {mc:.2f}")
        
        # Rolling b-value
        print("\nCalcolo rolling b-value...")
        b_df = rolling_b_value(df, window_events=300, step=50, m0=mc)
        
        # Salva risultati
        b_file = Path("data/processed/b_value_rolling.csv")
        b_df.to_csv(b_file, index=False)
        print(f"✓ Salvato: {b_file}")
        print(f"  B-value medio: {b_df['b_value'].mean():.3f} ± {b_df['b_value'].std():.3f}")
        
        return b_df
    except Exception as e:
        print(f"\n✗ Errore nel calcolo b-value: {e}")
        return None

def run_anomaly_detection(b_df):
    """Rileva anomalie nel b-value."""
    print("\n" + "=" * 60)
    print("[STEP 3] RILEVAMENTO ANOMALIE")
    print("=" * 60)
    
    try:
        from src.analysis.anomaly_bvalue import detect_anomalies
        
        print("\nApplicazione metodi di rilevamento anomalie...")
        anomalies_df = detect_anomalies(b_df)
        
        # Salva risultati
        anom_file = Path("data/processed/b_value_anomalies.csv")
        anomalies_df.to_csv(anom_file, index=False)
        print(f"✓ Salvato: {anom_file}")
        
        n_anomalies = anomalies_df["combined_anomaly"].sum() if "combined_anomaly" in anomalies_df.columns else 0
        print(f"  Anomalie rilevate: {n_anomalies}")
        
        return anomalies_df
    except Exception as e:
        print(f"\n✗ Errore nel rilevamento anomalie: {e}")
        return None

def run_multi_signal_fusion(df, b_df):
    """Fonde multipli segnali in un indice di unrest."""
    print("\n" + "=" * 60)
    print("[STEP 4] FUSIONE MULTI-SEGNALE")
    print("=" * 60)
    
    try:
        from src.analysis.multi_signal_model import (
            compute_seismic_rate_rolling,
            align_bvalue,
            build_unrest_index
        )
        
        # Calcola tasso sismico
        print("\nCalcolo tasso sismico rolling...")
        rate_df = compute_seismic_rate_rolling(df, window_days=30, step_days=7)
        
        # Allinea b-value
        print("Allineamento b-value...")
        b_aligned = align_bvalue(b_df, target_freq="W")
        
        # Prova a caricare dati uplift (se disponibili)
        uplift_file = Path("data/raw/uplift.csv")
        if uplift_file.exists():
            print("Caricamento dati uplift GNSS...")
            import pandas as pd
            uplift_df = pd.read_csv(uplift_file)
            uplift_aligned = pd.Series(uplift_df["uplift"].values, 
                                       index=pd.to_datetime(uplift_df["time"]))
        else:
            print("⚠️  Dati uplift non disponibili - solo analisi sismica")
            uplift_aligned = None
        
        # Calcola unrest index usando build_unrest_index
        print("Calcolo indice di unrest composito...")
        unrest_df = build_unrest_index(rate_df["seismic_rate"], b_aligned, uplift_aligned)
        
        # Salva risultati
        unrest_file = Path("data/processed/unrest_index.csv")
        unrest_df.to_csv(unrest_file, index=False)
        print(f"✓ Salvato: {unrest_file}")
        
        return unrest_df
    except Exception as e:
        print(f"\n✗ Errore nella fusione multi-segnale: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_early_warning(unrest_df, df):
    """Esegue il sistema di allerta precoce."""
    print("\n" + "=" * 60)
    print("[STEP 5] SISTEMA DI ALLERTA PRECOCE")
    print("=" * 60)
    
    try:
        from src.analysis.early_warning import build_ews, validate_ews
        
        print("\nCostruzione sistema di allerta...")
        ews_df = build_ews(unrest_df, df)
        
        print("Validazione prestazioni...")
        metrics = validate_ews(ews_df)
        
        # Salva risultati
        ews_file = Path("data/processed/early_warning_system.csv")
        ews_df.to_csv(ews_file, index=False)
        print(f"✓ Salvato: {ews_file}")
        
        # Stampa metriche
        print("\nMetriche di validazione:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.3f}")
        
        return ews_df
    except Exception as e:
        print(f"\n✗ Errore nel sistema di allerta: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_summary():
    """Genera un report riassuntivo."""
    print("\n" + "=" * 60)
    print("[STEP 6] GENERAZIONE REPORT")
    print("=" * 60)
    
    processed_dir = Path("data/processed")
    
    print("\nFile generati:")
    for f in sorted(processed_dir.glob("*.csv")):
        size_kb = f.stat().st_size / 1024
        print(f"  ✓ {f.name} ({size_kb:.1f} KB)")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETATA CON SUCCESSO")
    print("=" * 60)
    print(f"\nData/ora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Output disponibili in: data/processed/")
    print("\nNota: Questo sistema utilizza SOLO dati reali e validati.")

def main():
    print("=" * 60)
    print("CAMPI FLEGREI QUANTITATIVE MONITORING SYSTEM")
    print(f"Esecuzione: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("\nIMPORTANTE: Sistema basato su DATI REALI e VALIDATI")
    print("Nessun dato sintetico verrà generato o utilizzato.\n")
    
    # Verifica dati
    if not check_data_availability():
        print("\n✗ Dati insufficienti. Interrompo l'esecuzione.")
        sys.exit(1)
    
    # Step 1: Ingestion
    df = run_ingestion()
    if df is None:
        sys.exit(1)
    
    # Step 2: B-value analysis
    b_df = run_bvalue_analysis(df)
    if b_df is None:
        sys.exit(1)
    
    # Step 3: Anomaly detection
    anomalies_df = run_anomaly_detection(b_df)
    if anomalies_df is None:
        sys.exit(1)
    
    # Step 4: Multi-signal fusion
    unrest_df = run_multi_signal_fusion(df, b_df)
    if unrest_df is None:
        sys.exit(1)
    
    # Step 5: Early warning system
    ews_df = run_early_warning(unrest_df, df)
    if ews_df is None:
        sys.exit(1)
    
    # Step 6: Summary
    generate_summary()

if __name__ == "__main__":
    main()
