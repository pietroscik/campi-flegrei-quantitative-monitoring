import os
import logging
import traceback
import yaml
import pandas as pd
import sys
import subprocess
from src.ingestion.fetch_ingv import fetch_ingv_events, save_raw
from src.preprocessing.clean_catalog import build_catalog
from src.analysis.b_value import run_b_analysis
from src.analysis.anomaly_bvalue import run_anomaly
from src.analysis.multi_signal_model import run_multisignal
from src.analysis.early_warning import run_warning_system
from src.analysis.etas_mle import run_mle
from src.analysis.etas_model import run_etas
from src.deep_learning_models import run_dl_pipeline

try:
    from src.modeling.benioff import run_benioff_analysis
    from src.modeling.changepoint import run_changepoint_analysis
    from src.modeling.csd import run_csd_analysis
    from src.modeling.csi import run_csi_analysis
    from src.modeling.sarima import run_sarima_analysis
    MODELING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Modeling modules not fully available: {e}")
    MODELING_AVAILABLE = False

from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def pipeline():
    logging.info("[PIPELINE START]")

    # Assicurati che le cartelle di output esistano
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    try:
        # 1. INGESTION
        logging.info("Step 1: Ingestion...")
        
        # Use configurable temporal window (default to long-term for statistical validity)
        window_type = config.get('data', {}).get('temporal_window', 'long_term')
        window_days_map = {
            'short_term': config['data']['temporal_windows']['short_term'],
            'medium_term': config['data']['temporal_windows']['medium_term'],
            'long_term': config['data']['temporal_windows']['long_term'],
            'historical': None  # No limit - all available data
        }
        
        end = datetime.now()
        days_to_fetch = window_days_map.get(window_type, config['data']['temporal_windows']['long_term'])
        
        if days_to_fetch is None:
            # Historical mode: fetch maximum available data (e.g., 20+ years)
            start = end - timedelta(days=7300)  # 20 years default for historical
            logging.info(f"Using HISTORICAL window: 20+ years ({start.date()} to {end.date()})")
        else:
            start = end - timedelta(days=days_to_fetch)
            logging.info(f"Using {window_type.upper()} window: {days_to_fetch} days ({start.date()} to {end.date()})")

        # Chunking requests to avoid INGV API limits
        chunk_size_days = 365
        current_start = start
        dfs = []
        
        while current_start < end:
            current_end = min(current_start + timedelta(days=chunk_size_days), end)
            logging.info(f"Fetching API chunk: {current_start.date()} to {current_end.date()}")
            chunk_df = fetch_ingv_events(
                starttime=current_start.isoformat(),
                endtime=current_end.isoformat(),
                minmag=config['b_value']['m0'],
                maxlat=config['data']['bbox']['max_lat'],
                minlat=config['data']['bbox']['min_lat'],
                maxlon=config['data']['bbox']['max_lon'],
                minlon=config['data']['bbox']['min_lon']
            )
            if chunk_df is not None and not chunk_df.empty:
                dfs.append(chunk_df)
            current_start = current_end
            
        if not dfs:
            raise ValueError("No data retrieved from INGV API.")
            
        df = pd.concat(dfs, ignore_index=True)

        save_raw(df)

        # 2. CLEANING
        logging.info("Step 2: Cleaning...")
        clean_df = build_catalog("data/raw/ingv_events.csv")

        # 3. B-VALUE with configurable window sizes
        logging.info("Step 3: B-Value Analysis...")
        window_events = config['b_value'].get('window_events_medium', 300)
        m0 = config['b_value']['m0']
        run_b_analysis(window_events=window_events, m0=m0, config=config)

        # 4. ANOMALY
        logging.info("Step 4: Anomaly Detection...")
        run_anomaly()

        # 5. MULTI-SIGNAL
        logging.info("Step 5: Multi-Signal Model...")
        run_multisignal()

        # 6. ETAS MLE
        logging.info("Step 6: ETAS MLE & Intensity Modeling...")
        run_mle()
        run_etas()

        # 7. DEEP LEARNING
        logging.info("Step 7: Deep Learning Forecasting & Anomalies...")
        run_dl_pipeline()

        # 8. EARLY WARNING
        logging.info("Step 8: Early Warning System (Hybrid)...")
        run_warning_system()

        # 9. ADVANCED MODELING
        if MODELING_AVAILABLE:
            logging.info("Step 9: Advanced Modeling (CSD, CSI, Benioff, SARIMA, Changepoint)...")
            run_benioff_analysis()
            run_changepoint_analysis()
            run_csd_analysis()
            run_csi_analysis()
            run_sarima_analysis()
        else:
            logging.warning("Skipping Advanced Modeling due to missing dependencies.")

        # 10. VALIDATION
        logging.info("Step 10: Running Validation Engine...")
        # Eseguito come modulo per permettere l'importazione da src.deep_learning_models
        subprocess.run([sys.executable, "-m", "src.validation_engine"])

        # 11. FIGURES & REPORT
        logging.info("Step 11: Generating Figures and Reports...")
        subprocess.run([sys.executable, "scripts/generate_paper_figures.py"])
        subprocess.run([sys.executable, "scripts/generate_summary_figure.py"])
        subprocess.run([sys.executable, "scripts/generate_report.py"])

        logging.info("[PIPELINE COMPLETE]")
    except Exception as e:
        logging.error(f"[PIPELINE FAILED] Error: {e}")
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    pipeline()
