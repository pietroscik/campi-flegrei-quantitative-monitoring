import os
import logging
import traceback
import yaml
from src.ingestion.fetch_ingv import fetch_ingv_events, save_raw
from src.preprocessing.clean_catalog import build_catalog
from src.analysis.b_value import run_b_analysis
from src.analysis.anomaly_bvalue import run_anomaly
from src.analysis.multi_signal_model import run_multisignal
from src.analysis.early_warning import run_warning_system
from src.analysis.etas_mle import run_mle

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
        end = datetime.now()
        start = end - timedelta(days=365)

        df = fetch_ingv_events(
            starttime=start.isoformat(),
            endtime=end.isoformat(),
            minmag=config['b_value']['m0'],
            maxlat=config['data']['bbox']['max_lat'],
            minlat=config['data']['bbox']['min_lat'],
            maxlon=config['data']['bbox']['max_lon'],
            minlon=config['data']['bbox']['min_lon']
        )

        save_raw(df)

        # 2. CLEANING
        logging.info("Step 2: Cleaning...")
        clean_df = build_catalog("data/raw/ingv_events.csv")

        # 3. B-VALUE
        logging.info("Step 3: B-Value Analysis...")
        run_b_analysis()

        # 4. ANOMALY
        logging.info("Step 4: Anomaly Detection...")
        run_anomaly()

        # 5. MULTI-SIGNAL
        logging.info("Step 5: Multi-Signal Model...")
        run_multisignal()

        # 6. EARLY WARNING
        logging.info("Step 6: Early Warning System...")
        run_warning_system()

        # 7. ETAS MLE
        logging.info("Step 7: ETAS MLE...")
        run_mle()

        logging.info("[PIPELINE COMPLETE]")
    except Exception as e:
        logging.error(f"[PIPELINE FAILED] Error: {e}")
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    pipeline()
