"""
Worker Cycle Service for Campi Flegrei Monitoring System

This module executes a complete analysis cycle on recent seismic data,
suitable for periodic execution (e.g., cron job, scheduled task).
"""

import os
import sys
import logging
import yaml
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ingestion.fetch_ingv import fetch_ingv_events, save_raw
from src.preprocessing.clean_catalog import build_catalog
from src.analysis.b_value import run_b_analysis
from src.analysis.anomaly_bvalue import run_anomaly
from src.analysis.multi_signal_model import run_multisignal
from src.analysis.early_warning import run_warning_system
from src.analysis.etas_mle import run_mle

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def run_cycle(days_back=7, config_path="config.yaml"):
    """
    Execute a complete monitoring cycle on recent data.
    
    Parameters
    ----------
    days_back : int
        Number of days to look back for seismic events
    config_path : str
        Path to configuration YAML file
    """
    logging.info(f"[WORKER CYCLE START] - Analyzing last {days_back} days")
    
    # Ensure output directories exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    try:
        # 1. INGESTION - Fetch recent events from INGV
        logging.info("Step 1: Ingestion...")
        end = datetime.now()
        start = end - timedelta(days=days_back)
        
        df = fetch_ingv_events(
            starttime=start.isoformat(),
            endtime=end.isoformat(),
            minmag=config['b_value']['m0'],
            maxlat=config['data']['bbox']['max_lat'],
            minlat=config['data']['bbox']['min_lat'],
            maxlon=config['data']['bbox']['max_lon'],
            minlon=config['data']['bbox']['min_lon']
        )
        
        if df is None or len(df) == 0:
            logging.warning("No events found in the specified time window")
            return
        
        save_raw(df)
        logging.info(f"  → Fetched {len(df)} events")
        
        # 2. CLEANING - Quality control and preprocessing
        logging.info("Step 2: Cleaning...")
        clean_df = build_catalog("data/raw/ingv_events.csv")
        logging.info(f"  → Clean catalog: {len(clean_df)} events")
        
        # 3. B-VALUE - Gutenberg-Richter parameter estimation
        logging.info("Step 3: B-Value Analysis...")
        run_b_analysis()
        
        # 4. ANOMALY - Statistical anomaly detection
        logging.info("Step 4: Anomaly Detection...")
        run_anomaly()
        
        # 5. MULTI-SIGNAL - Composite unrest index
        logging.info("Step 5: Multi-Signal Model...")
        run_multisignal()
        
        # 6. EARLY WARNING - Alert state evaluation
        logging.info("Step 6: Early Warning System...")
        run_warning_system()
        
        # 7. ETAS MLE - Stochastic modeling
        logging.info("Step 7: ETAS MLE...")
        run_mle()
        
        logging.info("[WORKER CYCLE COMPLETE] ✔")
        
    except Exception as e:
        logging.error(f"[WORKER CYCLE FAILED] Error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise


def main():
    """Entry point for worker service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run monitoring cycle")
    parser.add_argument(
        "--days", 
        type=int, 
        default=7, 
        help="Number of days to analyze (default: 7)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml", 
        help="Path to config file (default: config.yaml)"
    )
    
    args = parser.parse_args()
    run_cycle(days_back=args.days, config_path=args.config)


if __name__ == "__main__":
    main()
