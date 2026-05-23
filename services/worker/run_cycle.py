"""
Worker Cycle Service for Campi Flegrei Monitoring System

This module executes a complete analysis cycle on recent seismic data,
suitable for periodic execution (e.g., cron job, scheduled task).
Includes integrated alerting system based on analysis results.
"""

import os
import sys
import logging
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ingestion.fetch_ingv import fetch_ingv_events, save_raw
from src.preprocessing.clean_catalog import build_catalog
from src.analysis.b_value import run_b_analysis
from src.analysis.anomaly_bvalue import run_anomaly
from src.analysis.multi_signal_model import run_multisignal
from src.analysis.early_warning import run_warning_system
from src.analysis.etas_mle import run_mle
from services.alerts.engine import AlertEngine

# Configuration
BASE_DIR = Path(__file__).parent.parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"
STATE_FILE = BASE_DIR / "data" / "processed" / ".last_alert_state.json"

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_cycle(
    days_back: int = 7,
    config_path: str = CONFIG_PATH,
    enable_alerts: bool = True,
    send_notifications: bool = True
) -> Dict[str, Any]:
    """
    Execute a complete monitoring cycle on recent data.
    
    Parameters
    ----------
    days_back : int
        Number of days to look back for seismic events
    config_path : str
        Path to configuration YAML file
    enable_alerts : bool
        Enable alert evaluation after analysis
    send_notifications : bool
        Send notifications if alerts are triggered
    
    Returns
    -------
    dict
        Cycle execution results including alert status
    """
    logger.info(f"[WORKER CYCLE START] - Analyzing last {days_back} days")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "days_analyzed": days_back,
        "events_found": 0,
        "steps_completed": [],
        "alerts_triggered": False,
        "alert_details": None,
        "notifications_sent": False,
        "errors": []
    }
    
    try:
        # Ensure output directories exist
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        
        # Load configuration
        config = load_config(config_path)
        
        # 1. INGESTION - Fetch recent events from INGV
        logger.info("Step 1: Ingestion...")
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
            logger.warning("No events found in the specified time window")
            results["errors"].append("No seismic events found in the time window")
            return results
        
        save_raw(df)
        results["events_found"] = len(df)
        results["steps_completed"].append("ingestion")
        logger.info(f"  → Fetched {len(df)} events")
        
        # 2. CLEANING - Quality control and preprocessing
        logger.info("Step 2: Cleaning...")
        clean_df = build_catalog("data/raw/ingv_events.csv")
        results["steps_completed"].append("cleaning")
        logger.info(f"  → Clean catalog: {len(clean_df)} events")
        
        # 3. B-VALUE - Gutenberg-Richter parameter estimation
        logger.info("Step 3: B-Value Analysis...")
        run_b_analysis()
        results["steps_completed"].append("b_value")
        
        # 4. ANOMALY - Statistical anomaly detection
        logger.info("Step 4: Anomaly Detection...")
        run_anomaly()
        results["steps_completed"].append("anomaly_detection")
        
        # 5. MULTI-SIGNAL - Composite unrest index
        logger.info("Step 5: Multi-Signal Model...")
        run_multisignal()
        results["steps_completed"].append("multi_signal")
        
        # 6. EARLY WARNING - Alert state evaluation
        logger.info("Step 6: Early Warning System...")
        warning_df = run_warning_system()
        results["steps_completed"].append("early_warning")
        
        # 7. ETAS MLE - Stochastic modeling
        logger.info("Step 7: ETAS MLE...")
        run_mle()
        results["steps_completed"].append("etas_mle")
        
        # 8. ALERT EVALUATION - Check if notifications should be sent
        if enable_alerts:
            logger.info("Step 8: Alert Evaluation...")
            alert_engine = AlertEngine(state_file=STATE_FILE)
            alert_result = alert_engine.check_and_notify(
                warning_df=warning_df,
                send_notification=send_notifications,
                config=config
            )
            
            results["alerts_triggered"] = alert_result["alert_triggered"]
            results["alert_details"] = alert_result["alert_details"]
            results["notifications_sent"] = alert_result["notification_sent"]
            
            if alert_result["alert_triggered"]:
                logger.warning(f"🚨 ALERT TRIGGERED: {alert_result['alert_details']}")
                if alert_result["notification_sent"]:
                    logger.info("✓ Notification sent successfully")
                else:
                    logger.warning("⚠ Alert triggered but notification failed")
            else:
                logger.info("✓ No alerts triggered - system within normal parameters")
        
        results["steps_completed"].append("alert_evaluation")
        logger.info("[WORKER CYCLE COMPLETE] ✔")
        
    except Exception as e:
        error_msg = f"[WORKER CYCLE FAILED] Error: {e}"
        logger.error(error_msg)
        results["errors"].append(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    return results


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
        default=str(CONFIG_PATH), 
        help="Path to config file"
    )
    parser.add_argument(
        "--no-alerts",
        action="store_true",
        help="Disable alert evaluation"
    )
    parser.add_argument(
        "--no-notifications",
        action="store_true",
        help="Disable sending notifications (alerts still evaluated)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without executing analysis (test mode)"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("[DRY RUN MODE] - No analysis will be executed")
        logger.info("Configuration loaded successfully")
        return
    
    run_cycle(
        days_back=args.days,
        config_path=args.config,
        enable_alerts=not args.no_alerts,
        send_notifications=not args.no_notifications
    )


if __name__ == "__main__":
    main()
