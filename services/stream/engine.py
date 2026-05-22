"""
Stream Service for Campi Flegrei Monitoring System

This module provides real-time streaming of monitoring state,
suitable for dashboard integration or live monitoring displays.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Definizioni colori ANSI per una CLI Dashboard moderna
class C:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARN = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    MAGENTA = '\033[95m'

def get_sparkline(series, length=21):
    """Genera un mini grafico testuale (sparkline) per mostrare il trend temporale."""
    if series is None or len(series) == 0:
        return "N/A"
    series = series.dropna().tail(length).values
    if len(series) == 0:
        return "N/A"
    bars = ' ▂▃▄▅▆▇█'
    min_val, max_val = float(min(series)), float(max(series))
    if max_val == min_val:
        return bars[3] * len(series)
    scaled = [int((x - min_val) / (max_val - min_val) * 7) for x in series]
    return ''.join(bars[i] for i in scaled)


def get_latest_state(data_path="data/processed/early_warning_system.csv"):
    """
    Retrieve the latest monitoring state from processed data.
    
    Parameters
    ----------
    data_path : str
        Path to the early warning system CSV file
    
    Returns
    -------
    dict
        Latest state information including alert level, unrest index, and timestamp
    """
    if not os.path.exists(data_path):
        logging.warning(f"Data file not found: {data_path}")
        return None
    
    try:
        df = pd.read_csv(data_path)
        if len(df) == 0:
            return None
        
        latest = df.iloc[-1].to_dict()
        
        # Parse timestamp if present
        if 'timestamp' in latest or 'time' in latest:
            time_col = 'timestamp' if 'timestamp' in latest else 'time'
            try:
                latest['parsed_time'] = pd.to_datetime(latest[time_col]).isoformat()
            except:
                latest['parsed_time'] = str(latest[time_col])
        
        return latest
    
    except Exception as e:
        logging.error(f"Error reading state: {e}")
        return None


def get_bvalue_state(data_path="data/processed/b_value_rolling.csv"):
    """
    Retrieve the latest b-value state.
    
    Parameters
    ----------
    data_path : str
        Path to the b-value rolling CSV file
    
    Returns
    -------
    dict
        Latest b-value information
    """
    if not os.path.exists(data_path):
        return None
    
    try:
        df = pd.read_csv(data_path)
        if len(df) == 0:
            return None
        
        latest = df.iloc[-1].to_dict()
        return latest
    
    except Exception as e:
        logging.error(f"Error reading b-value state: {e}")
        return None


def get_unrest_state(data_path="data/processed/unrest_index.csv"):
    """
    Retrieve the latest unrest index state.
    
    Parameters
    ----------
    data_path : str
        Path to the unrest index CSV file
    
    Returns
    -------
    dict
        Latest unrest index information
    """
    if not os.path.exists(data_path):
        return None
    
    try:
        df = pd.read_csv(data_path)
        if len(df) == 0:
            return None
        
        latest = df.iloc[-1].to_dict()
        return latest
    
    except Exception as e:
        logging.error(f"Error reading unrest state: {e}")
        return None


def stream_state(interval=10, output_format="console"):
    """
    Continuously stream monitoring state at specified interval.
    
    Parameters
    ----------
    interval : int
        Time between updates in seconds (default: 10)
    output_format : str
        Output format: "console", "json", or "api"
    """
    logging.info(f"[STREAM SERVICE START] - Update interval: {interval}s")
    
    iteration = 0
    while True:
        iteration += 1
        timestamp = datetime.now().isoformat()
        
        # Gather all state components
        ew_state = get_latest_state()
        bvalue_state = get_bvalue_state()
        unrest_state = get_unrest_state()
        
        # Compose comprehensive state
        state = {
            "iteration": iteration,
            "timestamp": timestamp,
            "early_warning": ew_state,
            "b_value": bvalue_state,
            "unrest_index": unrest_state,
            "status": "active" if ew_state else "no_data"
        }
        
        # Output based on format
        if output_format == "json":
            print(json.dumps(state, indent=2, default=str))
        elif output_format == "api":
            # For API integration, could push to endpoint
            print(json.dumps(state, default=str))
        else:  # console
            # Caricamento dinamico dei dataframe per estrarre trend e risultati completi
            try:
                df_ew = pd.read_csv("data/processed/early_warning_system.csv") if os.path.exists("data/processed/early_warning_system.csv") else None
                df_b = pd.read_csv("data/processed/b_value_rolling_events.csv") if os.path.exists("data/processed/b_value_rolling_events.csv") else None
                df_dl = pd.read_csv("data/processed/dl_forecast.csv") if os.path.exists("data/processed/dl_forecast.csv") else None
                df_etas = pd.read_csv("data/processed/etas_params.csv") if os.path.exists("data/processed/etas_params.csv") else None
            except Exception:
                df_ew, df_b, df_dl, df_etas = None, None, None, None

            # Pulisce lo schermo ad ogni ciclo per un effetto "cruscotto di controllo" (Dashboard)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"{C.BOLD}{C.BLUE}{'='*65}{C.END}")
            print(f"{C.BOLD}🌋 CAMPI FLEGREI LIVE STREAM | Update #{iteration} - {timestamp[:19]}{C.END}")
            print(f"{C.BOLD}{C.BLUE}{'='*65}{C.END}")
            
            if df_ew is not None and not df_ew.empty:
                last_row = df_ew.iloc[-1]
                state_str = last_row.get('state', 'N/A')
                
                if state_str == "NORMAL": state_color = f"{C.GREEN}🟢 {state_str}"
                elif state_str == "ELEVATED": state_color = f"{C.CYAN}🟡 {state_str}"
                elif state_str == "HIGH": state_color = f"{C.WARN}🟠 {state_str}"
                elif state_str == "CRITICAL": state_color = f"{C.FAIL}🔴 {state_str}"
                else: state_color = state_str
                
                print(f"\n{C.BOLD}[ SYSTEM STATUS ]{C.END}")
                print(f"Alert Level    : {state_color}{C.END}")
                stat_alert = int(last_row.get('stat_alert_flag', 0))
                dl_alert = int(last_row.get('dl_alert_flag', 0))
                combo_alert = int(last_row.get('alert_flag', 0))
                print(f"Alert Flags    : Stat: {stat_alert} | DL: {dl_alert} | Combined: {combo_alert}")
                
                print(f"\n{C.BOLD}[ METRICS & TRENDS (Last 21 windows) ]{C.END}")
                if 'unrest_index' in df_ew.columns:
                    ui_val = last_row.get('unrest_index', 0.0)
                    ui_spark = get_sparkline(df_ew['unrest_index'], 21)
                    print(f"Unrest Index   : {ui_val:6.2f}  [{C.CYAN}{ui_spark}{C.END}]")
                
                if df_b is not None and not df_b.empty and 'b_value' in df_b.columns:
                    b_val = df_b.iloc[-1].get('b_value', 0.0)
                    b_spark = get_sparkline(df_b['b_value'], 21)
                    print(f"B-value        : {b_val:6.3f}  [{C.WARN}{b_spark}{C.END}]")
                
                if 'seismic_rate' in df_ew.columns:
                    rate_val = last_row.get('seismic_rate', 0.0)
                    rate_spark = get_sparkline(df_ew['seismic_rate'], 21)
                    print(f"Seismic Rate   : {rate_val:6.2f}  [{C.HEADER}{rate_spark}{C.END}]")
            else:
                print(f"{C.FAIL}No Early Warning System data available.{C.END}")
                
            print(f"\n{C.BOLD}[ DEEP LEARNING FORECAST (Next 7 days) ]{C.END}")
            if df_dl is not None and not df_dl.empty and 'forecasted_rate' in df_dl.columns:
                avg_rate = df_dl['forecasted_rate'].mean()
                dl_spark = get_sparkline(df_dl['forecasted_rate'], 7)
                print(f"Avg Daily Rate : {avg_rate:6.2f} events/day")
                print(f"Forecast Trend :        [{C.MAGENTA}{dl_spark}{C.END}]")
            else:
                print("Forecast data not available.")
                
            print(f"\n{C.BOLD}[ ETAS MODEL PARAMETERS ]{C.END}")
            if df_etas is not None and not df_etas.empty:
                p = df_etas.iloc[-1]
                print(f"μ: {p.get('mu',0):.4f} | K: {p.get('K',0):.4f} | α: {p.get('alpha',0):.4f} | c: {p.get('c',0):.4f} | p: {p.get('p',0):.4f}")
            else:
                print("ETAS parameters not available.")
                
            print(f"{C.BOLD}{C.BLUE}{'='*65}{C.END}\n")
        
        time.sleep(interval)


def main():
    """Entry point for stream service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stream monitoring state")
    parser.add_argument(
        "--interval", 
        type=int, 
        default=10, 
        help="Update interval in seconds (default: 10)"
    )
    parser.add_argument(
        "--format", 
        type=str, 
        choices=["console", "json", "api"], 
        default="console", 
        help="Output format (default: console)"
    )
    
    args = parser.parse_args()
    stream_state(interval=args.interval, output_format=args.format)


if __name__ == "__main__":
    main()
