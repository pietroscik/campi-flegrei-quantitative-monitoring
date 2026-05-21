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
            print(f"\n{'='*60}")
            print(f"[LIVE STATE] Update #{iteration} - {timestamp}")
            print(f"{'='*60}")
            
            if ew_state:
                alert_level = ew_state.get('alert_level', ew_state.get('state', 'N/A'))
                print(f"Alert Level: {alert_level}")
                
                if 'unrest_index' in ew_state:
                    print(f"Unrest Index: {ew_state['unrest_index']:.3f}")
                
                if 'persistence' in ew_state:
                    print(f"Persistence: {ew_state['persistence']:.1%}")
            
            if bvalue_state:
                if 'b_value' in bvalue_state:
                    print(f"B-value: {bvalue_state['b_value']:.3f}")
            
            if unrest_state:
                if 'composite_index' in unrest_state:
                    print(f"Composite Index: {unrest_state['composite_index']:.3f}")
            
            print(f"{'='*60}\n")
        
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
