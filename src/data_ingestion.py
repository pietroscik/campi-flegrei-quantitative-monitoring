"""
Data Ingestion Module for Campi Flegrei Monitoring Framework.

This module handles the loading and preprocessing of seismic catalog data.
It supports:
1. Direct fetching from INGV FDSN webservices (preferred method).
2. Loading local CSV files (standard INGV ISIDE format) as fallback.

NO synthetic data is generated. If no valid data is found, the pipeline stops.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
import glob
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration for Campi Flegrei Area
CF_BOUNDS = {
    "min_lat": 40.80,
    "max_lat": 40.95,
    "min_lon": 14.10,
    "max_lon": 14.25
}

class DataIngestionError(Exception):
    """Custom exception for data ingestion failures."""
    pass

def fetch_from_ingv_api(days_back: int = 365, min_magnitude: float = 0.0) -> Optional[pd.DataFrame]:
    """
    Fetch seismic events directly from INGV FDSN webservice.
    
    Parameters:
        days_back (int): Number of days to look back from today.
        min_magnitude (float): Minimum magnitude threshold.
        
    Returns:
        pd.DataFrame or None: DataFrame with events, or None if fetch fails.
    """
    try:
        from ingestion.fetch_ingv import fetch_ingv_events
        print(f"Attempting to fetch data from INGV FDSN webservice for the last {days_back} days...")
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days_back)
        
        chunk_days = 365
        current_start = start_time
        dfs = []
        
        while current_start < end_time:
            current_end = min(current_start + timedelta(days=chunk_days), end_time)
            print(f"  Fetching chunk: {current_start.date()} to {current_end.date()}...")
            chunk_df = fetch_ingv_events(
                starttime=current_start,
                endtime=current_end,
                minmag=min_magnitude,
                maxlat=CF_BOUNDS["max_lat"] + 0.1,
                minlat=CF_BOUNDS["min_lat"] - 0.1,
                maxlon=CF_BOUNDS["max_lon"] + 0.1,
                minlon=CF_BOUNDS["min_lon"] - 0.1
            )
            if chunk_df is not None and not chunk_df.empty:
                dfs.append(chunk_df)
            current_start = current_end
            
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            print(f"Successfully fetched {len(df)} events from INGV API.")
            # Standardize column names
            df.columns = df.columns.str.lower().str.strip()
            if 'magnitude' in df.columns:
                df.rename(columns={'magnitude': 'mag', 'latitude': 'lat', 'longitude': 'lon'}, inplace=True)
            return df
        else:
            print("No events returned from INGV API.")
            return None
            
    except ImportError:
        print("INGV fetch module not available. Falling back to local file.")
        return None
    except Exception as e:
        print(f"Warning: Failed to fetch from INGV API: {e}")
        print("Falling back to local file ingestion.")
        return None

def load_local_catalog(file_path: str) -> pd.DataFrame:
    """
    Load a seismic catalog from a local CSV file.
    Expects standard INGV ISIDE format or similar columns:
    ['time', 'lat', 'lon', 'depth', 'mag'] (case insensitive)
    
    Parameters:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with standardized column names.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Catalog file not found: {file_path}")
    
    print(f"Loading catalog from: {file_path}")
    
    # Try to read with standard separators
    try:
        # INGV CSVs often use commas, sometimes semicolons. 
        # We try comma first, then detect.
        df = pd.read_csv(file_path)
        if df.shape[1] == 1: # If only 1 column, likely wrong separator
            df = pd.read_csv(file_path, sep=';')
    except Exception as e:
        raise DataIngestionError(f"Failed to parse CSV {file_path}: {e}")
    
    # Standardize column names to lowercase
    df.columns = df.columns.str.lower().str.strip()
    
    # Map common INGV column names to standard names
    column_mapping = {
        'date': 'time',
        'latitude': 'lat',
        'longitude': 'lon',
        'magnitude': 'mag',
        'depth_km': 'depth',
        'depth': 'depth'
    }
    
    df.rename(columns=column_mapping, inplace=True)
    
    # Verify essential columns exist
    required_cols = ['time', 'lat', 'lon', 'mag']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise DataIngestionError(f"Missing required columns: {missing_cols}. Found: {list(df.columns)}")
    
    # Parse time
    # INGV format is typically 'YYYY-MM-DD HH:MM:SS.sss' or similar
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    if df['time'].isna().all():
        raise DataIngestionError("Failed to parse time column. Check date format.")
    
    # Drop rows with NaN in critical fields
    initial_count = len(df)
    df.dropna(subset=['time', 'lat', 'lon', 'mag'], inplace=True)
    dropped_count = initial_count - len(df)
    
    if dropped_count > 0:
        print(f"Warning: Dropped {dropped_count} rows due to missing values.")
        
    return df

def filter_campi_flegrei(df: pd.DataFrame, bounds: Optional[dict] = None) -> pd.DataFrame:
    """
    Filter the catalog to include only events within the Campi Flegrei caldera area.
    
    Parameters:
        df (pd.DataFrame): Input catalog.
        bounds (dict): Dictionary with min_lat, max_lat, min_lon, max_lon.
        
    Returns:
        pd.DataFrame: Filtered catalog.
    """
    if bounds is None:
        bounds = CF_BOUNDS
        
    print(f"Filtering for Campi Flegrei area: Lat[{bounds['min_lat']}-{bounds['max_lat']}], Lon[{bounds['min_lon']}-{bounds['max_lon']}]")
    
    mask = (
        (df['lat'] >= bounds['min_lat']) & 
        (df['lat'] <= bounds['max_lat']) &
        (df['lon'] >= bounds['min_lon']) & 
        (df['lon'] <= bounds['max_lon'])
    )
    
    filtered_df = df[mask].copy()
    
    print(f"Filtered catalog: {len(filtered_df)} events (from {len(df)} total).")
    
    if len(filtered_df) == 0:
        raise DataIngestionError("No events found within the specified Campi Flegrei bounds. Check coordinates or input data.")
        
    return filtered_df

def get_data_source(data_dir: str = "data/raw") -> str:
    """
    Locate the primary seismic catalog file in the data directory.
    Looks for files named like 'catalog.csv', 'iside_*.csv', or any .csv in raw/.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        raise FileNotFoundError(f"Data directory '{data_dir}' created but is empty. Please place your INGV CSV catalog here.")
    
    # Priority list of filenames
    priority_names = ['catalog.csv', 'iside.csv', 'seismic_catalog.csv']
    
    for name in priority_names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
            
    # Fallback: find any CSV
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if csv_files:
        if len(csv_files) > 1:
            print(f"Warning: Multiple CSV files found. Using the first one: {csv_files[0]}")
        return csv_files[0]
    
    raise FileNotFoundError(
        f"No CSV catalog found in '{data_dir}'.\n"
        "Please download the seismic catalog from INGV ISIDE (http://iside.rm.ingv.it) "
        "and save it as 'data/raw/catalog.csv'."
    )

def run_ingestion_pipeline(data_dir: str = "data/raw", output_path: str = "data/processed/cleaned_catalog.csv", use_api: bool = True) -> pd.DataFrame:
    """
    Main entry point for the ingestion pipeline.
    
    Strategy:
    1. Attempt to fetch data from INGV API (if use_api=True).
    2. If API fails or returns no data, fall back to local CSV files.
    3. Load and clean data.
    4. Filter for Campi Flegrei.
    5. Save processed catalog.
    
    Returns:
        pd.DataFrame: The final processed catalog.
    """
    print("--- Starting Data Ingestion Pipeline ---")
    
    df = None
    
    # STEP 1: Try API Fetch (Optional)
    if use_api:
        print("\n[Mode] Attempting direct INGV API fetch...")
        try:
            df = fetch_from_ingv_api(days_back=365, min_magnitude=0.0)
        except Exception as e:
            print(f"API fetch failed: {e}")
            df = None
    
    # STEP 2: Fallback to Local File if API failed or returned nothing
    if df is None or len(df) == 0:
        print("\n[Mode] Falling back to local file ingestion...")
        try:
            source_file = get_data_source(data_dir)
            df = load_local_catalog(source_file)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            raise
        except DataIngestionError as e:
            print(f"DATA ERROR: {e}")
            raise
    else:
        print("Using data fetched from INGV API.")
    
    # STEP 3: Filter for Campi Flegrei
    cf_df = filter_campi_flegrei(df)
    
    # STEP 4: Sort by time
    cf_df.sort_values('time', inplace=True)
    cf_df.reset_index(drop=True, inplace=True)
    
    # STEP 5: Save Processed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cf_df.to_csv(output_path, index=False)
    print(f"Processed catalog saved to: {output_path}")
    print(f"Date range: {cf_df['time'].min()} to {cf_df['time'].max()}")
    print(f"Magnitude range: {cf_df['mag'].min():.1f} to {cf_df['mag'].max():.1f}")
    print("--- Ingestion Complete ---")
    
    return cf_df

if __name__ == "__main__":
    try:
        df = run_ingestion_pipeline()
        print(df.head())
    except Exception as e:
        print(f"Pipeline failed: {e}")
        exit(1)
