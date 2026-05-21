"""
Main Entry Point for Campi Flegrei Monitoring Framework.

This script orchestrates the full pipeline:
1. Data Ingestion (Real data only, no synthetic generation)
2. Statistical Analysis (b-value, rates)
3. Visualization
4. Reporting

Usage:
    python src/main.py [--data-dir <path>] [--output-dir <path>]
"""

import argparse
import os
import sys
from datetime import datetime

# Import pipeline components
from data_ingestion import run_ingestion_pipeline, DataIngestionError
from analysis import run_statistical_analysis
from visualization import generate_all_plots

def main():
    parser = argparse.ArgumentParser(description="Campi Flegrei Seismic Monitoring Pipeline")
    parser.add_argument('--data-dir', type=str, default='data/raw', help='Directory containing raw INGV CSV catalogs')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='Directory for processed outputs')
    parser.add_argument('--figures-dir', type=str, default='figures', help='Directory for generated plots')
    args = parser.parse_args()

    print("=" * 60)
    print("CAMPI FLEGREI QUANTITATIVE MONITORING FRAMEWORK")
    print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    # STEP 1: DATA INGESTION
    print("\n[STEP 1] DATA INGESTION")
    print("-" * 40)
    try:
        catalog_path = os.path.join(args.output_dir, "cleaned_catalog.csv")
        df = run_ingestion_pipeline(data_dir=args.data_dir, output_path=catalog_path)
        print(f"Successfully loaded {len(df)} events.")
    except FileNotFoundError as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("\nACTION REQUIRED:")
        print("1. Go to http://iside.rm.ingv.it/iside/standard/index.jsp")
        print("2. Define your search area (Campi Flegrei: Lat 40.8-40.95, Lon 14.1-14.25)")
        print("3. Download the CSV catalog.")
        print(f"4. Save it in the '{args.data_dir}' folder.")
        sys.exit(1)
    except DataIngestionError as e:
        print(f"\nDATA ERROR: {e}")
        sys.exit(1)

    # STEP 2: STATISTICAL ANALYSIS
    print("\n[STEP 2] STATISTICAL ANALYSIS")
    print("-" * 40)
    try:
        results = run_statistical_analysis(df)
        print("Statistical parameters computed successfully.")
    except Exception as e:
        print(f"Analysis failed: {e}")
        sys.exit(1)

    # STEP 3: VISUALIZATION
    print("\n[STEP 3] GENERATING VISUALIZATIONS")
    print("-" * 40)
    try:
        generate_all_plots(df, results, output_dir=args.figures_dir)
        print(f"Plots saved to '{args.figures_dir}'")
    except Exception as e:
        print(f"Visualization failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Outputs available in: {args.output_dir}")
    print(f"Figures available in: {args.figures_dir}")

if __name__ == "__main__":
    main()
