#!/usr/bin/env python3
"""
Generate synthetic summary figure for Campi Flegrei Monitoring System
Creates a 4-panel figure showing key results:
(A) SARIMA Modeling & Forecast
(B) Benioff Strain & Changepoints
(C) Coulomb Stress Drop (CSD)
(D) Critical Seismicity Index (CSI)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os

# Create results directory if not exists
os.makedirs('results', exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13

# Create figure with 4 panels
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Campi Flegrei - Advanced Structural & Forecasting Models', fontsize=16, fontweight='bold')

# Load Real Modeling Data
def load_if_exists(filepath, date_col='time'):
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])
            return df
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
    return None

print("Loading Advanced Modeling outputs...")
sarima_df = load_if_exists('data/processed/sarima_output.csv', 'date')
benioff_df = load_if_exists('data/processed/benioff_output.csv')
cp_df = load_if_exists('data/processed/changepoint_output.csv')
csd_df = load_if_exists('data/processed/csd_output.csv')
csi_df = load_if_exists('data/processed/csi_output.csv')

# ===== PANEL A: SARIMA Forecasting =====
ax1 = fig.add_subplot(2, 2, 1)
if sarima_df is not None:
    ax1.plot(sarima_df['date'], sarima_df['observed'], label='Observed Rate', alpha=0.6, color='steelblue')
    ax1.plot(sarima_df['date'], sarima_df['fitted'], color='red', label='SARIMA Fit', linewidth=1.5)
    
    if 'forecast' in sarima_df.columns:
        mask = sarima_df['forecast'].notna()
        if mask.any():
            ax1.plot(sarima_df.loc[mask, 'date'], sarima_df.loc[mask, 'forecast'], color='magenta', linestyle='--', label='Forecast')
            ax1.fill_between(sarima_df.loc[mask, 'date'], sarima_df.loc[mask, 'lower'], sarima_df.loc[mask, 'upper'], color='magenta', alpha=0.2)
            
ax1.set_title('(A) SARIMA Seismic Rate Modeling & Forecast', fontweight='bold')
ax1.set_ylabel('Events/Day')
ax1.legend(loc='upper left')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ===== PANEL B: Benioff Strain & Changepoints =====
ax2 = fig.add_subplot(2, 2, 2)
if benioff_df is not None:
    ax2.plot(benioff_df['time'], benioff_df['strain_rate'], color='purple', label='Benioff Strain Rate')

if cp_df is not None and 'changepoint' in cp_df.columns:
    cp_dates = cp_df[cp_df['changepoint'] == 1]['time']
    for cp in cp_dates:
        ax2.axvline(cp, color='orange', linestyle='--', alpha=0.7)
    if not cp_dates.empty:
        ax2.plot([], [], color='orange', linestyle='--', label='Detected Changepoints')

ax2.set_title('(B) Benioff Strain Release & Structural Changepoints', fontweight='bold')
ax2.set_ylabel('Strain Rate')
ax2.legend(loc='upper left')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ===== PANEL C: Coulomb Stress Drop (CSD) =====
ax3 = fig.add_subplot(2, 2, 3)
if csd_df is not None and 'csd_rolling' in csd_df.columns:
    ax3.scatter(csd_df['time'], csd_df['csd'], color='lightgray', s=10, alpha=0.5, label='Event CSD')
    ax3.plot(csd_df['time'], csd_df['csd_rolling'], color='darkblue', linewidth=2, label='Rolling CSD (50 events)')

ax3.set_title('(C) Coulomb Stress Drop (CSD)', fontweight='bold')
ax3.set_ylabel('Stress Drop (MPa)')
ax3.legend(loc='upper left')
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ===== PANEL D: Critical Seismicity Index (CSI) =====
ax4 = fig.add_subplot(2, 2, 4)
if csi_df is not None and 'csi_normalized' in csi_df.columns:
    ax4.plot(csi_df['time'], csi_df['csi_normalized'], color='teal', label='CSI')
    ax4.axhline(0.8, color='red', linestyle='--', label='Critical Threshold')
    
    crit = csi_df[csi_df['critical'] == 1]
    if not crit.empty:
        ax4.scatter(crit['time'], crit['csi_normalized'], color='red', s=20, zorder=5)

ax4.set_title('(D) Critical Seismicity Index (CSI)', fontweight='bold')
ax4.set_ylabel('Normalized Index')
ax4.legend(loc='upper left')
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save figure
output_path = 'results/summary_figure.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"[OK] Summary figure saved: {output_path}")

# Also save as PDF for publication quality
output_pdf = 'results/summary_figure.pdf'
plt.savefig(output_pdf, bbox_inches='tight')
print(f"[OK] PDF version saved: {output_pdf}")

plt.close()

# Generate caption file
caption_text = """
Figure 2: Advanced Modeling Results for Campi Flegrei

(A) SARIMA Forecasting: Observed daily seismic rate modeled with a Seasonal ARIMA process. 
    The magenta line represents the short-term forecast with 95% confidence intervals.
    
(B) Benioff Strain & Changepoints: Rolling Benioff strain release rate over time. 
    Vertical orange dashed lines indicate structural changepoints detected using 
    CUSUM and variance-penalty algorithms.
    
(C) Coulomb Stress Drop (CSD): Estimated stress drop per event (gray dots) and rolling 
    average (dark blue). Variations indicate changes in fault rupture mechanics.
    
(D) Critical Seismicity Index (CSI): Normalized metric combining moment release, event rate, 
    and magnitude variance. Red markers indicate periods exceeding the critical threshold (0.8), 
    often associated with approaching tipping points.
"""

with open('results/figure_caption.txt', 'w', encoding='utf-8') as f:
    f.write(caption_text)

print("[OK] Figure caption saved: results/figure_caption.txt")
print("\n=== SUMMARY ===")
print("Generated files:")
print("  - results/summary_figure.png (PNG format, 300 DPI)")
print("  - results/summary_figure.pdf (PDF format, publication quality)")
print("  - results/figure_caption.txt (detailed caption)")
