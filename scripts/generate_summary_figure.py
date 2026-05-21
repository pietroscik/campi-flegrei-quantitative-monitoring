#!/usr/bin/env python3
"""
Generate synthetic summary figure for Campi Flegrei Monitoring System
Creates a 4-panel figure showing key results:
(A) Seismicity map
(B) b-value temporal evolution
(C) Unrest index with thresholds
(D) ETAS model comparison
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
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

# Create figure with 4 panels
fig = plt.figure(figsize=(14, 10))
fig.suptitle('Campi Flegrei Monitoring System - Summary Results', fontsize=14, fontweight='bold')

# Generate synthetic data for demonstration (since real data may not be available)
np.random.seed(42)
n_points = 365  # 1 year of daily data

# Time axis
dates = pd.date_range(start='2023-01-01', periods=n_points, freq='D')

# Synthetic b-value series with realistic variations
b_base = 1.2
b_trend = 0.3 * np.sin(np.linspace(0, 4*np.pi, n_points))
b_noise = np.random.normal(0, 0.15, n_points)
b_values = b_base + b_trend + b_noise

# Anomaly periods (marked)
anomaly_mask = np.abs(b_values - np.mean(b_values)) > 2 * np.std(b_values)

# Unrest index
rate_norm = np.random.normal(0, 1, n_points)
b_norm = (b_values - np.mean(b_values)) / np.std(b_values)
uplift_norm = 0.5 * np.sin(np.linspace(0, 2*np.pi, n_points)) + np.random.normal(0, 0.3, n_points)
unrest_index = 0.4 * rate_norm - 0.3 * b_norm + 0.3 * uplift_norm

# Thresholds
q50 = np.percentile(unrest_index, 50)
q75 = np.percentile(unrest_index, 75)
q90 = np.percentile(unrest_index, 90)

# ETAS synthetic comparison
time_numeric = np.arange(n_points)
observed_rate = np.random.poisson(lam=5, size=n_points) + 2 * np.sin(np.linspace(0, 3*np.pi, n_points))
background_mu = 3.5
triggered = 1.5 * np.exp(-time_numeric/50) + np.random.normal(0, 0.5, n_points)
modeled_rate = background_mu + np.maximum(triggered, 0)

# ===== PANEL A: Seismicity Map (schematic) =====
ax1 = fig.add_subplot(2, 2, 1)

# Create schematic map of Campi Flegrei
lat_range = [40.75, 40.95]
lon_range = [14.05, 14.25]

# Generate synthetic earthquake locations
n_events = 500
synth_lat = np.random.normal(40.85, 0.05, n_events)
synth_lon = np.random.normal(14.15, 0.05, n_events)
synth_mag = np.random.exponential(0.5, n_events) + 0.5

# Clip to bounding box
synth_lat = np.clip(synth_lat, lat_range[0], lat_range[1])
synth_lon = np.clip(synth_lon, lon_range[0], lon_range[1])

# Scatter plot with magnitude-based sizing
scatter = ax1.scatter(synth_lon, synth_lat, c=synth_mag, cmap='Reds', 
                      s=synth_mag*30, alpha=0.6, edgecolors='k', linewidth=0.5)

# Add caldera outline (approximate)
caldera_center = (14.15, 40.85)
caldera = plt.Circle(caldera_center, 0.06, color='blue', fill=False, linewidth=2, linestyle='--')
ax1.add_patch(caldera)

ax1.set_xlabel('Longitude (°E)')
ax1.set_ylabel('Latitude (°N)')
ax1.set_title('(A) Seismicity Distribution (M ≥ 0.5)', fontweight='bold')
ax1.set_xlim(lon_range)
ax1.set_ylim(lat_range)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax1, label='Magnitude')
cbar.ax.tick_params(labelsize=9)

# ===== PANEL B: b-value Temporal Evolution =====
ax2 = fig.add_subplot(2, 2, 2)

ax2.plot(dates, b_values, 'b-', linewidth=1, label='Rolling b-value')
ax2.axhline(y=np.mean(b_values), color='gray', linestyle='--', linewidth=1.5, label=f'Mean: {np.mean(b_values):.2f}')
ax2.axhline(y=np.mean(b_values) - 2*np.std(b_values), color='orange', linestyle=':', linewidth=1.5, label='±2σ')
ax2.axhline(y=np.mean(b_values) + 2*np.std(b_values), color='orange', linestyle=':', linewidth=1.5)

# Highlight anomalies
anomaly_dates = dates[anomaly_mask]
anomaly_b = b_values[anomaly_mask]
ax2.scatter(anomaly_dates, anomaly_b, c='red', s=30, alpha=0.7, label='Anomalies', zorder=5)

ax2.set_xlabel('Date')
ax2.set_ylabel('b-value')
ax2.set_title('(B) Temporal b-value Evolution with Anomalies', fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ===== PANEL C: Unrest Index =====
ax3 = fig.add_subplot(2, 2, 3)

ax3.plot(dates, unrest_index, 'k-', linewidth=1, label='Unrest Index')
ax3.axhline(y=q50, color='green', linestyle='-', linewidth=2, label='NORMAL (≤50th)')
ax3.axhline(y=q75, color='orange', linestyle='--', linewidth=2, label='ELEVATED (75th)')
ax3.axhline(y=q90, color='red', linestyle='-.', linewidth=2, label='CRITICAL (>90th)')

# Fill regions
ax3.fill_between(dates, unrest_index.min(), q50, alpha=0.2, color='green', label='_nolegend_')
ax3.fill_between(dates, q50, q75, alpha=0.2, color='yellow', label='_nolegend_')
ax3.fill_between(dates, q75, q90, alpha=0.2, color='orange', label='_nolegend_')
ax3.fill_between(dates, q90, unrest_index.max(), alpha=0.2, color='red', label='_nolegend_')

ax3.set_xlabel('Date')
ax3.set_ylabel('Normalized Units')
ax3.set_title('(C) Multi-Signal Unrest Index', fontweight='bold')
ax3.legend(loc='upper right', fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(unrest_index.min() - 0.5, unrest_index.max() + 0.5)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ===== PANEL D: ETAS Model Comparison =====
ax4 = fig.add_subplot(2, 2, 4)

x = np.arange(n_points)
ax4.bar(x, observed_rate, width=1, alpha=0.5, color='steelblue', label='Observed Rate')
ax4.plot(x, modeled_rate, 'r-', linewidth=2.5, label='ETAS Modeled Rate')
ax4.plot(x, np.ones(n_points) * background_mu, 'g--', linewidth=2, label=f'Background μ={background_mu:.2f}')

ax4.set_xlabel('Time (days)')
ax4.set_ylabel('Event Rate (events/day)')
ax4.set_title('(D) ETAS Model Fit: Observed vs Modeled', fontweight='bold')
ax4.legend(loc='upper right', fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, n_points)

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
Figure 1: Summary of Campi Flegrei Monitoring System Results

(A) Seismicity Distribution: Spatial distribution of earthquakes (M ≥ 0.5) within the 
    Campi Flegrei caldera region. Circle sizes are proportional to magnitude. 
    The dashed blue circle indicates the approximate caldera boundary.

(B) Temporal b-value Evolution: Rolling b-value estimates (window = 100 events) over 
    the analysis period. Gray dashed line shows the mean b-value. Orange dotted lines 
    indicate ±2 standard deviations. Red dots highlight statistically significant 
    anomalies detected by Z-score and quantile methods.

(C) Multi-Signal Unrest Index: Composite index integrating seismicity rate, b-value, 
    and ground uplift. Green region: NORMAL state (≤50th percentile). Yellow region: 
    ELEVATED state (50th-75th percentile). Orange region: HIGH state (75th-90th 
    percentile). Red region: CRITICAL state (>90th percentile). Alert flags are 
    triggered when CRITICAL state persists >60% over 7-day window.

(D) ETAS Model Fit: Comparison of observed daily seismicity rate (blue bars) with 
    ETAS model predictions (red line). Green dashed line shows the background 
    seismicity rate (μ parameter). The ETAS model captures both background activity 
    and triggered sequences following larger events.

Parameters shown are from synthetic data for demonstration purposes. 
Run the full pipeline with real INGV data for actual monitoring results.
"""

with open('results/figure_caption.txt', 'w') as f:
    f.write(caption_text)

print("[OK] Figure caption saved: results/figure_caption.txt")
print("\n=== SUMMARY ===")
print("Generated files:")
print("  - results/summary_figure.png (PNG format, 300 DPI)")
print("  - results/summary_figure.pdf (PDF format, publication quality)")
print("  - results/figure_caption.txt (detailed caption)")
