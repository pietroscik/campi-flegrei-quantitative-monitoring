"""
Generate Paper-Ready Figures for Campi Flegrei Monitoring System

This script produces the minimum set of figures required for a scientific publication:
1. Seismicity rate over time
2. Gutenberg-Richter fit with cumulative distribution
3. b-value temporal evolution (rolling window)
4. Magnitude distribution stability (histograms by time windows)
5. Multi-signal unrest index with alert thresholds
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Create output directory
os.makedirs("figures", exist_ok=True)

# Set scientific style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Load data
print("Loading data...")
catalog = pd.read_csv("data/processed/catalog_clean.csv")
b_value_df = pd.read_csv("data/processed/b_value_rolling.csv")
unrest_df = pd.read_csv("data/processed/unrest_index.csv")
anomalies_df = pd.read_csv("data/processed/b_value_anomalies.csv")
ew_df = pd.read_csv("data/processed/early_warning_system.csv")

# Convert timestamps
catalog['time'] = pd.to_datetime(catalog['time'])
b_value_df['time'] = pd.to_datetime(b_value_df['time'])
unrest_df['time'] = pd.to_datetime(unrest_df['time'])
anomalies_df['time'] = pd.to_datetime(anomalies_df['time'])
ew_df['time'] = pd.to_datetime(ew_df['time'])

print(f"Catalog: {len(catalog)} events")
print(f"B-value samples: {len(b_value_df)}")
print(f"Unrest index days: {len(unrest_df)}")

# ============================================================
# FIGURE 1: Seismicity Rate Time Series
# ============================================================
print("\nGenerating Figure 1: Seismicity Rate...")

fig1, ax1 = plt.subplots(figsize=(12, 5))

# Compute daily seismic rate
catalog_indexed = catalog.set_index('time')
daily_rate = catalog_indexed.resample('D').size()

ax1.fill_between(daily_rate.index, daily_rate.values, alpha=0.3, color='steelblue', label='Daily count')
ax1.plot(daily_rate.index, daily_rate.rolling(window=7, min_periods=1).mean(), 
         color='darkblue', linewidth=2, label='7-day moving average')

ax1.set_xlabel('Date')
ax1.set_ylabel('Events per day')
ax1.set_title('Campi Flegrei - Seismicity Rate Evolution')
ax1.legend(loc='upper right')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/01_seismicity_rate.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: figures/01_seismicity_rate.png")

# ============================================================
# FIGURE 2: Gutenberg-Richter Fit
# ============================================================
print("Generating Figure 2: Gutenberg-Richter Fit...")

fig2, ax2 = plt.subplots(figsize=(10, 6))

# Compute cumulative magnitude distribution
magnitudes = catalog['magnitude'].values
m_min = 1.0  # completeness magnitude
mags_filtered = magnitudes[magnitudes >= m_min]

# Bin edges
bins = np.arange(m_min, mags_filtered.max() + 0.1, 0.1)
cum_counts = []
bin_centers = []

for i in range(len(bins)-1):
    count = np.sum(mags_filtered >= bins[i])
    cum_counts.append(count)
    bin_centers.append((bins[i] + bins[i+1]) / 2)

# Fit Gutenberg-Richter
from scipy.stats import linregress
log_counts = np.log10(cum_counts)
slope, intercept, r_value, p_value, std_err = linregress(bin_centers, log_counts)
b_value_fit = -slope

ax2.scatter(bin_centers, log_counts, s=30, alpha=0.7, color='navy', label='Observed')
ax2.plot(bin_centers, slope * np.array(bin_centers) + intercept, 
         'r-', linewidth=2, label=f'GR Fit (b={b_value_fit:.2f}, $R^2$={r_value**2:.3f})')

ax2.set_xlabel('Magnitude M')
ax2.set_ylabel('log₁₀ N(M ≥ m)')
ax2.set_title('Campi Flegrei - Gutenberg-Richter Frequency-Magnitude Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/02_gutenberg_richter_fit.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: figures/02_gutenberg_richter_fit.png")

# ============================================================
# FIGURE 3: B-value Temporal Evolution
# ============================================================
print("Generating Figure 3: B-value Temporal Evolution...")

fig3, ax3 = plt.subplots(figsize=(12, 5))

ax3.plot(b_value_df['time'], b_value_df['b_value'], 
         color='darkgreen', linewidth=1.5, alpha=0.8, label='Rolling b-value (N=100)')

# Highlight anomalies if present
if 'anomaly_score' in anomalies_df.columns:
    anomaly_mask = anomalies_df['anomaly_score'] > 0
    if anomaly_mask.any():
        ax3.scatter(anomalies_df.loc[anomaly_mask, 'time'], 
                   anomalies_df.loc[anomaly_mask, 'b_value'],
                   c='red', s=40, alpha=0.7, label='Anomalies', zorder=5)

# Add mean and std bands
mean_b = b_value_df['b_value'].mean()
std_b = b_value_df['b_value'].std()
ax3.axhline(mean_b, color='gray', linestyle='--', linewidth=1.5, label=f'Mean (b={mean_b:.3f})')
ax3.axhline(mean_b - std_b, color='orange', linestyle=':', linewidth=1, alpha=0.7, label='±1σ')
ax3.axhline(mean_b + std_b, color='orange', linestyle=':', linewidth=1, alpha=0.7)

ax3.set_xlabel('Date')
ax3.set_ylabel('b-value')
ax3.set_title('Campi Flegrei - Temporal Evolution of Gutenberg-Richter b-value')
ax3.legend(loc='upper right')
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/03_bvalue_evolution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: figures/03_bvalue_evolution.png")

# ============================================================
# FIGURE 4: Magnitude Distribution Stability
# ============================================================
print("Generating Figure 4: Magnitude Distribution Stability...")

fig4, axes4 = plt.subplots(2, 2, figsize=(14, 8))

# Split catalog into 4 time windows
n_windows = 4
catalog_sorted = catalog.sort_values('time')
window_size = len(catalog_sorted) // n_windows

axes4 = axes4.flatten()
colors = ['steelblue', 'darkgreen', 'darkorange', 'purple']

for i in range(n_windows):
    start_idx = i * window_size
    end_idx = start_idx + window_size if i < n_windows - 1 else len(catalog_sorted)
    window_data = catalog_sorted.iloc[start_idx:end_idx]
    
    mags = window_data['magnitude'].values
    mags = mags[mags >= 1.0]  # Apply completeness threshold
    
    axes4[i].hist(mags, bins=np.arange(0.5, 4.5, 0.2), color=colors[i], alpha=0.7, edgecolor='black')
    axes4[i].set_xlabel('Magnitude')
    axes4[i].set_ylabel('Count')
    time_start = window_data['time'].min()
    time_end = window_data['time'].max()
    axes4[i].set_title(f'Window {i+1}: {time_start.strftime("%Y-%m")} to {time_end.strftime("%Y-%m")}\nN={len(mags)} events')
    axes4[i].grid(True, alpha=0.3)

plt.suptitle('Campi Flegrei - Magnitude Distribution Stability Across Time Windows', fontsize=14)
plt.tight_layout()
plt.savefig('figures/04_magnitude_distribution_stability.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: figures/04_magnitude_distribution_stability.png")

# ============================================================
# FIGURE 5: Multi-Signal Unrest Index
# ============================================================
print("Generating Figure 5: Multi-Signal Unrest Index...")

fig5, ax5 = plt.subplots(figsize=(12, 6))

# Plot unrest index
ax5.plot(unrest_df['time'], unrest_df['unrest_index'], 
         color='darkred', linewidth=2, label='Unrest Index')

# Add threshold lines
thresholds = {
    'NORMAL': unrest_df['unrest_index'].quantile(0.50),
    'ELEVATED': unrest_df['unrest_index'].quantile(0.75),
    'HIGH': unrest_df['unrest_index'].quantile(0.90),
    'CRITICAL': unrest_df['unrest_index'].quantile(0.95)
}

colors_threshold = {'NORMAL': 'green', 'ELEVATED': 'orange', 'HIGH': 'red', 'CRITICAL': 'darkred'}
linestyles = {'NORMAL': '--', 'ELEVATED': '-.', 'HIGH': '-', 'CRITICAL': '-'}

for state, value in thresholds.items():
    ax5.axhline(value, color=colors_threshold[state], linestyle=linestyles[state], 
                linewidth=1.5, alpha=0.8, label=f'{state} threshold ({value:.2f})')

# Color background by alert state
if 'state' in ew_df.columns:
    for idx, row in ew_df.iterrows():
        if pd.notna(row['unrest_index']):
            color_map = {'NORMAL': 'lightgreen', 'ELEVATED': 'lightyellow', 
                        'HIGH': 'lightsalmon', 'CRITICAL': 'lightcoral'}
            ax5.axvspan(row['time'], row['time'] + pd.Timedelta(days=1), 
                       color=color_map.get(row['state'], 'white'), 
                       alpha=0.2, linewidth=0)

ax5.set_xlabel('Date')
ax5.set_ylabel('Normalized Unrest Index')
ax5.set_title('Campi Flegrei - Composite Unrest Index with Alert Thresholds')
ax5.legend(loc='upper right', fontsize=9)
ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/05_unrest_index.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: figures/05_unrest_index.png")

# ============================================================
# FIGURE 6: Summary Dashboard (All-in-One)
# ============================================================
print("Generating Figure 6: Summary Dashboard...")

fig6, axes6 = plt.subplots(3, 2, figsize=(16, 14))

# Panel A: Map-like overview (seismicity rate)
axes6[0, 0].fill_between(daily_rate.index, daily_rate.values, alpha=0.4, color='steelblue')
axes6[0, 0].plot(daily_rate.index, daily_rate.rolling(7, min_periods=1).mean(), 
                 color='darkblue', linewidth=2)
axes6[0, 0].set_ylabel('Events/day')
axes6[0, 0].set_title('(A) Seismicity Rate')
axes6[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
axes6[0, 0].tick_params(axis='x', rotation=45)

# Panel B: Gutenberg-Richter
axes6[0, 1].scatter(bin_centers, log_counts, s=30, alpha=0.7, color='navy')
axes6[0, 1].plot(bin_centers, slope * np.array(bin_centers) + intercept, 'r-', linewidth=2)
axes6[0, 1].set_xlabel('Magnitude')
axes6[0, 1].set_ylabel('log₁₀ N(M)')
axes6[0, 1].set_title(f'(B) Gutenberg-Richter Fit (b={b_value_fit:.2f})')
axes6[0, 1].grid(True, alpha=0.3)

# Panel C: B-value evolution
axes6[1, 0].plot(b_value_df['time'], b_value_df['b_value'], color='darkgreen', linewidth=1.5)
axes6[1, 0].axhline(mean_b, color='gray', linestyle='--', linewidth=1.5)
axes6[1, 0].axhline(mean_b - std_b, color='orange', linestyle=':', linewidth=1, alpha=0.7)
axes6[1, 0].axhline(mean_b + std_b, color='orange', linestyle=':', linewidth=1, alpha=0.7)
axes6[1, 0].set_ylabel('b-value')
axes6[1, 0].set_title('(C) B-value Temporal Evolution')
axes6[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
axes6[1, 0].tick_params(axis='x', rotation=45)

# Panel D: Anomaly score
if 'anomaly_score' in anomalies_df.columns:
    axes6[1, 1].bar(anomalies_df['time'], anomalies_df['anomaly_score'], 
                    color='coral', alpha=0.7, width=1.0)
    axes6[1, 1].set_ylabel('Anomaly Score')
    axes6[1, 1].set_title('(D) Statistical Anomalies')
    axes6[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes6[1, 1].tick_params(axis='x', rotation=45)

# Panel E: Unrest index
axes6[2, 0].plot(unrest_df['time'], unrest_df['unrest_index'], color='darkred', linewidth=2)
for state, value in thresholds.items():
    axes6[2, 0].axhline(value, color=colors_threshold[state], linestyle='--', 
                        linewidth=1, alpha=0.7, label=state)
axes6[2, 0].set_ylabel('Unrest Index')
axes6[2, 0].set_title('(E) Composite Unrest Index')
axes6[2, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
axes6[2, 0].tick_params(axis='x', rotation=45)
axes6[2, 0].legend(fontsize=8, loc='upper right')

# Panel F: Alert states
if 'state' in ew_df.columns and 'alert_flag' in ew_df.columns:
    alert_dates = ew_df[ew_df['alert_flag'] == 1]['time']
    axes6[2, 1].plot(ew_df['time'], ew_df['unrest_index'], color='gray', alpha=0.5)
    if len(alert_dates) > 0:
        axes6[2, 1].scatter(alert_dates, ew_df.loc[ew_df['alert_flag'] == 1, 'unrest_index'],
                           c='red', s=150, marker='^', label='ALERT', zorder=5)
    else:
        # No alerts - show annotation
        axes6[2, 1].text(0.5, 0.5, 'No alerts triggered', 
                        transform=axes6[2, 1].transAxes, ha='center', va='center',
                        fontsize=12, color='green', alpha=0.7)
    axes6[2, 1].set_ylabel('Unrest Index')
    axes6[2, 1].set_title('(F) Early Warning Alerts')
    axes6[2, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes6[2, 1].tick_params(axis='x', rotation=45)
    axes6[2, 1].legend()

plt.suptitle('Campi Flegrei Quantitative Monitoring System - Summary Dashboard', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('figures/06_summary_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: figures/06_summary_dashboard.png")

print("\n" + "="*60)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("="*60)
print("\nGenerated files:")
for f in os.listdir('figures'):
    filepath = os.path.join('figures', f)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"  - {f} ({size_kb:.1f} KB)")
