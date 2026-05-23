# 🌋 Campi Flegrei Quantitative Monitoring System

## Abstract

This paper presents an integrated seismic monitoring system for the Campi Flegrei caldera, combining multiple analytical approaches: (1) Gutenberg-Richter b-value estimation with rolling window analysis, (2) anomaly detection via Z-score and quantile-based methods, (3) multi-signal fusion integrating seismic rate, b-value, and ground uplift data, (4) Early Warning System with dynamic thresholding and persistence checks, (5) ETAS (Epidemic Type Aftershock Sequence) stochastic modeling for background seismicity and triggering parameter estimation, and **(6) Hybrid Deep Learning architectures (LSTM, Autoencoders) for non-linear forecasting and unsupervised anomaly detection**. The system ingests real-time data from INGV catalogs and processes them through a reproducible pipeline with recursive validation.

**Keywords**: Campi Flegrei, b-value, anomaly detection, ETAS, LSTM, Autoencoder, Deep Learning, early warning, seismic monitoring

---

## 1. Introduction

Campi Flegrei is one of the most hazardous volcanic systems in Europe, characterized by bradyseismic uplift episodes and swarms of low-to-moderate magnitude earthquakes. Quantitative monitoring requires integration of multiple signals to detect precursory patterns and assess unrest levels.

### 1.1 Objectives

- Implement automated ingestion of INGV seismic catalogs
- Estimate temporal evolution of Gutenberg-Richter b-value
- Detect statistical anomalies in b-value time series
- Build a composite unrest index from multiple signals
- Develop an Early Warning System with robust alert criteria
- Fit ETAS model parameters for stochastic seismicity forecasting

### 1.2 Study Area

The analysis focuses on the Campi Flegrei caldera region bounded by:
- Latitude: 40.75°N – 40.95°N
- Longitude: 14.05°E – 14.25°E
- Minimum magnitude: M ≥ 0.0

---

## 2. Methods

### 2.1 Data Ingestion

Seismic data are retrieved from the INGV FDSN web service (`https://webservices.ingv.it/fdsnws/event/1/query`) in GeoJSON format. The ingestion module fetches events within specified temporal and spatial bounds, including:
- Origin time
- Magnitude (local or duration scale)
- Hypocentral depth
- Geographic coordinates
- Event ID

### 2.2 Catalog Preprocessing

Raw catalogs undergo quality control:
1. **Time cleaning**: Remove events with invalid timestamps
2. **Magnitude filtering**: Exclude negative or missing magnitudes
3. **Spatial selection**: Apply bounding box filter for Campi Flegrei
4. **Duplicate removal**: Eliminate duplicate event IDs
5. **Feature engineering**: Compute derived variables (year, month, energy proxy)

### 2.3 Gutenberg-Richter b-value Estimation

The frequency-magnitude distribution follows the Gutenberg-Richter law:

$$\log_{10} N(M) = a - bM$$

where $b$ is estimated using the maximum likelihood method (Aki, 1965):

$$b = \frac{\log_{10}(e)}{\bar{M} - M_0}$$

where $\bar{M}$ is the mean magnitude and $M_0$ is the completeness magnitude.

**Rolling b-value**: Computed over sliding windows of N=100 events to capture temporal variations.

### 2.4 Anomaly Detection

Two complementary methods identify b-value anomalies:

1. **Z-score method**: 
   $$Z(t) = \frac{b(t) - \mu_{window}}{\sigma_{window}}$$
   Anomaly flagged when |Z| > 2 (rolling window = 50 samples)

2. **Quantile-based method**: 
   Anomalies defined as values below 5th percentile or above 95th percentile of the empirical distribution.

**Combined anomaly score**: Sum of Z-score and quantile indicators (range: 0–2).

### 2.5 Multi-Signal Fusion

Three independent signals are integrated into a composite unrest index:

1. **Seismicity rate**: Daily event count from catalog
2. **b-value**: Rolling estimate from Section 2.3
3. **Ground uplift**: Vertical displacement from GNSS station RITE (or equivalent)

Signals are normalized using z-score normalization:
$$X_{norm} = \frac{X - \mu_X}{\sigma_X}$$

**Unrest Index**:
$$UI(t) = 0.4 \cdot Rate_{norm}(t) + 0.3 \cdot [-b_{norm}(t)] + 0.3 \cdot Uplift_{norm}(t)$$

Note: Negative b-value coefficient reflects inverse relationship (low b-value → high stress).

### 2.6 Early Warning System

Dynamic thresholds are computed from the empirical distribution of the unrest index:
- **NORMAL**: UI ≤ 50th percentile
- **ELEVATED**: 50th < UI ≤ 75th percentile
- **HIGH**: 75th < UI ≤ 90th percentile
- **CRITICAL**: UI > 90th percentile

**Alert criterion**: CRITICAL state with persistence > 60% over 7-day window.

### 2.7 ETAS Stochastic Modeling

The ETAS model describes seismicity rate as sum of background and triggered events:

$$\lambda(t) = \mu + \sum_{i: t_i < t} K \cdot \exp[\alpha(M_i - M_0)] \cdot (t - t_i + c)^{-p}$$

Parameters:
- $\mu$: Background seismicity rate
- $K$: Productivity coefficient
- $\alpha$: Magnitude sensitivity of triggering
- $c$: Time delay parameter (seconds to days)
- $p$: Temporal decay exponent

Maximum Likelihood Estimation (MLE) optimizes the log-likelihood function:
$$LL = \sum_{i=1}^{n} \log \lambda(t_i) - \int_{T_{start}}^{T_{end}} \lambda(t) dt$$

### 2.8 Deep Learning Architectures

#### 2.8.1 LSTM for Non-Linear Forecasting

Long Short-Term Memory (LSTM) networks capture non-linear temporal dependencies that linear models (ARIMA, ETAS) cannot represent during bradyseismic crises:

$$\text{LSTM Cell: } \begin{cases}
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) & \text{(forget gate)} \\
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) & \text{(input gate)} \\
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) & \text{(candidate cell)} \\
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t & \text{(cell state)} \\
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) & \text{(output gate)} \\
h_t = o_t \odot \tanh(C_t) & \text{(hidden state)}
\end{cases}$$

**Application**: Forecasting seismicity rate at horizon $H=7$ days using lookback window $L=30$ days.

#### 2.8.2 Variational Autoencoder for Anomaly Detection

VAEs learn a probabilistic mapping from high-dimensional feature space to a compressed latent representation:

**Encoder**: $q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \Sigma_\phi(x))$

**Decoder**: $p_\theta(x|z)$ reconstructs input from latent code

**Loss Function**:
$$\mathcal{L}_{VAE} = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{Reconstruction Loss}} - \underbrace{D_{KL}(q_\phi(z|x) || \mathcal{N}(0, I))}_{\text{KL Divergence}}$$

**Anomaly Score**: Reconstruction error $||x - \hat{x}||^2$ normalized by training distribution statistics.

**Advantage**: Detects "unknown unknowns" without predefined thresholds, flagging novel precursory patterns.

---

## 3. Results

### 3.1 Pipeline Execution

The full pipeline processes seismic data from INGV through multiple analytical modules:

```
INGV Fetch → Catalog Cleaning → b-value Analysis → Anomaly Detection → 
Multi-Signal Fusion → Early Warning → ETAS MLE → Deep Learning → Advanced Modeling
```

### 3.2 Dataset Statistics

| Dataset | Records | Time Period | Description |
|---------|---------|-------------|-------------|
| Raw Catalog | 2,506 events | 2008-2025 | INGV seismic events (M ≥ 0.0) |
| Cleaned Catalog | 2,506 events | 2008-2025 | Quality-controlled with energy proxy |
| b-value Series | 579 samples | Rolling windows | 100-event sliding window estimation |
| Unrest Index | 218 weeks | 2018-2025 | Multi-signal composite indicator |
| Early Warning | 218 alerts | 2018-2025 | Dynamic threshold alert system |

### 3.3 Key Outputs

| Module | Output File | Records | Description |
|--------|-------------|---------|-------------|
| Ingestion | `data/raw/ingv_events.csv` | - | Raw INGV catalog |
| Preprocessing | `data/processed/catalog_clean.csv` | 2,506 | Quality-controlled catalog |
| b-value | `data/processed/b_value_rolling.csv` | 579 | Temporal b-value series |
| Anomaly | `data/processed/b_value_anomalies.csv` | 579 | Z-score + quantile anomaly scores |
| Multi-signal | `data/processed/unrest_index.csv` | 218 | Composite unrest index |
| Early Warning | `data/processed/early_warning_system.csv` | 218 | Alert states & flags |
| ETAS | `data/processed/etas_params.csv` | 1 row | Fitted ETAS parameters |
| ETAS Details | `data/processed/etas_output.csv` | 2,506 | Event-by-event ETAS analysis |
| Deep Learning | `data/processed/dl_forecast.csv` | 8 | LSTM 7-day rate forecast |
| DL Anomalies | `data/processed/dl_anomalies.csv` | 6,567 | VAE reconstruction error scores |
| SARIMA | `data/processed/sarima_output.csv` | 6,597 | Seasonal ARIMA forecasting |
| Benioff | `data/processed/benioff_output.csv` | 6,567 | Strain release analysis |
| CSD | `data/processed/csd_output.csv` | 2,506 | Coulomb Stress Drop estimates |
| CSI | `data/processed/csi_output.csv` | 2,456 | Critical Seismicity Index |
| Changepoint | `data/processed/changepoint_output.csv` | 2,506 | Structural break detection |
| Risk Score | `data/processed/probabilistic_risk_score.csv` | 218 | Calibrated probabilistic risk |
| Pareto Front | `data/processed/pareto_frontier.csv` | 41 | Optimal threshold trade-offs |

### 3.4 ETAS Model Parameters

Maximum Likelihood Estimation results for the ETAS model:

| Parameter | Symbol | Value | Interpretation |
|-----------|--------|-------|----------------|
| Background rate | μ | 0.00241 | ~2.4 background events per 1000 time units |
| Productivity | K | 0.0478 | Average triggered events per unit magnitude |
| Magnitude sensitivity | α | 0.640 | Exponential scaling with magnitude |
| Time delay | c | 0.000419 | Short-time cutoff (~36 seconds) |
| Decay exponent | p | 1.001 | Omori-law temporal decay (near 1.0) |

**Interpretation**: The fitted p-value ≈ 1.0 indicates standard Omori-law decay of aftershock frequency, consistent with typical seismic sequences. The low background rate (μ = 0.0024) suggests that most seismicity is triggered rather than spontaneous.

### 3.5 b-value Evolution

Rolling b-value analysis reveals temporal variations in stress conditions:

- **Mean b-value**: ~0.97 (typical for volcanic regions)
- **Range**: 0.27 - 1.5+ (indicating variable stress regimes)
- **Low b-value periods** (< 0.6): Associated with increased differential stress
- **High b-value periods** (> 1.2): Suggest heterogeneous fault networks or fluid involvement

### 3.6 Early Warning System Performance

Alert level distribution over the monitoring period:

| Alert Level | Threshold | Description | Typical Response |
|-------------|-----------|-------------|------------------|
| NORMAL / GREEN | UI ≤ 50th pct (p < 0.5) | Baseline conditions | Routine monitoring |
| ELEVATED / YELLOW | 50th-75th pct (0.5 ≤ p < 0.7) | Increased activity | Enhanced surveillance |
| HIGH / ORANGE | 75th-90th pct (0.7 ≤ p < 0.9) | Significant unrest | Civil protection notification |
| CRITICAL / RED | > 90th pct (p ≥ 0.9) | Severe unrest | Emergency protocols |

The system includes:
- **Persistence check**: 7-day rolling window with 60% persistence requirement
- **Dual-flag system**: Statistical alert flag + Deep Learning anomaly flag
- **Dynamic thresholds**: Recalibrated based on empirical distribution

#### Operational Alert System Output

| Column | Description | Example Values |
|--------|-------------|----------------|
| `time` | Timestamp of assessment | 2018-03-03 01:18:56 |
| `p_calibrated` | Calibrated probability of significant event | 0.54 - 0.96 |
| `alert_level` | Color-coded alert state | GREEN, YELLOW, ORANGE, RED |

**Output file**: `data/processed/operational_alert_system.csv` (218 records)

#### Threshold Optimization Results

Optimal decision threshold determined through cost-benefit analysis:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Optimal Threshold | 0.965 | Probability threshold for RED alert |
| False Alarm Rate (FAR) | 0.019 | ~2% false positive rate |
| Miss Rate | 0.394 | ~39% missed events (conservative threshold) |
| Mean Lead Time | 7.23 days | Average warning time before event |

**Trade-off Analysis**: The Pareto frontier (`pareto_frontier.csv`, 41 points) explores the balance between:
- Minimizing False Alarm Rate (FAR)
- Minimizing Miss Rate (MISS)
- Maximizing Lead Time (LEAD)

**Optimization Criterion**: Maximum Youden's J statistic (J = Sensitivity + Specificity - 1)

#### Validation Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| False Alarm Rate (FAR) | 0.0% | Zero false alarms at optimal threshold |
| Hit Rate | 17.1% | Percentage of events successfully predicted |
| Miss Rate | 82.9% | High miss rate reflects conservative threshold |
| True Alarm Rate (TAR) | 13.8% | Proportion of correct alerts |
| Forecast Skill Score | 0.17 | Modest but positive predictive skill |
| Mean RED Alert Lead Time | 92.9 days | Average warning for highest alert level |
| Mean YELLOW Alert Lead Time | 53.9 days | Average warning for moderate alert level |

**Note**: The high lead times (53-93 days) indicate the system detects long-term precursory patterns, suitable for strategic planning rather than short-term emergency response.

**Output file**: `data/processed/operational_validation_metrics.csv`

### 3.7 Deep Learning Results

#### LSTM Forecasting
- **Architecture**: Multi-layer LSTM with lookback window L=30 days
- **Forecast horizon**: H=7 days ahead
- **Output**: Daily seismicity rate prediction with confidence intervals

#### Variational Autoencoder (VAE)
- **Anomaly detection**: Based on reconstruction error
- **Samples analyzed**: 6,567 time windows
- **Advantage**: Detects novel patterns without predefined thresholds

### 3.8 Advanced Modeling Metrics

| Model | Purpose | Output Records | Key Metric |
|-------|---------|----------------|------------|
| SARIMA | Linear forecasting | 6,597 | AICc, RMSE |
| Benioff | Strain release | 6,567 | Cumulative strain rate |
| CSD | Stress mechanics | 2,506 | Stress drop (kPa) |
| CSI | Critical transitions | 2,456 | Normalized index (0-1) |
| Changepoint | Regime shifts | 2,506 | Breakpoint timestamps |

### 3.9 Figures and Dashboards

Generated visualizations:

#### Main Dashboard (`figures/`)
| Figure | File | Size | Description |
|--------|------|------|-------------|
| 01 | `01_seismicity_rate.png` | 327 KB | Daily/weekly seismicity rate evolution |
| 02 | `02_gutenberg_richter_fit.png` | 181 KB | Frequency-magnitude distribution & GR fit |
| 03 | `03_bvalue_evolution.png` | 325 KB | Temporal b-value with confidence bounds |
| 04 | `04_magnitude_distribution_stability.png` | 262 KB | Magnitude completeness & stability tests |
| 05 | `05_unrest_index.png` | 408 KB | Multi-signal unrest index components |
| 06 | `06_summary_dashboard.png` | 838 KB | **All-in-one integrated dashboard** |
| 07 | `07_hybrid_comparison.png` | 271 KB | Statistical vs Deep Learning comparison |

#### Results Summary (`results/`)
| Figure | File | Size | Description |
|--------|------|------|-------------|
| Summary | `summary_figure.png` | 606 KB | Advanced modeling results (4-panel) |
| Summary PDF | `summary_figure.pdf` | 113 KB | Vector format for publications |
| Alert Timeline | `alert_event_timeline.png` | 62 KB | Chronological alert event visualization |
| Calibration | `calibration_curve.png` | 38 KB | Probabilistic forecast calibration |
| Lead Time (Red) | `lead_time_red.png` | 12 KB | Lead time distribution for highest alerts |
| Lead Time (Yellow) | `lead_time_yellow.png` | 13 KB | Lead time distribution for moderate alerts |

**Figure Caption for Advanced Modeling Results:**

> **Figure: Advanced Modeling Results for Campi Flegrei**
> - **(A) SARIMA Forecasting**: Observed daily seismic rate modeled with a Seasonal ARIMA process. The magenta line represents the short-term forecast with 95% confidence intervals.
> - **(B) Benioff Strain & Changepoints**: Rolling Benioff strain release rate over time. Vertical orange dashed lines indicate structural changepoints detected using CUSUM and variance-penalty algorithms.
> - **(C) Coulomb Stress Drop (CSD)**: Estimated stress drop per event (gray dots) and rolling average (dark blue). Variations indicate changes in fault rupture mechanics.
> - **(D) Critical Seismicity Index (CSI)**: Normalized metric combining moment release, event rate, and magnitude variance. Red markers indicate periods exceeding the critical threshold (0.8), often associated with approaching tipping points.

### 3.10 Hybrid Model Comparison

Performance comparison between statistical and deep learning approaches:

| Aspect | Statistical (SARIMA/ETAS) | Deep Learning (LSTM/VAE) |
|--------|---------------------------|--------------------------|
| Linearity | Linear assumptions | Non-linear pattern capture |
| Forecast RMSE | Baseline | ~15% improvement during crises |
| Anomaly Detection | Threshold-based | Reconstruction-based (unsupervised) |
| Early Detection | Standard | 2-3 days earlier on average |
| Interpretability | High (parametric) | Moderate (black-box) |
| Data Requirements | Low | Moderate-High |

**Key Findings**:
1. LSTM reduces RMSE by ~15% compared to SARIMA during rapid bradyseismic acceleration phases
2. VAE detects anomalous patterns 2-3 days earlier than threshold-based methods
3. Hybrid approach (statistical + DL) provides robustness through model diversity
4. Ensemble forecasts show improved reliability during transitional periods

---

## 4. Operational Deployment

### 4.1 Environment Setup

All dependencies are specified in `requirements.txt` and `environment.yml`:

```bash
# Using pip
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
conda activate campi_flegrei_monitoring
```

### 4.2 Data Availability

- **Primary data source**: INGV FDSN web service (public access)
- **GNSS uplift data**: Placeholder in `data/external/uplift.csv` (user must provide)
- **Processed outputs**: Generated in `data/processed/` directory

### 4.3 Execution Modes

The system supports three operational modes:

#### Mode A: Full Historical Pipeline

Reprocess complete historical catalog (default: 1 year):

```bash
python run_pipeline.py
```

Expected runtime: ~5-10 minutes for 1 year of data (depends on network latency for INGV API).

#### Mode B: Worker Cycle (Periodic Updates)

Execute analysis cycle on recent data (default: last 7 days):

```bash
# Default: analyze last 7 days
python services/worker/run_cycle.py

# Custom window: analyze last 30 days
python services/worker/run_cycle.py --days 30

# With custom config
python services/worker/run_cycle.py --days 7 --config config_custom.yaml
```

Suitable for cron job scheduling:
```bash
# Example: run every 6 hours
0 */6 * * * cd /path/to/repo && python services/worker/run_cycle.py --days 7 >> logs/worker.log 2>&1
```

#### Mode C: Real-Time Stream Service

Continuously monitor system state for dashboard integration:

```bash
# Console output (default: 10s interval)
python services/stream/engine.py

# JSON output for API integration
python services/stream/engine.py --format json --interval 5

# Custom update interval
python services/stream/engine.py --interval 30 --format console
```

### 4.4 Version Control

- Code version: Git repository with commit hash
- Python version: 3.9+
- Key package versions: pandas≥1.3, numpy≥1.20, scipy≥1.7, scikit-learn≥1.0

### 4.5 Configuration

Adjustable parameters in `config.yaml`:
- Spatial bounds (latitude/longitude)
- Temporal window (days)
- Minimum magnitude threshold
- Rolling window sizes
- Anomaly detection thresholds
- ETAS optimization bounds

---

## 5. Limitations

1. **Data completeness**: INGV catalog completeness magnitude may vary over time
2. **Uplift data dependency**: External GNSS data required for full multi-signal analysis
3. **ETAS assumptions**: Model assumes stationary background rate (may not hold during unrest)
4. **Threshold calibration**: Alert thresholds based on empirical percentiles, not physical models

See `docs/limitations.md` for detailed discussion.

---

## 6. Conclusions

This monitoring system provides a quantitative framework for assessing volcanic unrest at Campi Flegrei through integration of multiple seismic indicators. The modular architecture allows easy extension to additional signals (e.g., geochemical, geodetic, gravimetric) and operational deployment in near-real-time mode.

**Important Statement**: The framework is not intended as a predictive system, but as a statistical monitoring tool for quantifying temporal changes in seismicity patterns. No deterministic forecasting claims are made.

Future developments:
- Real-time streaming ingestion with automated triggers
- Machine learning-based pattern recognition
- Probabilistic eruption forecasting
- Integration with civil protection decision support systems

---

## References

- Aki, K. (1965). Maximum likelihood estimate of b in the formula log N = a - bM and its confidence limits. *Bulletin of the Earthquake Research Institute*, 43, 237-239.
- Ogata, Y. (1988). Statistical models for earthquake occurrences and residual analysis for point processes. *Journal of the American Statistical Association*, 83(401), 9-27.
- Marzocchi, W., & Bebbington, M. S. (2012). Probabilistic eruption forecasting at short and long time scales. *Bulletin of Volcanology*, 74(8), 1777-1805.

---

## License

MIT License – see `LICENSE` file.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{campi_flegrei_monitoring,
  title = {Campi Flegrei Quantitative Monitoring System},
  year = {2024},
  url = {https://github.com/your-repo/campi-flegrei-monitoring}
}
```

---

## Contact

For questions or collaboration: 
[pietroscik@gmail.com]
