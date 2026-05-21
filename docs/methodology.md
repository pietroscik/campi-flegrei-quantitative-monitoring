# Methodology — Campi Flegrei Quantitative Monitoring System

## 1. Overview

This project implements a statistical–computational framework for the analysis of volcanic seismicity in the Campi Flegrei caldera. The system integrates classical seismological laws with stochastic point-process modeling and multi-signal risk indicators.

The pipeline is designed to be reproducible, modular, and suitable for near-real-time monitoring simulations.

---

## 2. Data Source

Seismic events are retrieved from the INGV (Istituto Nazionale di Geofisica e Vulcanologia) FDSN web service.

**Data retrieval:**
- Endpoint: `https://webservices.ingv.it/fdsnws/event/1/query`
- Format: GeoJSON
- Spatial bounds: 40.75°N – 40.95°N, 14.05°E – 14.25°E (Campi Flegrei caldera)
- Minimum magnitude: M ≥ 0.0

Each event includes:
- origin time (UTC)
- magnitude (ML or Md)
- latitude and longitude
- depth (when available)
- unique event ID

**Quality control:**
1. Remove events with invalid timestamps
2. Exclude negative or missing magnitudes
3. Apply spatial bounding box filter
4. Remove duplicate event IDs
5. Compute derived features (energy proxy, temporal bins)

---

## 3. Gutenberg–Richter Law (b-value)

The frequency–magnitude distribution follows:

$$\log_{10} N(M) = a - bM$$

where:
- $N(M)$ = cumulative number of events with magnitude ≥ M
- $a$ = seismic productivity parameter
- $b$ = slope quantifying relative proportion of small vs. large events

**Estimation method:** Maximum Likelihood Estimation (Aki, 1965):

$$b = \frac{\log_{10}(e)}{\bar{M} - M_0}$$

where $\bar{M}$ is the mean magnitude and $M_0$ is the completeness magnitude.

**Temporal analysis:** Rolling window approach with N=100 events to capture stress regime variations.

---

## 4. ETAS Model

The Epidemic-Type Aftershock Sequence (ETAS) model describes seismicity rate as:

$$\lambda(t) = \mu + \sum_{i: t_i < t} K \cdot \exp[\alpha(M_i - M_0)] \cdot (t - t_i + c)^{-p}$$

Parameters:
- $\mu$: background seismicity rate (events/day)
- $K$: productivity coefficient
- $\alpha$: magnitude sensitivity of triggering
- $c$: time delay parameter (days)
- $p$: temporal decay exponent

**Estimation:** Maximum Likelihood Estimation (MLE) optimizes the log-likelihood function over the observation period.

---

## 5. Anomaly Detection

Two complementary methods identify b-value anomalies:

**Z-score method:**
$$Z(t) = \frac{b(t) - \mu_{window}}{\sigma_{window}}$$
Anomaly flagged when |Z| > 2 (rolling window = 50 samples)

**Quantile-based method:**
Anomalies defined as values below 5th percentile or above 95th percentile.

**Combined anomaly score:** Sum of Z-score and quantile indicators (range: 0–2).

---

## 6. Multi-Signal Unrest Index

A composite index integrates three independent signals:

1. **Seismicity rate**: Daily event count
2. **b-value**: Rolling estimate (inverse relationship: low b → high stress)
3. **Ground uplift**: Vertical displacement from GNSS station RITE

**Normalization:** Z-score standardization for each signal:
$$X_{norm} = \frac{X - \mu_X}{\sigma_X}$$

**Unrest Index formula:**
$$UI(t) = 0.4 \cdot Rate_{norm}(t) + 0.3 \cdot [-b_{norm}(t)] + 0.3 \cdot Uplift_{norm}(t)$$

---

## 7. Early Warning System

Dynamic thresholds based on empirical percentiles of the unrest index:

| State | Condition |
|-------|-----------|
| NORMAL | UI ≤ 50th percentile |
| ELEVATED | 50th < UI ≤ 75th percentile |
| HIGH | 75th < UI ≤ 90th percentile |
| CRITICAL | UI > 90th percentile |

**Alert criterion:** CRITICAL state with persistence > 60% over 7-day window.

---

## 8. Pipeline Architecture

```
INGV FDSN API → Raw Catalog → Quality Control → 
b-value Analysis → Anomaly Detection → 
Multi-Signal Fusion → Early Warning → ETAS MLE → Outputs
```

**Execution modes:**
- Full historical pipeline (default: 1 year of data)
- Worker cycle (periodic updates on recent data)
- Real-time stream (continuous monitoring for dashboard)

---

## 9. Output Products

| Module | Output File | Description |
|--------|-------------|-------------|
| Ingestion | `data/raw/ingv_events.csv` | Raw INGV catalog |
| Preprocessing | `data/processed/catalog_clean.csv` | Quality-controlled catalog |
| b-value | `data/processed/b_value_rolling.csv` | Temporal b-value series |
| Anomaly | `data/processed/b_value_anomalies.csv` | Anomaly scores |
| Multi-signal | `data/processed/unrest_index.csv` | Composite unrest index |
| Early Warning | `data/processed/early_warning_system.csv` | Alert states & flags |
| ETAS | `data/processed/etas_params.csv` | Fitted ETAS parameters |

**Figures:**
- `figures/01_seismicity_rate.png` - Seismicity rate evolution
- `figures/02_gutenberg_richter_fit.png` - GR frequency-magnitude distribution
- `figures/03_bvalue_evolution.png` - Temporal b-value with anomalies
- `figures/04_magnitude_distribution_stability.png` - Magnitude histograms by time windows
- `figures/05_unrest_index.png` - Composite unrest index with thresholds
- `figures/06_summary_dashboard.png` - All-in-one summary panel
- `figures/07_hybrid_comparison.png` - Statistical vs Deep Learning comparison

---

## 10. Deep Learning Extensions

### 10.1 LSTM for Non-Linear Forecasting

Long Short-Term Memory (LSTM) networks address limitations of linear models during rapid bradyseismic acceleration:

**Architecture:**
- Input: Sequence of length $L=30$ days with multi-dimensional features
- Hidden layers: 2 stacked LSTM layers (64 and 32 units) with Layer Normalization
- Output: Forecast horizon $H=7$ days
- Regularization: Dropout (0.2) and early stopping (patience=15)

**Training protocol:**
- Optimizer: Adam (learning rate = 0.001, ReduceLROnPlateau)
- Loss: Mean Squared Error (MSE)
- Validation: Expanding window cross-validation (5 folds)

### 10.2 Variational Autoencoder for Anomaly Detection

VAEs provide unsupervised detection of novel precursory patterns:

**Encoder architecture:**
- Input: 7-dimensional feature vector (seismicity rate, magnitude statistics, depth, energy)
- Hidden layers: [32, 16] with Batch Normalization and Dropout
- Latent space: 8 dimensions (probabilistic: mean + log-variance)

**Decoder architecture:**
- Symmetric to encoder
- Output: Reconstruction of input features

**Anomaly scoring:**
$$\text{Anomaly Score} = \frac{||x - \hat{x}||^2 - \mu_{error}}{\sigma_{error}}$$

Threshold set at 95th percentile of training reconstruction errors.

### 10.3 Hybrid Validation Framework

Recursive backtesting compares statistical and deep learning models:

**Metrics:**
- **RMSE, MAE**: Forecasting accuracy
- **AICc**: Model selection (statistical models only)
- **F1-Score**: Anomaly detection performance
- **Reconstruction Error**: VAE anomaly indicator

**Validation protocol:**
- Expanding window: minimum 100 samples training, 30 samples test
- 5 consecutive folds
- No look-ahead bias (strict temporal ordering)

---

## 11. Reproducibility

All analyses can be reproduced by executing:

```bash
pip install -r requirements.txt
python run_pipeline.py
python scripts/generate_paper_figures.py
```

Configuration parameters are specified in `config.yaml`.

---

## References

- Aki, K. (1965). Maximum likelihood estimate of b in the formula log N = a - bM and its confidence limits. *Bulletin of the Earthquake Research Institute*, 43, 237-239.
- Ogata, Y. (1988). Statistical models for earthquake occurrences and residual analysis for point processes. *Journal of the American Statistical Association*, 83(401), 9-27.
