# Campi Flegrei Quantitative Monitoring System

## Abstract

This paper presents an integrated seismic monitoring system for the Campi Flegrei caldera, combining multiple analytical approaches: (1) Gutenberg-Richter b-value estimation with rolling window analysis, (2) anomaly detection via Z-score and quantile-based methods, (3) multi-signal fusion integrating seismic rate, b-value, and ground uplift data, (4) Early Warning System with dynamic thresholding and persistence checks, and (5) ETAS (Epidemic Type Aftershock Sequence) stochastic modeling for background seismicity and triggering parameter estimation. The system ingests real-time data from INGV catalogs and processes them through a reproducible pipeline.

**Keywords**: Campi Flegrei, b-value, anomaly detection, ETAS, early warning, seismic monitoring

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

---

## 3. Results

### 3.1 Pipeline Execution

The full pipeline processes ~1 year of INGV data through 7 sequential modules:

```
INGV Fetch → Catalog Cleaning → b-value Analysis → Anomaly Detection → 
Multi-Signal Fusion → Early Warning → ETAS MLE
```

### 3.2 Key Outputs

| Module | Output File | Description |
|--------|-------------|-------------|
| Ingestion | `data/raw/ingv_events.csv` | Raw INGV catalog |
| Preprocessing | `data/processed/catalog_clean.csv` | Quality-controlled catalog |
| b-value | `data/processed/b_value_rolling.csv` | Temporal b-value series |
| Anomaly | `data/processed/b_value_anomalies.csv` | Anomaly scores |
| Multi-signal | `data/processed/unrest_index.csv` | Composite unrest index |
| Early Warning | `data/processed/early_warning_system.csv` | Alert states & flags |
| ETAS | `data/processed/etas_params.csv` | Fitted ETAS parameters |

### 3.3 Figure: Synthetic Results Summary

See `results/summary_figure.png` for visualization of:
- (A) Map of seismicity with b-value spatial distribution
- (B) Temporal evolution of b-value with anomaly highlights
- (C) Multi-signal unrest index with threshold bands
- (D) ETAS model fit comparison (observed vs. modeled rate)

---

## 4. Reproducibility Statement

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

### 4.3 Pipeline Execution

Full reproducibility achieved by running:

```bash
python run_pipeline.py
```

Expected runtime: ~5-10 minutes for 1 year of data (depends on network latency for INGV API).

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

Future developments:
- Real-time streaming ingestion
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

For questions or collaboration: [your-email@example.com]
